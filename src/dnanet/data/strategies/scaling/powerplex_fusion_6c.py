from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from dnanet.data.preprocessing.peaks import find_peaks_above_threshold, find_peak_near_idx
from dnanet.data.strategies.scaling.kit import PPF6C_KIT, STRKit
from dnanet.data.strategies.scaling.scaling import (
    ScalingStrategy,
    SizeStandardParseResult,
)

# Indices of the size standard peaks that are higher than the other peaks.
HIGH_PEAKS_IDXS = (2, 7, 11, 15, 19)
# The expected difference between size standard peaks, based on annotated size standard lanes.
MEAN_DIFFS = [
    162.8750, 215.1458, 204.6667, 208.2917, 208.1042, 203.7500, 202.4583, 248.0417, 241.8958, 237.8750,
    242.1458, 233.9375, 226.0000, 229.6458, 218.5000, 223.0417, 208.9167, 210.0208, 205.5000
]


class PowerPlexFusion6CStrategy(ScalingStrategy):
    """Scaling strategy for the PowerPlex Fusion 6C kit.

    PPF6C has 6 dye channels total: 5 analysis dyes + 1 size standard.
    HID dye indices: 1,2,3,4,6 (5 is skipped).
    """

    def __init__(
        self, kit: STRKit | None = None, scanpoint_resolution: int = 4096, **kwargs
    ) -> None:
        if kit is None:
            kit = PPF6C_KIT
        super().__init__(
            kit, basepair_start=65, basepair_end=475, scanpoint_resolution=scanpoint_resolution
        )

    def marker_name_to_dye_idx(self) -> dict[str, int]:
        return {
            'AMEL': 0,
            'D3S1358': 0,
            'D1S1656': 0,
            'D2S441': 0,
            'D10S1248': 0,
            'D13S317': 0,
            'Penta E': 0,
            'D16S539': 1,
            'D18S51': 1,
            'D2S1338': 1,
            'CSF1PO': 1,
            'Penta D': 1,
            'TH01': 2,
            'vWA': 2,
            'D21S11': 2,
            'D7S820': 2,
            'D5S818': 2,
            'TPOX': 2,
            'D8S1179': 3,
            'D12S391': 3,
            'D19S433': 3,
            'SE33': 3,
            'D22S1045': 3,
            'DYS391': 4,
            'FGA': 4,
            'DYS576': 4,
            'DYS570': 4,
        }

    def cache_signature(self) -> dict:
        return {'class': self.__class__.__name__, 'scanpoint_resolution': self._scanpoint_resolution}

    def parse_size_standard(self, size_standard_lane: np.ndarray) -> SizeStandardParseResult | None:
        """Parse WEN ILS size standard from the PPF6C kit."""
        lane = np.asarray(size_standard_lane).reshape(-1)
        expected_bps = self.kit.size_standard.expected_bps

        peak_idxs = self._extract_ss_peaks(lane)
        if not self._validate_ss_peaks(peak_idxs, expected_bps):
            return None

        rescaled_indices, scaler = self.interpolate(peak_idxs, expected_bps, lane)
        return SizeStandardParseResult(
            rescaled_indices=rescaled_indices, scaler=scaler, fit_error=0.0
        )

    def _extract_ss_peaks(self, ss_lane: np.ndarray) -> np.ndarray:
        """Extract the size standard peaks from the size standard dye lane based on the expected differences between peaks.

        The algorithm is as follows:
        1. We want to find peaks from right to left, therefore we start by finding the final peak.
        2. Based on the expected difference between peaks, we try to find a peak within the expected search range.
        3. We apply correction to the extracted peaks, as sometimes artefacts are found instead of the actual size standard
        peak, or we found dipped or flat peak. We try to select the peak based on the mean of the peak heights, as the
        size standard peaks are expected to have similar height.
        """
        # We want to find the last peak, as this peak is often easy to find.
        last_two_peaks = find_peaks_above_threshold(ss_lane, 80)[-2:]
        if last_two_peaks.size < 2:
            return np.array([])
        # The distance between the final peaks should be around the mean difference. But sometimes there is a peak at
        # the end with larger distance to the ILS peaks. We do not want to select this peak but the peak following that.
        final_peak_idx = last_two_peaks[0] if last_two_peaks[1] - last_two_peaks[0] > MEAN_DIFFS[-1] * 1.5 else last_two_peaks[1]
        found_peak_idxs = self.find_peaks_by_mean_differences(ss_lane, final_peak_idx)

        if np.unique(found_peak_idxs).size != found_peak_idxs.size:
            # We found the same peak twice, and missed a different peak we were actually looking for. Quit parsing.
            return np.array([])

        mean_high_peaks, mean_low_peaks = compute_mean_high_low_peak_heights(ss_lane, found_peak_idxs)
        if mean_high_peaks < mean_low_peaks:
            # The SS has peaks at specific location that are higher than the other. If the mean of the higher peaks is
            # below the mean of the lower peaks, there is probably an issue with the ILS. Quit parsing.
            return np.array([])

        found_peak_idxs = self.correct_extracted_peaks(ss_lane, found_peak_idxs)
        return np.array(found_peak_idxs[-20:-1])  # last 19 peaks, excluding final

    @staticmethod
    def find_peaks_by_mean_differences(ss_lane: np.ndarray, final_peak_location: float) -> np.ndarray:
        """Find peaks from right to left by searching within a range around the expected location.

        The expected location is based on the mean difference between peaks that are retrieved from size standard
        annotations.
        """
        found_peak_idxs = [final_peak_location]
        for peak_nr in range(-1, -20, -1):  # Iterate from right to left.
            prev_peak_idx = found_peak_idxs[-1]
            # Find a search range based on the mean of the difference between peaks. Get the rfu values in this range.
            start, end = prev_peak_idx - MEAN_DIFFS[peak_nr] - 45, prev_peak_idx - MEAN_DIFFS[peak_nr] + 45
            search_range = np.arange(int(start), int(end) + 1)
            rfu_values = ss_lane[search_range]
            # Search for a peak index with high rfu as there is probably a peak close.
            peak_idx_max_rfu = search_range[np.argmax(rfu_values)]
            # Find a peak near the maximum rfu, which is often already the peak we are looking for, but sometimes
            # the peak is just around the border of the index_range. We do demand the peak being above half the mean
            # peak heights we already found.
            found_peak_idx = find_peak_near_idx(ss_lane.flatten(), peak_idx_max_rfu, float(np.mean(ss_lane[found_peak_idxs]) / 2))
            found_peak_idxs.append(int(found_peak_idx[0]))

        # Sort the peaks from left to right, so they are in order.
        return np.sort(found_peak_idxs)

    @staticmethod
    def correct_extracted_peaks(ss_lane: np.ndarray, found_peak_idxs: list[float]) -> list[float]:
        """Apply correction and validation to the extracted peaks. We apply some simple checks to check for
        abnormalities. If we found any, an empty list will be returned. This will lead to invalidation in a later stage.

        The following problems might occur:
        - We found a peak that is way too low. Then quit parsing as this ILS is probably too hard to parse.
        - We found a peak that is way too high in comparison with the other found peaks. Find the peak that
        matches the other peaks better.
        - We found a dipped (or flat) peak, then find the actual peak top.
        """
        W = 45  # window size
        for i, found_idx in enumerate(found_peak_idxs):
            mean_all_peaks = np.mean(ss_lane[found_peak_idxs])
            mean_high_peaks, mean_low_peaks = compute_mean_high_low_peak_heights(ss_lane, found_peak_idxs)
            # Find all peaks above half of the mean peak heights, within a range around the found peak.
            new_idx, new_height = find_peaks(ss_lane[found_idx - W:found_idx + W + 1], mean_all_peaks / 2)
            if new_idx.size > 1:  # Multiple peaks found
                diffs = np.diff(new_idx)
                if all(diffs < 5):  # Probably just a flat or dipped peak, select the actual peak top.
                    arg = np.argmax(new_height['peak_heights'])
                else:
                    # We have multiple peaks, select the one closest to the mean of the low/high peaks.
                    mean = mean_high_peaks if i in HIGH_PEAKS_IDXS else mean_low_peaks
                    arg = np.argmin(abs(mean - new_height['peak_heights']))
                corrected_idx = new_idx[arg] + found_idx - W
            elif new_idx.size == 1:
                # We found the peak we already got, or just +-1 pixel distance (flat peak).
                corrected_idx = new_idx[0] + found_idx - W
            else:
                # We found no peak in the window above the threshold, this ILS is probably too complicated. Quit parsing.
                return []

            # We apply simple checks to check for abnormal/unexpected peak heights.
            if i not in HIGH_PEAKS_IDXS and ss_lane[corrected_idx] > mean_high_peaks:
                # A small peak should not be higher than a high peak. Quit parsing.
                return []
            if ss_lane[corrected_idx] > 2 * mean_all_peaks:
                # Peaks should not be abnormally high in general. Quit parsing.
                return []
            found_peak_idxs[i] = corrected_idx
        return found_peak_idxs

    # This function is comment out, because it will only be used to rescale profiles that are already parsed.
    # When that is done, this function will be removed.
    # @staticmethod
    # def _extract_ss_peaks(signal: np.ndarray, threshold=180) -> np.ndarray:
    #     """Find size-standard peaks with adaptive thresholding for tail peaks."""
    #     peak_idxs = find_peaks_above_threshold(signal, threshold)
    #
    #     # The final two SS peaks are often lower — search with reduced threshold
    #     split_idx = 8200
    #     if len(peak_idxs) > 0 and peak_idxs[-1] <= split_idx:
    #         tail_peaks = find_peaks_above_threshold(signal[split_idx:], 120) + split_idx
    #         peak_idxs = np.union1d(peak_idxs, tail_peaks)
    #
    #     # Remove flat/close peaks (within 15 scan points)
    #     close = np.where(np.diff(peak_idxs) <= 15)[0]
    #     return np.delete(peak_idxs, close)

    @staticmethod
    def _validate_ss_peaks(peak_idxs: np.ndarray, expected_bps: np.ndarray) -> bool:
        """Validate that detected peaks match expected size standard pattern."""
        if peak_idxs.size != 19:
            return False

        peak_dists = np.abs(np.diff(peak_idxs))
        bp_dists = np.abs(np.diff(expected_bps))
        relative = peak_dists / bp_dists

        # Pixels-per-basepair should be between 7 and 13
        return bool(np.all((relative <= 13) & (relative >= 7)))

def compute_mean_high_low_peak_heights(array: np.ndarray, peak_idxs: list[float]) -> tuple[float, float]:
    """Compute the mean peak heights for the `high` and `low` peaks of a size standard separately."""
    peak_heights = array[peak_idxs]
    mean_high_peaks = np.mean([peak_heights[ind] for ind in range(19) if ind in HIGH_PEAKS_IDXS])
    mean_low_peaks = np.mean([peak_heights[ind] for ind in range(19) if ind not in HIGH_PEAKS_IDXS])
    return float(mean_high_peaks), float(mean_low_peaks)
