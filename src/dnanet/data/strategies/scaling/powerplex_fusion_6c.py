from __future__ import annotations

import numpy as np

from dnanet.data.preprocessing.peaks import find_peaks_above_threshold
from dnanet.data.strategies.scaling.kit import PPF6C_KIT, STRKit
from dnanet.data.strategies.scaling.scaling import (
    ScalingStrategy,
    SizeStandardParseResult,
)


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

        peak_idxs = self._extract_ss_peaks(lane, 120)
        peak_idxs = peak_idxs[-20:-1]  # last 19 peaks, excluding final

        if not self._validate_ss_peaks(peak_idxs, expected_bps):
            return None
        print(peak_idxs, lane[peak_idxs])
        rescaled_indices, scaler = self.interpolate(peak_idxs, expected_bps, lane)
        return SizeStandardParseResult(
            rescaled_indices=rescaled_indices, scaler=scaler, fit_error=0.0
        )

    @staticmethod
    def _extract_ss_peaks(signal: np.ndarray, threshold=180) -> np.ndarray:
        """Find size-standard peaks with adaptive thresholding for tail peaks."""
        peak_idxs = find_peaks_above_threshold(signal, threshold)

        # The final two SS peaks are often lower — search with reduced threshold
        split_idx = 8200
        if len(peak_idxs) > 0 and peak_idxs[-1] <= split_idx:
            tail_peaks = find_peaks_above_threshold(signal[split_idx:], 120) + split_idx
            peak_idxs = np.union1d(peak_idxs, tail_peaks)

        # Remove flat/close peaks (within 15 scan points)
        close = np.where(np.diff(peak_idxs) <= 15)[0]
        return np.delete(peak_idxs, close)

    @staticmethod
    def _validate_ss_peaks(peak_idxs: np.ndarray, expected_bps: np.ndarray) -> bool:
        """Validate that detected peaks match expected size standard pattern."""
        if len(peak_idxs) != 19:
            return False

        peak_dists = np.abs(np.diff(peak_idxs))
        bp_dists = np.abs(np.diff(expected_bps))
        relative = peak_dists / bp_dists

        # Pixels-per-basepair should be between 7 and 13
        return bool(np.all((relative <= 13) & (relative >= 7)))
