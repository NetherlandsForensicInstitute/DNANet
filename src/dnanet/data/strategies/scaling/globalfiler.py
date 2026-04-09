from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from dnanet.data.preprocessing.peaks import find_peaks_above_threshold
from dnanet.data.strategies.scaling.kit import GLOBALFILER_KIT
from dnanet.data.strategies.scaling.scaling import (
    ScalingStrategy,
    SizeStandardParseResult,
)


class GlobalFilerStrategy(ScalingStrategy):
    """Scaling strategy for the GlobalFiler kit.

    GlobalFiler has 6 dye channels total: 5 analysis dyes + 1 size standard.
    HID dye indices: 1,2,3,4,6 (5 is skipped, same as PPF6C).
    """

    def __init__(
        self,
        max_shrinkages: int = 10,
        validation_threshold: float = 5.0,
        **kwargs,
    ) -> None:
        kit = GLOBALFILER_KIT
        super().__init__(kit, basepair_start=60, basepair_end=480, scanpoint_resolution=4096)
        self._max_shrinkages = max_shrinkages
        self._validation_threshold = validation_threshold

    def marker_name_to_dye_idx(self) -> dict[str, int]:
        return {
            'D3S1358': 0,
            'vWA': 0,
            'D16S539': 0,
            'CSF1PO': 0,
            'TPOX': 0,
            'Y-Indel': 1,
            'AMEL': 1,
            'D8S1179': 1,
            'D21S11': 1,
            'D18S51': 1,
            'DYS391': 1,
            'D2S441': 2,
            'D19S433': 2,
            'TH01': 2,
            'FGA': 2,
            'D22S1045': 3,
            'D5S818': 3,
            'D13S317': 3,
            'D7S820': 3,
            'SE33': 3,
            'D10S1248': 4,
            'D1S1656': 4,
            'D12S391': 4,
            'D2S1338': 4,
        }

    def dye_channel_colors(self) -> list[str]:
        return ['blue', 'green', 'black', 'red', 'purple', 'orange']

    def parse_size_standard(self, size_standard_lane: np.ndarray) -> SizeStandardParseResult | None:
        """Parse GeneScan 600 LIZ with iterative fit shrinking."""
        lane = np.asarray(size_standard_lane).reshape(-1)
        expected_bps = self.kit.size_standard.expected_bps

        peak_idxs = self._extract_ss_peaks(lane)
        peak_idxs, bps, diff = self._attempt_fit(
            peak_idxs,
            expected_bps,
            self._validation_threshold,
            self._max_shrinkages,
        )

        rescaled_indices, scaler = self.interpolate(peak_idxs, bps, lane)
        return SizeStandardParseResult(
            rescaled_indices=rescaled_indices, scaler=scaler, fit_error=diff
        )

    @staticmethod
    def _extract_ss_peaks(signal: np.ndarray) -> np.ndarray:
        """Detect size-standard peaks and filter close duplicates."""
        peak_idxs = find_peaks_above_threshold(signal, 300)
        close = np.where(np.diff(peak_idxs) <= 15)[0]
        return np.delete(peak_idxs, close)

    @staticmethod
    def _attempt_fit(
        peak_idxs: np.ndarray,
        expected_bps: np.ndarray,
        threshold: float,
        max_shrinkages: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Iteratively trim trailing peaks until the polynomial fit is good.

        If the fit doesn't converge within ``max_shrinkages`` iterations,
        returns the best fit found (with a warning) rather than raising.
        This matches the original behavior where marginally bad fits are
        still usable for size-standard calibration.

        Returns:
            Tuple of ``(trimmed_peak_idxs, trimmed_bps, max_deviation)``.
        """
        bps = expected_bps
        best_trimmed, best_bps, best_diff = None, None, float('inf')

        # Need at least 3 peaks for a degree-2 polynomial fit
        if len(peak_idxs) < 3:
            logger.warning(
                'Too few size standard peaks ({}) for polynomial fit',
                len(peak_idxs),
            )
            return peak_idxs, bps[: len(peak_idxs)], float('inf')

        shrinkages = 0
        while shrinkages < max_shrinkages:
            trimmed = peak_idxs[-len(bps) :]
            # If fewer peaks than expected bps, trim bps to match
            if len(trimmed) < len(bps):
                bps = bps[-len(trimmed) :]
            if len(trimmed) < 3:
                break
            coeffs = np.polyfit(trimmed, bps, 2)
            fitted = np.polyval(coeffs, trimmed)
            diff = float(np.max(np.abs(fitted - bps)))

            # Track the best fit seen so far
            if diff < best_diff:
                best_trimmed, best_bps, best_diff = trimmed, bps, diff

            if diff < threshold:
                if shrinkages > 0:
                    logger.info(
                        'Size standard shrunk {} times; max diff {:.2f} bp',
                        shrinkages,
                        diff,
                    )
                return trimmed, bps, diff

            bps = bps[:-1]
            shrinkages += 1

        logger.warning(
            f'Size standard fit did not converge: {best_diff:.2f} bp deviation after '
            f'{max_shrinkages} shrinkages (threshold={threshold:.1f}). Using best fit found.'
        )
        return best_trimmed, best_bps, best_diff
