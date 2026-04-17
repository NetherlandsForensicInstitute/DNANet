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
        scanpoint_resolution: int = 4096,
        **kwargs,
    ) -> None:
        kit = GLOBALFILER_KIT
        super().__init__(kit, basepair_start=60, basepair_end=480, scanpoint_resolution=scanpoint_resolution)
        self._max_shrinkages = max_shrinkages
        self._validation_threshold = validation_threshold

    # fmt: off
    def marker_name_to_dye_idx(self) -> dict[str, int]:
        return {
            'D3S1358': 0,'vWA': 0,'D16S539': 0,'CSF1PO': 0,'TPOX': 0,
            'Yindel': 1,'AMEL': 1,'D8S1179': 1,'D21S11': 1,'D18S51': 1,'DYS391': 1,
            'D2S441': 2,'D19S433': 2,'TH01': 2,'FGA': 2,
            'D22S1045': 3,'D5S818': 3,'D13S317': 3,'D7S820': 3,'SE33': 3,
            'D10S1248': 4,'D1S1656': 4,'D12S391': 4,'D2S1338': 4,
        }
    # fmt: on


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

    # TODO: Can we use this function for other STR kits as well?
    @staticmethod
    def _attempt_fit(
        peak_idxs: np.ndarray,
        expected_bps: np.ndarray,
        threshold: float,
        max_shrinkages: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Align detected peaks to expected base pairs via sliding-window search.

        For each candidate window size (starting at ``len(expected_bps)`` and
        shrinking by one up to ``max_shrinkages`` times), every consecutive
        sub-sequence of that many detected peaks is fitted against the expected
        bp values using a degree-2 polynomial.  The window with the lowest
        max-absolute residual is kept.  If that residual is below ``threshold``
        the alignment is accepted; otherwise the window size shrinks and the
        search repeats.  Searching all windows instead of anchoring to one end
        makes the algorithm robust to noise peaks anywhere in the signal.

        Returns:
            Tuple of ``(peak_idxs, bps, max_deviation)`` for the best alignment.
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
            n = len(bps)
            if n < 3:
                break

            # Try every consecutive window of n peaks and pick the best-fitting one.
            # This handles extra noise peaks anywhere in the signal — not just at the
            # ends — by exhaustively searching all possible alignments.
            window_best_diff = float('inf')
            window_best_peaks: np.ndarray | None = None

            if len(peak_idxs) >= n:
                for start in range(len(peak_idxs) - n + 1):
                    window = peak_idxs[start : start + n]
                    coeffs = np.polyfit(window, bps, 2)
                    fitted = np.polyval(coeffs, window)
                    diff = float(np.max(np.abs(fitted - bps)))
                    if diff < window_best_diff:
                        window_best_diff = diff
                        window_best_peaks = window
            else:
                # Fewer peaks than expected bps: fit all peaks against first N bps.
                trimmed_bps = bps[: len(peak_idxs)]
                coeffs = np.polyfit(peak_idxs, trimmed_bps, 2)
                fitted = np.polyval(coeffs, peak_idxs)
                window_best_diff = float(np.max(np.abs(fitted - trimmed_bps)))
                window_best_peaks = peak_idxs
                bps = trimmed_bps

            if window_best_diff < best_diff:
                best_diff = window_best_diff
                best_trimmed = window_best_peaks
                best_bps = bps

            if window_best_diff < threshold:
                if shrinkages > 0:
                    logger.trace(
                        'Size standard shrunk {} times; max diff {:.2f} bp',
                        shrinkages,
                        window_best_diff,
                    )
                return window_best_peaks, bps, window_best_diff  # type: ignore[return-value]

            bps = bps[:-1]
            shrinkages += 1

        logger.warning(
            f'Size standard fit did not converge: {best_diff:.2f} bp deviation after '
            f'{max_shrinkages} shrinkages (threshold={threshold:.1f}). Using best fit found.'
        )
        if best_trimmed is None or best_bps is None:
            raise RuntimeError('Fit attempt failed')
        return best_trimmed, best_bps, best_diff
