"""Scaling strategies for electropherogram base-pair calibration.

Design pattern: **Strategy** (classic GoF)
    ``ScalingStrategy`` is the abstract base. Each concrete subclass
    (``PowerPlexFusion6CStrategy``, ``GlobalFilerStrategy``) encapsulates
    kit-specific logic for:
    - Detecting size-standard peaks
    - Validating the detection
    - Interpolating base-pair positions for every scan point
    - Rescaling the profile to a uniform base-pair grid

    The strategy is selected at configuration time and injected into the
    data pipeline — no conditional branching on kit name needed.

Design pattern: **Template Method**
    ``ScalingStrategy.interpolate()`` defines the *skeleton* of the
    interpolation algorithm (interpolate → rescale → return). Subclasses
    override ``parse_size_standard()`` and ``extract_ss_peaks()`` to
    customize the variable parts. The invariant part (the spline math)
    stays in the base class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.interpolate
import scipy.ndimage
from loguru import logger

from dnanet.core.panel import Panel
from dnanet.data.preprocessing.peaks import find_peaks_above_threshold
from dnanet.data.strategies.kit import STRKit
from dnanet.data.strategies.size_standard import (
    GENESCAN_600_LIZ,
    WEN_ILS,
    SizeStandard,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SizeStandardParseResult:
    """Output of size-standard parsing.

    Attributes:
        rescaled_indices: Indices into the original profile for rescaling.
        scaler: Base-pair value per rescaled scan point (1D).
        fit_error: Maximum deviation between fitted and expected base pairs.
    """

    rescaled_indices: np.ndarray
    scaler: np.ndarray
    fit_error: float


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ScalingStrategy(ABC):
    """Base class for kit-specific profile scaling.

    Args:
        kit: Kit configuration (defines size standard, panel, markers).
        basepair_start: Start of the target base-pair range.
        basepair_end: End of the target base-pair range.
        scanpoint_resolution: Number of scan points in the rescaled profile.
    """

    def __init__(
        self,
        kit: STRKit,
        basepair_start: int,
        basepair_end: int,
        scanpoint_resolution: int = 4096,
    ) -> None:
        self.kit = kit
        self.panel = kit.panel
        self._scanpoint_resolution = scanpoint_resolution
        self._basepair_start = basepair_start
        self._basepair_end = basepair_end

    @property
    def scanpoint_resolution(self) -> int:
        return self._scanpoint_resolution

    @property
    def basepair_start(self) -> int:
        return self._basepair_start

    @property
    def basepair_end(self) -> int:
        return self._basepair_end

    # -- Abstract interface (overridden by each kit) ---------------------- #

    @abstractmethod
    def parse_size_standard(
        self, size_standard_lane: np.ndarray
    ) -> SizeStandardParseResult | None:
        """Parse the size-standard dye lane into calibration data."""

    @abstractmethod
    def marker_name_to_dye_idx(self) -> dict[str, int]:
        """Map marker names to 0-based dye channel indices."""

    @abstractmethod
    def dye_channel_colors(self) -> list[str]:
        """Return display colors for each dye channel."""

    # -- Template method: shared interpolation logic ---------------------- #

    def interpolate(
        self,
        peak_idxs: np.ndarray,
        bps: np.ndarray,
        size_standard_lane: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate base-pair values and rescale to target range.

        This is the invariant part of the calibration algorithm. Subclasses
        provide ``peak_idxs`` and ``bps`` by overriding ``parse_size_standard``.

        Returns:
            Tuple of ``(rescaled_indices, scaler)``.
        """
        interpolator = self.basepair_interpolator(
            indices=peak_idxs, original_x_values=bps, extrapolate=False
        )
        interpolated_bp = interpolator(np.arange(len(size_standard_lane)))

        rescaled_indices = self.rescale_dye(
            interpolated_bp,
            rescale_size=self.scanpoint_resolution,
            target_range=(self.basepair_start, self.basepair_end),
        )
        scaler = interpolated_bp[rescaled_indices]
        return rescaled_indices, scaler

    # -- Shared utilities ------------------------------------------------- #

    @property
    def marker_names(self) -> list[str]:
        return list(self.marker_name_to_dye_idx().keys())

    @staticmethod
    def basepair_interpolator(
        indices: np.ndarray | list[float],
        original_x_values: np.ndarray | list[float],
        extrapolate: bool = False,
    ) -> Callable[[np.ndarray | float], np.ndarray]:
        """Build a cubic-spline interpolator for base-pair calibration.

        Args:
            indices: Scan-point indices where base pairs are known.
            original_x_values: Known base-pair values at those indices.
            extrapolate: If False, values outside the calibration range are zeroed.

        Returns:
            A callable that maps scan-point indices to base-pair values.
        """
        indices = np.asarray(indices)
        original_x_values = np.asarray(original_x_values)

        if indices.ndim != 1 or original_x_values.ndim != 1:
            raise ValueError("indices and original_x_values must be 1D")
        if indices.size != original_x_values.size:
            raise ValueError("indices and original_x_values must have equal length")

        order = np.argsort(indices)
        sorted_idx = indices[order]
        sorted_bp = original_x_values[order]

        spline = scipy.interpolate.CubicSpline(
            sorted_idx, sorted_bp, bc_type="natural", extrapolate=True
        )
        lo, hi = sorted_idx[0], sorted_idx[-1]

        def _interpolate(x: np.ndarray | float) -> np.ndarray:
            x_arr = np.asarray(x, dtype=float)
            values = np.asarray(spline(x_arr))
            if not extrapolate:
                values = np.where((x_arr < lo) | (x_arr > hi), 0.0, values)
            return np.atleast_1d(values)

        return _interpolate

    @staticmethod
    def rescale_dye(
        basepairs: np.ndarray,
        rescale_size: int,
        target_range: tuple[int, int],
    ) -> np.ndarray:
        """Map interpolated base-pair positions to uniform pixel indices.

        Args:
            basepairs: Interpolated base-pair value for each original scan point.
            rescale_size: Number of pixels in the output.
            target_range: ``(bp_start, bp_end)`` defining the output range.

        Returns:
            Array of indices into the original profile.
        """
        bp_start, bp_end = target_range
        target = np.linspace(bp_start, bp_end, rescale_size)

        sort_order = np.argsort(basepairs)
        sorted_bp = basepairs[sort_order]

        insertion = np.searchsorted(sorted_bp, target, side="left")
        insertion = np.clip(insertion, 1, len(sorted_bp) - 1)

        left_idx = insertion - 1
        right_idx = insertion
        left_delta = np.abs(sorted_bp[left_idx] - target)
        right_delta = np.abs(sorted_bp[right_idx] - target)

        return np.where(
            left_delta <= right_delta,
            sort_order[left_idx],
            sort_order[right_idx],
        )


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

_PPF6C_PANEL_PATH = Path("resources/kit_panels/SGPanel_PPF6C.xml")
_GLOBALFILER_PANEL_PATH = Path("resources/kit_panels/SGPanel_Globalfiler_Panel.xml")


class PowerPlexFusion6CStrategy(ScalingStrategy):
    """Scaling strategy for the PowerPlex Fusion 6C kit.

    PPF6C has 6 dye channels total: 5 analysis dyes + 1 size standard.
    HID dye indices: 1,2,3,4,6 (5 is skipped).
    """

    _HID_DYE_MAPPING = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4}

    def __init__(self, panel_path: Path = _PPF6C_PANEL_PATH, **kwargs) -> None:
        kit = STRKit(
            name="PPF6C",
            size_standard=WEN_ILS,
            panel=Panel.from_xml(panel_path, hid_dye_mapping=self._HID_DYE_MAPPING),
            num_dyes=5,
            hid_dye_mapping=self._HID_DYE_MAPPING,
            panel_path=panel_path,
            description="PowerPlex Fusion 6C using WEN ILS size standard.",
            hid_file_data_columns_raw=["DATA_1", "DATA_2", "DATA_3", "DATA_4", "DATA_106", "DATA_105"],
            hid_file_data_columns_analyzed=["DATA_9", "DATA_10", "DATA_11", "DATA_12", "DATA_206", "DATA_205"]
        )
        super().__init__(kit, basepair_start=65, basepair_end=475, scanpoint_resolution=4096)

    def marker_name_to_dye_idx(self) -> dict[str, int]:
        return {
            "AMEL": 0, "D3S1358": 0, "D1S1656": 0, "D2S441": 0,
            "D10S1248": 0, "D13S317": 0, "Penta E": 0,
            "D16S539": 1, "D18S51": 1, "D2S1338": 1, "CSF1PO": 1, "Penta D": 1,
            "TH01": 2, "vWA": 2, "D21S11": 2, "D7S820": 2, "D5S818": 2, "TPOX": 2,
            "D8S1179": 3, "D12S391": 3, "D19S433": 3, "SE33": 3, "D22S1045": 3,
            "DYS391": 4, "FGA": 4, "DYS576": 4, "DYS570": 4,
        }

    def dye_channel_colors(self) -> list[str]:
        return ["blue", "green", "black", "red", "purple", "orange"]

    def parse_size_standard(
        self, size_standard_lane: np.ndarray
    ) -> SizeStandardParseResult | None:
        """Parse WEN ILS size standard from the PPF6C kit."""
        lane = np.asarray(size_standard_lane).reshape(-1)
        expected_bps = self.kit.size_standard.expected_bps

        peak_idxs = self._extract_ss_peaks(lane)
        peak_idxs = peak_idxs[-20:-1]  # last 19 peaks, excluding final

        if not self._validate_ss_peaks(peak_idxs, expected_bps):
            return None

        rescaled_indices, scaler = self.interpolate(peak_idxs, expected_bps, lane)
        return SizeStandardParseResult(
            rescaled_indices=rescaled_indices, scaler=scaler, fit_error=0.0
        )

    @staticmethod
    def _extract_ss_peaks(signal: np.ndarray) -> np.ndarray:
        """Find size-standard peaks with adaptive thresholding for tail peaks."""
        peak_idxs = find_peaks_above_threshold(signal, 180)

        # The final two SS peaks are often lower — search with reduced threshold
        split_idx = 8200
        if len(peak_idxs) > 0 and peak_idxs[-1] <= split_idx:
            tail_peaks = find_peaks_above_threshold(signal[split_idx:], 120) + split_idx
            peak_idxs = np.union1d(peak_idxs, tail_peaks)

        # Remove flat/close peaks (within 15 scan points)
        close = np.where(np.diff(peak_idxs) <= 15)[0]
        return np.delete(peak_idxs, close)

    @staticmethod
    def _validate_ss_peaks(
        peak_idxs: np.ndarray, expected_bps: np.ndarray
    ) -> bool:
        """Validate that detected peaks match expected size standard pattern."""
        if len(peak_idxs) != 19:
            return False

        peak_dists = np.abs(np.diff(peak_idxs))
        bp_dists = np.abs(np.diff(expected_bps))
        relative = peak_dists / bp_dists

        # Pixels-per-basepair should be between 7 and 13
        return bool(np.all((relative <= 13) & (relative >= 7)))


class GlobalFilerStrategy(ScalingStrategy):
    """Scaling strategy for the GlobalFiler kit.

    GlobalFiler has 6 dye channels total: 5 analysis dyes + 1 size standard.
    HID dye indices: 1,2,3,4,6 (5 is skipped, same as PPF6C).
    """

    _HID_DYE_MAPPING = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4}

    def __init__(
        self,
        panel_path: Path = _GLOBALFILER_PANEL_PATH,
        max_shrinkages: int = 10,
        validation_threshold: float = 5.0,
        **kwargs,
    ) -> None:
        kit = STRKit(
            name="GlobalFiler",
            size_standard=GENESCAN_600_LIZ,
            panel=Panel.from_xml(panel_path, hid_dye_mapping=self._HID_DYE_MAPPING),
            num_dyes=5,
            hid_dye_mapping=self._HID_DYE_MAPPING,
            panel_path=panel_path,
            description="GlobalFiler using GeneScan 600 LIZ size standard.",
            hid_file_data_columns_raw=["DATA_1", "DATA_2", "DATA_3", "DATA_4", "DATA_106", "DATA_105"],
            hid_file_data_columns_analyzed=["DATA_9", "DATA_10", "DATA_11", "DATA_12", "DATA_206", "DATA_205"]
        )
        super().__init__(kit, basepair_start=60, basepair_end=480, scanpoint_resolution=4096)
        self._max_shrinkages = max_shrinkages
        self._validation_threshold = validation_threshold

    def marker_name_to_dye_idx(self) -> dict[str, int]:
        return {
            "D3S1358": 0, "vWA": 0, "D16S539": 0, "CSF1PO": 0, "TPOX": 0,
            "Y-Indel": 1, "AMEL": 1, "D8S1179": 1, "D21S11": 1, "D18S51": 1, "DYS391": 1,
            "D2S441": 2, "D19S433": 2, "TH01": 2, "FGA": 2,
            "D22S1045": 3, "D5S818": 3, "D13S317": 3, "D7S820": 3, "SE33": 3,
            "D10S1248": 4, "D1S1656": 4, "D12S391": 4, "D2S1338": 4,
        }

    def dye_channel_colors(self) -> list[str]:
        return ["blue", "green", "black", "red", "purple", "orange"]

    def parse_size_standard(
        self, size_standard_lane: np.ndarray
    ) -> SizeStandardParseResult | None:
        """Parse GeneScan 600 LIZ with iterative fit shrinking."""
        lane = np.asarray(size_standard_lane).reshape(-1)
        expected_bps = self.kit.size_standard.expected_bps

        peak_idxs = self._extract_ss_peaks(lane)
        peak_idxs, bps, diff = self._attempt_fit(
            peak_idxs, expected_bps,
            self._validation_threshold, self._max_shrinkages,
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
        best_trimmed, best_bps, best_diff = None, None, float("inf")

        # Need at least 3 peaks for a degree-2 polynomial fit
        if len(peak_idxs) < 3:
            logger.warning(
                "Too few size standard peaks ({}) for polynomial fit",
                len(peak_idxs),
            )
            return peak_idxs, bps[:len(peak_idxs)], float("inf")

        shrinkages = 0
        while shrinkages < max_shrinkages:
            trimmed = peak_idxs[-len(bps):]
            # If fewer peaks than expected bps, trim bps to match
            if len(trimmed) < len(bps):
                bps = bps[-len(trimmed):]
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
                        "Size standard shrunk {} times; max diff {:.2f} bp",
                        shrinkages, diff,
                    )
                return trimmed, bps, diff

            bps = bps[:-1]
            shrinkages += 1

        logger.warning(
            "Size standard fit did not converge: {:.2f} bp deviation after "
            "{} shrinkages (threshold={:.1f}). Using best fit found.",
            best_diff, max_shrinkages, threshold,
        )
        return best_trimmed, best_bps, best_diff


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

SCALING_STRATEGIES: dict[str, type[ScalingStrategy]] = {
    "PPF6C": PowerPlexFusion6CStrategy,
    "GLOBALFILER": GlobalFilerStrategy,
}


def get_scaling_strategy(name: str, **kwargs) -> ScalingStrategy:
    """Instantiate a scaling strategy by name.

    Args:
        name: Strategy name (e.g. "PPF6C", "GLOBALFILER").
        **kwargs: Additional arguments passed to the strategy constructor.

    Raises:
        ValueError: If the name is not recognized.
    """
    cls = SCALING_STRATEGIES.get(name.upper())
    if cls is None:
        raise ValueError(
            f"Unknown scaling strategy '{name}'. "
            f"Available: {list(SCALING_STRATEGIES.keys())}"
        )
    return cls(**kwargs)
