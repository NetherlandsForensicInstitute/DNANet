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
from typing import TYPE_CHECKING, Dict, Callable
from dataclasses import dataclass

import numpy as np
import scipy.interpolate


if TYPE_CHECKING:
    from dnanet.data.strategies.scaling.kit import STRKit


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
    def parse_size_standard(self, size_standard_lane: np.ndarray) -> SizeStandardParseResult | None:
        """Parse the size-standard dye lane into calibration data."""

    @abstractmethod
    def marker_name_to_dye_idx(self) -> dict[str, int]:
        """Map marker names to 0-based dye channel indices."""

    @abstractmethod
    def cache_signature(self) -> dict:
        """Return a JSON-serializable dict uniquely identifying this strategy's configuration.

        Used to build the dataset cache key. Must change whenever the strategy
        configuration changes in a way that affects loaded data.
        """

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

    @property
    def marker_to_idx(self) -> Dict[str, int]:
        return {name: idx for idx, name in enumerate(self.marker_names)}

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
            raise ValueError('indices and original_x_values must be 1D')
        if indices.size != original_x_values.size:
            raise ValueError('indices and original_x_values must have equal length')

        order = np.argsort(indices)
        sorted_idx = indices[order]
        sorted_bp = original_x_values[order]

        spline = scipy.interpolate.CubicSpline(
            sorted_idx, sorted_bp, bc_type='natural', extrapolate=True
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

        insertion = np.searchsorted(sorted_bp, target, side='left')
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
