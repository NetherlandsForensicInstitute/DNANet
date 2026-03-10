"""
EPG scaling strategies encapsulate how to parse the size standard and rescale
electropherograms (EPGs). While the size standard comes from a kit, the choice
of scaling strategy is dataset/EPG-characteristic dependent, so we keep the
focus on scaling behavior rather than general kit logic.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from DNAnet.data.kit_compatibility.kit import Kit
from DNAnet.data.kit_compatibility.kit import get_standard_kit
from DNAnet.data.kit_compatibility.lane_standards import (
    BASE_PAIR_END,
    BASE_PAIR_START,
    RESCALE_SIZE,
    VAL_THRESHOLD,
    InternalSizeStandard,
    get_size_standard_bps,
)
from DNAnet.data.utils import (
    basepair_interpolator,
    extract_ss_peaks_simple,
    get_interpolated_basepairs,
    rescale_dye,
)

LOGGER = logging.getLogger("dnanet")


@dataclass
class SizeStandardParseResult:
    """
    Container for size-standard parsing outputs.

    Attributes:
        rescaled_indices: Indices into the original profile used for rescaling.
        scaler: Base-pair value per rescaled pixel (1D).
        fit_error: Max absolute deviation between fitted and expected base pairs.
    """
    rescaled_indices: np.ndarray
    scaler: np.ndarray
    fit_error: float


class EPGScalingStrategy(ABC):
    """
    Base contract for EPG scaling.

    Currently focuses on size-standard parsing; additional kit-specific
    behaviors can be added when needed.

    :param kit: Kit configuration to use (defines size standard/panel/markers).
    :param panel_path: Optional panel path override; if provided, it replaces
        the kit's panel_path.
    """

    def __init__(self, kit: Kit):
        self.kit = kit
        self.size_standard: InternalSizeStandard = self.kit.size_standard
        self.panel = self.kit.panel

    @abstractmethod
    def parse_size_standard(
        self,
        size_standard_lane: np.ndarray,
    ) -> SizeStandardParseResult:
        """Parse the size-standard dye lane into interpolation/rescaling data."""


class ProvedItEPGScalingStrategy(EPGScalingStrategy):
    """
    Kit strategy for ProvedIt-style data.

    Mirrors the size-standard parsing logic from `ProvedItHIDImage._read`:
    - Detect size-standard peaks.
    - Fit expected base pairs, trimming trailing peaks if the fit is poor.
    - Produce interpolated base pairs for every scan point and rescale indices.
    """

    def __init__(
        self,
        kit: Kit | None = None,
        max_shrinkages: int = 10,
        validation_threshold: float = VAL_THRESHOLD,
    ):
        if kit is None:
            kit = get_standard_kit("PPF6C")
        super().__init__(kit)
        self.max_shrinkages = max_shrinkages
        self.validation_threshold = validation_threshold

    def parse_size_standard(
        self,
        size_standard_lane: np.ndarray,
    ) -> SizeStandardParseResult:
        # Ensure 1D array for peak finding.
        size_standard_lane = np.asarray(size_standard_lane).reshape(-1)

        # Detect peaks in the size-standard dye.
        size_standard_peaks_idxs = extract_ss_peaks_simple(size_standard_lane)
        expected_bps = get_size_standard_bps(self.size_standard)

        diff: float = self.validation_threshold + 1
        shrinkages = 0
        bps = expected_bps
        peak_idxs = size_standard_peaks_idxs

        # Attempt to fit; trim trailing peaks if the fit is too poor.
        while shrinkages < self.max_shrinkages:
            peak_idxs = peak_idxs[-len(bps):]
            coeffs = np.polyfit(peak_idxs, bps, 2)
            fitted = np.polyval(coeffs, peak_idxs)
            diff = float(np.max(np.abs(fitted - bps)))
            if diff < self.validation_threshold:
                break
            else:
                bps = bps[:-1]
                shrinkages += 1

        if diff >= self.validation_threshold:
            raise ValueError(
                f"Size standard differs {diff:.2f} from expected for {self.size_standard}"
            )
        
        if shrinkages > 0:
            LOGGER.info(
                "Size standard was shrunk %d times to fit the profile; max diff %.2f bp.",
                shrinkages,
                diff,
            )

        # Interpolate base pairs for all scan points.
        interpolator = basepair_interpolator(
            indices=peak_idxs,
            original_x_values=bps,
            extrapolate=False,
        )
        # interpolated_base_pairs shape: (num_scan_points,)
        # This list contains the base-pair value for each original scan point.
        # For points outside the range of detected peaks, the value will be 0 because the interpolator does not extrapolate.
        # For example: interpolated_base_pairs[3805:3810] = [0. ,  0. , 60. , 60.07, 60.14]
        # In this case, the first two points are outside the detected peak range, hence 0.
        # The next three points fall within the range, hence they have interpolated base-pair values.
        # Index 3807 corresponds to a base-pair value of 60.
        interpolated_base_pairs = interpolator(np.arange(len(size_standard_lane)))

        # Rescale to the target base-pair range and cache scaler.
        # rescaled_indices shape: (RESCALE_SIZE,)
        # This list contains the indices into the original profile that correspond to the rescaled size standard.
        # For example: rescaled_indices[0:5] = [3807, 3812, 3817, 3822, 3827]
        # This means that the first five points in the rescaled EPG correspond to these indices in the original profile.
        rescaled_indices = rescale_dye(
            interpolated_base_pairs,
            rescale_size=RESCALE_SIZE,
            target_range=(BASE_PAIR_START, BASE_PAIR_END),
        )
        # maps the rescaled indices to base-pair values.
        # For example: scaler[0:5] = [60. , 61.07, 62.14, 63.21, 64.28]
        # This means that the first five points in the rescaled EPG correspond to these base-pair values.
        scaler = interpolated_base_pairs[rescaled_indices]

        return SizeStandardParseResult(
            rescaled_indices=rescaled_indices,
            scaler=scaler,
            fit_error=diff,
        )


class NfiEPGScalingStrategy(EPGScalingStrategy):
    """
    Kit strategy for the original NFI/R&D pipeline (legacy HIDImage).

    Relies on the classic size-standard validation/interpolation and rescaling.
    """

    def __init__(self, kit: Kit | None = None):
        if kit is None:
            kit = get_standard_kit("GLOBALFILER")
        super().__init__(kit)

    def parse_size_standard(
        self,
        size_standard_lane: np.ndarray
    ) -> SizeStandardParseResult:
        """_summary_

        Args:
            size_standard_lane: The dye lane containing the size standard of the EPG

        Raises:
            ValueError: When interpolating the size standard fails

        Returns:
            SizeStandardParseresult containing the rescaled indices and scaler.
        """
        size_standard_lane = np.asarray(size_standard_lane).reshape(-1)

        interpolated_base_pairs = get_interpolated_basepairs(size_standard_lane)
        if interpolated_base_pairs is None:
            raise ValueError("Invalid size standard: interpolation failed validation.")

        rescaled_indices = rescale_dye(interpolated_base_pairs, rescale_size=4096, target_range=(65, 475))
        scaler = interpolated_base_pairs[rescaled_indices]

        return SizeStandardParseResult(
            rescaled_indices=rescaled_indices,
            scaler=scaler,
            fit_error=0.0,
        )
