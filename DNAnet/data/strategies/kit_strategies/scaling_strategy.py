"""
scaling strategies encapsulate how to parse the size standard and rescale
electropherograms (dna profiles).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union, Tuple, Callable, List

import numpy as np
import scipy

from DNAnet.data.data_models import Panel
from DNAnet.data.strategies.kit_strategies.internal_size_standard import GENESCAN_600_LIZ_BPS, WEN_ILS_BPS
from DNAnet.data.strategies.kit_strategies.str_kit import STRKit
from DNAnet.data.utils import (
    find_peaks_above_threshold
)

LOGGER = logging.getLogger("dnanet")

_POWER_PLEX_FUSION_6C_PANEL_PATH = Path("resources/kit_panels/SGPanel_PPF6C.xml")
_GLOBALFILER_PANEL_PATH = Path("resources/kit_panels/SGPanel_Globalfiler_Panel.xml")


class ScalingStrategyOptions(Enum):
    PPF6C = "POWER_PLEX_FUSION_6C"
    GLOBALFILER = "GLOBALFILER"


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


def get_scaling_strategy(strat_name: str, **kwargs) -> ScalingStrategy:
    if strat_name == ScalingStrategyOptions.PPF6C.name:
        return PowerPlexFusion6CScalingStrategy(**kwargs)

    if strat_name == ScalingStrategyOptions.GLOBALFILER.name:
        return GlobalFilerScalingStrategy(**kwargs)

    raise ValueError(f"Kit name is not in the standards: {strat_name}")



class ScalingStrategy(ABC):
    """
    Base contract for profile scaling.

    Currently focuses on size-standard parsing; additional kit-specific
    behaviors can be added when needed.

    :param kit: Kit configuration to use (defines size standard/panel/markers).
    :param basepair_start: Starting base pair for rescaling (e.g. 65).
    :param basepair_end: Ending base pair for rescaling (e.g. 475).
    :param scanpoint_resolution: Number of scan points in the rescaled profile (e.g. 4096).
    """

    def __init__(self, kit: STRKit, basepair_start: int, basepair_end: int, scanpoint_resolution: int = 4096):
        self.kit = kit
        self.panel = self.kit.panel
        self._scanpoint_resolution = scanpoint_resolution
        self._basepair_start = basepair_start
        self._basepair_end = basepair_end

    @abstractmethod
    def parse_size_standard(
        self,
        size_standard_lane: np.ndarray,
    ) -> Optional[SizeStandardParseResult]:
        """Parse the size-standard dye lane into interpolation/rescaling data."""
        raise NotImplementedError

    @property
    def scanpoint_resolution(self) -> int:
        return self._scanpoint_resolution

    @property
    def basepair_start(self) -> int:
        return self._basepair_start

    @property
    def basepair_end(self) -> int:
        return self._basepair_end

    @abstractmethod
    def dye_channel_colors(self) -> List[str]:
        raise NotImplementedError

    @staticmethod
    def basepair_interpolator(indices: Union[np.ndarray, list[float]],
                              original_x_values: Union[np.ndarray, list[float]],
                              extrapolate: bool = False) \
            -> Callable[[Union[np.ndarray, float]], np.ndarray]:
        """Generate a cubic-spline interpolator for translating pixel indices.

           GeneMarker's sizing uses a cubic spline that passes exactly through every
           size-standard point. To mimic this behaviour we construct a natural cubic
           spline between the detected size-standard peak indices (``indices``) and
           the expected base-pair values (``original_x_values``). Outside of the
           calibrated range we zero-out the values unless explicit extrapolation is
           requested, reproducing the previous behaviour of returning zeros beyond the
           last standard fragment.

           :param indices: indices for which a value is present
           :param original_x_values: known x_values between which to interpolate
           :param extrapolate: whether to use extrapolation or not
           :return: callable that evaluates the spline at the requested positions
           """
        indices = np.asarray(indices)
        original_x_values = np.asarray(original_x_values)

        if indices.ndim != 1 or original_x_values.ndim != 1:
            raise ValueError("`indices` and `original_x_values` must be one-dimensional arrays")

        if indices.size != original_x_values.size:
            raise ValueError("`indices` and `original_x_values` must contain the same number of points")

        sort_order = np.argsort(indices)
        sorted_indices = indices[sort_order]
        sorted_basepairs = original_x_values[sort_order]

        spline = scipy.interpolate.CubicSpline(
            sorted_indices,
            sorted_basepairs,
            bc_type="natural",
            extrapolate=True,
        )

        lower_bound = sorted_indices[0]
        upper_bound = sorted_indices[-1]

        def _interpolate(x_new: Union[np.ndarray, float]) -> np.ndarray:
            x_new_arr = np.asarray(x_new, dtype=float)
            values = np.asarray(spline(x_new_arr))

            if not extrapolate:
                outside_mask = (x_new_arr < lower_bound) | (x_new_arr > upper_bound)
                values = np.where(outside_mask, 0.0, values)

            return np.atleast_1d(values)

        return _interpolate

    def interpolate(self, peak_idxs: np.ndarray, bps: np.ndarray, size_standard_lane: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolates base-pair values for all scan points and rescales to target base-pair range.

        :param peak_idxs: Indices of detected size-standard peaks.
        :param bps: Expected base-pair values corresponding to the detected peaks.
        :param size_standard_lane: The dye lane containing the size standard of the profile.

        :return: A tuple containing:
            - rescaled_indices: Indices into the original profile corresponding to the rescaled size standard
            - scaler: Base-pair value per rescaled scan point (1D array)


        """
        # Interpolate base pairs for all scan points.
        interpolator = self.basepair_interpolator(
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
        # rescaled_indices shape: (scanpoint_resolution,)
        # This list contains the indices into the original profile that correspond to the rescaled size standard.
        # For example: rescaled_indices[0:5] = [3807, 3812, 3817, 3822, 3827]
        # This means that the first five points in the rescaled EPG correspond to these indices in the original profile.
        rescaled_indices = self.rescale_dye(
            interpolated_base_pairs,
            rescale_size=self.scanpoint_resolution,
            target_range=(self.basepair_start, self.basepair_end),
        )
        # maps the rescaled indices to base-pair values.
        # For example: scaler[0:5] = [60. , 61.07, 62.14, 63.21, 64.28]
        # This means that the first five points in the rescaled EPG correspond to these base-pair values.
        scaler = interpolated_base_pairs[rescaled_indices]

        return rescaled_indices, scaler


    @staticmethod
    def rescale_dye(
            basepairs: np.ndarray,
            rescale_size: int,
            target_range: Tuple[int, int]
    ) -> np.ndarray:
        """Map base-pair positions to scaled pixel indices.

        ``basepairs`` is an array containing the interpolated base-pair value for
        each scan point of the electropherogram. ``target_range`` defines the base
        pair range of the rescaled profile (defaults to the size standard range when
        ``None``).
        """
        bp_start, bp_end = target_range
        target_linspace = np.linspace(bp_start, bp_end, rescale_size)

        # Presorting interpolated base pairs
        sort_indices = np.argsort(basepairs)
        sorted_basepairs = basepairs[sort_indices]

        # Find insertion indices
        insertion_indices = np.searchsorted(
            sorted_basepairs,
            target_linspace,
            side='left'
        )

        # Adjust indices for boundary conditions
        insertion_indices = np.clip(
            insertion_indices,
            1,
            len(sorted_basepairs) - 1
        )

        # Determine the closest index prior or after based on value proximity
        left_indices = insertion_indices - 1
        right_indices = insertion_indices

        left_deltas = np.abs(sorted_basepairs[left_indices] - target_linspace)
        right_deltas = np.abs(sorted_basepairs[right_indices] - target_linspace)

        return np.where(
            (left_deltas < right_deltas) | (left_deltas == right_deltas),
            sort_indices[left_indices],
            sort_indices[right_indices],
            )

class GlobalFilerScalingStrategy(ScalingStrategy):
    """
    Scaling strategy for GlobalFiler data.

    Functionality:
    - Detect size-standard peaks.
    - Fit expected base pairs, trimming trailing peaks if the fit is poor.
    - Produce interpolated base pairs for every scan point and rescale indices.
    """

    def __init__(
        self,
        max_shrinkages: int = 10,
        validation_threshold: float = 5.0,
    ):
        kit = STRKit(
            name="GlobalFiler",
            size_standard=GENESCAN_600_LIZ_BPS,
            num_dyes=6,
            panel_path= _GLOBALFILER_PANEL_PATH,
            panel=Panel(_GLOBALFILER_PANEL_PATH),
            markers=None,  # fill in with marker names for quick validation
            description="GlobalFiler kit using GENESCAN_600_LIZ size standard.",
        )
        super().__init__(kit, basepair_start=60, basepair_end=480, scanpoint_resolution=4096)
        self.max_shrinkages = max_shrinkages
        self.validation_threshold = validation_threshold

    def dye_channel_colors(self) -> List[str]:
        return ['blue', 'green', 'black', 'red', 'purple', 'orange']

    @staticmethod
    def attempt_fit(peak_idxs: np.ndarray, expected_bps: np.ndarray, validation_threshold: float, max_shrinkages: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Attempt to fit expected base pairs to detected peaks, with iterative shrinking if the fit is poor.
        The fit is evaluated based on the maximum absolute deviation between the fitted and expected base pairs.
        """
        diff: float = validation_threshold + 1
        shrinkages = 0
        bps = expected_bps

        # Attempt to fit; trim trailing peaks if the fit is too poor.
        while shrinkages < max_shrinkages:
            peak_idxs = peak_idxs[-len(bps) :]
            coeffs = np.polyfit(peak_idxs, bps, 2)
            fitted = np.polyval(coeffs, peak_idxs)
            diff = float(np.max(np.abs(fitted - bps)))
            if diff < validation_threshold:
                break
            else:
                bps = bps[:-1]
                shrinkages += 1

        if diff >= validation_threshold:
            raise ValueError(
                f"Size standard differs {diff:.2f} from expected values after {shrinkages} shrinkages; unable to fit profile."
            )

        if shrinkages > 0:
            LOGGER.info(
                "Size standard was shrunk %d times to fit the profile; max diff %.2f bp.",
                shrinkages,
                diff,
            )

        return peak_idxs, bps, diff

    def parse_size_standard(
        self,
        size_standard_lane: np.ndarray,
    ) -> Optional[SizeStandardParseResult]:
        # Ensure 1D array for peak finding.
        size_standard_lane = np.asarray(size_standard_lane).reshape(-1)

        # Detect peaks in the size-standard dye.
        size_standard_peaks_idxs = self.extract_ss_peaks(size_standard_lane)
        expected_bps = self.kit.size_standard.expected_bps

        # Fit expected base pairs to detected peaks, with iterative shrinking if the fit is poor.
        peak_idxs, bps, diff = self.attempt_fit(size_standard_peaks_idxs, expected_bps, self.validation_threshold, self.max_shrinkages)

        rescaled_indices, scaler = self.interpolate(peak_idxs, bps, size_standard_lane)

        return SizeStandardParseResult(
            rescaled_indices=rescaled_indices,
            scaler=scaler,
            fit_error=diff,
        )

    @staticmethod
    def extract_ss_peaks(array: np.ndarray) -> np.ndarray:
        peak_idxs = find_peaks_above_threshold(array, 300)

        close_idxs = np.where(np.diff(peak_idxs) <= 15)[0]
        return np.delete(peak_idxs, close_idxs)



class PowerPlexFusion6CScalingStrategy(ScalingStrategy):
    """
    Scaling strategy for PowerPlex Fusion 6C data.
    """

    def __init__(self, **kwargs):
        kit = STRKit(
            name="PPF6C",
            size_standard=WEN_ILS_BPS,
            num_dyes=6,
            panel_path=_POWER_PLEX_FUSION_6C_PANEL_PATH,
            panel=Panel(
                _POWER_PLEX_FUSION_6C_PANEL_PATH
            ),  # fill in when you load the panel
            markers=None,  # fill in with marker names for quick validation
            description="PPF6C dataset using WEN_ILS size standard.",
        )
        super().__init__(kit, basepair_start=65, basepair_end=475, scanpoint_resolution=4096)

    def dye_channel_colors(self) -> List[str]:
        return ['blue', 'green', 'black', 'red', 'purple', 'orange']

    def parse_size_standard(
        self, size_standard_lane: np.ndarray
    ) -> Optional[SizeStandardParseResult]:
        """_summary_

        Args:
            size_standard_lane: The dye lane containing the size standard of the EPG

        Raises:
            ValueError: When interpolating the size standard fails

        Returns:
            SizeStandardParseresult containing the rescaled indices and scaler.
        """
        size_standard_lane = np.asarray(size_standard_lane).reshape(-1)
        size_standard_expected = self.kit.size_standard.expected_bps

        # find the peaks in the size standard array
        size_standard_peaks_idxs = self.extract_ss_peaks(size_standard_lane)
        # only take the last 19 peaks excluding the final peak, validate by comparing their
        # relative distances to SIZE_STANDARD_BPS
        size_standard_peaks_idxs = size_standard_peaks_idxs[-20:-1]
        if not self.validate_ss_peaks(size_standard_peaks_idxs, size_standard_expected):
            return None


        rescaled_indices, scaler = self.interpolate(size_standard_peaks_idxs, size_standard_expected, size_standard_lane)

        return SizeStandardParseResult(
            rescaled_indices=rescaled_indices,
            scaler=scaler,
            fit_error=0.0,
        )

    @staticmethod
    def validate_ss_peaks(size_standard_peaks_idxs: np.ndarray, size_standard_expected: np.ndarray) -> bool:
        """
        Validate whether the identified peaks in the size standard are likely to
        correspond with the SIZE_STANDARD_BPS array, by checking whether 19 peaks
        have been identified and the relative distances between base pairs is
        between certain values. Returns True if all checks are passed, False otherwise.
        """

        if len(size_standard_peaks_idxs) != 19:
            # The number of selected peaks should be 19
            return False
        distances_between_peaks = np.absolute(
            np.diff((size_standard_peaks_idxs,
                     scipy.ndimage.shift(size_standard_peaks_idxs, shift=1, mode='nearest')
                     ), axis=0))[0, 1:]
        distance_basepairs = np.absolute(
            np.diff((size_standard_expected,
                     scipy.ndimage.shift(size_standard_expected, shift=1, mode='nearest')
                     ), axis=0))[0, 1:]
        relative_distances = distances_between_peaks / distance_basepairs
        if not np.all((relative_distances <= 13) & (relative_distances >= 7)):
            # The pixels per basepair are expected to be between 7 and 13
            return False
        return True

    @staticmethod
    def extract_ss_peaks(array: np.ndarray) -> np.ndarray:
        """
        Takes an array and extracts the indices of the size standard peaks, by comparing each
        value with the neighbours and a threshold. We may find 'flat' peaks (e.g.
        [500, 520, 520, 510]) or a peak within a close distance of another
        peak, therefore we filter the found indices based on distance.
        """
        peak_idxs = find_peaks_above_threshold(array, 180)
        # the final two peaks in the size standard are often lower than the other peaks, therefore we
        # try to find those in the end of the array with a lower threshold if we haven't found them yet
        split_idx = 8200  # TODO: can we find this dynamically or something?
        if len(peak_idxs) > 0 and peak_idxs[-1] <= split_idx:
            final_peak_idxs = find_peaks_above_threshold(array[split_idx:], 120) + split_idx
            peak_idxs = np.union1d(peak_idxs, final_peak_idxs)
        # look for peak that are close (within 15 pixels) and delete the peak on the first index. This
        # may go wrong when we have a situation like [1000, 1001, 800, 800, 799], then we
        # ideally want to keep the highest peak (1001), but now this one gets deleted and we keep 1000.
        close_idxs = np.where(np.diff(peak_idxs) <= 15)[0]
        return np.delete(peak_idxs, close_idxs)
