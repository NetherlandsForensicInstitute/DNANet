import csv
import logging
from binascii import crc32
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks

from DNAnet.data.data_models import Allele, Annotation, Marker, Panel
from DNAnet.data.data_models.base import Image
from DNAnet.data.parsing import get_peak_data, parse_called_alleles
from DNAnet.data.utils import (
    assert_image_data_valid_format,
    basepair_interpolator,
    find_peak_boundary,
    find_peak_idx_near_or_in_range,
    get_interpolated_basepairs,
    rescale_dye,
)
from DNAnet.typing import PathLike


LOGGER = logging.getLogger("dnanet")


class HIDImage(Image):
    """
    Image representation of the raw peaks from a HID file serving as a DNA profile

    :param path: location of HID file
    :param panel: the panel to be used
    :param annotations_file: the path of the csv/txt file that contains
        the annotations of the HID file.
    :param include_size_standard: include size standard in data
        if `true` all six dyes are included
        if `false` only the first five dyes are included
    :param annotation: any Annotation belonging to the image
    :param use_cache: whether retrieved peaks should be cached
    :param meta: meta information of the HID file.
    """
    THRESHOLD = 40  # 40 rfu is the lowest detection threshold

    def __init__(self,
                 path: PathLike,
                 panel: Optional[Panel] = None,
                 annotations_file: PathLike = None,
                 include_size_standard: bool = False,
                 annotation: Optional[Annotation] = None,
                 use_cache: bool = True,
                 meta: MutableMapping[str, Any] = None):
        self.path = path if isinstance(path, Path) else Path(path)
        self.annotations_file = annotations_file
        self.include_size_standard = include_size_standard
        self.use_cache = use_cache
        self.root = self.path.parent
        self._data: Optional[np.ndarray] = None
        self._annotation = annotation
        self._meta = meta or dict()
        self._scaler: Optional[np.ndarray] = None
        self._panel = panel

    @property
    def data(self) -> np.ndarray:
        if self.use_cache:
            if self._data is None:
                self._data = self._read()
            return self._data
        return self._read()

    @cached_property
    def dimensions(self) -> Tuple[int, int]:
        """
        Returns a `(height, width)` tuple of the dimensions of the image.
        """
        return self.data.shape[0], self.data.shape[1]

    @property
    def annotation(self):
        return self._annotation

    @property
    def meta(self) -> MutableMapping[str, Any]:
        return self._meta

    def _read(self) -> Optional[np.ndarray]:
        """
        Parse the raw hid image, validate the size standard and parse called alleles into a
        segmentation, if annotations are present.
        """
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))

        # Parse the raw hid image into a numpy array.
        if (profile := get_peak_data(self.path)) is None:
            return None
        # Use the size standard to translate the location in the profile (array) to base pairs
        interpolated_base_pairs = get_interpolated_basepairs(np.array(profile[-1]))
        if interpolated_base_pairs is None:
            # If the size standard does not pass validation, interpolated_base_pairs
            # becomes None and the image will be skipped when creating a dataset
            return None
        # Scale the profile using the size standard
        data = self._rescale_profile(profile,
                                     interpolated_base_pairs,
                                     self.include_size_standard)
        # Create a scaler, which is used to map a pixel index in the profile to a base pair
        # location, i.e. the first pixel is in fact BASE_PAIR_START, the last pixel is BASE_PAIR_END
        self._scaler = interpolated_base_pairs[rescale_dye(interpolated_base_pairs)]

        called_alleles = None
        # Determine the called alleles from the annotations file
        if self.annotations_file and self._panel and \
                (annotations_name := self.meta.get('annotations_name')):
            called_alleles = parse_called_alleles(self.annotations_file,
                                                  self._panel,
                                                  annotations_name)

        if called_alleles and self.annotation is None:
            # Parse the called alleles into a segmentation
            segmentation = self._get_segmentation(called_alleles, data.shape)
            self._annotation = Annotation(image=segmentation)
            self._meta['called_alleles'] = called_alleles

        if data is None:
            raise ValueError(f'Reading {self.path} resulted in None')
        try:
            assert_image_data_valid_format(data, n_color_channels=1)
        except ValueError as e:
            # TODO: Convert to uint8 successfully (strange behavior)
            if 'dtype of `data` must be' in str(e):
                pass
            else:
                raise

        # Cache the dimensions in case `use_cache` is False, so that we don't
        # have to reload the entire image when the dimensions are requested separately.
        self._dimensions = data.shape[:2]
        return data

    @property
    def hash(self) -> int:
        return crc32("/".join(self.path.relative_to(self.root).parts).encode())

    @property
    def scaler(self) -> np.ndarray:
        """
        Array in which the value are the base pairs and
        the index represents its position within the
        (scaled) array/data of the profile, e.g.:

        [65, 66, 67, ...,  474, 474.5, 475]
        in this example base-pair 67 should be placed on
        index 3 of an array. The size of the scalar depends
        on the `utils.RESCALE_SIZE` constant

        TODO: INCLUDE LOGIC
        TODO: | np.argmin(np.abs(self.scaler - allele.bin)
        """
        if self._scaler is None:
            # to avoid missing the scaler when we have not yet read the file.
            self._read()
        return self._scaler[np.newaxis, :]

    @staticmethod
    def _rescale_profile(profile: np.ndarray,
                         interpolated_base_pairs: np.ndarray,
                         include_standard: bool) -> np.ndarray:
        """
        Rescale profile based on interpolated base pairs.

        :param profile: array of dyes in chronological order
        :param interpolated_base_pairs: the interpolated base pairs
        :param include_standard: if the size standard should be included
            in the final profile/data
        :return: parsed profile as array
        """
        # Select profile based on include_standard flag
        selected_profile = profile if include_standard else profile[:-1]
        data = selected_profile[:, rescale_dye(interpolated_base_pairs)]
        return data[..., np.newaxis]

    def _get_segmentation(self,
                          called_alleles: Sequence[Marker],
                          shape: Tuple[int, ...]) -> np.ndarray:
        """
        Creates a binary mask based on the locations of called alleles in the annotation. Use
        the scaler to determine for an allele bin (a single base pair), the pixel location in
        the segmentation array.
        """
        image = np.zeros(shape, dtype=np.int8)
        for marker in called_alleles:
            for allele in marker.alleles:
                image[
                    marker.dye_row,
                    slice(*tuple(np.argmin(np.abs(self.scaler - allele.bin), axis=1))),
                    0
                ] = 1
        return image

    def adjust_annotations(self, adjustment_type: str = 'top') -> 'HIDImage':
        """
        Adjust the annotation of the image or the spu annotation in case of 'adjust_spu' is True.
        If `adjustment_type` is 'top', (by default) we label the top of the peak, instead of the
        entire bin. If the type is 'complete', we find the entire peak and label this.
        Note that the original image annotations are overwritten.
        """
        profile = self.data  # force data to be read to generate annotations
        annotations = self.annotation.image
        if annotations is None:
            LOGGER.warning(f"No annotations found for file {self.path} when "
                           f"adjusting annotations.")
            return self

        for layer, dye in enumerate(profile):
            # find indices of groups of positive annotations
            _annotations, _ = np.where(annotations[layer] == 1)
            if _annotations.size == 0:  # no annotation present in this dye
                continue
            annotation_groups = np.split(_annotations, np.where(np.diff(_annotations) != 1)[0] + 1)
            for ann_group in annotation_groups:
                annotations[layer, ann_group, 0] = 0.
                peak_idx = find_peak_idx_near_or_in_range(dye, ann_group,
                                                          self.THRESHOLD)

                if peak_idx.size == 0:
                    LOGGER.warning(f"No peak found above {self.THRESHOLD}rfu. "
                                   f"Original annotation is removed "
                                   "and no adjustment is applied "
                                   f"({self.path}, dye {layer}, bin {ann_group}, "
                                   f"rfus {dye[ann_group].flatten()}).")
                else:
                    if adjustment_type == 'complete':
                        # find the boundary of the peak and annotate the range
                        start, end = find_peak_boundary(dye, int(peak_idx),
                                                        self.THRESHOLD)
                        annotations[layer, np.arange(start, end + 1), 0] = 1.
                    elif adjustment_type == 'top':
                        # label only the top of the peak
                        annotations[layer, peak_idx, 0] = 1.
                    else:
                        raise ValueError("Unknown adjustment type found: "
                                         f"{adjustment_type}. Please provide"
                                         " either `top` or `complete`.")
        return self

    def __repr__(self):
        return f"HIDImage({self.path.name})"


class Ladder(HIDImage):
    """
    Base class for a ladder .hid image. The ladder contains (almost) all alleles present on
    every dye (these can be found in the `alleles_in_ladder` csv file) and by finding the exact
    locations of these peaks in every dye, we can rescale the .hid images of actual DNA profiles.
    This is necessary since sometimes alleles are not located at the base pairs we expect them to
    be, i.e. the base pair locations of the default panel (loaded from the .xml file).
    """
    _alleles_in_ladder = None

    def __init__(self,
                 path: PathLike,
                 default_panel: Panel,
                 use_cache: bool = True):
        super().__init__(path=path,
                         panel=default_panel,
                         include_size_standard=True,
                         use_cache=use_cache)
        if Ladder._alleles_in_ladder is None:
            # load the alleles that should be present in every ladder
            Ladder._alleles_in_ladder = self.read_alleles_in_ladder()

        # we can try to find peaks if the ladder has valid data
        self.peak_indices = self.get_peak_indices() if self.data is not None else None
        # we can adjust the panel and find the corrected base pair locations of alleles if the
        # ladder has the correct number of peaks on every dye
        self._panel = self.adjust_panel(default_panel) if self.peak_indices else None

    @property
    def alleles_in_ladder(self) -> Dict:
        return self._alleles_in_ladder

    @property
    def panel(self) -> Panel:
        return self._panel

    @staticmethod
    def read_alleles_in_ladder() -> Dict[int, List[Tuple[str, str]]]:
        """
        Read 'Ladder digitale profielenstraat.csv' (with columns 'Marker', 'Allele' and 'Dye') into
        a dictionary with the dye rows as keys and Marker/Allele names as values. This file
        contains allele alleles that should be present in the ladder. Note that this file, and
        therefore neither the ladder, do not contain all alleles possible alleles.
        """
        lines = defaultdict(list)
        with open('resources/data/Ladder digitale profielenstraat.csv',
                  mode='r', newline='', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                lines[int(row['Dye'])].append((row['Marker'], row['Allele']))
        return lines

    def get_peak_indices(self) -> Optional[List[np.array]]:
        """
        Retrieve all peak indices per dye (using a simple peak finding algorithm) and check per
        dye whether the number of peaks in the ladder correspond to the expected number of peaks
        provided in `alleles_in_ladder`. If this is not the case, return None. The peak indices
        are pixel indices on the array.
        """
        all_peaks = []
        for dye_row in range(self.data.shape[0] - 1):
            dye = self.data[dye_row, :, 0]
            dynamic_threshold = np.max(dye) * 0.6

            # To be able to detect peaks at index 0, we pad the data with a 0
            padded_data = np.concatenate((np.zeros(1), dye))
            peaks, _ = find_peaks(padded_data, height=dynamic_threshold)
            peaks = peaks - 1  # remove padding

            if len(peaks) == len(self.alleles_in_ladder[dye_row]):
                all_peaks.append(peaks)
            else:
                LOGGER.warning(f"Expected {len(self.alleles_in_ladder[dye_row])} peaks on "
                               f"dye row {dye_row}, but found {len(peaks)} for ladder {self.path}.")
                return None
        return all_peaks

    def adjust_panel(self, original_panel: Panel) -> Panel:
        """
        Create an adjusted panel (meaning we adjust the base pairs of the alleles) using the
        found peak indices of the peaks in the ladder, using inter- and extrapolation.

        For more information on panel adjustment see:
        https://softgenetics.com/PDF/CalibratingPanelautoadjust_icon.pdf

        :param original_panel: The original panel, loaded from
        `resources/data/SGPanel_PPF6C_SPOOR.xml` from which we know all possible alleles.
        :return: Panel with the same alleles as the original panel, but adjusted base pairs based
        on the peaks in the ladder.
        """
        adjusted_panel = []
        for dye_row in range(self.data.shape[0] - 1):
            alleles_ladder_dye = self.alleles_in_ladder[dye_row]
            panel_markers = [marker for marker in original_panel._panel
                             if marker.dye_row == dye_row]

            # map marker/allele names to the base pair locations in the ladder
            marker_allele_to_bp = {
                (marker, allele): self.scaler[0, peak_idx]
                for (marker, allele), peak_idx in
                zip(alleles_ladder_dye, self.peak_indices[dye_row])
            }

            for marker in panel_markers:
                adjusted_alleles = []
                for idx, allele in enumerate(marker.alleles):
                    if allele.name in [a.name for a in adjusted_alleles]:
                        # we have already adjusted this allele
                        continue
                    ladder_base_pair = marker_allele_to_bp.get((marker.name, allele.name))
                    if ladder_base_pair:
                        # this allele was present in the ladder, therefore we can directly adjust
                        # the location with the ladder base pair location.
                        adjusted_alleles.append(
                            Allele(name=allele.name,
                                   base_pair=float(ladder_base_pair),
                                   left_bin=allele.left_bin,
                                   right_bin=allele.right_bin))
                    else:
                        # otherwise we should inter- or extrapolate using the other alleles
                        # present on the marker
                        result_allele = self._handle_missing_bp_in_ladder(
                            marker, allele, idx, marker_allele_to_bp)
                        adjusted_alleles.extend(result_allele)

                adjusted_panel.append(Marker(dye_row=dye_row,
                                             name=marker.name,
                                             alleles=adjusted_alleles))

        return Panel(panel_contents=adjusted_panel)

    def _handle_missing_bp_in_ladder(self,
                                     marker: Marker,
                                     allele: Allele,
                                     index: int,
                                     marker_allele_to_bp: Dict[Tuple[str, str], float]) -> \
            Sequence[Allele]:
        """
        Inter-/extrapolate to find the base pair from an allele in the panel that is not present
        in the ladder.

        :param marker: The marker to inter-/extrapolate basepair for.
        :param allele: The allele to inter-/extrapolate basepair for.
        :param index: The index of the allele in the marker.
        :param marker_allele_to_bp: mapping from marker/alleles to base pairs from the ladder.
        :return: A list of Alleles containing the inter-/extrapolated basepair.
        """
        # find all alleles after the provided alleles we can possibly use for inter-/extrapolation
        following_alleles = [
            allele for allele in marker.alleles[index + 1:]
            if marker_allele_to_bp.get((marker.name, allele.name))
        ]

        adjusted_alleles = []
        if index == 0:
            # We are at the beginning of the profile, therefore extrapolate with the next two
            # alleles with known ladder base pairs, that should always be present
            bp_in_panel = [allele.base_pair for allele in following_alleles[:2]]
            bp_in_ladder = [marker_allele_to_bp.get((marker.name, allele.name))
                        for allele in following_alleles[:2]]

            adjusted_alleles.append(self._extrapolate_base_pair(
                bp_in_panel, bp_in_ladder, marker, allele, marker_allele_to_bp))
        else:
            if not following_alleles:
                # We are at the end of the profile, therefore extrapolate with two previous
                # alleles with known ladder base pairs
                previous_alleles = marker.alleles[index - 2: index]
                bp_in_panel = [allele.base_pair for allele in previous_alleles]
                bp_in_ladder = [marker_allele_to_bp.get((marker.name, allele.name))
                            for allele in previous_alleles]

                adjusted_alleles.append(self._extrapolate_base_pair(
                    bp_in_panel, bp_in_ladder, marker, allele, marker_allele_to_bp))
            else:
                # Interpolate between previous and next allele with known ladder base pair. There
                # is always a previous allele since we move from left to right. This interpolation
                # may result in the adjustment of multiple alleles that have no ladder base pair,
                # in between the `previous` and `next` allele.
                prev_allele = marker.alleles[index - 1]
                prev_bp_in_ladder = marker_allele_to_bp[(marker.name, prev_allele.name)]
                prev_bp_in_panel = prev_allele.base_pair

                next_allele = following_alleles[0]
                next_bp_in_ladder = marker_allele_to_bp[(marker.name, next_allele.name)]
                next_bp_in_panel = next_allele.base_pair
                next_idx = marker.alleles.index(next_allele)

                # Interpolate all alleles in between the previous and next
                interp = basepair_interpolator(
                    indices=[prev_bp_in_panel, next_bp_in_panel],
                    original_x_values=[prev_bp_in_ladder, next_bp_in_ladder]
                )
                for following_allele in marker.alleles[index:next_idx]:
                    interpolated_bp = interp(following_allele.base_pair)
                    marker_allele_to_bp[(marker.name, following_allele.name)] = \
                        float(interpolated_bp)

                    adjusted_alleles.append(Allele(
                        name=following_allele.name,
                        base_pair=float(interpolated_bp),
                        left_bin=following_allele.left_bin,
                        right_bin=following_allele.right_bin))

        return adjusted_alleles

    @staticmethod
    def _extrapolate_base_pair(indices: List[float],
                               x_values: List[float],
                               marker: Marker,
                               allele: Allele,
                               marker_allele_to_bp: Dict[
                                   Tuple[str, str], float]) -> Allele:
        """
        Extrapolates base pair for specified allele and returns an `Allele` object with the
        newly computed basepair location.
        """
        interp = basepair_interpolator(indices=indices,
                                       original_x_values=x_values,
                                       extrapolate=True)
        extrapolated_bp = interp(allele.base_pair)
        # update the marker/allele to basepair mapping
        marker_allele_to_bp[(marker.name, allele.name)] = float(extrapolated_bp)

        return Allele(name=allele.name,
                      base_pair=float(extrapolated_bp),
                      left_bin=allele.left_bin,
                      right_bin=allele.right_bin)

    def __eq__(self, other):
        if not isinstance(other, Ladder):
            return False

        return self.path == other.path and np.array_equal(self.data, other.data) and all(
            np.array_equal(s, o) for s, o in zip(self.peak_indices, other.peak_indices))

    def __repr__(self):
        return f"Ladder({self.path.name})"
