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


class OldHIDImage(Image):
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
            segmentation = self._get_segmentation(self.scaler, called_alleles, data.shape)
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

    @classmethod
    def _get_segmentation(
        cls,
        scaler, 
        called_alleles: Sequence[Marker],
        shape: Tuple[int, ...]
    ) -> np.ndarray:
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
                    slice(*tuple(np.argmin(np.abs(scaler - allele.bin), axis=1))),
                    0
                ] = 1
        return image

    def adjust_annotations(self, adjustment_type: str = 'top') -> 'OldHIDImage':
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
        return f"OldHIDImage({self.path.name})"


