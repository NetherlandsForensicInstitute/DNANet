import csv
import logging
from binascii import crc32
from functools import cached_property
from pathlib import Path
from typing import Any, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from DNAnet.data.dataset_compatibility.dataset_strategy import DatasetStrategy
from DNAnet.data.kit_compatibility.kit_strategy import KitStrategy, NfiKitStrategy
from DNAnet.data.dataset_compatibility.dataset_strategy import NFI_RND_DatasetStrategy

from DNAnet.data.data_models import Annotation, Marker, Panel
from DNAnet.data.data_models.base import Image
from DNAnet.data.kit_compatibility.lane_standards import InternalSizeStandard
from DNAnet.data.parsing import get_peak_data, parse_called_alleles
from DNAnet.data.utils import (
    assert_image_data_valid_format,
    basepair_interpolator,
    find_peak_boundary,
    find_peak_idx_near_or_in_range,
    extract_ss_peaks_new_unify,
    rescale_dye_new_unify
)


LOGGER = logging.getLogger("dnanet")


class ProvedItHIDImage(Image):
    """
    Image representation of the raw peaks from a HID file serving as a DNA profile

    :param path: location of HID file
    :param panel: the panel to be used
    :param annotations_file: the path of the csv/txt file that contains
        the annotations of the HID file.
    :param include_size_standard: include size standard in the data attribute.
        if `true` all six dyes are included. For inspection of the HID file.
        if `false` only the first five dyes are included. For training + testing models.
    :param annotation: any Annotation belonging to the image
    :param use_cache: whether retrieved peaks should be cached
    :param meta: meta information of the HID file.
    """
    THRESHOLD = 40  # 40 rfu is the lowest detection threshold

    def __init__(self,
                 path: Path,
                 dataset_strategy: Optional[DatasetStrategy] = None,
                 kit_strategy: Optional[KitStrategy] = None,
                 panel: Optional[Panel] = None,
                 annotations_file: Path = None,
                 include_size_standard: bool = False,
                 annotation: Optional[Annotation] = None,
                 use_cache: bool = True,
                 meta: MutableMapping[str, Any] = None):
        
        # Provide legacy-friendly defaults when strategies are not supplied.
        if dataset_strategy is None:
            if panel is None:
                raise ValueError("Panel is required when dataset_strategy is not provided.")
            dataset_strategy = NFI_RND_DatasetStrategy(
                panel=panel,
                genotypes_path="resources/data/2p_5p_Dataset_NFI/References",
            )
        if kit_strategy is None:
            kit_strategy = NfiKitStrategy(
                size_standard=InternalSizeStandard.WEN_ILS
            )

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
        self.dataset_strategy = dataset_strategy
        self.kit_strategy = kit_strategy

    @property
    def data(self) -> np.ndarray:
        if self.use_cache:
            if self._data is None:
                self._data = self._read()
            return self._data
        return self._read()
    
    @data.setter
    def data(self, value: np.ndarray):
        self._data = value

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
        self.profile = profile = get_peak_data(self.path)
        if profile is None:
            return None
        

        size_standard_dye_lane = np.array(profile[-1])
        try:
            ss = self.kit_strategy.parse_size_standard(size_standard_dye_lane)
        except ValueError as e:
            LOGGER.warning(f"Size standard invalid for {self.path.name}: {e}")
            return None

        data = self._rescale_profile(
            profile,
            ss.rescaled_indices,
            self.include_size_standard,
        )
        self._scaler = ss.scaler
        

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
            self._annotation = Annotation(image=segmentation) # where the annotation is ASSIGNED
            self._meta['called_alleles'] = called_alleles

        # But what if there is no annotations file, only genotype info?
        # This is ofc hardcoded for the ProvedIt dataset for now
        if self.annotation is None and self._panel:
            try:
                true_alleles = self.dataset_strategy.load_donor_alleles(self.path.stem)
                segmentation = self._get_segmentation(true_alleles, data.shape)
                self._annotation = Annotation(image=segmentation)
                self._meta["called_alleles"] = true_alleles
            except ValueError as e:
                LOGGER.warning(f"Could not load true alleles for {self.path}: {e}")
                # If we cannot load the true alleles, we do not set the annotation.


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
    def _rescale_profile(
        profile: np.ndarray,
        rescale_indices: np.ndarray,
        include_standard: bool,
    ) -> np.ndarray:
        """Rescale profile based on precomputed rescale indices.

        :param profile: array of dyes in chronological order
        :param rescale_indices: indices of the original profile corresponding to
            each pixel in the rescaled profile
        :param include_standard: if the size standard should be included in the
            final profile/data
        :return: parsed profile as array
        """
        # Select profile based on include_standard flag
        selected_profile = profile if include_standard else profile[:-1]
        data = selected_profile[:, rescale_indices]
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

    def adjust_annotations(self, adjustment_type: str = 'top') -> 'ProvedItHIDImage':
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
                    # LOGGER.warning(f"No peak found above {self.THRESHOLD}rfu. "
                    #                f"Original annotation is removed "
                    #                "and no adjustment is applied "
                    #                f"({self.path}, dye {layer}, bin {ann_group}, "
                    #                f"rfus {dye[ann_group].flatten()}).")
                    pass
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
        return f"ProvedItHIDImage({self.path.name})"
