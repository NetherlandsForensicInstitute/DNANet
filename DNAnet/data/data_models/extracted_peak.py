from typing import Optional, Sequence, MutableSequence, MutableMapping, Any

import numpy as np
import scipy

from DNAnet.data.data_models import Marker, Allele, Annotation
from DNAnet.data.data_models.base import Image
from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.data.preprocessing.peak_utils import build_peak_data, find_bin, markers_contain_allele, slice_with_padding
from DNAnet.evaluation.visualizations import _get_marker_bin

SCAN_TO_BASE = scipy.interpolate.interp1d(
    [0, 4096], [65, 475], fill_value="extrapolate"
)

class ExtractedPeak(Image):
    """
    Represents a peak extracted from a HIDImage, containing the data around the peak center and its annotation.
     The annotation is based on the presence of an allele in the annotation image at the peak center (with optional padding).
     If ground truth called alleles are available in the HIDImage metadata, the annotation is based on whether the predicted
     allele (based on the peak base pair) is present in the ground truth called alleles for the corresponding marker.

    """
    def __init__(self,
                 image: HIDImage,
                 dye_index: int,
                 peak_center: int,
                 window_size: int,
                 peak_height: int,
                 annotation_padding: int = 1,
                 include_raw_annotation: bool = False,
                 use_ground_truth: bool = True,
                 include_max_pool_dyes: bool = False):
        self.window_start = peak_center - window_size // 2
        self.peak_height = peak_height
        self.window_size = window_size
        self.original_peak_center_index = peak_center
        self.dye_index = dye_index
        self.include_max_pool_dyes = include_max_pool_dyes
        self.peak_basepair = SCAN_TO_BASE(peak_center)
        self.markers = image._panel._panel
        self.marker: Optional[Marker] = self._find_marker()
        self._data = build_peak_data(image.data, dye_index, self.window_start, window_size, include_max_pool_dyes)
        self._annotation = self._create_annotation(image, padding=annotation_padding, include_raw=include_raw_annotation, use_ground_truth=use_ground_truth)
        self.is_normalized = False


    def get_marker_name(self) -> str:
        """
        Returns the name of the marker that contains the peak base pair.
        Returns "Out of Bin" if no marker is found.

        """
        if marker := self.get_marker():
            return marker.name

        return "Out of Bin"

    def get_marker(self) -> Optional[Marker]:
        """
        Returns the marker that contains the peak base pair.
        If no marker is found, returns None.
        """
        return self.marker

    def _find_marker(self) -> Optional[Marker]:
        for marker in self.markers:
            lbin, rbin = _get_marker_bin(marker).squeeze()
            if marker.dye_row == self.dye_index and lbin <= self.peak_basepair <= rbin:
                return marker

        return None

    def get_allele(self) -> Optional[Allele]:
        marker = self.get_marker()
        if marker is None:
            return None

        allele = find_bin(marker, self.peak_basepair)
        if allele is None:
            return None

        return Allele(allele.name, allele.base_pair, allele.left_bin, allele.right_bin, self.peak_height)


    @property
    def data(self) -> np.ndarray:
        return self._data


    def _create_annotation(self, img: HIDImage, padding: int, include_raw: bool = True, use_ground_truth: bool = True) -> Annotation:
        """
        Adds a label based on the annotation image.
        If the peak center (with optional padding) is annotated as an allele, the label is "allele".
        Otherwise, the label is "noise".
        Args:
            padding: padding around the peak center to consider for annotation.
        """
        full_annotation = img.annotation.image

        # If ground truth alleles are available, use them for labeling
        if img.meta['called_alleles'] is not None and use_ground_truth:
            ground_truth_alleles: Sequence[Marker] = img.meta['called_alleles']
            if markers_contain_allele(ground_truth_alleles, self.get_allele(), self.get_marker()):
                label = "allele"
            else:
                label = "noise"
            if include_raw:
                cropped_annotation = slice_with_padding(full_annotation, self.dye_index, self.window_start, self.window_size)
                return Annotation(labels=label, image=cropped_annotation)
            return Annotation(labels=label)

        cropped_annotation = slice_with_padding(full_annotation, self.dye_index, self.window_start, self.window_size)

        start = self.window_size // 2 - padding
        end = self.window_size // 2 + padding + 1
        if cropped_annotation[start:end].sum() > 0:
            label = "allele"
        else:
            label = "noise"

        if include_raw:
            return Annotation(labels=label, image=cropped_annotation)
        return Annotation(labels=label)


    @property
    def annotation(self) -> Annotation:
        return self._annotation

    @property
    def annotations(self) -> MutableSequence[Annotation]:
        return [self._annotation]

    @property
    def is_allele(self) -> bool:
        return self._annotation.label == "allele"

    @property
    def meta(self) -> MutableMapping[str, Any]:
        return {
            "dye_index": self.dye_index,
            "original_peak_center_index": self.original_peak_center_index,
            "window_size": self.window_size,
            "peak_basepair": self.peak_basepair,
            "marker": self.get_marker(),
        }

    @property
    def hash(self) -> int:
        return hash((self.dye_index, self.original_peak_center_index, self.window_size, self.peak_height, self.include_max_pool_dyes))

    def __eq__(self, other):
        if not isinstance(other, ExtractedPeak):
            return NotImplemented
        return (self.dye_index == other.dye_index and
                self.original_peak_center_index == other.original_peak_center_index and
                self.window_size == other.window_size and
                self.peak_height == other.peak_height and
                self.include_max_pool_dyes == other.include_max_pool_dyes)

    def __hash__(self):
        return self.hash
