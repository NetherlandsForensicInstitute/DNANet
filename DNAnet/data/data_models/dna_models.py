import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Iterable, MutableMapping, Optional, Sequence, Set, Tuple, Union

import numpy as np

from DNAnet.typing import PathLike


DYE_MAPPING = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4}

LOGGER = logging.getLogger("dnanet")


@dataclass
class Allele:
    """
    Stores information regarding a measured allele

    :param name: name of allele
    :param base_pair: indicates the base pair position (size)
    :param left_bin: range of bin left of the base pair position
    :param right_bin: range of bin right of the base pair position
    :param height: peak height in RFU
    """
    name: str
    base_pair: float = None
    left_bin: float = None
    right_bin: float = None
    height: float = None

    @property
    def bin(self) -> np.ndarray:
        """
        Left and right bin of base pair position (size)
        """
        return np.array([self.left_bin, self.right_bin])[:, np.newaxis]


@dataclass
class Marker:
    """
    Stores information regarding a marker

    :param dye_row: row number of dye in which marker is measured
    :param name: name of marker
    :param alleles: Sequence of Allele within marker
    """
    dye_row: int
    name: str
    alleles: Sequence[Allele]


class Panel:
    """
    A class that retrieves information regarding alleles and markers. The panel can be
    read from file or can be directly instantiated by providing the raw contents.

    :param panel_path: path to a panel xml file with markers
    :param panel_contents: the direct contents of a panel
    """

    def __init__(self,
                 panel_path: Optional[PathLike] = None,
                 panel_contents: Optional[Sequence[Marker]] = None):
        self._panel_path = panel_path
        if panel_path:
            self._panel = self._parse_panel(panel_path)
        elif panel_contents:
            self._panel = panel_contents
        else:
            raise ValueError("Cannot instantiate Panel object, since panel path nor panel contents "
                             "are provided.")

    def get_allele_info(self, marker_name: str, allele_name: str) \
            -> Tuple[float, float, float]:
        """
        Retrieve allele information for a given allele name and
        corresponding marker name.

        :param marker_name: name of the marker
        :param allele_name: name of the allele
        :return: base pair position (size) of allele, left bin (base pair) of
            allele, and right bin (base pair) of allele
        """
        for marker in self._panel:
            for allele in marker.alleles:
                if marker.name == marker_name and allele.name == allele_name:
                    return (allele.base_pair,
                            allele.base_pair - allele.left_bin,
                            allele.base_pair + allele.right_bin)

        # could be rare allele. see if we can find the base (e.g. 21 for 21.1)
        if '.' in allele_name and len(allele_name.split('.')) == 2:
            base_allele, extra_base_pairs = allele_name.split('.')
            mid, left, right = self.get_allele_info(marker_name, base_allele)
            return (mid + int(extra_base_pairs),
                    left + int(extra_base_pairs),
                    right + int(extra_base_pairs))
        # TODO better fix - should not occur if we have the right panels
        if marker_name.startswith('DYS'):
            # could be Y profile
            return 10, 9, 11
        LOGGER.debug(f'Marker: {marker_name} and allele: {allele_name} '
                     f'not found in panel.')
        return 10, 9, 11

    def get_dye_row(self, marker_name: str) -> Optional[int]:
        """
        Get corresponding dye row the given marker name

        :param marker_name: name of the marker
        :return: dye (row) on which the marker is present, or None if
        the marker is unknown (e.g. from a different kit)
        """
        dye = [_marker.dye_row for _marker in self._panel
               if _marker.name == marker_name]
        if len(dye) > 1:
            raise ValueError(f'More than one dye found for marker {marker_name}')
        if len(dye) == 0:
            # unknown marker, e.g. Y profile
            return None
        return dye[0]

    def get_allele_name_by_dye_and_bp(self, dye_row: int, base_pair: float) \
            -> Tuple[str, str]:
        """
        Get the allele and marker name by the index of the dye row and
        the base pair.
        """
        if not self.dye_bp_to_allele_mapping.get(dye_row) or not \
                self.dye_bp_to_allele_mapping[dye_row].get(base_pair):
            raise ValueError(f"Unknown dye row {dye_row} or base_pair "
                             f"{base_pair} found.")

        return self.dye_bp_to_allele_mapping[dye_row][base_pair]

    @cached_property
    def dye_bp_to_allele_mapping(self) -> \
            Dict[int, Dict[float, Tuple[str, str]]]:
        """
        A mapping from dye index and base pair to the marker and allele name.
        """
        dye_bp_to_marker_allele = defaultdict(lambda: defaultdict(tuple))
        for marker in self._panel:
            for allele in marker.alleles:
                dye_bp_to_marker_allele[marker.dye_row][allele.base_pair] = \
                    (marker.name, allele.name)
        return dye_bp_to_marker_allele

    @staticmethod
    def _parse_panel(panel_path: PathLike) -> Sequence[Marker]:
        """
        Reads panel file with marker information

        :param panel_path: path to the panel file with markers
        :return: all Markers within the panel and the present Alleles
        """
        panel = []
        for event, elem in ET.iterparse(panel_path, events=('end',)):
            if elem.tag == 'Locus':
                alleles = elem.findall('Allele')
                panel.append(
                    Marker(
                        DYE_MAPPING.get(int(elem.find('DyeIndex').text)),
                        elem.find('MarkerTitle').text,
                        [
                            Allele(
                                a.attrib['Label'],
                                float(a.attrib['Size']),
                                float(a.attrib['Left_Binning']),
                                float(a.attrib['Right_Binning'])
                            )
                            for a in alleles
                        ]
                    )
                )
                elem.clear()  # Clear the element from the memory once you're done with it
        return panel


class Annotation:
    """
    This class represents a way of handling image annotations.

    :param labels: One or more labels that have been annotated for the image.
    :param image: A matrix annotation representing an image. To be used in
        segmentation tasks, where predictions are made at the pixel level.
    :param meta: A mapping where additional metadata about this annotation
        can be stored.
    """
    def __init__(self,
                 labels: Union[str, Iterable[str]] = None,
                 image: np.ndarray = None,
                 meta: MutableMapping[str, Any] = None):
        if labels is not None and not isinstance(labels, Set):
            labels = {labels} if isinstance(labels, str) else set(labels)
        self.labels: Set[str] = labels
        self.image: np.ndarray = image
        self.meta: MutableMapping[str, Any] = meta or dict()

    @property
    def label(self) -> str:
        """
        Returns the single label in this annotation as a `str`. Fails if no
        labels or multiple labels were annotated.
        """
        if not self.labels:
            raise TypeError("No labels in annotation")
        if len(self.labels) > 1:
            raise ValueError(f"Multiple ({len(self.labels)}) annotated labels")
        return next(iter(self.labels))

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) \
               and self.labels == other.labels \
               and np.array_equal(self.image, other.image) \
               and self.meta == other.meta

    def __hash__(self) -> int:
        labels = None if self.labels is None else frozenset(self.labels)
        return hash((
            labels,
            self.image.tobytes() if self.image is not None else None,
        ))

    def __str__(self) -> str:
        return f"Annotation(" \
               f"labels={self.labels}, " \
               f"image={self.image}, " \
               f"meta={self.meta}" \
               f")"

    def __repr__(self) -> str:
        return str(self)
