import logging
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Iterable, MutableMapping, Optional, Sequence, Set, Tuple, Union, List

import numpy as np

from DNAnet.data.strategies.strategy_registry import StrategyRegistry
from DNAnet.data.utils import fill_lut_range
from DNAnet.typing import PathLike

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
    base_pair: float | None = None
    left_bin: float | None = None
    right_bin: float | None = None
    height: float | None = None

    @property
    def bin(self) -> np.ndarray:
        """
        Left and right bin of base pair position (size)
        """
        return np.array([self.left_bin, self.right_bin])[:, np.newaxis]

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the Allele.
        """
        return {
            "name": self.name,
            "base_pair": self.base_pair,
            "left_bin": self.left_bin,
            "right_bin": self.right_bin,
            "height": self.height
        }

    @classmethod
    def from_dict(cls, allele_dict: Dict[str, Any]):
        """
        Creates an Allele instance from a dictionary representation.
        """
        return cls(
            name=allele_dict['name'],
            base_pair=allele_dict.get('base_pair'),
            left_bin=allele_dict.get('left_bin'),
            right_bin=allele_dict.get('right_bin'),
            height=allele_dict.get('height')
        )


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

    @property
    def is_autosomal(self) -> bool:
        """
        Returns True if the marker is autosomal (i.e., not AMEL or DYS-prefixed).
        """
        return not (self.name == "AMEL" or self.name.startswith("DYS"))

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the Marker.
        """
        return {
            "dye_row": self.dye_row,
            "name": self.name,
            "alleles": [allele.to_dict() for allele in self.alleles]
        }

    @classmethod
    def from_dict(cls, marker_dict: Dict[str, Any]):
        """
        Creates a Marker instance from a dictionary representation.
        """
        alleles = [Allele.from_dict(a) for a in marker_dict['alleles']]
        return cls(
            dye_row=marker_dict['dye_row'],
            name=marker_dict['name'],
            alleles=alleles
        )

    @property
    def min_bp(self) -> float:
        """Returns the leftmost bin edge of the smallest allele in the marker."""
        if not self.alleles:
            return float('inf')
        return min(a.base_pair - a.left_bin for a in self.alleles if a.base_pair is not None)

    @property
    def max_bp(self) -> float:
        """Returns the rightmost bin edge of the largest allele in the marker."""
        if not self.alleles:
            return float('-inf')
        return max(a.base_pair + a.right_bin for a in self.alleles if a.base_pair is not None)


class Panel:
    """
    A class that retrieves information regarding alleles and markers. The panel can be
    read from file or can be directly instantiated by providing the raw contents.

    Optionally builds a lookup table for the lookup of marker names by dye row and base pair.
    The marker name is retrieved when the basepair value is between the leftmost point of the smallest allele bin
    and the rightmost point of the largest allele bin in the marker.
    This lookup table can be efficiently used in by models during training as retrieval is O(1).
    The lookup table can be accessed by the 'get_marker_name_by_dye_and_bp' method.

    :param panel_path: path to a panel xml file with markers
    :param panel_contents: the direct contents of a panel
    :param precomputer_marker_lookup: whether to precompute marker lookup table.
    """

    def __init__(self,
                 panel_path: Optional[PathLike] = None,
                 panel_contents: Optional[Sequence[Marker]] = None,
                 precomputer_marker_lookup: bool = True):
        self._panel_path = panel_path
        if panel_path:
            self._panel = self._parse_panel(panel_path)
        elif panel_contents:
            self._panel = panel_contents
        else:
            raise ValueError("Cannot instantiate Panel object, since panel path nor panel contents "
                             "are provided.")

        if precomputer_marker_lookup:
            # Precompute O(1) marker lookup by (dye_row, base_pair)
            self._marker_bp_scale: int = 10 # 0.1 bp resolution
            self._marker_name_lut: Dict[int, List[Optional[str]]] = {}
            self._marker_name_lut_offset: Dict[int, int] = {}
            self._build_marker_name_lut()


    def get_allele_basepair_and_bins(self, marker_name: str, allele_name: str) \
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
            mid, left, right = self.get_allele_basepair_and_bins(marker_name, base_allele)
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
        dye_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4} # TODO: should this be hardcoded?
        panel = []
        for event, elem in ET.iterparse(panel_path, events=('end',)):
            if elem.tag == 'Locus':
                alleles = elem.findall('Allele')
                panel.append(
                    Marker(
                        dye_mapping.get(int(elem.find('DyeIndex').text)),
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

    def fill_allele_bins(self, called_alleles: Sequence[Marker]) -> None:
        """
        Fill in the base pair, left bin and right bin information for each allele
        in the called_alleles based the allele names. Also fills in the dye_row for each marker.
        The called_alleles are modified in place.
        Args:
            called_alleles: Sequence[Marker] - The called alleles to fill in.
        """
        for marker in called_alleles:
            for allele in marker.alleles:
                basepair, left_bin, right_bin = self.get_allele_info(marker.name, allele.name)
                allele.base_pair = basepair
                allele.left_bin = left_bin
                allele.right_bin = right_bin

            if marker.dye_row == -1:
                marker.dye_row = StrategyRegistry.get_scaling_strategy().marker_name_to_dye_idx().get(marker.name, -1)

    def get_marker_name_by_dye_and_bp(self, dye_row: int, base_pair: float) -> str:
        """
        O(1) lookup: return marker.name where dye_row matches and base_pair lies
        inside the marker's overall allele bin coverage.
        returns "Out of Bin" if no marker found.
        """
        if not hasattr(self, '_marker_name_lut'):
            raise ValueError("Marker name lookup table not built.")
        lut = self._marker_name_lut.get(dye_row)
        if lut is None or len(lut) == 0:
            return "Out of Bin"

        k = int(round(base_pair * self._marker_bp_scale))
        i = k + self._marker_name_lut_offset[dye_row]
        if i < 0 or i >= len(lut) or lut[i] is None:
            return "Out of Bin"

        return lut[i]


    def _build_marker_name_lut(self) -> None:
        """
        Builds a per-dye lookup table mapping base-pair positions to marker names
        using O(1) array indexing.
        """
        by_dye: Dict[int, List[Marker]] = defaultdict(list)
        for m in self._panel:
            by_dye[m.dye_row].append(m)

        for dye_row, markers in by_dye.items():
            # Collect valid ranges using the Marker properties
            ranges = []
            for m in markers:
                start, end = m.min_bp, m.max_bp
                # Skip markers with no valid alleles (inf/-inf)
                if start == float('inf') or end == float('-inf'):
                    continue
                ranges.append((start, end, m.name))

            if not ranges:
                self._marker_name_lut[dye_row] = []
                self._marker_name_lut_offset[dye_row] = 0
                continue

            # Determine global bounds for the LUT
            global_min = min(r[0] for r in ranges)
            global_max = max(r[1] for r in ranges)

            # Discretize bounds to integer indices
            min_idx = int(math.floor(global_min * self._marker_bp_scale))
            max_idx = int(math.ceil(global_max * self._marker_bp_scale))

            # Initialize LUT
            lut_len = max_idx - min_idx + 1
            lut = [None] * lut_len

            # offset shifts a discretized bp index to 0-based list index
            # usage: lut[bp_idx + offset]
            offset = -min_idx

            for start, end, name in ranges:
                s_k = int(math.floor(start * self._marker_bp_scale))
                e_k = int(math.ceil(end * self._marker_bp_scale))

                # Fill the range in the LUT, handling overlaps by splitting evenly
                fill_lut_range(lut, offset, s_k, e_k, name, ranges, self._marker_bp_scale)

            self._marker_name_lut[dye_row] = lut
            self._marker_name_lut_offset[dye_row] = offset



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
