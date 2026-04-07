"""Panel — the reference allele catalogue for a forensic DNA kit.

Design pattern: **Repository**
    A Panel acts as a *repository* for marker and allele information. It loads
    data from an external source (XML panel file) and provides query methods
    to look up alleles by marker name, dye row, or base-pair position.

    The original Panel class mixed three concerns:
    1. Parsing XML files
    2. Storing the panel data
    3. Building a lookup table (LUT) for O(1) marker lookups

    In this clean version we keep these together (they're tightly coupled),
    but clarify the design:
    - Construction: `Panel.from_xml(path)` (class method, explicit factory)
    - Querying: clear method names with type-safe signatures
    - LUT: built lazily on first access via `@cached_property`

Design pattern: **Factory Method**
    `Panel.from_xml()` is a *factory method* — it encapsulates the XML parsing
    logic and returns a fully constructed Panel. This separates "how to parse
    an XML file" from "what a Panel is", and makes it easy to add other
    factories later (e.g. `Panel.from_json()`, `Panel.from_dataframe()`).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from functools import cached_property
from typing import Sequence

from loguru import logger

from dnanet.core.allele import Allele
from dnanet.core.marker import Marker
from dnanet.core.types import PathLike

# Default dye mapping: PPF6C style (1-based HID indices, skipping dye 5)
_DEFAULT_HID_DYE_MAPPING: dict[int, int] = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4}


class Panel:
    """Reference panel of markers and alleles for a forensic DNA kit.

    A panel defines which markers exist, what alleles are expected at each
    marker, and the base-pair positions / bin boundaries for each allele.

    Args:
        markers: The complete list of markers in this panel.
    """

    def __init__(self, markers: Sequence[Marker] = ()) -> None:
        self._markers = tuple(markers)

    # -- Factory methods -------------------------------------------------- #

    @classmethod
    def from_xml(
        cls,
        path: PathLike,
        hid_dye_mapping: dict[int, int] | None = None,
    ) -> Panel:
        """Parse a panel from an SGPanel XML file.

        This is the standard format exported by forensic DNA analysis
        software (e.g. GeneMapper). Each ``<Locus>`` element contains
        allele definitions with size and bin boundaries.

        Args:
            path: Path to the panel XML file.
            hid_dye_mapping: Maps 1-based HID dye indices to 0-based channel
                rows. If ``None``, uses the default PPF6C mapping
                ``{1: 0, 2: 1, 3: 2, 4: 3, 6: 4}``.

        Returns:
            A fully constructed Panel instance.
        """
        mapping = hid_dye_mapping or _DEFAULT_HID_DYE_MAPPING
        markers: list[Marker] = []

        for _event, elem in ET.iterparse(str(path), events=("end",)):
            if elem.tag != "Locus":
                continue
            
            hid_idx = int(elem.find("DyeIndex").text) # type: ignore
            if hid_idx not in mapping:
                # Skip dye channels not in the mapping (e.g. size standard)
                elem.clear()
                continue
            dye_index = mapping[hid_idx]
            marker_name = elem.find("MarkerTitle").text # type: ignore
            if marker_name is None:
                continue
            alleles = frozenset(
                Allele(
                    name=a.attrib["Label"],
                    base_pair=float(a.attrib["Size"]),
                    left_bin=float(a.attrib["Left_Binning"]),
                    right_bin=float(a.attrib["Right_Binning"]),
                )
                for a in elem.findall("Allele")
            )

            markers.append(Marker(name=marker_name, dye_row=dye_index, alleles=alleles))
            elem.clear()

        logger.debug("Parsed panel from {}: {} markers", path, len(markers))
        return cls(markers)

    # -- Public query interface ------------------------------------------- #

    @property
    def markers(self) -> tuple[Marker, ...]:
        """All markers in the panel."""
        return self._markers

    def get_dye_row(self, marker_name: str) -> int | None:
        """Return the dye row for a marker, or None if unknown."""
        for marker in self._markers:
            if marker.name == marker_name:
                return marker.dye_row
        return None

    def get_allele_basepair_and_bins(
        self, marker_name: str, allele_name: str
    ) -> tuple[float, float, float]:
        """Look up an allele's (midpoint, left_edge, right_edge) in base pairs.

        Handles micro-variant alleles (e.g. "21.1") by offsetting from the
        base allele if the exact allele isn't found in the panel.

        Returns:
            A tuple of (base_pair, left_bin_edge, right_bin_edge).
        """
        for marker in self._markers:
            if marker.name != marker_name:
                continue
            for allele in marker.alleles:
                if (
                    allele.base_pair is None or
                    allele.left_bin is None or
                    allele.right_bin is None
                ):
                    raise AttributeError(f'Allele does not contain basepair/bin information: {allele}')
                if allele.name == allele_name:
                    return (
                        allele.base_pair,
                        allele.base_pair - allele.left_bin,
                        allele.base_pair + allele.right_bin,
                    )

        # Micro-variant fallback: "21.1" -> base on allele "21" + 1 bp
        if "." in allele_name:
            parts = allele_name.split(".")
            if len(parts) == 2:
                base_name, extra_bp = parts
                mid, left, right = self.get_allele_basepair_and_bins(marker_name, base_name)
                offset = int(extra_bp)
                return (mid + offset, left + offset, right + offset)

        # Y-STR fallback
        if marker_name.startswith("DYS"):
            logger.debug("Y-STR marker {} allele {} not in panel, using default bins", marker_name, allele_name)
            return (10, 9, 11)

        logger.debug("Marker {} allele {} not found in panel", marker_name, allele_name)
        return (10, 9, 11)

    def get_marker_name_by_dye_and_bp(self, dye_row: int, base_pair: float) -> str:
        """O(1) lookup: find which marker covers a given dye + base-pair position.

        Returns:
            The marker name, or ``"Out of Bin"`` if no marker covers that position.
        """
        lut_entry = self._marker_lut.get(dye_row)
        if lut_entry is None:
            return "Out of Bin"

        lut, offset = lut_entry
        index = int(round(base_pair * self._LUT_SCALE)) + offset
        if index < 0 or index >= len(lut) or lut[index] is None:
            return "Out of Bin"
        if (marker_name := lut[index]) is None:
            raise ValueError(f'Did not catch OOB: {lut}')
        return marker_name

    @cached_property
    def dye_bp_to_allele_mapping(self) -> dict[int, dict[float, tuple[str, str]]]:
        """Mapping from (dye_row, base_pair) -> (marker_name, allele_name)."""
        result: dict[int, dict[float, tuple[str, str]]] = defaultdict(dict)
        for marker in self._markers:
            for allele in marker.alleles:
                if allele.base_pair is None:
                    logger.warning(f'Allele has no base_pair: {marker=}, {allele=}')
                    continue
                result[marker.dye_row][allele.base_pair] = (marker.name, allele.name)
        return dict(result)

    # -- Private: LUT construction ---------------------------------------- #

    _LUT_SCALE: int = 10  # 0.1 bp resolution

    @cached_property
    def _marker_lut(self) -> dict[int, tuple[list[str | None], int]]:
        """Build per-dye lookup tables mapping base-pair positions to marker names.

        Design pattern: **Lazy Initialization**
            The LUT is expensive to build but not always needed (e.g. during
            evaluation-only runs). Using `@cached_property` means it's built
            exactly once, on first access, and cached thereafter.

        Returns:
            Dict mapping dye_row -> (lut_array, offset) where
            lut_array[bp_index + offset] gives the marker name.
        """
        by_dye: dict[int, list[Marker]] = defaultdict(list)
        for marker in self._markers:
            by_dye[marker.dye_row].append(marker)

        result: dict[int, tuple[list[str | None], int]] = {}

        for dye_row, markers in by_dye.items():
            ranges = [
                (m.min_bp, m.max_bp, m.name)
                for m in markers
                if m.min_bp != float("inf") and m.max_bp != float("-inf")
            ]
            if not ranges:
                continue

            min_idx = int(math.floor(min(r[0] for r in ranges) * self._LUT_SCALE))
            max_idx = int(math.ceil(max(r[1] for r in ranges) * self._LUT_SCALE))
            offset = -min_idx
            lut: list[str | None] = [None] * (max_idx - min_idx + 1)

            for start, end, name in ranges:
                s = int(math.floor(start * self._LUT_SCALE))
                e = int(math.ceil(end * self._LUT_SCALE))
                for i in range(s + offset, min(e + offset + 1, len(lut))):
                    lut[i] = name

            result[dye_row] = (lut, offset)

        return result

    # -- Dunder methods --------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._markers)

    def __iter__(self):
        return iter(self._markers)

    def __repr__(self) -> str:
        return f"Panel(markers={len(self._markers)})"
