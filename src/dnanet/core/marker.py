"""Marker domain model.

Design pattern: **Composite**
    A Marker is a *composite* — it's a domain object that contains a collection
    of child objects (Alleles). The composite pattern lets you treat a single
    Marker as a meaningful unit, while still being able to iterate over its
    parts. This is a natural fit for DNA loci: a Marker (e.g. "D3S1358") is
    a region on a chromosome that contains one or more Alleles.

    Key simplifications from the original:
    1. `frozen=True` — Markers are value objects too. Use `dataclasses.replace()`
       to create modified copies.
    2. `alleles` is a `tuple` instead of a mutable `Sequence` — reinforcing
       immutability and making Markers hashable by default.
    3. The `is_autosomal` check uses the constants module instead of inline
       string checks, so the list of non-autosomal markers is defined once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dnanet.core.allele import Allele
from dnanet.core.constants import NON_AUTOSOMAL_MARKERS, NON_AUTOSOMAL_PREFIXES


@dataclass(frozen=True, slots=True)
class Marker:
    """A DNA marker (locus) containing one or more alleles.

    Attributes:
        name: Marker designation (e.g. "D3S1358", "AMEL").
        dye_row: 0-based index of the fluorescence channel this marker
            appears on.
        alleles: The alleles detected at this locus.
    """

    name: str
    dye_row: int
    alleles: tuple[Allele, ...] = ()

    @property
    def is_autosomal(self) -> bool:
        """True if this marker is autosomal (not sex-linked or Y-STR)."""
        if self.name in NON_AUTOSOMAL_MARKERS:
            return False
        return not any(self.name.startswith(prefix) for prefix in NON_AUTOSOMAL_PREFIXES)

    @property
    def min_bp(self) -> float:
        """Leftmost bin edge across all alleles (in base pairs)."""
        return min(
            (a.base_pair - a.left_bin for a in self.alleles if a.base_pair is not None),
            default=float("inf"),
        )

    @property
    def max_bp(self) -> float:
        """Rightmost bin edge across all alleles (in base pairs)."""
        return max(
            (a.base_pair + a.right_bin for a in self.alleles if a.base_pair is not None),
            default=float("-inf"),
        )

    # -- Serialization --

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "name": self.name,
            "dye_row": self.dye_row,
            "alleles": [a.to_dict() for a in self.alleles],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Marker:
        """Deserialize from a plain dictionary."""
        return cls(
            name=data["name"],
            dye_row=data["dye_row"],
            alleles=tuple(Allele.from_dict(a) for a in data.get("alleles", [])),
        )
