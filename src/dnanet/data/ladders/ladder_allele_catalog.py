import csv
from collections import defaultdict
from dataclasses import field, dataclass

from dnanet.core import Panel, Allele
from dnanet.core.types import PathLike


@dataclass(frozen=True)
class LadderAlleleCatalog:
    """Catalog of alleles expected in a ladder, organized by dye row.

    Loaded from a CSV file with columns: Marker, Allele, Dye.

    Note: This is a separate class to prevent multiple I/O operations for reading the
    same CSV file and allow the use of the csv's information inside the Ladder class.
    """

    alleles_by_dye: dict[int, list[tuple[str, str]]] = field(default_factory=dict)

    def expected_count(self, dye_row: int) -> int:
        """Number of alleles expected on a given dye row."""
        return len(self.alleles_by_dye.get(dye_row, []))

    @classmethod
    def from_panel(cls, panel: Panel) -> "LadderAlleleCatalog":
        """Create a catalog from a panel."""
        alleles: dict[int, list[tuple[str, str]]] = defaultdict(list)
        for marker in panel.markers:
            for allele in marker.alleles:
                if allele.in_ladder:
                    alleles[marker.dye_row].append((marker.name, allele.name))

        return cls(alleles_by_dye=dict(alleles))

    def __hash__(self):
        return hash(
            tuple(
                (dye, tuple(alleles))
                for dye, alleles in sorted(self.alleles_by_dye.items())
            )
        )

    def __eq__(self, other):
        if not isinstance(other, LadderAlleleCatalog):
            return NotImplemented
        return self.alleles_by_dye == other.alleles_by_dye
