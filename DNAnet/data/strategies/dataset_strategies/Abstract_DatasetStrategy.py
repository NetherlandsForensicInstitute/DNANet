import csv
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Literal
from typing import List


from DNAnet.data.data_models.dna_models import Marker, Panel

FileCategory = Literal["sample", "ladder", "control", "unknown"]


class DatasetStrategy(ABC):
    """
    Unified strategy interface for dataset-specific behavior such as
    file categorization, contributor parsing and allele loading.
    """

    # TODO: Do we really need an instance of this class for the load_donor_alleles function?
    def __init__(self, panel: Panel, genotypes_path: Path):
        self.panel = panel
        self.genotypes_path = Path(genotypes_path)

    @classmethod
    @abstractmethod
    def categorize_file(cls, file_name: str) -> FileCategory:
        """Return the category (sample/ladder/control/unknown) for a given file name."""
        ...

    @classmethod
    @abstractmethod
    def get_contributors(cls, file_name: str) -> List[str]:
        """Derive contributor file stems from the HID filename."""
        ...

    @classmethod
    @abstractmethod
    def parse_annotation_file(
        cls, path: str | Path, sample_name: str | None = None
    ) -> List[Marker] | None: ...

    @abstractmethod
    def build_marker(self, marker_name: str, allele_names: Iterable[str]) -> Marker:
        """Construct a Marker with Alleles using the provided panel metadata."""
        ...

    def load_donor_alleles(self, file_name: str) -> List[Marker]:
        """
        Load donor alleles for the provided file by reading reference genotype CSVs.
        :param file_name: .hid file name (not full path) to load actual donors for
        :return: list of Markers with combined alleles from all contributors
        """

        marker_allele_strings = defaultdict(set)
        for file_stem in self.get_contributors(file_name):
            # TODO: Fix this self. reference, is this something that the ABC Strategy should have? Isn't this a static class?
            reference_profiles_path = self.genotypes_path / f"{file_stem}.csv"
            with reference_profiles_path.open("r") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    marker_allele_strings[row["Marker"]].update(
                        [row["Allele1"], row["Allele2"]]
                    )

        return [
            self.build_marker(marker_name, alleles)
            for marker_name, alleles in marker_allele_strings.items()
        ]

    def serialize(self) -> dict:
        """Return a lightweight serialization of the strategy."""
        return {
            "class": self.__class__.__name__,
            "genotypes_path": str(self.genotypes_path),
            "panel_path": str(getattr(self.panel, "_panel_path", "")),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(genotypes_path={self.genotypes_path})"
