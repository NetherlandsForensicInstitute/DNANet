import csv
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Literal, Optional

from DNAnet.data.data_models.dna_models import Allele, Marker, Panel
from DNAnet.utils import DONORS_PER_DATASET_NR, get_prefix_from_filename, is_rd_hid_filename

FileCategory = Literal["sample", "ladder", "control", "unknown"]


class DatasetStrategy(ABC):
    """
    Unified strategy interface for dataset-specific behavior such as
    file categorization, contributor parsing and allele loading.
    """

    def __init__(self, panel: Panel, genotypes_path: Path):
        self.panel = panel
        self.genotypes_path = Path(genotypes_path)

    @abstractmethod
    def categorize_file(self, file_name: str) -> FileCategory:
        """Return the category (sample/ladder/control/unknown) for a given file name."""
        raise NotImplementedError

    @abstractmethod
    def get_contributors(self, file_name: str) -> list[str]:
        """Derive contributor file stems from the HID filename."""
        raise NotImplementedError

    @abstractmethod
    def build_marker(self, marker_name: str, allele_names: Iterable[str]) -> Marker:
        """Construct a Marker with Alleles using the provided panel metadata."""
        raise NotImplementedError

    def load_donor_alleles(self, file_name: str) -> list[Marker]:
        """
        Load donor alleles for the provided file by reading reference genotype CSVs.
        :param file_name: .hid file name (not full path) to load actual donors for
        :return: list of Markers with combined alleles from all contributors
        """

        marker_allele_strings = defaultdict(set)
        for file_stem in self.get_contributors(file_name):
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


class NFI_RND_DatasetStrategy(DatasetStrategy):
    """
    Strategy tailored to the NFI R&D dataset.
    """

    def categorize_file(self, file_name: str) -> FileCategory:
        OTHER_KITS = ("ppy23", "minifiler", "hdplex")

        fname = file_name.lower()
        # Ladder files
        if "ladder" in fname and not any(kit in fname for kit in OTHER_KITS):
            return "ladder"
        # Controls/blanks (implement your own logic or import from utils)
        if (
            "blanco" in fname
            or "ladder" in fname
            or "pocon" in fname
            or "controle" in fname
            or fname.startswith('a')
        ):
            return "control"
        # Valid sample HID file (using is_rd_hid_filename logic)
        if len(re.findall(r'\d[ABCDEF]\d', file_name[:3])) > 0:
            return "sample"
        # Unknown or unhandled
        return "unknown"

    def get_contributors(self, file_name: str) -> list[str]:
        if not is_rd_hid_filename(file_name):
            raise ValueError(
                f"Cannot load donor alleles for non-RD sample. Found file name {file_name}"
            )

        mixture_type = get_prefix_from_filename(file_name)
        dataset_nr, nr_donors = mixture_type[0], int(mixture_type[2])
        return [
            f"{dataset_nr}{letter}"
            for letter in DONORS_PER_DATASET_NR[dataset_nr][:nr_donors]
        ]

    # From previous implementation, but I believe this code is broken or deprecated.
    def build_marker(self, marker_name: str, allele_names: Iterable[str]) -> Marker:
        dye_row = self.panel.get_dye_row(marker_name)
        return Marker(dye_row, marker_name, [Allele(a) for a in sorted(allele_names)])


class ProvedItDatasetStrategy(DatasetStrategy):
    """
    Strategy tailored to the ProvedIt dataset.
    """

    def categorize_file(self, file_name: str) -> FileCategory:
        if "Ladder" in file_name:
            return "ladder"
        if "LEA" in file_name:
            return "control"
        if len(self.get_contributors(file_name)) > 0:
            return "sample"
        return "unknown"

    def get_contributors(self, file_name: str) -> list[str]:
        """
        Extracts all contributor IDs from a ProvedIt filename.
        Contributors are the numbers separated by underscores after 'RD14-0003-'
        and before the next '-'.
        Example:
            F07_RD14-0003-30_31_32_33_34-1;1;1;1;1-M3e-0.075GF-Q2.0_06.5sec.hid
            -> ['30', '31', '32', '33', '34']
        """
        match = re.search(r"RD14-0003-([\d_]+)-", file_name)
        if not match:
            raise ValueError(f"Cannot extract contributors from provided file name: {file_name}")
        contributors = match.group(1).split('_')
        if not (2 <= len(contributors) <= 5):
            raise ValueError(f"Expected 2-5 contributors, found {len(contributors)} in {file_name}")
        return contributors

    def build_marker(self, marker_name: str, allele_names: Iterable[str]) -> Marker:
        dye_row: Optional[int] = self.panel.get_dye_row(marker_name)
        if dye_row is None:
            raise TypeError(
                f"Marker {marker_name} not found in panel {self.panel}. "
                "Please check the panel or the marker name."
            )
        
        new_alleles = [
            Allele(name, *self.panel.get_allele_info(marker_name, name))
            for name in sorted(allele_names)
        ]
        return Marker(dye_row, marker_name, new_alleles)
