from DNAnet.data.data_models.dna_models import Allele, Marker
from DNAnet.data.strategies.dataset_strategies.Abstract_DatasetStrategy import (
    DatasetStrategy,
    FileCategory,
)


import re
from pathlib import Path
from typing import Iterable, List, Optional


class ProvedItDatasetStrategy(DatasetStrategy):
    """
    Strategy tailored to the ProvedIt dataset.
    """

    @classmethod
    def categorize_file(cls, file_name: str) -> FileCategory:
        if "Ladder" in file_name:
            return "ladder"
        if "LEA" in file_name:
            return "control"
        try:
            if len(cls.get_contributors(file_name)) > 0:
                return "sample"
        except ValueError:
            # If we cannot parse contributors, treat the file as unknown instead of failing.
            return "unknown"
        return "unknown"

    @classmethod
    def get_contributors(cls, file_name: str) -> list[str]:
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
            raise ValueError(
                f"Cannot extract contributors from provided file name: {file_name}"
            )
        contributors = match.group(1).split("_")
        if not (2 <= len(contributors) <= 5):
            raise ValueError(
                f"Expected 2-5 contributors, found {len(contributors)} in {file_name}"
            )
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

    @classmethod
    def parse_annotation_file(
        cls, path: str | Path, sample_name: str | None = None
    ) -> List[Marker] | None:
        raise NotImplementedError("To be done")
