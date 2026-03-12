import openpyxl

from DNAnet.data.data_models.dna_models import Allele, Marker
from DNAnet.data.data_models.structs import AlleleAnnotation
from DNAnet.data.strategies.dataset_strategies.Abstract_DatasetStrategy import (
    DatasetStrategy,
    FileCategory,
)


import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from DNAnet.data.strategies.strategy_registry import StrategyRegistry


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

    @classmethod
    def parse_annotation_file(cls, path: str | Path) -> Dict[str, List[Marker]] | None:
        path = Path(path)
        # Check if it's the standard xlsx format
        if not path.suffix in ('.xlsx', '.xls'):
            raise ValueError("PROVEDIt dataset annotations should be in Excel format")
        
        excel_file = openpyxl.open(path)
        sheet_values = [
            [column.value for column in row]
            for row in excel_file.worksheets[0].rows
        ]
        headers = sheet_values[0]
        rows = sheet_values[1:]
        
        _kit_strategy = StrategyRegistry.get_kit()
        
        annotation_mapping = {}
        for row in rows:
            markers = []
            research_id, sample_id = None, None
            for header, col in zip(headers, row, strict=True):
                if header == "Research ID":
                    research_id = col
                elif header == "Sample ID":
                    sample_id = col
                else:
                    marker_name = str(header)
                    markers.append(cls.build_marker(marker_name, allele_names=str(col).split(",")))
            annotation_mapping[str(sample_id)] = markers
        
        return dict(annotation_mapping)

    @classmethod
    def create_annotation_for_sample(cls, annotation_mapping: Dict[str, List[Marker]], sample_name: str) -> AlleleAnnotation:
        sample_ids = cls.get_contributors(sample_name)
        sample_markers = [annotation_mapping.get(sample) for sample in sample_ids]
        return AlleleAnnotation(annotation=sample_markers)