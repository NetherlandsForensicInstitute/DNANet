from collections import defaultdict
from itertools import chain

import openpyxl

from DNAnet.data.data_models.dna_models import Marker
from DNAnet.data.data_models.structs import AlleleAnnotation
from DNAnet.data.strategies.dataset_strategies.Abstract_DatasetStrategy import (
    DatasetStrategy,
    FileCategory,
)


import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from DNAnet.data.strategies.strategy_registry import StrategyRegistry


class ProvedItDatasetStrategy(DatasetStrategy):
    """
    Strategy tailored to the ProvedIt dataset.
    """
    
    @classmethod
    def collect_dataset_files(cls, path: str | Path, **kwargs) -> Tuple[List[Path], Mapping, Mapping]:
        path = Path(path)
        hid_files = list(path.rglob('*.hid'))
        genotypes_file = cls._find_genotypes_file(path)
        annotation_mapping = cls.parse_annotation_file(genotypes_file)
        if annotation_mapping is None:
            raise RuntimeError(f'Annotation mapping failed to create for {genotypes_file}')
        
        hid_file_mapping: Dict[str, List[Path]] = defaultdict(list)
        for hid_file in hid_files:
            _file_category = cls.categorize_file(file_name=hid_file.stem)
            hid_file_mapping[_file_category].append(hid_file)
        
        ladder_mapping = cls._find_connected_ladder(hid_file_mapping["sample"], hid_file_mapping["ladder"])
        
        return list(hid_file_mapping["sample"]), annotation_mapping, ladder_mapping
    
    @classmethod
    def _find_genotypes_file(cls, path: Path) -> Path:
        possible_extensions = {'xlsx', 'csv'}
        possible_files = list(filter(
            lambda x: "genotypes" in x.name.lower(),
            chain(*(path.rglob(f'*.{ext}') for ext in possible_extensions))
        ))
        if len(possible_files) != 1:
            raise RuntimeError(f'No genotypes file or multiple found: {possible_files}')

        return possible_files[0]
    
    @classmethod
    def _find_connected_ladder(cls, hid_files: List[Path], ladder_files: List[Path]) -> Dict[str, Path | None]:
        ladder_mapping: Dict[str, Dict[str, Path]] = defaultdict(dict)
        for ladder in ladder_files:
            ladder_mapping[ladder.parts[-2]][ladder.stem[0]] = ladder
        ladder_mapping = dict(ladder_mapping)

        return {
            hid_file.stem: ladder_mapping[hid_file.parent.stem].get(hid_file.name[0], None)
            for hid_file in hid_files
        }
        

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
        sample_markers = [annotation_mapping[sample] for sample in sample_ids]
        return AlleleAnnotation(annotation=sample_markers) # type: ignore