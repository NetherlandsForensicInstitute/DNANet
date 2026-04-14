"""ProvedIt dataset strategy.

Handles the ProvedIt dataset conventions:
- File naming: ``{well}_{sample-description}_{injection-time}.hid``
  e.g. ``B03_RD14-0003-34d1-0.5IP-Q0.75ng_05sec.hid``
  The sample description encodes contributor count, ratio, quality, etc.
- Ladders: filenames containing ``Ladder-GF`` (GlobalFiler ladder)
- Annotations: Single XLSX genotype file for the entire dataset
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, List, Tuple, Mapping, Generator
from pathlib import Path
from functools import reduce
from itertools import groupby

import openpyxl
from loguru import logger

from dnanet.core.allele import Allele
from dnanet.core.marker import Marker
from dnanet.core.annotation import Annotation, AlleleAnnotation, ScanpointAnnotation
from dnanet.data.strategies.registry import StrategyRegistry
from dnanet.data.strategies.datasets.dataset import FileCategory, DatasetStrategy


if TYPE_CHECKING:
    from dnanet.core.types import PathLike

class ProvedItStrategy(DatasetStrategy):
    """Strategy for the ProvedIt dataset (GlobalFiler kit)."""
    
    # ProvedIt ladder pattern
    _LADDER_PATTERN = re.compile(r"Ladder", re.IGNORECASE)
    # Following pattern layed out in https://lftdi.camden.rutgers.edu/wp-content/uploads/2019/12/PROVEDIt-Database-Naming-Convention-Laboratory-Methodsv1.pdf
    _SAMPLE_PATTERN = re.compile(r'([A-Z]\d{2})_(RD1[24]-0003)-((?:\d{1,2})(?:_\d{1,2})+)-(\d(?:;\d)+)')
    
    @classmethod
    def collect_dataset_files(
        cls,
        root_path: str | Path, **kwargs
    ) -> Generator[Tuple[Path, ScanpointAnnotation | AlleleAnnotation | None, Path | None], None, None]:
        path = Path(root_path)
        
        dataset_hid_files = list(path.rglob("*.hid"))
        
        # Groupy all HID files by their category (control, ladder, sample)
        hid_file_types = {
            key: [*paths] for key, paths in
            groupby(
                sorted(dataset_hid_files, key=lambda f: cls.categorize_file(f.stem)),
                key=lambda f: cls.categorize_file(f.stem)
            )
        }
        
        # Check for an annotations file and parse it into a dict with Annotations
        annotations_file = cls._find_annotation_file(path)
        annotation_mapping = cls.parse_annotations(annotations_file)
        
        for sample in hid_file_types['sample']:
            sample_annotation = cls._combine_contributors_into_annotation(sample, annotation_mapping)
            sample_ladder = cls.find_ladder_for_sample(sample)
            yield sample, sample_annotation, sample_ladder

    
    @classmethod
    def categorize_file(cls, file_name: str) -> FileCategory:
        """Classify based on ProvedIt naming conventions.

        - Ladders contain ``Ladder`` (e.g. ``B03_Ladder-GF_02.5sec.hid``)
        - Negative controls contain ``neg`` or ``NTC``
        - Everything else is a sample
        """
        stem = Path(file_name).stem

        if cls._LADDER_PATTERN.search(stem):
            return "ladder"
        if cls._SAMPLE_PATTERN.search(stem):
            return "sample"
        lower = stem.lower()
        if "neg" in lower or "ntc" in lower:
            return "control"
        logger.trace(f'Unknown file found: {file_name}')
        return "unknown"

    @classmethod
    def get_number_of_contributors(cls, file_name: str) -> str | None:
        """Extract contributor count from ProvedIt naming.

        ProvedIt encodes contributors in the sample description,
        e.g. ``RD14-0003-34d1`` → 2 contributors (digits 3 and 4 in ``34``).
        The exact parsing depends on the specific ProvedIt naming convention.

        Returns:
            String like ``"2p"`` or ``None`` if not determinable.
        """
        stem = Path(file_name).stem
        # ProvedIt samples have format: well_description_time
        parts = stem.split("_")
        if len(parts) < 2:
            return None

        description = parts[1]
        # Count unique contributor digits in the ratio section
        # e.g. "RD14-0003-34d1" has "34" → 2 contributors
        # e.g. "RD14-0003-1d1" has "1" → 1 contributor
        ratio_match = re.search(r"-(\d+)d\d", description)
        if ratio_match:
            ratio_str = ratio_match.group(1)
            return f"{len(ratio_str)}p"
        return None

    @classmethod
    def get_sample_id(cls, file_name: str) -> str:
        """Extract sample identifier from ProvedIt filename.

        Returns the description part (between well and injection time),
        e.g. ``B03_RD14-0003-34d1-0.5IP-Q0.75ng_05sec.hid``
        → ``"RD14-0003-34d1-0.5IP-Q0.75ng"``
        """
        stem = Path(file_name).stem
        parts = stem.split("_")
        if len(parts) >= 3:
            # Join everything between well and injection time
            return "_".join(parts[1:-1])
        return stem

    @classmethod
    def _find_annotation_file(
        cls, path: Path
    ) -> Path:
        """Find the genotype XLSX file for ProvedIt.

        ProvedIt uses a single XLSX file for all genotype data.
        """
        annotations_file = list(path.glob("*Known Genotypes*"))
        match len(annotations_file):
            case 0:
                raise FileNotFoundError('No annotations file found for ProvedIt dataset')
            case 1:
                return annotations_file[0]
            case _:
                raise RuntimeError(f'Found multiple annotation files: {annotations_file}')

    @classmethod
    def parse_annotations(
        cls,
        annotation_source: PathLike,
    ) -> Mapping[str, AlleleAnnotation]:
        """Load called alleles from the ProvedIt XLSX genotype file.

        The ProvedIt XLSX contains genotype data for all samples.
        Each row typically maps sample→marker→alleles.

        Note:
            This is a stub — full XLSX parsing will be implemented when
            ProvedIt integration is tested end-to-end.
        """
        path = Path(annotation_source)
        # Check if it's the standard xlsx format
        if not path.suffix in ('.xlsx', '.xls'):
            raise ValueError('PROVEDIt dataset annotations should be in Excel format')

        excel_file = openpyxl.open(path)
        sheet_values = [[column.value for column in row] for row in excel_file.worksheets[0].rows]
        headers = sheet_values[0]
        rows = sheet_values[1:]

        annotation_mapping: Dict[str, AlleleAnnotation] = {}
        marker_2_dye = cls._get_scaling_strategy().marker_name_to_dye_idx()
        for row in rows:
            markers: List[Marker] = []
            research_id, sample_id = None, None
            for header, col in zip(headers, row, strict=True):
                if header == 'Research ID':
                    research_id = col
                elif header == 'Sample ID':
                    sample_id = col
                else:
                    _marker = Marker(
                        name=str(header),
                        dye_row=marker_2_dye[str(header)],
                        alleles=frozenset(
                            Allele(name=allele) for allele in str(col).split(',')
                        )
                    )
                    markers.append(_marker)
            annotation_mapping[str(sample_id)] = AlleleAnnotation(markers)

        return annotation_mapping

    @staticmethod
    def _get_scaling_strategy():
        from dnanet.data.strategies.registry import StrategyRegistry
        return StrategyRegistry.get_scaling_strategy()
    
    @classmethod
    def find_ladder_for_sample(
        cls, sample_path: Path, ladder_mapping: dict[str, Path] | None = None
    ) -> Path | None:
        """Find the ladder for a ProvedIt sample.

        ProvedIt ladders share the same injection directory and well prefix.
        E.g. sample ``B03_RD14-...`` uses ladder ``B03_Ladder-GF_...`` from
        the same run.
        """
        stem = sample_path.stem
        if ladder_mapping and stem in ladder_mapping:
            return ladder_mapping[stem]

        # Extract well prefix (e.g. "B03")
        well = stem.split("_")[0] if "_" in stem else None
        if well is None:
            return None

        # Look in same directory for a ladder with matching well
        parent = sample_path.parent
        for f in parent.glob("*.hid"):
            if "ladder" in f.name.lower() and f.stem.startswith(well):
                return f

        # Fallback: any ladder in the directory
        ladders = [f for f in parent.glob("*.hid") if "ladder" in f.name.lower()]
        return ladders[0] if ladders else None

    @classmethod
    def _combine_contributors_into_annotation(cls, sample_file: Path, annotation_mapping: Mapping[str, AlleleAnnotation]) -> AlleleAnnotation:
        for part in sample_file.stem.split("-"):
            if re.match(r'(\d{1,2})(_\d{1,2})+', part):
                return reduce(lambda x, y: x + y, [annotation_mapping[c] for c in part.split('_')])
        raise ValueError(f'Could not extract contributors from sample: {sample_file.stem}')
    
    @staticmethod
    def get_annotation_classes() -> list[str]:
        return ["noise", "allele"]
