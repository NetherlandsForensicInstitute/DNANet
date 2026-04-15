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
from torch.utils.data import Subset
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from dnanet.core.allele import Allele
from dnanet.core.marker import Marker
from dnanet.core.annotation import AlleleAnnotation, ScanpointAnnotation
from dnanet.data.strategies.datasets.dataset import FileCategory, DatasetStrategy


if TYPE_CHECKING:
    from dnanet.core.types import PathLike
    from dnanet.data.dataset import TransformableDataset


class ProvedItStrategy(DatasetStrategy):
    """Strategy for the ProvedIt dataset (GlobalFiler kit)."""

    # Constant suffix pattern for HID files
    _HID_SUFFIX = '*.hid'
    # ProvedIt ladder pattern
    _LADDER_PATTERN = re.compile(r'Ladder', re.IGNORECASE)
    # Following pattern layed out in https://lftdi.camden.rutgers.edu/wp-content/uploads/2019/12/PROVEDIt-Database-Naming-Convention-Laboratory-Methodsv1.pdf
    _SAMPLE_PATTERN = re.compile(r'([A-Z]\d{2})_(RD1[24]-0003)-(\d{1,2}(?:_\d{1,2})+)-(\d(?:;\d)+)')

    @classmethod
    def collect_dataset_files(
        cls, root_path: str | Path
    ) -> Generator[
        Tuple[Path, ScanpointAnnotation | AlleleAnnotation | None, Path | None], None, None
    ]:
        """Collects the HID, Annotation (optional), and Ladder (optional) files for the ProvedIt dataset.

        Args:
            root_path: The root folder in which all neccesarry files are located.

        Yields:
            A tuple containing the Path to the HID file, its (optional) Annotation, and its (optional) Ladder
        """
        path = Path(root_path)

        dataset_hid_files = path.rglob(cls._HID_SUFFIX)

        # Groupy all HID files by their category (control, ladder, sample)
        hid_file_types = {
            key: [*paths]
            for key, paths in groupby(
                sorted(dataset_hid_files, key=lambda f: cls.categorize_file(f.stem)),
                key=lambda f: cls.categorize_file(f.stem),
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
            return 'ladder'
        if cls._SAMPLE_PATTERN.search(stem):
            return 'sample'
        lower = stem.lower()
        if 'neg' in lower or 'ntc' in lower:
            return 'control'
        logger.trace(f'Unknown file found: {file_name}')
        return 'unknown'

    @classmethod
    def get_number_of_contributors(cls, file_name: str) -> int:
        """Extract contributor count from ProvedIt naming.

        ProvedIt encodes contributors in the sample description,
        e.g. ``RD14-0003-34d1`` → 2 contributors (digits 3 and 4 in ``34``).
        The exact parsing depends on the specific ProvedIt naming convention.

        Returns:
            String like ``"2p"`` or ``None`` if not determinable.
        """
        stem = Path(file_name).stem
        # Example filename:
        # A02_RD14-0003-31_32-1;1-M2c-0.03GF-Q2.0_01.5sec
        # We want to extract the 31_32 here to conclude 2 contributors (with 1:1 mixture prop.)
        parts = stem.split('-')
        if len(parts) < 2:
            raise ValueError(f'Could not extract NoC from {file_name=}')

        # We check each of the file's parts
        for p in parts:
            # And if it matches the pattern of:
            # contributor 1 + (underscore + contributer{i}) * i_contributors (i ranging from 1 to 4)
            if re.match(r'(\d{1,2})(_\d{1,2})+', p):
                # We return the number of distinct contributor ID's
                return len(p.split('_'))
        raise ValueError(f'Could not extract NoC from {file_name=}')

    @classmethod
    def get_sample_id(cls, file_name: str) -> str:
        """Extract sample identifier from ProvedIt filename.

        Returns the description part (between well and injection time),
        e.g. ``B03_RD14-0003-34d1-0.5IP-Q0.75ng_05sec.hid``
        → ``"RD14-0003-34d1-0.5IP-Q0.75ng"``
        """
        stem = Path(file_name).stem
        parts = stem.split('_')
        if len(parts) >= 3:
            # Join everything between well and injection time
            return '_'.join(parts[1:-1])
        return stem

    @classmethod
    def _extract_contributors(cls, file_stem: str) -> List[int]:
        """Return the list of contributor IDs encoded in a sample's filename."""
        match = cls._SAMPLE_PATTERN.search(file_stem)
        if match is None:
            raise ValueError(f'Filename stem {file_stem!r} does not match expected pattern')
        return [int(x) for x in match.group(3).split('_')]

    @classmethod
    def _find_annotation_file(cls, path: Path) -> Path:
        """Find the genotype XLSX file for ProvedIt.

        ProvedIt uses a single XLSX file for all genotype data.
        """
        annotations_file = list(path.glob('*Known Genotypes*'))
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
        multiple_research_ids: bool = False,
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
        if path.suffix not in ('.xlsx', '.xls'):
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
                elif col == 'N/A':
                    logger.debug('Skipping Y-Marker with N/A (for a female sample)')
                else:
                    _marker = Marker(
                        name=str(header),
                        dye_row=marker_2_dye[str(header)],
                        alleles=frozenset(Allele(name=allele) for allele in str(col).split(',')),
                    )
                    markers.append(_marker)

            _annotation = AlleleAnnotation(markers)
            if multiple_research_ids:
                annotation_mapping[f'{research_id}-{sample_id}'] = _annotation
            else:
                annotation_mapping[str(sample_id)] = _annotation

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
        well = stem.split('_')[0] if '_' in stem else None
        if well is None:
            return None

        # Look in same directory for a ladder with matching well
        parent = sample_path.parent
        for f in parent.glob(cls._HID_SUFFIX):
            if 'ladder' in f.name.lower() and f.stem.startswith(well):
                return f

        # Fallback: any ladder in the directory
        ladders = [f for f in parent.glob(cls._HID_SUFFIX) if 'ladder' in f.name.lower()]
        return ladders[0] if ladders else None

    @classmethod
    def _combine_contributors_into_annotation(
        cls, sample_file: Path, annotation_mapping: Mapping[str, AlleleAnnotation]
    ) -> AlleleAnnotation:
        for part in sample_file.stem.split('-'):
            if re.match(r'(\d{1,2})(_\d{1,2})+', part):
                return reduce(lambda x, y: x + y, [annotation_mapping[c] for c in part.split('_')])
        raise ValueError(f'Could not extract contributors from sample: {sample_file.stem}')

    @staticmethod
    def get_annotation_classes() -> list[str]:
        return ['noise', 'allele']

    @classmethod
    def _split(
        cls,
        dataset,
        fraction: float | None = None,
        seed: int | None = None,
        k_folds: int | None = None,
        stratify_noc: bool = True,
        test_fraction: float = 0.0,
        **kwargs,
    ) -> Tuple[Subset, Subset] | Tuple[Subset, Subset, Subset] | List[Tuple[Subset, Subset]]:
        """Split the ProvedIt dataset for train/val or train/val/test.

        Args:
            dataset: A HIDDataset
            fraction: Percentage for the train fraction. Defaults to None.
            seed: Random seed to have reproducability. Defaults to None.
            k_folds: KFold splitting for cross-validation. Defaults to None.
            stratify_noc: Balance the NoC over the split(s). Defaults to True.
            test_fraction: Fraction of total data held out as test set. Defaults to 0.0.
            **kwargs

        Raises:
            ValueError: When fraction and/or k_folds parameters aren't valid.

        Returns:
            A (list of) split of train/val datasets, or a (train, val, test) 3-tuple
            when test_fraction > 0.
        """
        # FIXME: See section 3.7 of https://resolver.tudelft.nl/uuid:d07c1be2-cfa1-44d5-892f-c2d110e0c9a0 for genotype aware splitting
        logger.warning('Genotype Aware splitting is not implemented for this dataset')

        match (fraction, k_folds):
            case (float(), None) if 0 < fraction < 1:
                return cls._fractional_split(
                    dataset=dataset,
                    fraction=fraction,
                    stratify_noc=stratify_noc,
                    seed=seed,
                    test_fraction=test_fraction,
                )
            case (None, int()) if 2 <= k_folds < len(dataset):
                if test_fraction > 0.0:
                    raise ValueError('test_fraction is not supported with k-fold splitting')
                return cls._kfold_split(
                    dataset=dataset,
                    k_folds=k_folds,
                    stratify_noc=stratify_noc,
                    seed=seed,
                )
            case _:
                raise ValueError(
                    f'Provide either a fraction in (0, 1) or 2 <= k_folds < {len(dataset)=}, bot both or none. Got {fraction=}, {k_folds=}'
                )

    @classmethod
    def _fractional_split(
        cls,
        dataset: TransformableDataset,
        fraction: float,
        stratify_noc: bool,
        seed: int | None,
        test_fraction: float = 0.0,
    ) -> Tuple[Subset, Subset] | Tuple[Subset, Subset, Subset]:
        indices = list(range(len(dataset.images)))
        nocs = [cls.get_number_of_contributors(file_name=img.path.stem) for img in dataset.images]

        if test_fraction > 0.0:
            main_idx, test_idx = train_test_split(
                indices,
                train_size=1.0 - test_fraction,
                random_state=seed,
                stratify=nocs if stratify_noc else None,
            )
            adjusted_fraction = fraction / (1.0 - test_fraction)
            main_nocs = [nocs[i] for i in main_idx] if stratify_noc else None
            train_idx, val_idx = train_test_split(
                main_idx,
                train_size=adjusted_fraction,
                random_state=seed,
                stratify=main_nocs,
            )
            logger.info(
                f'Fractional 3-way split | {fraction:.0%} train / '
                f'{1 - fraction - test_fraction:.0%} val / {test_fraction:.0%} test | '
                f'stratify={"noc" if stratify_noc else "none"}'
            )
            return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)

        logger.info(
            f'Fractional split | {fraction:.0%} train | stratify={"noc" if stratify_noc else "none"}'
        )
        train_idx, val_idx = train_test_split(
            indices, train_size=fraction, random_state=seed, stratify=nocs if stratify_noc else None
        )
        return Subset(dataset, train_idx), Subset(dataset, val_idx)

    @classmethod
    def _kfold_split(
        cls, dataset: TransformableDataset, k_folds: int, stratify_noc: bool, seed: int | None
    ):
        indices = list(range(len(dataset.images)))
        nocs = [cls.get_number_of_contributors(file_name=img.path.stem) for img in dataset.images]
        splitter = (
            StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
            if stratify_noc
            else KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        )

        logger.info(f'K-Fold split | {k_folds} folds | stratify={"noc" if stratify_noc else "none"}')
        return [
            (
                Subset(dataset, [indices[i] for i in train]),
                Subset(dataset, [indices[i] for i in val]),
            )
            for train, val in splitter.split(indices, nocs if stratify_noc else indices)
        ]
