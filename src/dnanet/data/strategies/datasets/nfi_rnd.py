"""NFI R&D dataset strategy.

Handles the NFI Research & Development dataset conventions:
- File naming: ``{NOC}{batch}{contributors}_{well}_{rep}.hid``
  e.g. ``1A2_A01_01.hid`` means NOC=2, batch=A, contributors=1A2
- Ladders: filenames starting with ``ladder_`` (case-insensitive)
- Controls: ``blanco``, ``pocon``, ``controle``, or filenames starting with ``A``
- Annotations: Tab-separated TXT files (AlleleReport format), one per injection
"""

from __future__ import annotations

import csv
import io
import os
import re
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple, Iterable, Generator

from loguru import logger
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    train_test_split,
)
from torch.utils.data import Subset

from dnanet.core.allele import Allele
from dnanet.core.annotation import Annotation, AlleleAnnotation
from dnanet.core.marker import Marker
from dnanet.data.strategies.datasets.dataset import DatasetStrategy

if TYPE_CHECKING:
    from dnanet.core.types import PathLike
    from dnanet.data.strategies import ScalingStrategy
    from dnanet.data.hid_dataset import HIDDataset
    from dnanet.data.strategies.datasets.dataset import FileCategory


class NFIRnDStrategy(DatasetStrategy):
    """Strategy for the NFI R&D mixture dataset."""

    READ_ANNOTATION_HEIGHTS: bool = False
    # R&D filename pattern: digit + letter + digit (e.g. "1A2")
    _RD_PREFIX_RE = re.compile(r'^\d[A-F]\d')

    @classmethod
    def collect_dataset_files(
        cls,
        root_path: PathLike,
        scaling_strategy: ScalingStrategy,
        analysis_treshold_type: str = 'DTH',
        **kwargs
    ) -> Generator[Tuple[Path, Annotation | None, Path | None]]:
        """Collect the dataset files for this specific dataset strategy.

        Creates a generator that yields paths to HIDImage's, Annotations (optional),
        and corresponding Ladder file (optional).

        Args:
            root_path: The path to the root of this dataset
            scaling_strategy: The scaling strategy to use for the annotations.
            analysis_treshold_type: Whether to take annotations that were made with high (DTH) or low (DTL) analytical tresholds.
        """
        path = Path(root_path)

        hid_to_annotation_paths = list(path.rglob('*hid_to_annotation*'))
        hid_to_ladder_paths = list(path.rglob('*best_ladder_paths*'))
        if not hid_to_annotation_paths or not hid_to_ladder_paths:
            raise ValueError(
                'Path does not contain the neccessary mapping files (annotation & ladder)'
            )

        # Allele Report for Annotations
        annotation_txt_files = list(path.rglob('*AlleleReport.txt'))
        annotation_name_to_annotation: Dict[str, Annotation] = {}
        for txt_file in annotation_txt_files:
            _annotation = cls.parse_annotations(txt_file, scaling_strategy)

            if _annotation:
                annotation_name_to_annotation.update(_annotation)

        # HID to Annotation mapping
        hta_header, hta_values = cls._read_csv_file(hid_to_annotation_path[0])
        analysis_treshold_type_column = [
            i for i, head in enumerate(hta_header) if analysis_treshold_type in head
        ]
        if len(analysis_treshold_type_column) != 1:
            raise RuntimeError(
                f'Could not infer the analysis treshold type column for annotation mapping: {hta_header}'
            )
        hid_to_annotation = dict(
            [
                (
                    v[0].replace('.hid', ''),
                    annotation_name_to_annotation.get(v[analysis_treshold_type_column[0]]),
                )
                for v in hta_values
            ]
        )

        # Hid to Ladder mapping
        _, htl_values = cls._read_csv_file(hid_to_ladder_path[0])
        hid_to_ladder = {hid: path / ladder for hid, ladder in htl_values}

        hid_files = list(path.rglob('*.hid'))
        hid_file_samples = list(filter(lambda x: cls.categorize_file(x.name) == 'sample', hid_files))

        for hid_file in hid_file_samples:
            yield (
                hid_file,
                hid_to_annotation.get(str(hid_file.stem)),
                hid_to_ladder.get(hid_file.stem),
            )

    @classmethod
    def categorize_file(cls, file_name: str) -> FileCategory:
        r"""Classify based on NFI R&D naming conventions.

        - Ladders start with ``ladder_``
        - Controls contain ``blanco``, ``pocon``, ``controle``, or start with ``A``
        - Samples match the ``\\d[A-F]\\d`` prefix pattern
        """
        lower = file_name.lower()
        stem = Path(file_name).stem

        if lower.startswith('ladder') or 'ladder' in lower:
            return 'ladder'
        if 'blanco' in lower or 'pocon' in lower or 'controle' in lower:
            return 'control'
        if file_name.startswith('A'):
            return 'control'
        if cls._RD_PREFIX_RE.match(stem):
            return 'sample'
        return 'unknown'

    @classmethod
    def get_number_of_contributors(cls, file_name: str) -> int:
        """Extract NOC from R&D filename: ``1A2`` → ``"2p"``."""
        stem = Path(file_name).stem
        if cls._RD_PREFIX_RE.match(stem):
            return int(stem[2])
        raise ValueError(f'Could not extract NoC from filename: {file_name}')

    @classmethod
    def get_sample_id(cls, file_name: str) -> str:
        """Extract profile prefix: ``1A2_A01_01.hid`` → ``"1A2"``."""
        stem = Path(file_name).stem
        if cls._RD_PREFIX_RE.match(stem):
            return stem.split('_')[0]
        raise ValueError(f'Cannot extract sample ID from R&D filename: {file_name}')

    @classmethod
    def find_annotation_file(cls, sample_path: Path, annotation_dir: Path) -> Path | None:
        """Find the TXT annotation file that contains this sample.

        R&D annotations are stored as ``{injection_name}.txt`` files,
        typically in the same directory structure as the HID files.
        """
        # The annotation file lives alongside the sample's injection directory
        injection_dir = sample_path.parent
        txt_files = list(annotation_dir.glob('*.txt'))
        if not txt_files:
            txt_files = list(injection_dir.glob('*.txt'))
        return txt_files[0] if txt_files else None

    @classmethod
    def find_ladder_for_sample(
        cls, sample_path: Path, ladder_mapping: dict[str, Path] | None = None
    ) -> Path | None:
        """Find the ladder for an R&D sample.

        Uses the pre-built mapping (from ``best_ladder_paths_DTH.csv``)
        if provided, otherwise looks in the same injection directory.
        """
        stem = sample_path.stem
        if ladder_mapping and stem in ladder_mapping:
            return ladder_mapping[stem]

        # Fallback: look for ladder files in the same directory
        parent = sample_path.parent
        ladders = [f for f in parent.glob('*.hid') if 'ladder' in f.name.lower()]
        return ladders[0] if ladders else None

    @staticmethod
    def load_ladder_mapping(csv_path: Path) -> dict[str, Path]:
        """Load a sample→ladder mapping from ``best_ladder_paths_DTH.csv``.

        Returns:
            Dict mapping sample stems to ladder paths.
        """
        mapping = {}
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapping[row['image_path']] = Path(row['ladder_path'])
        return mapping


    @classmethod
    def parse_annotations(
        cls,
        annotation_source: PathLike,
        scaling_strategy: ScalingStrategy
    ) -> Dict[str, Annotation]:
        """Parse manually called alleles from an annotation text file.

        The annotation file may contain calls for multiple samples. This function
        finds the rows matching ``sample_name`` and returns the parsed markers.

        Args:
            annotation_source: Path to the annotation CSV/TSV/TXT file.
            scaling_strategy: The scaling strategy to use for the annotations.

        Returns:
            List of Markers with their alleles, or ``None`` if not found.
        """
        if os.stat(annotation_source).st_size == 0:
            logger.debug('Empty annotation file: {}', annotation_source)
            raise RuntimeError('Annotations file is emtpy')

        annotation_mapping: Dict[str, Annotation] = {}
        with open(annotation_source, 'r') as f:
            header_result = cls._parse_csv_header(f)
            if header_result is None:
                return {}
            delimiter, allele_cols, height_cols = header_result

            reader = csv.reader(f, delimiter=delimiter)
            for sample, rows in groupby(reader, lambda row: row[0]):
                sample_annotation = cls._parse_sample_annotations(rows, allele_cols, height_cols, scaling_strategy)
                annotation_mapping[sample] = AlleleAnnotation(sample_annotation)

        return annotation_mapping

    @classmethod
    def _parse_sample_annotations(
        cls,
        rows,
        allele_cols: Iterable[int],
        height_cols: Iterable[int],
        scaling_strategy: ScalingStrategy
    ) -> list[Marker]:
        """Parse annotation rows for a single sample into Markers."""
        markers: list[Marker] = []

        marker_2_dye = scaling_strategy.marker_name_to_dye_idx()
        for row in rows:
            marker_name = row[1]
            dye_row = marker_2_dye.get(marker_name)
            if dye_row is None:
                continue

            alleles = frozenset(
                Allele(
                    name=allele_name,
                    height=float(row[height_col]) if cls.READ_ANNOTATION_HEIGHTS else None,
                )
                for allele_col, height_col in zip(allele_cols, height_cols, strict=True)
                if (allele_name := row[allele_col].strip('OB_'))
            )
            markers.append(Marker(name=marker_name, dye_row=dye_row, alleles=alleles))

        return markers

    @classmethod
    def _parse_csv_header(cls, file: io.TextIOWrapper) -> tuple[str, list[int], list[int]] | None:
        """Detect delimiter and locate Allele/Height columns in the header."""
        header = next(file)

        for delimiter in [',', ';', '\t']:
            columns = header.split(delimiter)
            allele_cols = [i for i, col in enumerate(columns) if col.startswith('Allele')]
            if allele_cols:
                height_cols = [i for i, col in enumerate(columns) if col.startswith('Height')]
                return delimiter, allele_cols, height_cols
        logger.trace(f'Could not parse Annotation header for: {file.name}')
        return None

    @classmethod
    def _read_csv_file(cls, csv_file: str | Path) -> Tuple[List[str], List[List[str]]]:
        """Read a csv file with a header row.

        Args:
            csv_file: The path to the csv file to be read

        Returns:
            Tuple with the header row and (filtered on empty) value rows.
        """
        with open(csv_file, 'r') as htap:
            reader = csv.reader(htap)
            rows = list(reader)
        headers, values = rows[0], filter(lambda r: len(r) > 0, rows[1:])
        return headers, list(values)

    @staticmethod
    def get_annotation_classes() -> list[str]:
        return ['noise', 'allele']

    @classmethod
    def _split(
        cls,
        dataset: HIDDataset,
        fraction: float | None = None,
        seed: int | None = None,
        k_folds: int | None = None,
        stratify_noc: bool = False,
        genotype_aware: bool = True,
        test_fraction: float = 0.0,
        **kwargs,
    ):
        """Replica-aware split that keeps sample prefixes together and balances NoC.

        Possible options are:
        1. Simple fractional split (2-way or 3-way with test_fraction)
        2. K-Fold split
        3. Above splits with optional:
            - Replica's grouped (to prevent data-leakage)
            - Number of Contributors balanced over splits
        """
        # TODO: If genotype-aware splitting is true, we can only have 3 or 6 folds since we have 6 mixture datasets
        match (fraction, k_folds):
            case (float(), None) if 0 < fraction < 1:
                return cls._fractional_split(
                    dataset, fraction, seed, stratify_noc, genotype_aware, test_fraction
                )
            case (None, int()) if 2 <= k_folds <= 6:
                if test_fraction > 0.0:
                    raise ValueError('test_fraction is not supported with k-fold splitting')
                if k_folds not in (2, 3, 6):
                    logger.warning(
                        f'Splitting the NFI R&D into {k_folds} folds results in uneven splits (2, 3, or 6 will)'
                    )
                return cls._kfold_split(dataset, k_folds, seed, stratify_noc, genotype_aware)
            case _:
                raise ValueError(
                    f'Provide either a fraction in (0, 1) or 2 <= k_folds <= 6, not both. Got {fraction=}, {k_folds=}'
                )

    # -- Fractional splitting ------
    @classmethod
    def _fractional_split(
        cls,
        dataset: HIDDataset,
        fraction: float,
        seed: int | None,
        stratify_noc: bool,
        genotype_aware: bool,
        test_fraction: float = 0.0,
    ) -> Tuple[Subset, Subset] | Tuple[Subset, Subset, Subset]:
        if genotype_aware:
            if stratify_noc:
                logger.warning(
                    '`stratify_noc` is set to True but cannot be combined with `genotype_aware`'
                )

            _, group_indices = cls._get_mixture_dataset_groups(dataset)

            if test_fraction > 0.0:
                main_groups, test_groups = train_test_split(
                    group_indices,
                    train_size=1.0 - test_fraction,
                    random_state=seed,
                    stratify=None,
                )
                adjusted_fraction = fraction / (1.0 - test_fraction)
                train_groups, val_groups = train_test_split(
                    main_groups,
                    train_size=adjusted_fraction,
                    random_state=seed,
                    stratify=None,
                )
                train_idx = [i for group in train_groups for i in group]
                val_idx = [i for group in val_groups for i in group]
                test_idx = [i for group in test_groups for i in group]
                logger.info(
                    f'Fractional 3-way split | {fraction:.0%} train / '
                    f'{1 - fraction - test_fraction:.0%} val / {test_fraction:.0%} test | '
                    'Genotype aware'
                )
                return (
                    Subset(dataset, train_idx),
                    Subset(dataset, val_idx),
                    Subset(dataset, test_idx),
                )

            train_groups, val_groups = train_test_split(
                group_indices, train_size=fraction, random_state=seed, stratify=None
            )
            train_idx = [i for group in train_groups for i in group]
            val_idx = [i for group in val_groups for i in group]

            actual_fraction = len(train_idx) / (len(train_idx) + len(val_idx))

            logger.info(
                f'Fractional split | {fraction:.0%} train (actual {actual_fraction:.1%}) | '
                'Genotype aware'
            )

        else:
            indices = list(range(len(dataset.images)))
            nocs = [
                cls.get_number_of_contributors(file_name=img.path.stem) for img in dataset.images
            ]

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
                return (
                    Subset(dataset, train_idx),
                    Subset(dataset, val_idx),
                    Subset(dataset, test_idx),
                )

            logger.info(
                f'Fractional split | {fraction:.0%} train | stratify={"noc" if stratify_noc else "none"}'
            )
            train_idx, val_idx = train_test_split(
                indices,
                train_size=fraction,
                random_state=seed,
                stratify=nocs if stratify_noc else None,
            )
        return Subset(dataset, train_idx), Subset(dataset, val_idx)

    # -- K-Fold --------------------

    @classmethod
    def _kfold_split(
        cls,
        dataset: HIDDataset,
        k_folds: int,
        seed: int | None,
        stratify_noc: bool,
        genotype_aware: bool,
    ) -> List[Tuple[Subset, Subset]]:
        if genotype_aware:
            _, group_indices = cls._get_mixture_dataset_groups(dataset)

            splitter = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
            group_range = list(range(len(group_indices)))
            logger.info(f'K-Fold split | {k_folds} folds | Genotype aware')
            return [
                (
                    Subset(dataset, [i for gi in train for i in group_indices[gi]]),
                    Subset(dataset, [i for gi in val for i in group_indices[gi]]),
                )
                for train, val in splitter.split(group_range)  # type: ignore
            ]

        indices = list(range(len(dataset.images)))
        sample_nocs = [cls.get_number_of_contributors(dataset.images[i].path.stem) for i in indices]
        if any(n is None for n in sample_nocs) and stratify_noc:
            raise AttributeError(
                "NoC couldn't be inferred for every sample, stratify=noc not possible"
            )
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
            for train, val in splitter.split(indices, sample_nocs if stratify_noc else indices)  # type: ignore
        ]

    # -- Splitting helpers ---------
    @classmethod
    def _get_mixture_dataset_groups(cls, dataset: HIDDataset) -> Tuple[List[str], List[List[int]]]:
        """Return (group_names, group_index_lists) where each group is a mixture dataset."""
        group_map: Dict[str, List[int]] = {}
        mixture_folder_pattern = re.compile(r'Mixture dataset \d')
        for i, img in enumerate(dataset.images):
            # parent folder name is the mixture dataset identifier
            group_key = mixture_folder_pattern.search(str(img.path.absolute()))
            if group_key is None:
                raise ValueError(
                    f'Failed to retrieve mixture dataset group key from: {img.path.absolute()}'
                )
            group_map.setdefault(group_key.group(), []).append(i)

        group_names = list(group_map.keys())
        group_indices = [group_map[name] for name in group_names]
        return group_names, group_indices
