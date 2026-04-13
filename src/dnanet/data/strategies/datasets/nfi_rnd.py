"""NFI R&D dataset strategy.

Handles the NFI Research & Development dataset conventions:
- File naming: ``{NOC}{batch}{contributors}_{well}_{rep}.hid``
  e.g. ``1A2_A01_01.hid`` means NOC=2, batch=A, contributors=1A2
- Ladders: filenames starting with ``ladder_`` (case-insensitive)
- Controls: ``blanco``, ``pocon``, ``controle``, or filenames starting with ``A``
- Annotations: Tab-separated TXT files (AlleleReport format), one per injection
"""

from __future__ import annotations

import os
import re
import csv
from typing import TYPE_CHECKING, Dict, List, Tuple, Iterable, Generator
from pathlib import Path
from itertools import groupby

import numpy as np
from loguru import logger
from torch.utils.data import Subset, Dataset
from sklearn.model_selection import (
    KFold,
    GroupKFold,
    StratifiedKFold,
    StratifiedGroupKFold,
    train_test_split,
)

from dnanet.core.allele import Allele
from dnanet.core.marker import Marker
from dnanet.core.annotation import Annotation, AlleleAnnotation
from dnanet.data.strategies.datasets.dataset import SplitResult, FileCategory, DatasetStrategy


# R&D filename pattern: digit + letter + digit (e.g. "1A2")
_RD_PREFIX_RE = re.compile(r"^\d[A-F]\d")



if TYPE_CHECKING:
    from dnanet.data import HIDDataset
    from dnanet.core.types import PathLike


class NFIRnDStrategy(DatasetStrategy):
    """Strategy for the NFI R&D mixture dataset."""

    READ_ANNOTATION_HEIGHTS: bool = False
    # R&D filename pattern: digit + letter + digit (e.g. "1A2")
    _RD_PREFIX_RE = re.compile(r'^\d[A-F]\d')

    @classmethod
    def collect_dataset_files(
        cls,
        root_path: PathLike,
        analysis_treshold_type: str = 'DTH',
        **kwargs
    ) -> Generator[Tuple[Path, Annotation | None, Path | None]]:
        """Collect the dataset files for this specific dataset strategy.

        Creates a generator that yields paths to HIDImage's, Annotations (optional),
        and corresponding Ladder file (optional).

        Args:
            root_path: The path to the root of this dataset
            analysis_treshold_type: Whether to take annotations that were made with high (DTH) or low (DTL) analytical tresholds.
        """
        path = Path(root_path)
        csv_files = list(path.rglob('*.csv'))

        hid_to_annotation_file_pattern = r'.*hid_to_annotation.*'
        hid_to_annotation_path = None
        hid_to_ladder_pattern = r'.*best_ladder_paths.*'
        hid_to_ladder_path = None

        analysis_treshold_type: str = kwargs.get('analysis_treshold_type', 'DTH')
        logger.info(f"Using treshold type: {analysis_treshold_type}")

        for csv_file in csv_files:
            if re.match(hid_to_annotation_file_pattern, csv_file.name):
                hid_to_annotation_path = csv_file.absolute()
            if re.match(hid_to_ladder_pattern, csv_file.name):
                hid_to_ladder_path = csv_file.absolute()
        if hid_to_annotation_path is None or hid_to_ladder_path is None:
            raise ValueError(
                'Path does not contain the neccessary mapping files (annotation & ladder)'
            )

        # Allele Report for Annotations
        annotation_txt_files = list(path.rglob('*AlleleReport.txt'))
        annotation_name_to_annotation: Dict[str, Annotation] = {}
        for txt_file in annotation_txt_files:
            _annotation = cls.parse_annotations(txt_file)

            if _annotation:
                annotation_name_to_annotation.update(_annotation)

        # HID to Annotation mapping
        hta_header, hta_values = cls._read_csv_file(hid_to_annotation_path)
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
        _, htl_values = cls._read_csv_file(hid_to_ladder_path)
        hid_to_ladder = {hid: Path(ladder) for hid, ladder in htl_values}

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
    def get_number_of_contributors(cls, file_name: str) -> int | None:
        """Extract NOC from R&D filename: ``1A2`` → ``"2p"``."""
        stem = Path(file_name).stem
        if cls._RD_PREFIX_RE.match(stem):
            return int(stem[2])
        return None

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

    @staticmethod
    def _get_scaling_strategy():
        """Get the active panel from the strategy registry."""
        from dnanet.data.strategies.registry import StrategyRegistry

        return StrategyRegistry.get_scaling_strategy()

    @classmethod
    def parse_annotations(
        cls,
        annotation_source: PathLike,
    ) -> Dict[str, Annotation]:
        """Parse manually called alleles from an annotation text file.

        The annotation file may contain calls for multiple samples. This function
        finds the rows matching ``sample_name`` and returns the parsed markers.

        Args:
            annotation_source: Path to the annotation CSV/TSV/TXT file.

        Returns:
            List of Markers with their alleles, or ``None`` if not found.
        """
        if os.stat(annotation_source).st_size == 0:
            logger.debug('Empty annotation file: {}', annotation_source)
            raise RuntimeError('Annotations file is emtpy')

        annotation_mapping: Dict[str, Annotation] = {}
        with open(annotation_source, 'r') as f:
            try:
                delimiter, allele_cols, height_cols = cls._parse_csv_header(f)
            except TypeError as e:
                logger.debug('Could not parse header of {}: {}', annotation_source, e)
                raise e

            reader = csv.reader(f, delimiter=delimiter)
            for sample, rows in groupby(reader, lambda row: row[0]):
                sample_annotation = cls._parse_sample_annotations(rows, allele_cols, height_cols)
                annotation_mapping[sample] = AlleleAnnotation(sample_annotation)

        return annotation_mapping

    @classmethod
    def _parse_sample_annotations(
        cls,
        rows,
        allele_cols: Iterable[int],
        height_cols: Iterable[int],
    ) -> list[Marker]:
        """Parse annotation rows for a single sample into Markers."""
        markers: list[Marker] = []

        marker_2_dye = cls._get_scaling_strategy().marker_name_to_dye_idx()
        for row in rows:
            marker_name = row[1]
            dye_row = marker_2_dye[marker_name]
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
    def _parse_csv_header(cls, file) -> tuple[str, list[int], list[int]]:
        """Detect delimiter and locate Allele/Height columns in the header."""
        header = next(file)

        for delimiter in [',', ';', '\t']:
            columns = header.split(delimiter)
            allele_cols = [i for i, col in enumerate(columns) if col.startswith('Allele')]
            if allele_cols:
                height_cols = [i for i, col in enumerate(columns) if col.startswith('Height')]
                return delimiter, allele_cols, height_cols

        raise TypeError(f'No valid delimiter found in header: {header!r}')

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
    def split(
        cls,
        dataset: HIDDataset,
        fraction: float | None = None,
        seed: int | None = None,
        k_folds: int | None = None,
        stratify_noc: bool = True,
        group_by_replica: bool = True,
    ) -> SplitResult:
        """Replica-aware split that keeps sample prefixes together and balances NoC.

        Possible options are:
        1. Simple fractional split
        2. K-Fold split
        3. Above splits with optional:
            - Replica's grouped (to prevent data-leakage)
            - Number of Contributors balanced over splits
        """
        match (fraction, k_folds):
            case (float(), None) if 0 < fraction < 1:
                return cls._fractional_split(dataset, fraction, seed, stratify_noc, group_by_replica)
            case (None, int()) if 2 <= k_folds < len(dataset):
                return cls._kfold_split(dataset, k_folds, seed, stratify_noc, group_by_replica)
            case (None, None):
                logger.info('No split | using entire dataset for training')
                train_set = Subset(dataset, list(range(len(dataset))))
                val_set = Subset(dataset, [])
                return train_set, val_set
            case _:
                raise ValueError(
                    f'Provide either a fraction in (0, 1) or 2 <= k_folds < {len(dataset)=}, not both. Got {fraction=}, {k_folds=}'
                )

    # -- Fractional splitting ------
    @classmethod
    def _fractional_split(cls, dataset: HIDDataset, fraction, seed, stratify_noc, group_by_replica):

        if not group_by_replica:
            indices = list(range(len(dataset)))
            nocs = [
                v
                for v in (
                    cls.get_number_of_contributors(file_name=img.path.stem) for img in dataset.data
                )
                if v is not None
            ]

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

        # Grouped: approximate via StratifiedGroupKFold / GroupKFold, take first fold
        replica_map = cls._build_replica_map(dataset)
        replica_ids = list(replica_map.keys())
        n_splits = max(2, round(1.0 / (1.0 - fraction)))
        dummy_X = np.arange(len(replica_ids))
        noc_labels = cls._replica_noc_labels(dataset, replica_map) if stratify_noc else dummy_X

        splitter = (
            StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            if stratify_noc
            else GroupKFold(n_splits=n_splits)
        )
        logger.info(
            f'Fractional grouped split | -{fraction:.0%} train | {n_splits=} | stratify={"noc" if stratify_noc else "none"}'
        )
        train_pos, val_pos = next(splitter.split(dummy_X, noc_labels, groups=replica_ids))
        return cls._subsets(dataset, replica_map, train_pos, val_pos)

    # -- K-Fold --------------------

    @classmethod
    def _kfold_split(
        cls,
        dataset: HIDDataset,
        k_folds: int,
        seed: int | None,
        stratify_noc: bool,
        group_by_replica: bool,
    ) -> SplitResult:
        replica_map = cls._build_replica_map(dataset)
        replica_ids = list(replica_map.keys())
        dummy_X = np.arange(len(replica_ids))
        noc_labels = cls._replica_noc_labels(dataset, replica_map) if stratify_noc else dummy_X

        if not group_by_replica:
            indices = [i for indices in replica_map.values() for i in indices]
            sample_nocs = [
                cls.get_number_of_contributors(dataset.data[i].path.stem) for i in indices
            ]
            if any(n is None for n in sample_nocs) and stratify_noc:
                raise AttributeError(
                    "NoC couldn't be inferred for every sample, stratify=noc not possible"
                )
            splitter = (
                StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
                if stratify_noc
                else KFold(n_splits=k_folds, shuffle=True, random_state=seed)
            )
            logger.info(
                f'K-Fold split | {k_folds} folds | stratify={"noc" if stratify_noc else "none"}'
            )
            return [
                (
                    Subset(dataset, [indices[i] for i in train]),
                    Subset(dataset, [indices[i] for i in val]),
                )
                for train, val in splitter.split(indices, sample_nocs if stratify_noc else indices)
            ]

        splitter = (
            StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=seed)
            if stratify_noc
            else GroupKFold(n_splits=k_folds)
        )
        logger.info(
            f'K-Fold grouped split | {k_folds} folds | stratify={"noc" if stratify_noc else "none"}'
        )
        return [
            cls._subsets(dataset, replica_map, train_pos, val_pos)
            for train_pos, val_pos in splitter.split(dummy_X, noc_labels, groups=replica_ids)
        ]

    # -- Splitting helpers ---------

    @classmethod
    def _build_replica_map(cls, dataset: HIDDataset) -> Dict[str, List[int]]:
        """Maps each replica_id to its list of sample indices."""
        replica_map: Dict[str, List[int]] = {}
        for i, img in enumerate(dataset.data):
            replica_id = cls.get_sample_id(img.path.stem)
            replica_map.setdefault(replica_id, []).append(i)
        return replica_map

    @classmethod
    def _replica_noc_labels(
        cls, dataset: HIDDataset, replica_map: Dict[str, List[int]]
    ) -> List[int]:
        """Majority-vote NoC label per replica, in replica_map insertion order."""

        def majority_noc(indices: List[int]) -> int:
            nocs = [
                cls.get_number_of_contributors(file_name=dataset.data[i].path.stem) for i in indices
            ]
            if any([n is None for n in nocs]):
                raise ValueError(
                    'Could not extract NoC for all samples, stratify on NoC not possible.'
                )
            return Counter(nocs).most_common(1)[0][0]  # type: ignore

        return [majority_noc(indices) for indices in replica_map.values()]

    @staticmethod
    def _subsets(
        dataset: Dataset, replica_map: dict, train_pos: Sequence[int], val_pos: Sequence[int]
    ) -> Tuple[Subset, Subset]:
        """Expand replica positions back to flat sample index lists."""
        replicas = list(replica_map.values())
        train_idx = [i for pos in train_pos for i in replicas[pos]]
        val_idx = [i for pos in val_pos for i in replicas[pos]]
        return Subset(dataset, train_idx), Subset(dataset, val_idx)

