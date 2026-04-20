"""Abstract dataset strategy — interface for dataset-specific behavior.

Design pattern: **Strategy** (abstract base for dataset variants)
    Each forensic dataset (NFI R&D, ProvedIt, etc.) has different conventions
    for file naming, annotation formats, ladder identification, and contributor
    metadata. This ABC defines the contract; concrete implementations (in
    ``dnanet.data.strategies.datasets``) provide the details.

    To support a new dataset, create a new subclass and pass it explicitly
    where a ``DatasetStrategy`` is required.
"""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Literal, Mapping, Generator
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from dnanet.core import LabelCategory

if typing.TYPE_CHECKING:
    from pathlib import Path

    from annotated_types import T

    from dnanet.core.types import PathLike
    from dnanet.core.annotation import Annotation, ScanpointAnnotation
    from dnanet.data.strategies.scaling import ScalingStrategy


FileCategory = Literal['sample', 'ladder', 'control', 'unknown']


class DatasetStrategy(ABC):
    """Interface for dataset-specific file handling and annotation parsing.

    Each method is a classmethod because dataset strategies are stateless —
    the behavior depends only on the dataset conventions, not on instance state.
    """

    @classmethod
    @abstractmethod
    def collect_dataset_files(
        cls, root_path: PathLike, scaling_strategy: ScalingStrategy, **kwargs
    ) -> Generator[Tuple[Path, Annotation | None, Path | None]]:
        """Collect the dataset files for this specific dataset strategy."""

    @classmethod
    @abstractmethod
    def categorize_file(cls, file_name: str) -> FileCategory:
        """Classify a file as sample, ladder, control, or unknown."""

    @classmethod
    @abstractmethod
    def get_number_of_contributors(cls, file_name: str) -> int | None:
        """Extract number-of-contributors info from a filename.

        Returns an int for the NoC or ``None`` if not determinable.
        """

    @classmethod
    @abstractmethod
    def get_sample_id(cls, file_name: str) -> str:
        """Extract a unique sample identifier from a filename.

        For R&D this is the prefix (e.g. ``"1A2"``), for ProvedIt it might
        be the full stem minus the injection suffix.
        """

    @classmethod
    @abstractmethod
    def parse_annotations(
        cls,
        annotation_source: PathLike,
        scaling_strategy: ScalingStrategy
    ) -> Mapping[str, Annotation]:
        """Load annotation from annotation sample to Annotation object.

        Args:
            annotation_source: Path to annotation file/directory.
            scaling_strategy: Scaling strategy to use for scaling.

        Returns:
            Annotation object (either AlleleAnnotation or ScanpointAnnotation)
        """

    @classmethod
    @abstractmethod
    def find_ladder_for_sample(
        cls, sample_path: Path, ladder_mapping: Dict[str, Path] | None = None
    ) -> Path | None:
        """Find the ladder file corresponding to a sample.

        Args:
            sample_path: Path to the sample HID file.
            ladder_mapping: Optional pre-built mapping of sample→ladder paths.

        Returns:
            Path to the ladder file, or ``None`` if not found.
        """


    @staticmethod
    @abstractmethod
    def get_annotation_classes() -> list[str]:
        """Return the list of annotation classes supported by this dataset.

        The first class is assumed to be the default (noise) class.
        """

    @classmethod
    @abstractmethod
    def _split(
        cls, dataset, **kwargs
    ) -> Tuple[Any, Any] | Tuple[Any, Any, Any] | List[Tuple[Any, Any]]:
        """Default: simple random fraction split.

        Override in strategies that have richer metadata (e.g. replica-aware).

        Returns:
            ``(train_subset, val_subset)`` as :class:`torch.utils.data.Subset`
            for a 2-way fractional split, or
            ``(train_subset, val_subset, test_subset)`` when ``test_fraction > 0``, or
            a list of ``(train_subset, val_subset)`` pairs for k-fold splits.
        """

    @classmethod
    def split(cls, dataset, **kwargs) -> Tuple[T, T] | Tuple[T, T, T] | List[Tuple[T, T]]:
        """Splitting wrapper.

        Uses a Strategy's _split implementation to split the data and
        converts the result to the correct dataset type if needed.
        """
        from dnanet.data.peak_dataset import PeakWindowDataset

        result = cls._split(dataset, **kwargs)

        if not isinstance(dataset, PeakWindowDataset):
            return result

        def convert(splits):
            return tuple(dataset.subset(s.indices) for s in splits)

        if isinstance(result, list):
            # K-Fold
            return [convert(pair) for pair in result]
        return convert(result)

    @property
    def annotation_to_idx(self) -> Dict[str, int]:
        return {name: idx for idx, name in enumerate(self.get_annotation_classes())}

    @classmethod
    def _parse_span_annotation(
            cls, span_annotations_path: Path, scaling_strategy: ScalingStrategy
    ) -> dict[str, ScanpointAnnotation | None]:

        _dye_name_to_dye_idx = {
            'blue': 0,
            'green': 1,
            'yellow': 2,
            'black': 2,
            'red': 3,
            'purple': 4,
            'orange': 5,
        }

        # collect files
        csv_files = list(span_annotations_path.rglob('*.csv'))
        if not csv_files:
            logger.warning('No span annotation CSV files found in {}', span_annotations_path)
            return {}

        # read csv files
        df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
        columns = df.columns.tolist()
        required_columns = {'profile', 'user', 'dye', 'x0', 'x1', 'category'}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(f'Missing span annotation columns: {sorted(missing_columns)}, found columns: {sorted(columns)}')

        len_before_drop = len(df)
        df = df.dropna(subset=['profile', 'user', 'dye', 'x0', 'x1', 'category']).copy()
        dropped = len_before_drop - len(df)


        logger.info(
            f'Found {len(df)} valid span annotations in {len(df["profile"].unique())} '
            f'profiles (dropped {dropped} rows)'
        )
        logger.info(f'Categories found in annotations: {df["category"].unique()}')

        # convert dye names to dye indices
        df['dye_idx'] = (
            df['dye']
            .astype(str)
            .str.strip()
            .str.lower()
            .map(_dye_name_to_dye_idx)
        )
        unknown_dyes = df.loc[df['dye_idx'].isna(), 'dye'].unique()
        if len(unknown_dyes) > 0:
            raise ValueError(f'Unknown dye values in span annotations: {unknown_dyes}')

        # convert category names to indices
        df['category_idx'] = df['category'].map(LabelCategory.display_name_to_index)
        unknown_categories = df.loc[df['category_idx'].isna(), 'category'].unique()
        if len(unknown_categories) > 0:
            raise ValueError(f'Unknown category values in span annotations: {unknown_categories}')

        df[['dye_idx', 'category_idx', 'x0', 'x1']] = df[
            ['dye_idx', 'category_idx', 'x0', 'x1']
        ].astype(int)

        # create span annotations grouped by file and annotator
        hid_file_name_to_span_annotations: dict[str, list[np.ndarray]] = {}
        for (hid_file_name, _annotator), hid_file_df in df.groupby(['profile', 'user'], sort=False):
            spannotation = cls._df_to_span_annotation(hid_file_df, scaling_strategy)
            hid_file_name_to_span_annotations.setdefault(hid_file_name, []).append(spannotation)

        # merge span annotations into a scanpoint annotation
        hid_to_annotation: dict[str, ScanpointAnnotation | None] = {}
        for hid_file_name, span_annotations in hid_file_name_to_span_annotations.items():
            if len(span_annotations) > 1:
                span_annotation = cls._merge_span_annotations(span_annotations, hid_file_name)
            else:
                span_annotation = span_annotations[0]

            hid_to_annotation[hid_file_name] = cls._span_to_scanpoint_annotation(span_annotation, hid_file_name)

        return hid_to_annotation

    @staticmethod
    def _df_to_span_annotation(df: pd.DataFrame, scaling_strategy: ScalingStrategy) -> np.ndarray:
        num_dyes = scaling_strategy.kit.num_dyes
        scanpoints = scaling_strategy.scanpoint_resolution
        num_classes = len(LabelCategory)
        spannotation = np.zeros((num_dyes, scanpoints, num_classes), dtype=np.int8)

        for row in df.itertuples(index=False):
            dye_idx = int(row.dye_idx)
            category_idx = int(row.category_idx)
            if not 0 <= dye_idx < num_dyes:
                raise ValueError(f'Dye index {dye_idx} outside annotation shape')
            if not 0 <= category_idx < num_classes:
                raise ValueError(f'Category index {category_idx} outside annotation shape')

            start, stop = sorted((int(row.x0), int(row.x1)))
            start = max(0, start)
            stop = min(scanpoints, stop)
            if start >= stop:
                continue

            spannotation[dye_idx, start:stop, category_idx] = 1

        return spannotation

    @staticmethod
    def _merge_span_annotations(spannotations: List[np.ndarray], hid_file_name: str) -> np.ndarray:
        logger.debug(
            f'Found multiple span annotations for {hid_file_name}. Merging by taking the first only'
        )
        return spannotations[0]

    @staticmethod
    def _span_to_scanpoint_annotation(span_annotation: np.ndarray, hid_file_name: str) -> ScanpointAnnotation:
        flattened = span_annotation.argmax(axis=-1)

        if np.any(span_annotation.sum(axis=-1) > 1):
            logger.debug(f'Found overlapping annotations for {hid_file_name}, taking the lowest class index')

        return ScanpointAnnotation(flattened.astype(np.int8, copy=False))
