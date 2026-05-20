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

import csv
import typing
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Literal, Mapping, Generator

import numpy as np
from loguru import logger

from dnanet.core import LabelCategory
from dnanet.core.annotation import Annotation, SpanAnnotation, ScanpointAnnotation
from dnanet.data.strategies.scaling import ScalingStrategy


if typing.TYPE_CHECKING:
    from pathlib import Path

    from annotated_types import T

    from dnanet.core.types import PathLike


FileCategory = Literal['sample', 'ladder', 'control', 'unknown']


class DatasetStrategy(ABC):
    """Interface for dataset-specific file handling and annotation parsing.

    Each method is a classmethod because dataset strategies are stateless —
    the behavior depends only on the dataset conventions, not on instance state.
    """

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
        cls, annotation_source: PathLike, scaling_strategy: ScalingStrategy
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

    @abstractmethod
    def get_annotation_classes(self) -> list[str]:
        """Return the list of annotation classes supported by this dataset.

        The first class is assumed to be the default (noise) class.
        """

    @abstractmethod
    def cache_signature(self) -> dict:
        """Return a JSON-serializable dict uniquely identifying this strategy's configuration.

        Used to build the dataset cache key. Must change whenever the strategy
        configuration changes in a way that affects collected files or annotations.
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

    @staticmethod
    def _unwrap(dataset) -> Tuple[Any, List[int]]:
        """Return (underlying dataset, index map from local position to underlying index).

        Handles both plain datasets (with .images) and torch Subset wrappers.
        """
        from torch.utils.data import Subset

        if isinstance(dataset, Subset):
            return dataset.dataset, list(dataset.indices)
        return dataset, list(range(len(dataset.images)))

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
    ) -> dict[str, SpanAnnotation | None]:
        """Parse span-annotation CSV files into per-profile span annotations.

        Span annotations are expected to contain ``profile``, ``user``, ``dye``,
        ``x0``, ``x1``, and ``category`` columns. Rows are grouped by profile and
        annotator, converted to span tensors, and optionally merged when multiple
        annotators labeled the same profile.  The result is a :class:`SpanAnnotation`
        wrapping the raw ``(num_dyes, scanpoints, num_classes)`` tensor; flattening
        to :class:`ScanpointAnnotation` is deferred to ``HIDDataset._load_images``
        so that per-class adjustment can run on the full 3-D tensor first.

        Args:
            span_annotations_path: Directory containing span-annotation CSV files.
            scaling_strategy: Scaling strategy that defines dye count and
                scanpoint resolution.

        Returns:
            Mapping from HID profile filename to scanpoint annotation. Returns an
            empty mapping when no CSV files are found.

        Raises:
            ValueError: If required columns are missing, dyes are unknown, or
                categories cannot be mapped to :class:`LabelCategory`.
        """
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
        rows = []
        required_columns = {'profile', 'user', 'dye', 'x0', 'x1', 'category'}
        for f in csv_files:
            with open(f, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                if reader.fieldnames is None:
                    continue
                columns = set(reader.fieldnames)
                missing_columns = required_columns.difference(columns)
                if missing_columns:
                    raise ValueError(
                        f'Missing span annotation columns: {sorted(missing_columns)}, found columns: {sorted(columns)} in {f}'
                    )
                for row in reader:
                    # check for NaNs (empty strings in csv.DictReader)
                    if any(not row.get(col) for col in required_columns):
                        continue
                    rows.append(row)

        if not rows:
            return {}

        profiles = {row['profile'] for row in rows}
        categories = {row['category'] for row in rows}

        logger.info(f'Found {len(rows)} valid span annotations in {len(profiles)} unique profiles')
        logger.info(f'Categories found in annotations: {categories}')

        # convert dye names to dye indices and category names to indices
        valid_rows = []
        for row in rows:
            dye_idx = _dye_name_to_dye_idx.get(str(row['dye']).strip().lower())
            if dye_idx is None:
                raise ValueError(f'Unknown dye values in span annotations: {row["dye"]}')

            category_idx = LabelCategory.display_name_to_index(row['category'])
            if category_idx is None:
                raise ValueError(f'Unknown category values in span annotations: {row["category"]}')

            row['dye_idx'] = int(dye_idx)
            row['category_idx'] = int(category_idx)
            row['x0'] = int(row['x0'])
            row['x1'] = int(row['x1'])
            valid_rows.append(row)

        # create span annotations grouped by file and annotator
        hid_file_name_to_span_annotations: dict[str, list[np.ndarray]] = {}

        # grouping logic
        groups: dict[tuple[str, str], list[dict]] = {}
        for row in valid_rows:
            profile = row['profile']
            if profile.lower().endswith('.hid'):
                profile = profile[:-4]
            key = (profile, row['user'])
            groups.setdefault(key, []).append(row)

        for (hid_file_name, _annotator), group_rows in groups.items():
            spannotation = cls._df_to_span_annotation(group_rows, scaling_strategy)
            hid_file_name_to_span_annotations.setdefault(hid_file_name, []).append(spannotation)

        # Merge span tensors per profile and wrap in SpanAnnotation.
        # Flattening to ScanpointAnnotation is intentionally deferred to
        # HIDDataset._load_images so that per-class adjustment can run first
        # on the full 3-D tensor, preserving class boundaries.
        hid_to_annotation: dict[str, SpanAnnotation | None] = {}
        for hid_file_name, span_annotations in hid_file_name_to_span_annotations.items():
            if len(span_annotations) > 1:
                span_annotation = cls._merge_span_annotations(span_annotations, hid_file_name)
            else:
                span_annotation = span_annotations[0]

            hid_to_annotation[hid_file_name] = SpanAnnotation(span_annotation)

        return hid_to_annotation

    @staticmethod
    def _df_to_span_annotation(rows: list[dict], scaling_strategy: ScalingStrategy) -> np.ndarray:
        """Convert one profile/annotator rows to a one-hot span tensor.

        Args:
            rows: Span rows for a single profile and annotator. Each row must
                already contain integer ``dye_idx`` and ``category_idx``.
            scaling_strategy: Scaling strategy that defines the output shape.

        Returns:
            A ``(num_dyes, scanpoints, num_classes)`` array with annotated spans
            marked as ``1``.

        Raises:
            ValueError: If a row contains a dye or category index outside the
                output tensor shape.
        """
        num_dyes = scaling_strategy.kit.num_dyes
        scanpoints = scaling_strategy.scanpoint_resolution
        num_classes = len(LabelCategory)
        spannotation = np.zeros((num_dyes, scanpoints, num_classes), dtype=np.int8)

        for row in rows:
            dye_idx = int(row['dye_idx'])
            category_idx = int(row['category_idx'])
            if not 0 <= dye_idx < num_dyes:
                # raise ValueError(f'Dye index {dye_idx} outside annotation shape')
                continue # FIXME parsing of y profiles fails
            if not 0 <= category_idx < num_classes:
                raise ValueError(f'Category index {category_idx} outside annotation shape')

            start, stop = sorted((int(row['x0']), int(row['x1'])))
            start = max(0, start)
            stop = min(scanpoints, stop)
            if start >= stop:
                continue

            spannotation[dye_idx, start:stop, category_idx] = 1

        return spannotation

    @staticmethod
    def _merge_span_annotations(spannotations: List[np.ndarray], hid_file_name: str) -> np.ndarray:
        """Merge multiple annotator span tensors for the same HID profile.

        The current merge policy keeps the first annotation and logs that
        multiple annotations were present.

        Args:
            spannotations: Span tensors collected for one HID profile.
            hid_file_name: HID profile filename used for logging.

        Returns:
            The selected span annotation tensor.
        """
        logger.debug(
            f'Found multiple span annotations for {hid_file_name}. Merging by taking the first only'
        )
        return spannotations[0]


