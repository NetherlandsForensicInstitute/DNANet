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


if typing.TYPE_CHECKING:
    from pathlib import Path

    from annotated_types import T

    from dnanet.core.types import PathLike
    from dnanet.core.annotation import Annotation
    from dnanet.data.strategies.scaling import ScalingStrategy


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

    @staticmethod
    @abstractmethod
    def get_annotation_classes() -> list[str]:
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
