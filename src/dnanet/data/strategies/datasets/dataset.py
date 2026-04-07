"""Abstract dataset strategy — interface for dataset-specific behavior.

Design pattern: **Strategy** (abstract base for dataset variants)
    Each forensic dataset (NFI R&D, ProvedIt, etc.) has different conventions
    for file naming, annotation formats, ladder identification, and contributor
    metadata. This ABC defines the contract; concrete implementations (in
    ``dnanet.data.strategies.datasets``) provide the details.

    To support a new dataset, create a new subclass and register it in the
    ``DATASET_STRATEGIES`` dict at the bottom of this file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Generator, Literal, Tuple

from dnanet.core.annotation import Annotation
from dnanet.core.types import PathLike

FileCategory = Literal["sample", "ladder", "control", "unknown"]


class DatasetStrategy(ABC):
    """Interface for dataset-specific file handling and annotation parsing.

    Each method is a classmethod because dataset strategies are stateless —
    the behavior depends only on the dataset conventions, not on instance state.
    """
    
    @classmethod
    @abstractmethod
    def collect_dataset_files(cls, root_path: PathLike, **kwargs) -> Generator[Tuple[Path, Annotation | None, Path | None]]:
        """Collect the dataset files for this specific dataset strategy."""
        
    @classmethod
    @abstractmethod
    def categorize_file(cls, file_name: str) -> FileCategory:
        """Classify a file as sample, ladder, control, or unknown."""

    @classmethod
    @abstractmethod
    def get_contributors(cls, file_name: str) -> str | None:
        """Extract number-of-contributors info from a filename.

        Returns a string like ``"2p"`` or ``None`` if not determinable.
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
    def find_annotation_file(
        cls, sample_path: Path, annotation_dir: Path
    ) -> Path | None:
        """Locate the annotation file for a given sample.

        Returns ``None`` if no annotation is available.
        """

    @classmethod
    @abstractmethod
    def parse_annotations(
        cls,
        annotation_source: Path,
    ) -> Dict[str, Annotation]:
        """Load annotation from annotation sample to Annotation object.

        Args:
            annotation_source: Path to annotation file/directory.
            sample_name: The sample identifier (from ``get_sample_id``).

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

    @classmethod
    @abstractmethod
    def find_annotation_for_sample(
        cls, sample_path: Path, annotation_mapping: Dict[str, Path] | None = None
    ) -> Path | None:
        """Find the appropriate annotation for a given sample."""

    @staticmethod
    @abstractmethod
    def get_annotation_classes() -> list[str]:
        """
        Return the list of annotation classes supported by this dataset.
        The first class is assumed to be the default (noise) class
        """

