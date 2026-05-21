"""HIDImage — domain model for a single DNA profile.

Design pattern: **Lazy Loading (Virtual Proxy)**
    An ``HIDImage`` wraps a path to a HID file. The heavy data (the numpy
    array of fluorescence signals) is NOT loaded in ``__init__``. Instead,
    it's loaded on first access to the ``data`` property.

    This means you can create thousands of ``HIDImage`` objects cheaply
    (e.g. when scanning a directory), and only pay the I/O cost when you
    actually need the pixel data.

    The ``use_cache`` flag controls whether the loaded data is kept in
    memory after the first read (default: yes).

Design pattern: **Null Object** (for annotation)
    If no annotation is available, ``annotation`` returns ``None`` rather
    than raising. Consumers check for ``None`` rather than catching exceptions.
"""

from __future__ import annotations

import abc
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, MutableMapping
from pathlib import Path
from functools import cached_property

import numpy as np
from loguru import logger

from dnanet.data.parsing import get_peak_data
from dnanet.data.strategies.scaling import ScalingStrategy


if TYPE_CHECKING:
    from dnanet.core.panel import Panel
    from dnanet.core.types import PathLike
    from dnanet.core.annotation import (
        Annotation,
        ClassAnnotation,
        AlleleAnnotation,
        ScanpointAnnotation,
    )


# Default RFU detection threshold
_DEFAULT_RFU_THRESHOLD = 40

class TrainableElement(abc.ABC):

    @property
    @abstractmethod
    def data(self) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def annotation(self) -> Annotation | ClassAnnotation | None:
        raise NotImplementedError



class HIDImage(TrainableElement):
    """A single DNA profile loaded from a HID file.

    This is the primary data container in DNANet. Each HIDImage wraps:
    - The raw fluorescence signal (5 dyes + optional size standard)
    - An optional segmentation annotation (binary mask)
    - Metadata (called alleles, ladder path, number of contributors, etc.)
    - A scaler array mapping pixel positions to base pairs

    Args:
        path: Path to the HID file.
        panel: Allele/marker panel for this profile.
        annotations_file: Path to annotation CSV (if available).
        include_size_standard: Include the 6th dye (size standard) in data.
        annotation: Pre-built annotation (e.g. from cache).
        use_cache: Cache the loaded data in memory.
        data_loading_strategy: One of "raw", "analyzed", "superior".
        rfu_threshold: Minimum RFU for peak detection.
        meta: Additional metadata dictionary.
    """

    def __init__(
        self,
        path: PathLike,
        scaling_strategy: ScalingStrategy,
        adjusted_panel: Panel | None = None,
        include_size_standard: bool = False,
        annotation: ScanpointAnnotation | None = None,
        allele_annotation: AlleleAnnotation | None = None,
        load_in_memory: bool = True,
        data_loading_strategy: str = "superior",
        rfu_threshold: float = _DEFAULT_RFU_THRESHOLD,
        meta: MutableMapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.include_size_standard = include_size_standard
        self.load_in_memory = load_in_memory
        self.data_loading_strategy = data_loading_strategy
        self.rfu_threshold = rfu_threshold
        self.scaling_strategy = scaling_strategy

        self._adjusted_panel = adjusted_panel
        self._data: np.ndarray | None = None
        self._annotation = annotation
        self._allele_annotation = allele_annotation
        self._meta: MutableMapping[str, Any] = meta or {}
        self._scaler: np.ndarray | None = None

    # -- Properties ------------------------------------------------------- #

    @property
    def data(self) -> np.ndarray | None:
        """The fluorescence signal array, loaded lazily.

        Shape: ``(num_dyes, signal_length, 1)`` — the trailing dimension
        is kept for backward compatibility with the segmentation pipeline.
        """
        if self.load_in_memory:
            if self._data is None:
                self._data = self._load()
            return self._data
        return self._load()

    @cached_property
    def dimensions(self) -> tuple[int, int]:
        """``(height, width)`` of the data array."""
        d = self.data
        return (d.shape[0], d.shape[1]) if d is not None else (0, 0)

    @property
    def allele_annotation(self) -> AlleleAnnotation | None:
        return self._allele_annotation


    @property
    def annotation(self) -> ScanpointAnnotation | None:
        return self._annotation

    @annotation.setter
    def annotation(self, annotation) -> None:
        self._annotation = annotation

    @property
    def meta(self) -> MutableMapping[str, Any]:
        return self._meta

    @property
    def scaler(self) -> np.ndarray:
        """Base-pair values for each pixel position. Shape: ``(1, signal_length)``."""
        if self._scaler is None:
            self._load()
        return self._scaler  # type: ignore

    @property
    def adjusted_panel(self) -> Panel | None:
        return self._adjusted_panel

    # -- Data loading ----------------------------------------------------- #

    def _load(self) -> np.ndarray | None:
        """Parse the HID file, validate size standard, build annotation."""
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))

        profile = get_peak_data(self.path, self.scaling_strategy, self.data_loading_strategy)
        if profile is None:
            return None

        # Parse size standard and rescale
        ss_lane = np.array(profile[-1])
        try:
            ss_result = self.scaling_strategy.parse_size_standard(ss_lane)
        except ValueError as e:
            logger.warning("Size standard invalid for {}: {}", self.path.name, e)
            return None

        if ss_result is None:
            logger.warning("Size standard parsing returned None for {}", self.path.name)
            return None

        selected = profile if self.include_size_standard else profile[:-1]
        data = selected[:, ss_result.rescaled_indices]
        self._scaler = ss_result.scaler

        return data

    # -- Dunder ----------------------------------------------------------- #

    def __repr__(self) -> str:
        return f"HIDImage({self.path.name})"
