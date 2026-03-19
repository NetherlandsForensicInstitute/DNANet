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

from functools import cached_property
from pathlib import Path
from typing import Any, MutableMapping

import numpy as np
from loguru import logger

from dnanet.core.annotation import Annotation
from dnanet.core.marker import Marker
from dnanet.core.panel import Panel
from dnanet.core.types import PathLike
from dnanet.data.parsing import get_peak_data, parse_called_alleles
from dnanet.data.preprocessing.peaks import find_peak_boundary, find_peak_idx_near_or_in_range
from dnanet.data.strategies.registry import StrategyRegistry


# Default RFU detection threshold
_DEFAULT_RFU_THRESHOLD = 40


class HIDImage:
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
        panel: Panel | None = None,
        annotations_file: PathLike | None = None,
        include_size_standard: bool = False,
        annotation: Annotation | None = None,
        use_cache: bool = True,
        data_loading_strategy: str = "superior",
        rfu_threshold: float = _DEFAULT_RFU_THRESHOLD,
        meta: MutableMapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.annotations_file = annotations_file
        self.include_size_standard = include_size_standard
        self.use_cache = use_cache
        self.data_loading_strategy = data_loading_strategy
        self.rfu_threshold = rfu_threshold

        self._panel = panel
        self._data: np.ndarray | None = None
        self._annotation = annotation
        self._meta: MutableMapping[str, Any] = meta or {}
        self._scaler: np.ndarray | None = None

    # -- Properties ------------------------------------------------------- #

    @property
    def data(self) -> np.ndarray | None:
        """The fluorescence signal array, loaded lazily.

        Shape: ``(num_dyes, signal_length, 1)`` — the trailing dimension
        is kept for backward compatibility with the segmentation pipeline.
        """
        if self.use_cache:
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
    def annotation(self) -> Annotation | None:
        return self._annotation

    @property
    def meta(self) -> MutableMapping[str, Any]:
        return self._meta

    @property
    def scaler(self) -> np.ndarray:
        """Base-pair values for each pixel position. Shape: ``(1, signal_length)``."""
        if self._scaler is None:
            self._load()
        return self._scaler[np.newaxis, :]

    @property
    def panel(self) -> Panel | None:
        return self._panel

    # -- Data loading ----------------------------------------------------- #

    def _load(self) -> np.ndarray | None:
        """Parse the HID file, validate size standard, build annotation."""
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))

        profile = get_peak_data(self.path, self.data_loading_strategy)
        if profile is None:
            return None

        # Parse size standard and rescale
        scaling = StrategyRegistry.get_scaling_strategy()
        ss_lane = np.array(profile[-1])
        try:
            ss_result = scaling.parse_size_standard(ss_lane)
        except ValueError as e:
            logger.warning("Size standard invalid for {}: {}", self.path.name, e)
            return None

        if ss_result is None:
            logger.warning("Size standard parsing returned None for {}", self.path.name)
            return None

        selected = profile if self.include_size_standard else profile[:-1]
        data = selected[:, ss_result.rescaled_indices][..., np.newaxis]
        self._scaler = ss_result.scaler

        # Build segmentation annotation from called alleles
        if self.annotations_file and self._panel:
            annotations_name = self._meta.get("annotations_name")
            if annotations_name:
                called_alleles = parse_called_alleles(
                    self.annotations_file, self._panel, annotations_name
                )
                if called_alleles and self._annotation is None:
                    mask = self._build_segmentation(
                        self.scaler, called_alleles, data.shape
                    )
                    self._annotation = Annotation(image=mask)
                    self._meta["called_alleles"] = called_alleles

        return data

    # -- Segmentation ----------------------------------------------------- #

    @staticmethod
    def _build_segmentation(
        scaler: np.ndarray,
        called_alleles: list[Marker],
        shape: tuple[int, ...],
    ) -> np.ndarray:
        """Create a binary mask from called alleles using the scaler.

        For each allele, the pixels between its left and right bin edges
        (mapped through the scaler) are set to 1.
        """
        mask = np.zeros(shape, dtype=np.int8)
        for marker in called_alleles:
            for allele in marker.alleles:
                if allele.base_pair is None or allele.left_bin is None:
                    continue
                bin_edges = np.array(
                    [allele.base_pair - allele.left_bin,
                     allele.base_pair + allele.right_bin]
                )[:, np.newaxis]
                pixel_range = np.argmin(np.abs(scaler - bin_edges), axis=1)
                mask[marker.dye_row, pixel_range[0] : pixel_range[1], 0] = 1
        return mask

    # -- Annotation adjustment -------------------------------------------- #

    def adjust_annotations(self, method: str = "top") -> HIDImage:
        """Refine the annotation by snapping to actual peak positions.

        Args:
            method: ``"top"`` to label only the peak apex, or ``"complete"``
                to label the entire peak from boundary to boundary.

        Returns:
            ``self`` (annotations are modified in-place for efficiency).
        """
        if self._annotation is None or self._annotation.image is None:
            logger.warning("No annotations to adjust for {}", self.path)
            return self

        _ = self.data  # ensure data is loaded
        annotations = self._annotation.image
        profile = self._data

        for layer, dye in enumerate(profile):
            ann_indices = np.where(annotations[layer] == 1)[0]
            if ann_indices.size == 0:
                continue

            # Split into contiguous groups
            groups = np.split(ann_indices, np.where(np.diff(ann_indices) != 1)[0] + 1)
            for group in groups:
                annotations[layer, group, 0] = 0.0
                peak_idx = find_peak_idx_near_or_in_range(
                    dye, group, self.rfu_threshold
                )

                if peak_idx.size == 0:
                    logger.warning(
                        "No peak above {}rfu in {} (dye {}, bin {})",
                        self.rfu_threshold, self.path, layer, group,
                    )
                elif method == "complete":
                    start, end = find_peak_boundary(
                        dye, int(peak_idx), self.rfu_threshold
                    )
                    annotations[layer, np.arange(start, end + 1), 0] = 1.0
                elif method == "top":
                    annotations[layer, peak_idx, 0] = 1.0
                else:
                    raise ValueError(f"Unknown adjustment method: {method!r}")

        return self

    # -- Dunder ----------------------------------------------------------- #

    def __repr__(self) -> str:
        return f"HIDImage({self.path.name})"
