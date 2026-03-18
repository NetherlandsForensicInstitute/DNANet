"""Annotation domain model.

Design pattern: **Value Object** (again)
    An Annotation represents ground-truth labeling of a DNA profile image.
    It can hold:
    - A set of categorical labels (for classification tasks)
    - A pixel-level mask image (for segmentation tasks)
    - Arbitrary metadata

    Compared to the original:
    - Made into a frozen dataclass for consistency with Allele/Marker.
    - `labels` is a `frozenset` (not a mutable `set`) for true immutability.
    - `meta` is kept as a regular dict (frozen only protects the reference,
      not the dict contents — this is a pragmatic trade-off).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class Annotation:
    """Ground-truth annotation for a DNA profile.

    Attributes:
        labels: Categorical labels (e.g. for classification tasks).
            Stored as a frozenset for immutability.
        image: Pixel-level annotation mask (e.g. for segmentation).
            Shape typically: (num_dyes, signal_length).
        meta: Additional metadata about this annotation (annotator ID,
            confidence, etc.).
    """

    labels: frozenset[str] = field(default_factory=frozenset)
    image: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    # -- Convenience ------------------------------------------------------- #

    @property
    def label(self) -> str:
        """The single label, if exactly one is present.

        Raises:
            ValueError: If there are zero or multiple labels.
        """
        if not self.labels:
            raise ValueError("No labels in annotation")
        if len(self.labels) > 1:
            raise ValueError(f"Multiple labels in annotation: {self.labels}")
        return next(iter(self.labels))

    # -- Construction helpers ---------------------------------------------- #

    @classmethod
    def from_labels(cls, labels: str | Iterable[str], **meta: Any) -> Annotation:
        """Create an annotation from one or more labels.

        Args:
            labels: A single label string or an iterable of label strings.
            **meta: Additional metadata key-value pairs.
        """
        if isinstance(labels, str):
            labels = frozenset({labels})
        else:
            labels = frozenset(labels)
        return cls(labels=labels, meta=meta)

    @classmethod
    def from_mask(cls, image: np.ndarray, **meta: Any) -> Annotation:
        """Create an annotation from a segmentation mask."""
        return cls(image=image, meta=meta)

    # -- Equality (numpy-aware) -------------------------------------------- #

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Annotation):
            return NotImplemented
        return (
            self.labels == other.labels
            and np.array_equal(self.image, other.image)
            and self.meta == other.meta
        )

    def __hash__(self) -> int:
        return hash((
            self.labels,
            self.image.tobytes() if self.image is not None else None,
        ))
