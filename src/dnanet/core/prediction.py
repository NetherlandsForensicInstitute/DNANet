"""Prediction domain model.

Design pattern: **Value Object** (with factory methods)
    A Prediction is the output of a model — structurally similar to an
    Annotation but representing model output rather than ground truth.

    Key simplifications from the original:
    1. Frozen dataclass for immutability.
    2. Factory methods (`for_segmentation`, `for_classification`) make it
       clear what kind of prediction you're constructing.
    3. Serialization kept for JSON I/O (needed for result persistence).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dnanet.core.marker import Marker
from dnanet.core.types import PathLike


@dataclass(frozen=True, slots=True)
class Prediction:
    """A single model prediction for a DNA profile.

    Attributes:
        classification: Label-to-confidence mapping (for classification tasks).
        image: Predicted pixel mask (for segmentation/reconstruction tasks).
        original_image_path: Path to the source HID file.
        called_alleles: Post-processed allele calls derived from the prediction.
    """

    classification: Mapping[str, float] | None = None
    image: np.ndarray | None = None
    original_image_path: Path | None = None
    called_alleles: tuple[Marker, ...] | None = None

    # -- Factory methods -------------------------------------------------- #

    @classmethod
    def for_segmentation(
        cls,
        image: np.ndarray,
        source_path: PathLike,
        called_alleles: Sequence[Marker] | None = None,
    ) -> Prediction:
        """Create a segmentation prediction."""
        return cls(
            image=image,
            original_image_path=Path(source_path),
            called_alleles=tuple(called_alleles) if called_alleles else None,
        )

    @classmethod
    def for_classification(
        cls,
        scores: Mapping[str, float],
        source_path: PathLike,
    ) -> Prediction:
        """Create a classification prediction."""
        return cls(
            classification=scores,
            original_image_path=Path(source_path),
        )

    # -- Query ------------------------------------------------------------- #

    @property
    def label(self) -> str:
        """The label with the highest confidence score."""
        if not self.classification:
            raise ValueError("No classification scores available")
        return max(self.classification, key=self.classification.get)

    @property
    def confidence(self) -> float:
        """The confidence of the top-predicted label."""
        return self.classification[self.label]

    # -- Serialization ----------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "classification": dict(self.classification) if self.classification else None,
            "image": self.image.tolist() if self.image is not None else None,
            "original_image_path": str(self.original_image_path) if self.original_image_path else None,
            "called_alleles": (
                [m.to_dict() for m in self.called_alleles]
                if self.called_alleles
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Prediction:
        """Deserialize from a plain dictionary."""
        return cls(
            classification=data.get("classification"),
            image=np.array(data["image"]) if data.get("image") is not None else None,
            original_image_path=(
                Path(data["original_image_path"])
                if data.get("original_image_path")
                else None
            ),
            called_alleles=(
                tuple(Marker.from_dict(m) for m in data["called_alleles"])
                if data.get("called_alleles")
                else None
            ),
        )

    # -- Dunder ------------------------------------------------------------ #

    def __hash__(self) -> int:
        return hash((
            tuple(self.classification.items()) if self.classification else (),
            self.image.tobytes() if self.image is not None else None,
        ))
