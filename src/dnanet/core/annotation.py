"""Annotation domain model.

The previous single ``Annotation`` value object has been split into three
task-specific value objects:

- ``ScanpointAnnotation`` for segmentation masks
- ``AlleleAnnotation`` for called-marker annotations
- ``ClassAnnotation`` for categorical labels
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np


if TYPE_CHECKING:
    from dnanet.core.marker import Marker


@dataclass(frozen=True, slots=True)
class ScanpointAnnotation:
    data: np.ndarray

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScanpointAnnotation):
            return NotImplemented
        return np.array_equal(self.data, other.data)


@dataclass(frozen=True, slots=True)
class AlleleAnnotation:
    data: list[Marker]

    def __add__(self, other: AlleleAnnotation) -> AlleleAnnotation:
        return _merge_allele_annotations(self, other)


@dataclass(frozen=True, slots=True)
class ClassAnnotation:
    data: str


Annotation = ScanpointAnnotation | AlleleAnnotation

# TODO: Test logic here
def _merge_allele_annotations(ann1: AlleleAnnotation, ann2: AlleleAnnotation) -> AlleleAnnotation:
    """Merges two allele annotations by combining markers with matching names.

    For each unique marker name across both annotations:
      - If the name exists in both, the markers are combined using the ``+`` operator.
      - If the name exists in only one, that marker is taken as-is.

    Args:
        ann1 (AlleleAnnotation): The first allele annotation containing a list of markers.
        ann2 (AlleleAnnotation): The second allele annotation containing a list of markers.

    Returns:
        AlleleAnnotation: A new allele annotation whose markers are the union of both inputs,
        with matching markers merged. Order is not guaranteed.
    """
    ann1_by_name = {marker.name: marker for marker in ann1.data}
    ann2_by_name = {marker.name: marker for marker in ann2.data}

    merged = {
        name: ann1_by_name[name] + ann2_by_name[name]
        if name in ann1_by_name and name in ann2_by_name
        else ann1_by_name.get(name) or ann2_by_name.get(name)
        for name in ann1_by_name.keys() | ann2_by_name.keys()
    }

    return AlleleAnnotation(data=list(merged.values()))  # type: ignore
