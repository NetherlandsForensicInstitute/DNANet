"""Annotation domain model.

The previous single ``Annotation`` value object has been split into three
task-specific value objects:

- ``ScanpointAnnotation`` for segmentation masks
- ``AlleleAnnotation`` for called-marker annotations
- ``ClassAnnotation`` for categorical labels
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

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


def _merge_allele_annotations(ann1: AlleleAnnotation, ann2: AlleleAnnotation) -> AlleleAnnotation:
    """
    Merges two allele annotations by combining markers with matching names from both annotations.

    Arguments:
        ann1 (AlleleAnnotation): The first allele annotation containing a list of markers.
        ann2 (AlleleAnnotation): The second allele annotation containing a list of markers.

    Returns:
        AlleleAnnotation: A new allele annotation containing combined markers with matching names
        from both input annotations.
    """
    ann2_by_name = {marker.name: marker for marker in ann2.data}
    out_ann = []

    for marker1 in ann1.data:
        marker2 = ann2_by_name.get(marker1.name)
        if marker2 is not None:
            new_marker = marker1 + marker2
            out_ann.append(new_marker)

    return AlleleAnnotation(data=out_ann)
