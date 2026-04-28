"""Stubs that satisfy ``MemmapCacheWriter.write``'s minimal contract.

The writer reads only ``image.path``, ``image.data``, ``image.scaler``,
``image.annotation.data`` (optional), ``image.allele_annotation.data`` (each
element exposing ``.to_dict()``), ``image.adjusted_panel.markers`` (each
exposing ``.to_dict()``), and ``image.meta``. Real ``HIDImage`` construction
goes through file I/O and panel resolution; bypassing it keeps these tests
fast and scoped to the cache layer.
"""

from __future__ import annotations

from typing import Any
from dataclasses import field, dataclass

import numpy as np
import pytest


@dataclass
class _Marker:
    name: str

    def to_dict(self) -> dict[str, Any]:
        return {'name': self.name, 'dye_row': 0, 'alleles': []}


@dataclass
class _Annotation:
    data: np.ndarray


@dataclass
class _AlleleAnnotation:
    data: list[_Marker]


@dataclass
class _Panel:
    markers: list[_Marker]


@dataclass
class _StubImage:
    path: str
    data: np.ndarray
    scaler: np.ndarray
    annotation: _Annotation | None = None
    allele_annotation: _AlleleAnnotation | None = None
    adjusted_panel: _Panel | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _make_image(
    path: str,
    *,
    shape: tuple[int, int] = (6, 16),
    annotated: bool = True,
    panel_id: str | None = 'A',
    allele_id: str | None = 'X',
    fill: int = 1,
) -> _StubImage:
    """Build a writer-compatible stub image with deterministic content."""
    data = np.full(shape, fill, dtype=np.int16)
    scaler = np.linspace(60, 480, shape[1], dtype=np.float32)
    annotation = _Annotation(np.full(shape, fill % 2, dtype=np.int8)) if annotated else None
    allele_annotation = (
        _AlleleAnnotation([_Marker(f'allele-{allele_id}')]) if allele_id is not None else None
    )
    adjusted_panel = (
        _Panel([_Marker(f'panel-{panel_id}-{i}') for i in range(3)])
        if panel_id is not None
        else None
    )
    return _StubImage(
        path=path,
        data=data,
        scaler=scaler,
        annotation=annotation,
        allele_annotation=allele_annotation,
        adjusted_panel=adjusted_panel,
    )


@pytest.fixture
def make_image():
    """Factory fixture so each test can specialize ``_make_image`` kwargs."""
    return _make_image
