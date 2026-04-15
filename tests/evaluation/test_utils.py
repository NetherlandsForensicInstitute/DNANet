"""Tests for evaluation utility functions."""

from __future__ import annotations

import pytest

from dnanet.core.allele import Allele
from dnanet.core.marker import Marker
from dnanet.evaluation.utils import (
    flatten_markers_to_allele_names,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _marker(name: str, dye: int, alleles: list[tuple[str, int | None]]) -> Marker:
    return Marker(
        name=name,
        dye_row=dye,
        alleles=frozenset(Allele(name=n, height=h) for n, h in alleles),
    )


# ---------------------------------------------------------------------------
# flatten_markers_to_allele_names
# ---------------------------------------------------------------------------

class TestFlattenMarkers:
    def test_basic_flatten(self):
        markers = [
            _marker("D5S818", 0, [("13", 500), ("15", 300)]),
            _marker("vWA", 1, [("16", 200)]),
        ]
        result = flatten_markers_to_allele_names(markers)
        assert result == frozenset({"D5S818_13", "D5S818_15", "vWA_16"})

    def test_filter_by_locus(self):
        markers = [
            _marker("D5S818", 0, [("13", 500)]),
            _marker("vWA", 1, [("16", 200)]),
        ]
        result = flatten_markers_to_allele_names(markers, locus="D5S818")
        assert result == frozenset({"D5S818_13"})

    def test_empty_markers(self):
        result = flatten_markers_to_allele_names([])
        assert result == frozenset()


    def test_returns_frozenset(self):
        markers = [_marker("D5S818", 0, [("13", 500)])]
        result = flatten_markers_to_allele_names(markers)
        assert isinstance(result, frozenset)
