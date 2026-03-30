"""Tests for evaluation utility functions."""

from __future__ import annotations

import pytest

from dnanet.core.allele import Allele
from dnanet.core.marker import Marker
from dnanet.evaluation.utils import (
    THRESHOLDS_PER_LOCUS,
    flatten_markers_to_allele_names,
    get_rfu_thresholds,
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
# get_rfu_thresholds
# ---------------------------------------------------------------------------

class TestGetRfuThresholds:
    def test_low_threshold_all_loci(self):
        thresholds = get_rfu_thresholds(low_or_high="low")
        assert thresholds["D5S818"] == 45
        assert thresholds["D8S1179"] == 80
        assert len(thresholds) == len(THRESHOLDS_PER_LOCUS)

    def test_high_threshold_single_locus(self):
        thresholds = get_rfu_thresholds(locus="D8S1179", low_or_high="high")
        assert thresholds == {"D8S1179": 135}

    def test_between_mode(self):
        thresholds = get_rfu_thresholds(locus="D5S818", low_or_high="between")
        assert thresholds == {"D5S818": {"low": 45, "high": 85}}

    def test_manual_threshold(self):
        thresholds = get_rfu_thresholds(min_rfu=100)
        assert all(v == 100 for v in thresholds.values())

    def test_no_threshold(self):
        thresholds = get_rfu_thresholds()
        assert all(v is None for v in thresholds.values())

    def test_unknown_locus_raises(self):
        with pytest.raises(ValueError, match="Unknown locus"):
            get_rfu_thresholds(locus="FAKE_MARKER")

    def test_invalid_low_or_high_raises(self):
        with pytest.raises(ValueError, match="Expected 'low'"):
            get_rfu_thresholds(low_or_high="wrong")


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

    def test_filter_by_min_rfu(self):
        markers = [
            _marker("D5S818", 0, [("13", 500), ("15", 30)]),
        ]
        result = flatten_markers_to_allele_names(markers, min_rfu=100)
        assert result == frozenset({"D5S818_13"})

    def test_filter_by_low_threshold(self):
        markers = [
            _marker("D5S818", 0, [("13", 500), ("15", 30)]),
        ]
        # D5S818 low threshold is 45, so "15" (height=30) gets filtered
        result = flatten_markers_to_allele_names(markers, low_or_high="low")
        assert result == frozenset({"D5S818_13"})

    def test_between_filter(self):
        markers = [
            _marker("D5S818", 0, [
                ("13", 500),   # above high (85) -> excluded
                ("15", 60),    # between 45 and 85 -> included
                ("16", 30),    # below low (45) -> excluded
            ]),
        ]
        result = flatten_markers_to_allele_names(markers, low_or_high="between")
        assert result == frozenset({"D5S818_15"})

    def test_empty_markers(self):
        result = flatten_markers_to_allele_names([])
        assert result == frozenset()

    def test_no_height_with_threshold_raises(self):
        markers = [
            Marker(name="D5S818", dye_row=0, alleles=frozenset([Allele(name="13"),])),
        ]
        with pytest.raises(ValueError, match="has no height"):
            flatten_markers_to_allele_names(markers, min_rfu=100)

    def test_returns_frozenset(self):
        markers = [_marker("D5S818", 0, [("13", 500)])]
        result = flatten_markers_to_allele_names(markers)
        assert isinstance(result, frozenset)
