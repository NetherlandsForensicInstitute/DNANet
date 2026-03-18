"""Tests for the Panel repository."""

import pytest

from dnanet.core import Panel


class TestPanel:
    """Test suite for Panel."""

    def test_creation_requires_markers(self):
        with pytest.raises(ValueError, match="at least one Marker"):
            Panel(markers=[])

    def test_len(self, sample_panel):
        assert len(sample_panel) == 2

    def test_iter(self, sample_panel):
        names = [m.name for m in sample_panel]
        assert "D3S1358" in names
        assert "AMEL" in names

    def test_get_dye_row_known(self, sample_panel):
        assert sample_panel.get_dye_row("D3S1358") == 0
        assert sample_panel.get_dye_row("AMEL") == 1

    def test_get_dye_row_unknown(self, sample_panel):
        assert sample_panel.get_dye_row("NONEXISTENT") is None

    def test_allele_lookup(self, sample_panel):
        mid, left, right = sample_panel.get_allele_basepair_and_bins("D3S1358", "12")
        assert mid == 120.0
        assert left < mid < right

    def test_allele_lookup_microvariant_fallback(self, sample_panel):
        """Micro-variants (e.g. 12.1) fall back to base allele + offset."""
        mid, left, right = sample_panel.get_allele_basepair_and_bins("D3S1358", "12.1")
        base_mid, _, _ = sample_panel.get_allele_basepair_and_bins("D3S1358", "12")
        assert mid == base_mid + 1

    def test_marker_name_by_dye_and_bp(self, sample_panel):
        name = sample_panel.get_marker_name_by_dye_and_bp(dye_row=0, base_pair=120.0)
        assert name == "D3S1358"

    def test_marker_name_out_of_bin(self, sample_panel):
        name = sample_panel.get_marker_name_by_dye_and_bp(dye_row=0, base_pair=999.0)
        assert name == "Out of Bin"

    def test_dye_bp_to_allele_mapping(self, sample_panel):
        mapping = sample_panel.dye_bp_to_allele_mapping
        assert 0 in mapping  # D3S1358 on dye 0
        assert 1 in mapping  # AMEL on dye 1
        assert mapping[1][107.0] == ("AMEL", "X")

    def test_repr(self, sample_panel):
        assert "2" in repr(sample_panel)
