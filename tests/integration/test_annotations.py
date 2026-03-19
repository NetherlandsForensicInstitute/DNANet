"""Integration tests for annotation parsing with real AlleleReport files."""

import pytest

from dnanet.core.panel import Panel
from dnanet.data.parsing.annotations import parse_called_alleles

from tests.conftest import PANEL_PATH, RD_DIR


class TestParseCalledAlleles:
    """Test annotation parsing against the real AlleleReport TXT files.

    These tests verify backward compatibility with the original DNANet
    annotation parser output.
    """

    @pytest.fixture
    def panel(self) -> Panel:
        return Panel.from_xml(PANEL_PATH)

    @pytest.fixture
    def allele_report(self) -> str:
        return str(RD_DIR / "Dataset 1 DTH_AlleleReport.txt")

    def test_parse_sample_1a2(self, panel, allele_report):
        """Parse sample '1_11148_1A2' — the primary test sample."""
        markers = parse_called_alleles(allele_report, panel, "1_11148_1A2")
        assert markers is not None
        assert len(markers) == 27  # matches original test

    def test_amel_alleles(self, panel, allele_report):
        """AMEL should have X and Y alleles with correct heights."""
        markers = parse_called_alleles(allele_report, panel, "1_11148_1A2")
        amel = markers[0]
        assert amel.name == "AMEL"
        assert amel.dye_row == 0
        assert len(amel.alleles) == 2

        x_allele = next(a for a in amel.alleles if a.name == "X")
        y_allele = next(a for a in amel.alleles if a.name == "Y")
        assert x_allele.height == 11148.0
        assert y_allele.height == 10495.0

    def test_d3s1358_alleles(self, panel, allele_report):
        """D3S1358 should have alleles 14, 15, 17."""
        markers = parse_called_alleles(allele_report, panel, "1_11148_1A2")
        d3 = next(m for m in markers if m.name == "D3S1358")
        allele_names = {a.name for a in d3.alleles}
        assert allele_names == {"14", "15", "17"}

    def test_d3s1358_heights(self, panel, allele_report):
        """D3S1358 allele heights should match reference."""
        markers = parse_called_alleles(allele_report, panel, "1_11148_1A2")
        d3 = next(m for m in markers if m.name == "D3S1358")
        height_map = {a.name: a.height for a in d3.alleles}
        assert height_map["14"] == 3950.0
        assert height_map["15"] == 8780.0
        assert height_map["17"] == 6486.0

    def test_alleles_have_basepair_info(self, panel, allele_report):
        """Each allele should have base_pair and bin edges populated."""
        markers = parse_called_alleles(allele_report, panel, "1_11148_1A2")
        for marker in markers:
            for allele in marker.alleles:
                assert allele.base_pair is not None, f"{marker.name}/{allele.name} missing bp"
                assert allele.left_bin is not None
                assert allele.right_bin is not None

    def test_unknown_sample_returns_none(self, panel, allele_report):
        """Requesting a non-existent sample should return None."""
        result = parse_called_alleles(allele_report, panel, "NONEXISTENT_SAMPLE")
        assert result is None

    def test_d1s1656_microvariant(self, panel, allele_report):
        """D1S1656 should parse micro-variants like 15.3 and 18.3."""
        markers = parse_called_alleles(allele_report, panel, "1_11148_1A2")
        d1 = next(m for m in markers if m.name == "D1S1656")
        allele_names = {a.name for a in d1.alleles}
        assert "15.3" in allele_names
        assert "18.3" in allele_names

    def test_second_sample(self, panel):
        """Parse the second sample '3_10196_1A2'."""
        report = str(RD_DIR / "Dataset 1 DTH_AlleleReport.txt")
        markers = parse_called_alleles(report, panel, "3_10196_1A2")
        assert markers is not None
        assert len(markers) > 0

    def test_markers_have_correct_dye_rows(self, panel, allele_report):
        """Dye rows from annotation parsing should match the panel."""
        markers = parse_called_alleles(allele_report, panel, "1_11148_1A2")
        for marker in markers:
            expected_dye = panel.get_dye_row(marker.name)
            assert marker.dye_row == expected_dye, (
                f"{marker.name}: expected dye {expected_dye}, got {marker.dye_row}"
            )
