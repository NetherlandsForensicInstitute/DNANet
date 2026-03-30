"""Integration tests for annotation parsing with real AlleleReport files."""

import pytest

from dnanet.core.panel import Panel

from dnanet.data.strategies.registry import StrategyRegistry
from tests.conftest import PANEL_PATH, RD_DIR


class TestParseCalledAlleles:
    """Test annotation parsing against the real AlleleReport TXT files.

    These tests verify backward compatibility with the original DNANet
    annotation parser output.
    """
    
    @pytest.fixture
    def nfi_rnd_kit(self):
        """Configure PPF6C kit + NFI_RND dataset strategy."""
        StrategyRegistry.configure_kit("PPF6C")
        StrategyRegistry.configure_dataset("NFI_RND")
        yield
        StrategyRegistry.reset()

    @pytest.fixture
    def panel(self) -> Panel:
        return Panel.from_xml(PANEL_PATH)

    @pytest.fixture
    def allele_report(self) -> str:
        return str(RD_DIR / "Dataset 1 DTH_AlleleReport.txt")

    def test_parse_sample_1a2(self, nfi_rnd_kit, allele_report):
        """Parse sample '1_11148_1A2' — the primary test sample."""
        markers = StrategyRegistry.get_dataset_strategy().create_annotation_to_markers(allele_report, "1_11148_1A2")
        assert markers is not None
        assert len(markers) == 27  # matches original test

    def test_amel_alleles(self, nfi_rnd_kit, allele_report):
        """AMEL should have X and Y alleles with correct heights."""
        dataset_strat = StrategyRegistry.get_dataset_strategy()
        markers = dataset_strat.create_annotation_to_markers(allele_report, "1_11148_1A2")
        amel = markers[0]
        assert amel.name == "AMEL"
        assert amel.dye_row == 0
        assert len(amel.alleles) == 2

        x_allele = next(a for a in amel.alleles if a.name == "X")
        y_allele = next(a for a in amel.alleles if a.name == "Y")
        assert x_allele.height == 11148.0
        assert y_allele.height == 10495.0

    def test_d3s1358_alleles(self, nfi_rnd_kit, allele_report):
        """D3S1358 should have alleles 14, 15, 17."""
        dataset_strat = StrategyRegistry.get_dataset_strategy()
        markers = dataset_strat.create_annotation_to_markers(allele_report, "1_11148_1A2")
        d3 = next(m for m in markers if m.name == "D3S1358")
        allele_names = {a.name for a in d3.alleles}
        assert allele_names == {"14", "15", "17"}

    def test_d3s1358_heights(self, nfi_rnd_kit, allele_report):
        """D3S1358 allele heights should match reference."""
        dataset_strat = StrategyRegistry.get_dataset_strategy()
        markers = dataset_strat.create_annotation_to_markers(allele_report, "1_11148_1A2")
        d3 = next(m for m in markers if m.name == "D3S1358")
        height_map = {a.name: a.height for a in d3.alleles}
        assert height_map["14"] == 3950.0
        assert height_map["15"] == 8780.0
        assert height_map["17"] == 6486.0

    def test_unknown_sample_raises_error(self, nfi_rnd_kit, allele_report):
        """Requesting a non-existent sample should return None."""
        dataset_strat = StrategyRegistry.get_dataset_strategy()
        with pytest.raises(RuntimeError):
            dataset_strat.create_annotation_to_markers(allele_report, "NONEXISTENT_SAMPLE")

    def test_d1s1656_microvariant(self, nfi_rnd_kit, allele_report):
        """D1S1656 should parse micro-variants like 15.3 and 18.3."""
        dataset_strat = StrategyRegistry.get_dataset_strategy()
        markers = dataset_strat.create_annotation_to_markers(allele_report, "1_11148_1A2")
        d1 = next(m for m in markers if m.name == "D1S1656")
        allele_names = {a.name for a in d1.alleles}
        assert "15.3" in allele_names
        assert "18.3" in allele_names

    def test_second_sample(self, nfi_rnd_kit):
        """Parse the second sample '3_10196_1A2'."""
        report = str(RD_DIR / "Dataset 1 DTH_AlleleReport.txt")
        dataset_strat = StrategyRegistry.get_dataset_strategy()
        markers = dataset_strat.create_annotation_to_markers(report, "3_10196_1A2")
        assert markers is not None
        assert len(markers) > 0

    def test_markers_have_correct_dye_rows(self, nfi_rnd_kit, panel, allele_report):
        """Dye rows from annotation parsing should match the panel."""
        dataset_strat = StrategyRegistry.get_dataset_strategy()
        markers = dataset_strat.create_annotation_to_markers(allele_report, "1_11148_1A2")
        for marker in markers:
            expected_dye = panel.get_dye_row(marker.name)
            assert marker.dye_row == expected_dye, (
                f"{marker.name}: expected dye {expected_dye}, got {marker.dye_row}"
            )
