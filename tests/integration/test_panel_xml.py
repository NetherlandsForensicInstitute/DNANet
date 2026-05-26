"""Integration tests for Panel.from_xml with real panel XML files."""

import pytest

from tests.conftest import PANEL_PATH, RESOURCES_DIR
from dnanet.core.panel import Panel


class TestPanelFromXML:
    """Test Panel parsing against the real PPF6C panel XML."""

    @pytest.fixture
    def panel(self) -> Panel:
        return Panel.from_xml(PANEL_PATH)

    def test_marker_count(self, panel):
        """PPF6C panel should have 27 markers."""
        assert len(panel) == 27

    def test_known_markers_present(self, panel):
        names = {m.name for m in panel.markers}
        expected = {'AMEL', 'D3S1358', 'D1S1656', 'TH01', 'vWA', 'FGA', 'SE33'}
        assert expected.issubset(names)

    def test_amel_alleles(self, panel):
        """AMEL should have X and Y alleles."""
        amel = next(m for m in panel.markers if m.name == 'AMEL')
        allele_names = {a.name for a in amel.alleles}
        assert 'X' in allele_names
        assert 'Y' in allele_names

    def test_d3s1358_dye_row(self, panel):
        """D3S1358 should be on dye row 0 (blue)."""
        assert panel.get_dye_row('D3S1358') == 0

    def test_allele_basepair_lookup(self, panel):
        """AMEL X should be near bp 81.5."""
        bp, left, right = panel.get_allele_basepair_and_bins('AMEL', 'X')
        assert 80.0 < bp < 83.0
        assert left < bp < right

    def test_marker_lut_lookup(self, panel):
        """LUT should resolve (dye=0, bp=81.5) → AMEL."""
        name = panel.get_marker_name_by_dye_and_bp(0, 81.5)
        assert name == 'AMEL'

    def test_dye_bp_to_allele_mapping(self, panel):
        mapping = panel.dye_bp_to_allele_mapping
        # AMEL is on dye 0 in PPF6C
        assert 0 in mapping
        # Check that at least one allele maps to AMEL
        amel_entries = [
            (marker, allele) for bp, (marker, allele) in mapping[0].items() if marker == 'AMEL'
        ]
        assert len(amel_entries) >= 2  # X and Y

    def test_custom_dye_mapping(self):
        """Panel should respect a custom HID dye mapping."""
        # Only map dye 1 → row 0, skip all others
        panel = Panel.from_xml(PANEL_PATH, hid_dye_mapping={1: 0})
        # Should only have markers from dye 1 (blue in PPF6C)
        dye_rows = {m.dye_row for m in panel.markers}
        assert dye_rows == {0}

    def test_globalfiler_panel(self):
        """GlobalFiler panel should also parse correctly."""
        gf_path = (
            RESOURCES_DIR.parents[1] / 'resources' / 'kit_panels' / 'SGPanel_Globalfiler_Panel.xml'
        )
        if not gf_path.exists():
            pytest.skip('GlobalFiler panel XML not available')
        panel = Panel.from_xml(gf_path)
        assert len(panel) > 0
        names = {m.name for m in panel.markers}
        assert 'D3S1358' in names
