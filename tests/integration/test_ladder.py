"""Integration tests for Ladder with real ladder HID + ladder_alleles.csv."""

import numpy as np
import pytest

from tests.conftest import RD_DIR, PANEL_PATH, LADDER_ALLELES_CSV
from dnanet.core.panel import Panel
from dnanet.data.parsing.hid import get_peak_data
from dnanet.data.ladders.ladder import Ladder
from dnanet.data.strategies.registry import StrategyRegistry
from dnanet.data.ladders.ladder_allele_catalog import LadderAlleleCatalog


class TestLadderAlleleCatalogCSV:
    """Test loading the real ladder_alleles.csv."""

    @pytest.fixture
    def catalog(self) -> LadderAlleleCatalog:
        return LadderAlleleCatalog.from_csv(LADDER_ALLELES_CSV)

    def test_loads_successfully(self, catalog):
        assert len(catalog.alleles_by_dye) > 0

    def test_has_five_dyes(self, catalog):
        """PPF6C ladder should have alleles on dyes 0-4."""
        assert set(catalog.alleles_by_dye.keys()) == {0, 1, 2, 3, 4}

    def test_dye0_has_amel(self, catalog):
        """Dye 0 should contain AMEL X and Y."""
        dye0_markers = [name for name, _ in catalog.alleles_by_dye[0]]
        assert 'AMEL' in dye0_markers

    def test_expected_counts_per_dye(self, catalog):
        """Should have the exact peak counts expected by the original tests."""
        expected = {0: 89, 1: 80, 2: 89, 3: 99, 4: 76}
        for dye, count in expected.items():
            actual = catalog.expected_count(dye)
            assert actual == count, f'Dye {dye}: expected {count}, got {actual}'


class TestLadderWithRealData:
    """Integration test: build a Ladder from a real ladder HID file.

    This requires the PPF6C kit to be configured (for size-standard parsing).
    """

    @pytest.fixture
    def ppf6c(self):
        StrategyRegistry.configure_kit('PPF6C')
        yield
        StrategyRegistry.reset()

    @pytest.fixture
    def catalog(self) -> LadderAlleleCatalog:
        return LadderAlleleCatalog.from_csv(LADDER_ALLELES_CSV)

    @pytest.fixture
    def default_panel(self) -> Panel:
        return Panel.from_xml(PANEL_PATH)

    def test_ladder_from_real_hid(self, ppf6c, catalog, default_panel):
        """Build a Ladder from Ladder_G03_21.hid and verify peak counts."""
        scaling = StrategyRegistry.get_scaling_strategy()

        # Load the ladder HID data
        raw_data = get_peak_data(RD_DIR / 'Ladder_G03_21.hid', strategy='raw')
        assert raw_data is not None

        # Parse size standard to get scaler and rescaled indices
        ss_lane = np.array(raw_data[-1])
        ss_result = scaling.parse_size_standard(ss_lane)
        assert ss_result is not None

        # Rescale all channels
        data = raw_data[:, ss_result.rescaled_indices][..., np.newaxis]
        scaler = ss_result.scaler

        # Build the ladder
        ladder = Ladder.from_hid_data(
            path=RD_DIR / 'Ladder_G03_21.hid',
            data=data,
            scaler=scaler,
            catalog=catalog,
            default_panel=default_panel,
            num_dyes=5,
        )
        assert ladder is not None, 'Ladder should have been built successfully'
        assert ladder.adjusted_panel is not None

        # Verify peak counts per dye match expected values from original tests
        expected_peaks = [89, 80, 89, 99, 76]
        for dye, (peak_idxs, expected) in enumerate(
            zip(ladder.peak_indices, expected_peaks, strict=True)
        ):
            assert len(peak_idxs) == expected, (
                f'Dye {dye}: expected {expected} peaks, got {len(peak_idxs)}'
            )

    def test_adjusted_panel_amel(self, ppf6c, catalog, default_panel):
        """Adjusted panel should have AMEL with calibrated bp values."""
        scaling = StrategyRegistry.get_scaling_strategy()
        raw_data = get_peak_data(RD_DIR / 'Ladder_G03_21.hid', strategy='raw')
        ss_result = scaling.parse_size_standard(np.array(raw_data[-1]))
        data = raw_data[:, ss_result.rescaled_indices][..., np.newaxis]

        ladder = Ladder.from_hid_data(
            path=RD_DIR / 'Ladder_G03_21.hid',
            data=data,
            scaler=ss_result.scaler,
            catalog=catalog,
            default_panel=default_panel,
            num_dyes=5,
        )
        assert ladder is not None

        # Find AMEL in adjusted panel
        amel = next((m for m in ladder.adjusted_panel.markers if m.name == 'AMEL'), None)
        assert amel is not None
        assert len(amel.alleles) >= 2  # X and Y

        # The adjusted bp should be close to but not identical to default
        default_amel = next(m for m in default_panel.markers if m.name == 'AMEL')
        x_default = next(a for a in default_amel.alleles if a.name == 'X')
        x_adjusted = next(a for a in amel.alleles if a.name == 'X')

        # Should be within a few bp of the default
        assert abs(x_adjusted.base_pair - x_default.base_pair) < 5.0

    def test_adjusted_panel_preserves_allele_count(self, ppf6c, catalog, default_panel):
        """Each marker in adjusted panel should have same allele count as default."""
        scaling = StrategyRegistry.get_scaling_strategy()
        raw_data = get_peak_data(RD_DIR / 'Ladder_G03_21.hid', strategy='raw')
        ss_result = scaling.parse_size_standard(np.array(raw_data[-1]))
        data = raw_data[:, ss_result.rescaled_indices][..., np.newaxis]

        ladder = Ladder.from_hid_data(
            path=RD_DIR / 'Ladder_G03_21.hid',
            data=data,
            scaler=ss_result.scaler,
            catalog=catalog,
            default_panel=default_panel,
            num_dyes=5,
        )
        assert ladder is not None

        for m_default in default_panel.markers:
            m_adjusted = next(
                (m for m in ladder.adjusted_panel.markers if m.name == m_default.name),
                None,
            )
            if m_adjusted is not None:
                assert len(m_adjusted.alleles) == len(m_default.alleles), (
                    f'{m_default.name}: allele count changed from '
                    f'{len(m_default.alleles)} to {len(m_adjusted.alleles)}'
                )
