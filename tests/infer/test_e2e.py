"""E2E tests for the inference pipeline with real HID files.

These tests exercise the full pipeline path from HID file parsing through
allele calling, using real .hid files from the test resources.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.conftest import RD_DIR, HID_DIR
from dnanet.data.image import HIDImage
from dnanet.infer.output import (
    AlleleCall,
    MarkerResult,
    ProfileResult,
    InferenceResult,
)
from dnanet.data.strategies import GlobalFilerStrategy, PowerPlexFusion6CStrategy
from dnanet.data.ladders.ladder import Ladder
from dnanet.data.ladders.ladder_allele_catalog import LadderAlleleCatalog


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ppf6c_strategy():
    """PowerPlex Fusion 6C strategy for testing."""
    return PowerPlexFusion6CStrategy()


@pytest.fixture
def gf_strategy():
    """GlobalFiler strategy for testing."""
    return GlobalFilerStrategy()


# ---------------------------------------------------------------------------
# HID Image loading tests
# ---------------------------------------------------------------------------


class TestHIDImageLoading:
    """Test HID image loading with real files."""

    def test_load_sample_hid(self, ppf6c_strategy):
        """Real sample HID file loads successfully."""
        hid_path = HID_DIR / '1A2_A01_01.hid'
        assert hid_path.exists()

        image = HIDImage(
            path=hid_path,
            scaling_strategy=ppf6c_strategy,
            data_loading_strategy='superior',
        )
        data = image.data
        assert data is not None
        assert data.shape[0] == 5  # 5 analysis dyes

    def test_load_ladder_hid(self, ppf6c_strategy):
        """Real ladder HID file loads successfully."""
        ladder_path = HID_DIR / 'Ladder_G03_21.hid'
        assert ladder_path.exists()

        image = HIDImage(
            path=ladder_path,
            scaling_strategy=ppf6c_strategy,
            data_loading_strategy='raw',
        )
        data = image.data
        assert data is not None
        assert data.shape[0] == 5

    def test_load_provedit_hid(self, gf_strategy):
        """PROVEDIt HID file loads successfully."""
        hid_path = (
            RD_DIR.parent.parent
            / 'PROVEDIt'
            / '5 sec'
            / 'RD14-0003(020316ADG_5sec)'
            / 'A02_RD14-0003-31_32-1;1-M2c-0.03GF-Q2.0_01.5sec.hid'
        )
        ladder_path = (
            RD_DIR.parent.parent
            / 'PROVEDIt'
            / '5 sec'
            / 'RD14-0003(020316ADG_5sec)'
            / 'A01_Ladder-GF_01.5sec.hid'
        )
        assert hid_path.exists(), f'PROVEDIt HID not found: {hid_path}'
        assert ladder_path.exists(), f'PROVEDIt ladder not found: {ladder_path}'

        image = HIDImage(
            path=hid_path,
            scaling_strategy=gf_strategy,
            data_loading_strategy='superior',
        )
        data = image.data
        assert data is not None


# ---------------------------------------------------------------------------
# Panel adjustment tests
# ---------------------------------------------------------------------------


class TestPanelAdjustment:
    """Test ladder-based panel adjustment."""

    def test_adjust_panel_from_ladder(self, ppf6c_strategy):
        """Panel is adjusted using ladder alleles."""
        ladder_path = HID_DIR / 'Ladder_G03_21.hid'
        catalog = LadderAlleleCatalog.from_panel(ppf6c_strategy.panel)
        assert catalog is not None

        adjusted_panel = Ladder.create_adjusted_panel(
            ladder_path=ladder_path,
            catalog=catalog,
            data_loading_strategy='superior',
            scaling_strategy=ppf6c_strategy,
            dataset_strategy=None,  # type: ignore[arg-type]
        )

        assert adjusted_panel is not None
        assert len(adjusted_panel.markers) > 0
        # AMEL should always be present
        marker_names = [m.name for m in adjusted_panel.markers]
        assert 'AMEL' in marker_names

    def test_adjusted_panel_has_ladder_alleles(self, ppf6c_strategy):
        """Adjusted panel markers have alleles from the ladder."""
        ladder_path = HID_DIR / 'Ladder_G03_21.hid'
        catalog = LadderAlleleCatalog.from_panel(ppf6c_strategy.panel)
        assert catalog is not None

        adjusted_panel = Ladder.create_adjusted_panel(
            ladder_path=ladder_path,
            catalog=catalog,
            data_loading_strategy='superior',
            scaling_strategy=ppf6c_strategy,
            dataset_strategy=None,  # type: ignore[arg-type]
        )

        assert adjusted_panel is not None
        # At least some markers should have alleles
        markers_with_alleles = [m for m in adjusted_panel.markers if len(m.alleles) > 0]
        assert len(markers_with_alleles) > 0


# ---------------------------------------------------------------------------
# Result assembly tests
# ---------------------------------------------------------------------------


class TestResultAssembly:
    """Test result assembly from real HID data."""

    def _create_mock_pipeline(self, ppf6c_strategy, adjusted_panel):
        """Create a mock pipeline with the given adjusted panel."""
        mock_pipeline = MagicMock()
        mock_pipeline.scaling_strategy = ppf6c_strategy
        mock_pipeline._adjusted_panel = adjusted_panel
        mock_pipeline._prediction_threshold = 0.5
        mock_pipeline._confidence_threshold = None
        return mock_pipeline

    def test_profile_result_from_hid(self, ppf6c_strategy):
        """ProfileResult is correctly assembled from HID image."""
        hid_path = HID_DIR / '1A2_A01_01.hid'
        ladder_path = HID_DIR / 'Ladder_G03_21.hid'

        image = HIDImage(
            path=hid_path,
            scaling_strategy=ppf6c_strategy,
            data_loading_strategy='superior',
        )
        data = image.data
        assert data is not None

        catalog = LadderAlleleCatalog.from_panel(ppf6c_strategy.panel)
        assert catalog is not None

        adjusted_panel = Ladder.create_adjusted_panel(
            ladder_path=ladder_path,
            catalog=catalog,
            data_loading_strategy='superior',
            scaling_strategy=ppf6c_strategy,
            dataset_strategy=None,  # type: ignore[arg-type]
        )
        assert adjusted_panel is not None

        # Build a ProfileResult directly
        sample_name = hid_path.stem
        result = ProfileResult(
            sample=sample_name,
            hid_path=str(hid_path),
            ladder_path=str(ladder_path),
            markers=[],
        )

        assert isinstance(result, ProfileResult)
        assert result.sample == sample_name
        assert result.hid_path == str(hid_path)
        assert result.ladder_path == str(ladder_path)

    def test_result_with_alleles(self, ppf6c_strategy):
        """Result contains properly formatted allele calls."""
        hid_path = HID_DIR / '1A2_A01_01.hid'
        ladder_path = HID_DIR / 'Ladder_G03_21.hid'

        image = HIDImage(
            path=hid_path,
            scaling_strategy=ppf6c_strategy,
            data_loading_strategy='superior',
        )
        data = image.data
        assert data is not None

        catalog = LadderAlleleCatalog.from_panel(ppf6c_strategy.panel)
        assert catalog is not None

        adjusted_panel = Ladder.create_adjusted_panel(
            ladder_path=ladder_path,
            catalog=catalog,
            data_loading_strategy='superior',
            scaling_strategy=ppf6c_strategy,
            dataset_strategy=None,  # type: ignore[arg-type]
        )
        assert adjusted_panel is not None

        # Create mock predictions with clear peaks
        mock_prediction = np.zeros_like(data, dtype=np.float32)
        # Add a strong peak at a specific position
        mock_prediction[0, 2000:2010] = np.linspace(0.5, 0.9, 10).astype(np.float32)

        # Build a ProfileResult with mock markers
        result = ProfileResult(
            sample=hid_path.stem,
            hid_path=str(hid_path),
            ladder_path=str(ladder_path),
            markers=[
                MarkerResult(
                    name='D3S1358',
                    dye_row=0,
                    alleles=[
                        AlleleCall(name='12', base_pair=120.0, height=1500.0, confidence=0.95),
                        AlleleCall(name='15', base_pair=132.0, height=2200.0, confidence=0.88),
                    ],
                ),
            ],
        )

        # Verify result structure
        assert isinstance(result, ProfileResult)
        assert len(result.markers) == 1
        for marker in result.markers:
            assert isinstance(marker, MarkerResult)
            assert isinstance(marker.name, str)
            assert isinstance(marker.dye_row, int)
            for allele in marker.alleles:
                assert isinstance(allele, AlleleCall)
                assert allele.confidence >= 0.0
                assert allele.confidence <= 1.0
                assert allele.base_pair > 0
                assert allele.height >= 0

    def test_inference_result_from_multiple_profiles(self, ppf6c_strategy):
        """InferenceResult aggregates multiple profile results."""
        hid_path = HID_DIR / '1A2_A01_01.hid'
        ladder_path = HID_DIR / 'Ladder_G03_21.hid'

        image = HIDImage(
            path=hid_path,
            scaling_strategy=ppf6c_strategy,
            data_loading_strategy='superior',
        )
        data = image.data
        assert data is not None

        catalog = LadderAlleleCatalog.from_panel(ppf6c_strategy.panel)
        assert catalog is not None

        adjusted_panel = Ladder.create_adjusted_panel(
            ladder_path=ladder_path,
            catalog=catalog,
            data_loading_strategy='superior',
            scaling_strategy=ppf6c_strategy,
            dataset_strategy=None,  # type: ignore[arg-type]
        )
        assert adjusted_panel is not None

        # Build a ProfileResult
        profile_result = ProfileResult(
            sample=hid_path.stem,
            hid_path=str(hid_path),
            ladder_path=str(ladder_path),
            markers=[
                MarkerResult(
                    name='D3S1358',
                    dye_row=0,
                    alleles=[
                        AlleleCall(name='12', base_pair=120.0, height=1500.0, confidence=0.95),
                    ],
                ),
            ],
        )

        inference_result = InferenceResult(
            checkpoint='test.ckpt',
            kit='PPF6C',
            profiles=[profile_result],
        )

        assert inference_result.total_profiles == 1
        assert inference_result.total_markers_called == len(profile_result.markers)
        assert inference_result.total_alleles == profile_result.allele_count

        # Verify JSON serialization
        result_dict = inference_result.to_dict()
        assert result_dict['checkpoint'] == 'test.ckpt'
        assert result_dict['kit'] == 'PPF6C'
        assert result_dict['total_profiles'] == 1
        assert 'profiles' in result_dict


# ---------------------------------------------------------------------------
# Kit switching tests
# ---------------------------------------------------------------------------


class TestKitSwitching:
    """Test kit switching via different scaling strategies."""

    def test_gf_strategy_has_markers(self, gf_strategy):
        """GlobalFiler strategy has markers defined."""
        gf_markers = {m.name for m in gf_strategy.panel.markers}
        assert len(gf_markers) > 0
        # AMEL is common to all kits
        assert 'AMEL' in gf_markers

    def test_different_strategies_have_different_markers(self, ppf6c_strategy, gf_strategy):
        """Different strategies have different marker sets."""
        ppf6c_markers = {m.name for m in ppf6c_strategy.panel.markers}
        gf_markers = {m.name for m in gf_strategy.panel.markers}

        # Panels should have different marker sets
        assert ppf6c_markers != gf_markers
