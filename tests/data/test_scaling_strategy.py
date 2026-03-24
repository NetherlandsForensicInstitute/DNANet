"""Extended tests for ScalingStrategy — GlobalFiler _attempt_fit edge cases."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from tests.conftest import RD_DIR, PROVEDIT_DIR
from dnanet.data.parsing.hid import get_peak_data
from dnanet.data.strategies.scaling import (
    GlobalFilerStrategy,
    PowerPlexFusion6CStrategy,
    get_scaling_strategy,
)
from dnanet.data.strategies.registry import StrategyRegistry


# ---------------------------------------------------------------------------
# GlobalFiler _attempt_fit edge cases
# ---------------------------------------------------------------------------

class TestGlobalFilerAttemptFitEdgeCases:
    def test_too_few_peaks_handles_gracefully(self):
        """With only 2 peaks, should not crash."""
        peak_idxs = np.array([100, 200])
        expected_bps = np.array([60, 160, 260, 360, 460])
        result = GlobalFilerStrategy._attempt_fit(
            peak_idxs, expected_bps, threshold=5.0, max_shrinkages=10,
        )
        trimmed, bps, diff = result
        assert len(trimmed) <= 2
        assert len(bps) == len(trimmed)

    def test_single_peak_handled(self):
        """A single peak should return gracefully."""
        peak_idxs = np.array([100])
        expected_bps = np.array([60, 160, 260])
        result = GlobalFilerStrategy._attempt_fit(
            peak_idxs, expected_bps, threshold=5.0, max_shrinkages=10,
        )
        trimmed, bps, diff = result
        assert len(trimmed) <= 1

    def test_max_shrinkages_returns_best_fit(self):
        """When fit never converges, should return best rather than raise."""
        # Create deliberately bad data
        peak_idxs = np.array([100, 500, 1500, 2000, 3000])
        expected_bps = np.array([60, 80, 100, 120, 140])
        result = GlobalFilerStrategy._attempt_fit(
            peak_idxs, expected_bps, threshold=0.001, max_shrinkages=3,
        )
        trimmed, bps, diff = result
        assert trimmed is not None
        assert bps is not None
        assert len(trimmed) == len(bps)

    def test_shrinkage_reduces_length(self):
        """Each shrinkage should reduce the number of points used."""
        # Make data that needs exactly one shrinkage
        peak_idxs = np.array([100, 200, 300, 400, 500])
        expected_bps = np.array([60, 160, 260, 360, 9999])  # last one is bad

        result = GlobalFilerStrategy._attempt_fit(
            peak_idxs, expected_bps, threshold=5.0, max_shrinkages=10,
        )
        trimmed, bps, diff = result
        assert len(bps) < len(expected_bps)

    def test_perfect_quadratic_fit(self):
        """A perfect quadratic relationship should converge immediately."""
        x = np.array([100, 200, 300, 400, 500])
        # y = 0.001*x^2 + 0.5*x + 10  (quadratic)
        y = 0.001 * x ** 2 + 0.5 * x + 10
        result = GlobalFilerStrategy._attempt_fit(
            x, y, threshold=5.0, max_shrinkages=10,
        )
        _, _, diff = result
        assert diff < 0.01


# ---------------------------------------------------------------------------
# GlobalFiler parse_size_standard
# ---------------------------------------------------------------------------

class TestGlobalFilerParseSizeStandard:
    @pytest.fixture
    def gf(self):
        StrategyRegistry.configure_kit("GLOBALFILER")
        yield StrategyRegistry.get_scaling_strategy()
        StrategyRegistry.reset()

    def test_parse_with_real_provedit_data(self, gf):
        """Parse size standard from a real ProvedIt ladder file."""
        ladder_path = PROVEDIT_DIR / "5 sec" / "RD14-0003(020316ADG_5sec)" / "A01_Ladder-GF_01.5sec.hid"
        if not ladder_path.exists():
            pytest.skip("ProvedIt test resource not available")

        data = get_peak_data(ladder_path, strategy="raw")
        assert data is not None
        ss_lane = np.array(data[-1])
        result = gf.parse_size_standard(ss_lane)
        assert result is not None
        assert result.rescaled_indices.shape[0] == 4096
        assert result.scaler.shape[0] == 4096

    def test_parse_returns_correct_shapes(self, gf):
        """Even with imperfect data, result should have correct shapes."""
        ladder_path = PROVEDIT_DIR / "5 sec" / "RD14-0003(020316ADG_5sec)" / "A01_Ladder-GF_01.5sec.hid"
        if not ladder_path.exists():
            pytest.skip("ProvedIt test resource not available")

        data = get_peak_data(ladder_path, strategy="raw")
        result = gf.parse_size_standard(np.array(data[-1]))
        assert result.rescaled_indices.shape == (4096,)
        assert result.scaler.shape == (4096,)
        assert isinstance(result.fit_error, float)


# ---------------------------------------------------------------------------
# PPF6C validate_ss_peaks
# ---------------------------------------------------------------------------

class TestPPF6CValidateSSPeaks:
    def test_valid_19_peaks(self):
        """19 evenly spaced peaks should pass validation."""
        # Create peaks with ~10 pixels/bp ratio
        peak_idxs = np.arange(19) * 100 + 1000
        expected_bps = np.arange(19) * 10 + 60
        result = PowerPlexFusion6CStrategy._validate_ss_peaks(peak_idxs, expected_bps)
        assert result is True

    def test_wrong_count_fails(self):
        """18 peaks should fail (need exactly 19)."""
        result = PowerPlexFusion6CStrategy._validate_ss_peaks(
            np.arange(18) * 100 + 1000,
            np.arange(18) * 10 + 60,
        )
        assert result is False

    def test_bad_spacing_fails(self):
        """Peaks with pixel/bp ratio outside 7-13 should fail."""
        # Ratio of ~2 pixels/bp (too small)
        peak_idxs = np.arange(19) * 20 + 1000
        expected_bps = np.arange(19) * 10 + 60
        result = PowerPlexFusion6CStrategy._validate_ss_peaks(peak_idxs, expected_bps)
        assert result is False


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

class TestScalingStrategyFactory:
    def test_ppf6c(self):
        s = get_scaling_strategy("PPF6C")
        assert isinstance(s, PowerPlexFusion6CStrategy)

    def test_globalfiler(self):
        s = get_scaling_strategy("GLOBALFILER")
        assert isinstance(s, GlobalFilerStrategy)

    def test_case_insensitive(self):
        s = get_scaling_strategy("ppf6c")
        assert isinstance(s, PowerPlexFusion6CStrategy)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_scaling_strategy("NONEXISTENT_KIT")
