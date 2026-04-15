"""Integration tests for ScalingStrategy with real size-standard data."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from dnanet.data.parsing.hid import get_peak_data
from dnanet.data.strategies.registry import StrategyRegistry
from dnanet.data.strategies.scaling.scaling import (
    ScalingStrategy,
)
from dnanet.data.strategies.scaling.globalfiler import GlobalFilerStrategy
from dnanet.data.strategies.scaling.powerplex_fusion_6c import PowerPlexFusion6CStrategy

from tests.conftest import RD_DIR


class TestPPF6CSizeStandard:
    """Test PPF6C size-standard parsing with real ladder data."""

    @pytest.fixture
    def ppf6c(self):
        StrategyRegistry.configure_kit("PPF6C")
        yield StrategyRegistry.get_scaling_strategy()
        StrategyRegistry.reset()

    def test_parse_size_standard_from_ladder(self, ppf6c):
        """Size standard parsing should succeed on a real ladder file."""
        data = get_peak_data(RD_DIR / "Ladder_G03_21.hid", data_loading_strategy="raw")
        assert data is not None

        ss_lane = np.array(data[-1])
        result = ppf6c.parse_size_standard(ss_lane)
        assert result is not None
        assert result.rescaled_indices.shape[0] == 4096
        assert result.scaler.shape[0] == 4096

    def test_scaler_is_monotonically_nondecreasing(self, ppf6c):
        """The scaler (bp values) should be monotonically non-decreasing.

        Due to discrete pixel→bp mapping, adjacent positions can have the
        same bp value, so we check >= rather than strictly >.
        """
        data = get_peak_data(RD_DIR / "Ladder_G03_21.hid", data_loading_strategy="raw")
        result = ppf6c.parse_size_standard(np.array(data[-1]))
        assert result is not None

        # Non-zero region of scaler should be non-decreasing
        nonzero = result.scaler[result.scaler > 0]
        assert np.all(np.diff(nonzero) >= 0)

    def test_scaler_range(self, ppf6c):
        """Scaler should cover approximately 65-475 bp for PPF6C."""
        data = get_peak_data(RD_DIR / "Ladder_G03_21.hid", data_loading_strategy="raw")
        result = ppf6c.parse_size_standard(np.array(data[-1]))
        assert result is not None

        nonzero = result.scaler[result.scaler > 0]
        assert nonzero[0] >= 60
        assert nonzero[-1] <= 500

    def test_extract_ss_peaks(self, ppf6c):
        """Peak extraction on real size-standard lane should match reference.

        Reference values from original DNANet test_utils.py.
        """
        data = get_peak_data(RD_DIR / "1A2_A01_01.hid", data_loading_strategy="analyzed")
        assert data is not None
        ss_lane = data[-1]

        peaks = PowerPlexFusion6CStrategy._extract_ss_peaks(ss_lane)
        expected = np.array([
            2796, 2813, 2901, 2917, 2961, 2989, 3038, 3095, 3144,
            3387, 3645, 3702, 3881, 4120, 4348, 4578, 4806, 5033,
            5257, 5531, 5798, 6059, 6323, 6579, 6827, 7077, 7318,
            7560, 7788, 8018, 8238,
        ])
        assert_array_equal(peaks, expected)

    def test_extract_ss_peaks_with_low_tail(self, ppf6c):
        """When tail peaks are below threshold, adaptive detection should find them.

        From original test: zero out signal after 8200 but keep final peak at 150rfu.
        """
        data = get_peak_data(RD_DIR / "1A2_A01_01.hid", data_loading_strategy="analyzed")
        assert data is not None
        ss_lane = data[-1].copy()

        expected = np.array([
            2796, 2813, 2901, 2917, 2961, 2989, 3038, 3095, 3144,
            3387, 3645, 3702, 3881, 4120, 4348, 4578, 4806, 5033,
            5257, 5531, 5798, 6059, 6323, 6579, 6827, 7077, 7318,
            7560, 7788, 8018, 8238,
        ])

        # Simulate low tail peak
        ss_lane[8199:] = 0
        ss_lane[expected[-1]] = 150

        peaks = PowerPlexFusion6CStrategy._extract_ss_peaks(ss_lane)
        assert_array_equal(peaks, expected)

    def test_marker_name_to_dye_idx(self, ppf6c):
        """Marker-to-dye mapping should cover all expected PPF6C markers."""
        mapping = ppf6c.marker_name_to_dye_idx()
        assert mapping["AMEL"] == 0
        assert mapping["TH01"] == 2
        assert mapping["FGA"] == 4
        assert len(mapping) == 27

    def test_dye_channel_colors(self, ppf6c):
        """Should return 6 colors (5 dyes + size standard)."""
        colors = ppf6c.dye_channel_colors()
        assert len(colors) == 6


class TestGlobalFilerAttemptFit:
    """Test the iterative fit logic from the original tests."""

    def test_perfect_fit(self):
        """With a perfect linear relationship, fit should converge immediately."""
        peak_idxs = np.array([100, 200, 300])
        expected_bps = np.array([60, 160, 260])

        new_peaks, new_bps, diff = GlobalFilerStrategy._attempt_fit(
            peak_idxs, expected_bps,
            threshold=5.0, max_shrinkages=10,
        )
        assert diff < 1.0
        assert_array_equal(new_peaks, peak_idxs)
        assert_array_equal(new_bps, expected_bps)


class TestBasepairInterpolatorRegression:
    """Regression tests ported from original test_utils.py.

    These verify backward compatibility of the interpolation math.
    """

    def test_integer_interpolation(self):
        """Linear indices [0,3,6] → bps [1,7,13]: should give 2bp per index."""
        indices = [0, 3, 6]
        bps = [1, 7, 13]
        interp = ScalingStrategy.basepair_interpolator(indices, bps)
        assert_array_equal(interp(np.arange(6)), np.array([1., 3., 5., 7., 9., 11.]))

    def test_extrapolation(self):
        """With extrapolate=True, should extend linearly beyond known range."""
        indices = [2, 3]
        bps = [12, 13]
        interp = ScalingStrategy.basepair_interpolator(indices, bps, extrapolate=True)
        assert_array_equal(interp(np.arange(3)), np.array([10., 11., 12.]))
        assert_array_equal(interp(0), np.array([10.]))
        assert_array_equal(interp(4), np.array([14.]))
        assert_array_equal(interp(5), np.array([15.]))

    def test_no_extrapolation(self):
        """Without extrapolation, out-of-range values should be 0."""
        indices = [10, 20, 30]
        bps = [100, 200, 300]
        interp = ScalingStrategy.basepair_interpolator(indices, bps)
        assert interp(10) == 100
        assert interp(20) == 200
        assert interp(15) == 150
        assert interp(5) == 0   # below range
        assert interp(35) == 0  # above range

    def test_float_interpolation(self):
        """Float-valued indices and bps from original test."""
        indices = [97.82, 102.13]
        bps = [97.64, 101.95]
        interp = ScalingStrategy.basepair_interpolator(indices, bps)
        assert_array_equal(interp(101.22), np.array([101.04]))

    def test_float_extrapolation(self):
        indices = [97.82, 102.13]
        bps = [97.64, 101.95]
        interp = ScalingStrategy.basepair_interpolator(indices, bps, extrapolate=True)
        assert_array_equal(interp(93.47), np.array([93.29]))


class TestRescaleDye:
    """Tests for ScalingStrategy.rescale_dye — ported from original."""

    def test_linear_basepairs(self):
        """Simple linear bp array: target [60,100] over 5 pixels."""
        basepairs = np.array([50, 60, 70, 80, 90, 100, 110, 120, 130, 140])
        rescaled = ScalingStrategy.rescale_dye(basepairs, 5, (60, 100))
        expected = np.array([1, 2, 3, 4, 5])
        assert_array_equal(rescaled, expected)
