"""Tests for strategy infrastructure."""

import numpy as np

from dnanet.data.strategies.scaling.scaling import ScalingStrategy
from dnanet.data.strategies.scaling.size_standard import WEN_ILS, GENESCAN_600_LIZ


class TestSizeStandard:
    def test_wen_ils_has_19_peaks(self):
        assert len(WEN_ILS.expected_bps) == 19

    def test_genescan_values_sorted(self):
        assert np.all(np.diff(GENESCAN_600_LIZ.expected_bps) > 0)


class TestBasepairInterpolator:
    def test_interpolates_correctly(self):
        indices = np.array([0, 100, 200])
        bps = np.array([65.0, 270.0, 475.0])
        interp = ScalingStrategy.basepair_interpolator(indices, bps)
        result = interp(100)
        assert abs(result[0] - 270.0) < 1e-6

    def test_no_extrapolation_by_default(self):
        indices = np.array([10, 20, 30])
        bps = np.array([100.0, 200.0, 300.0])
        interp = ScalingStrategy.basepair_interpolator(indices, bps, extrapolate=False)
        result = interp(0)
        assert result[0] == 0.0  # outside range → zero

    def test_extrapolation_when_enabled(self):
        indices = np.array([10, 20, 30])
        bps = np.array([150.0, 250.0, 350.0])
        interp = ScalingStrategy.basepair_interpolator(indices, bps, extrapolate=True)
        result = interp(5)  # extrapolate before first point
        assert result[0] != 0.0  # should extrapolate, not zero
