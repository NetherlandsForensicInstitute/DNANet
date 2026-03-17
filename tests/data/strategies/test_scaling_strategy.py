import numpy as np
import pytest
from DNAnet.data.strategies.kit_strategies.scaling_strategy import ScalingStrategy, GlobalFilerScalingStrategy
from unittest.mock import MagicMock

class ConcreteScalingStrategy(ScalingStrategy):
    def parse_size_standard(self, size_standard_lane):
        return None

@pytest.fixture
def mock_kit():
    kit = MagicMock()
    kit.panel = MagicMock()
    return kit

@pytest.fixture
def scaling_strategy(mock_kit):
    return ConcreteScalingStrategy(mock_kit, basepair_start=60, basepair_end=500, scanpoint_resolution=100)

def test_basepair_interpolator():
    indices = [10, 20, 30]
    bps = [100, 200, 300]
    interpolator = ScalingStrategy.basepair_interpolator(indices, bps)

    assert interpolator(10) == 100
    assert interpolator(20) == 200
    assert interpolator(15) == 150
    assert interpolator(5) == 0  # No extrapolation by default
    assert interpolator(35) == 0

def test_basepair_interpolator_extrapolate():
    indices = [10, 20]
    bps = [100, 200]
    interpolator = ScalingStrategy.basepair_interpolator(indices, bps, extrapolate=True)

    assert interpolator(5) == 50
    assert interpolator(25) == 250

def test_rescale_dye():
    # 10 scan points with bps from 50 to 140
    basepairs = np.array([50, 60, 70, 80, 90, 100, 110, 120, 130, 140])
    rescale_size = 5
    target_range = (60, 100)

    # target_linspace = [60, 70, 80, 90, 100]
    # Expected indices in 'basepairs' that are closest to target_linspace:
    # 60 -> index 1
    # 70 -> index 2
    # 80 -> index 3
    # 90 -> index 4
    # 100 -> index 5

    rescaled_indices = ScalingStrategy.rescale_dye(basepairs, rescale_size, target_range)
    expected_indices = np.array([1, 2, 3, 4, 5])

    np.testing.assert_array_equal(rescaled_indices, expected_indices)

def test_interpolate(scaling_strategy):
    # Setup
    peak_idxs = np.array([10, 20, 30])
    bps = np.array([100, 200, 300])
    size_standard_lane = np.zeros(40)

    rescaled_indices, scaler = scaling_strategy.interpolate(peak_idxs, bps, size_standard_lane)

    # scanpoint_resolution is 100
    assert len(rescaled_indices) == 100
    assert len(scaler) == 100

    assert np.all(scaler >= 0)

def test_global_filer_attempt_fit():
    # Perfect fit
    peak_idxs = np.array([100, 200, 300])
    expected_bps = np.array([60, 160, 260]) # Linear for simplicity

    new_peak_idxs, new_bps, diff = GlobalFilerScalingStrategy.attempt_fit(
        peak_idxs, expected_bps, validation_threshold=5.0, max_shrinkages=10
    )

    assert diff < 1.0
    np.testing.assert_array_equal(new_peak_idxs, peak_idxs)
    np.testing.assert_array_equal(new_bps, expected_bps)
