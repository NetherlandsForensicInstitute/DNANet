"""Tests for baseline estimation algorithms."""

import numpy as np
import pytest

from dnanet.data.preprocessing.baseline import (
    baseline_classic,
    baseline_enhanced,
    baseline_superior,
    fft_lowpass_smooth,
    get_baseline_method,
    percentile_window_baseline,
)


class TestPercentileWindowBaseline:
    """Tests for the core percentile-window baseline."""

    def test_output_shape_matches_input_2d(self, npy_rng):
        signal = npy_rng.randn(5, 4096).astype(np.float64)
        baseline = percentile_window_baseline(signal, window_size=101, percentile=20)
        assert baseline.shape == signal.shape

    def test_output_shape_matches_input_1d(self, npy_rng):
        signal = npy_rng.randn(4096).astype(np.float64)
        baseline = percentile_window_baseline(signal, window_size=101, percentile=20)
        assert baseline.shape == signal.shape

    def test_size_standard_channel_untouched(self):
        """When skip_last_channel=True, the last channel should have zero baseline."""
        signal = np.ones((6, 1000))
        baseline = percentile_window_baseline(
            signal, window_size=51, percentile=20, skip_last_channel=True
        )
        np.testing.assert_array_equal(baseline[5], 0)

    def test_all_channels_processed_by_default(self):
        """Without skip_last_channel, all channels get baseline estimation."""
        signal = np.ones((6, 1000))
        baseline = percentile_window_baseline(signal, window_size=51, percentile=20)
        # All channels should be processed (non-zero baseline for constant signal)
        assert np.any(baseline[5] != 0)

    def test_even_window_made_odd(self, npy_rng):
        """Even window sizes should be handled without error."""
        signal = npy_rng.randn(5, 100)
        baseline = percentile_window_baseline(signal, window_size=100, percentile=20)
        assert baseline.shape == signal.shape


class TestNamedBaselines:
    """Tests for the named baseline methods."""

    @pytest.fixture
    def signal(self, npy_rng):
        return npy_rng.randn(5, 4096).astype(np.float64)

    def test_classic(self, signal):
        result = baseline_classic(signal)
        assert result.shape == signal.shape

    def test_superior(self, signal):
        result = baseline_superior(signal)
        assert result.shape == signal.shape

    def test_enhanced(self, signal):
        result = baseline_enhanced(signal)
        assert result.shape == signal.shape


class TestFFTSmooth:
    def test_output_shape(self, npy_rng):
        signal = npy_rng.randn(5, 1000)
        smoothed = fft_lowpass_smooth(signal, keep_fraction=0.1)
        assert smoothed.shape == signal.shape

    def test_1d_input(self, npy_rng):
        signal = npy_rng.randn(1000)
        smoothed = fft_lowpass_smooth(signal, keep_fraction=0.1)
        assert smoothed.shape == signal.shape


class TestGetBaselineMethod:
    def test_known_methods(self):
        for name in ('classic', 'superior', 'enhanced'):
            method = get_baseline_method(name)
            assert callable(method)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match='Unknown baseline'):
            get_baseline_method('nonexistent')
