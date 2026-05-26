"""Tests for peak detection utilities.

Includes parametrized regression tests ported from the original DNANet.
"""

import numpy as np
import pytest

from dnanet.data.preprocessing.peaks import (
    find_peak_boundary,
    find_peak_idx_near_or_in_range,
    find_peak_near_idx,
    find_peaks_above_threshold,
)


class TestFindPeaksAboveThreshold:
    def test_single_peak(self):
        signal = np.array([0, 0, 10, 50, 100, 50, 10, 0, 0], dtype=float)
        peaks = find_peaks_above_threshold(signal, threshold=30)
        assert 4 in peaks

    def test_no_peaks_below_threshold(self):
        signal = np.array([1, 2, 3, 2, 1], dtype=float)
        peaks = find_peaks_above_threshold(signal, threshold=10)
        assert len(peaks) == 0

    def test_flat_top_peak(self):
        signal = np.array([0, 0, 50, 100, 100, 100, 50, 0, 0], dtype=float)
        peaks = find_peaks_above_threshold(signal, threshold=30)
        # The function doesn't crash on flat-top peaks
        assert isinstance(peaks, np.ndarray)

    @pytest.mark.parametrize(
        "array, expected",
        [
            (np.array([1000, 1001, 1010, 1003, 1002]), np.array([2])),
            (np.array([100, 101, 110, 103, 102]), np.array([])),
            (np.array([1000, 1003, 1003, 999]), np.array([1, 2])),
            (np.array([1000, 1001, 800, 800, 799]), np.array([1, 3])),
        ],
        ids=["single_peak", "below_threshold", "flat_top_pair", "two_peaks"],
    )
    def test_parametrized(self, array, expected):
        """Ported from original DNANet test_utils.py."""
        result = find_peaks_above_threshold(array, 200)
        np.testing.assert_array_equal(result, expected)


class TestFindPeakBoundary:
    def test_simple_peak(self):
        signal = np.array([0, 10, 30, 60, 100, 60, 30, 10, 0], dtype=float)
        start, end = find_peak_boundary(signal, peak_idx=4, threshold=20)
        assert start <= 4 <= end
        assert signal[start] >= 20 or start == 0

    @pytest.mark.parametrize(
        "array, idx, expected",
        [
            (np.array([110, 90, 101, 103, 110, 103, 90, 101]), 4, (2, 5)),
            (np.array([90, 90, 90, 90, 110, 90, 90, 90]), 4, (4, 4)),
            (np.array([110, 100, 101, 103, 110, 103, 100, 101]), 4, (0, 7)),
        ],
        ids=["mid_peak", "isolated_peak", "wide_peak"],
    )
    def test_parametrized(self, array, idx, expected):
        """Ported from original DNANet test_utils.py."""
        assert find_peak_boundary(array, idx, threshold=100) == expected


class TestFindPeakNearIdx:
    @pytest.mark.parametrize(
        "array, idx, expected",
        [
            (np.array([100, 101, 110, 103, 102, 101, 120, 101]), 1, np.array([2])),
            (np.array([100, 101, 110, 103, 102, 101, 120, 101]), 7, np.array([6])),
            (np.array([100, 101, 110, 103, 102, 101, 120, 101]), 4, np.array([2])),
        ],
        ids=["near_left_peak", "near_right_peak", "between_peaks"],
    )
    def test_parametrized(self, array, idx, expected):
        """Ported from original DNANet test_utils.py."""
        result = find_peak_near_idx(array, idx)
        np.testing.assert_array_equal(result, expected)


class TestFindPeakIdxNearOrInRange:
    @pytest.mark.parametrize(
        "array, range_, expected",
        [
            (
                np.array([100, 101, 110, 103, 102, 101, 120, 101]),
                np.array([1, 2, 3]),
                np.array([2]),
            ),
            (
                np.array([100, 101, 110, 103, 102, 101, 120, 101]),
                np.array([0, 1]),
                np.array([2]),
            ),
            (
                np.array([100, 101, 110, 103, 202, 101, 120, 101]),
                np.array([0, 1, 2, 3, 4, 5, 6]),
                np.array([4]),
            ),
            (
                np.array([80, 81, 90, 83, 82, 81, 90, 81]),
                np.array([1, 2, 3]),
                np.array([]),
            ),
        ],
        ids=["peak_in_range", "peak_just_after", "highest_peak", "below_threshold"],
    )
    def test_parametrized(self, array, range_, expected):
        """Ported from original DNANet test_utils.py."""
        result = find_peak_idx_near_or_in_range(array, range_, threshold=100)
        np.testing.assert_array_equal(result, expected)
