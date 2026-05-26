"""Unit tests for multi-class annotation adjustment.

Covers:
- find_valley_idx_in_range and find_absolute_peak_idx_in_range
- HIDDataset._adjust_and_flatten_span_annotation (per-class dispatch, flattening)
- The ScanpointAnnotation → one-hot → adjust fallback path (line-405 flow)
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from dnanet.core.constants import LabelCategory
from dnanet.core.annotation import SpanAnnotation, ScanpointAnnotation
from dnanet.data.hid_dataset import HIDDataset
from dnanet.data.preprocessing.peaks import (
    find_valley_idx_in_range,
    find_absolute_peak_idx_in_range,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dataset() -> HIDDataset:
    return object.__new__(HIDDataset)


def _profile(signal: np.ndarray | None) -> MagicMock:
    m = MagicMock()
    m.data = signal
    return m


def _span(num_dyes: int, scanpoints: int, num_classes: int) -> np.ndarray:
    return np.zeros((num_dyes, scanpoints, num_classes), dtype=np.int8)


# ---------------------------------------------------------------------------
# find_valley_idx_in_range
# ---------------------------------------------------------------------------


class TestFindValleyIdxInRange:
    def test_returns_minimum(self):
        signal = np.array([10, 5, 1, 8, 10], dtype=float)
        result = find_valley_idx_in_range(signal, np.array([0, 1, 2, 3, 4]), threshold=0)
        assert result[0] == 2

    def test_empty_range_returns_empty(self):
        signal = np.array([10, 5, 1], dtype=float)
        result = find_valley_idx_in_range(signal, np.array([]), threshold=0)
        assert result.size == 0

    def test_threshold_ignored(self):
        signal = np.array([100, 200, 50], dtype=float)
        r_low = find_valley_idx_in_range(signal, np.array([0, 1, 2]), threshold=0)
        r_high = find_valley_idx_in_range(signal, np.array([0, 1, 2]), threshold=999)
        np.testing.assert_array_equal(r_low, r_high)


# ---------------------------------------------------------------------------
# find_absolute_peak_idx_in_range
# ---------------------------------------------------------------------------


class TestFindAbsolutePeakIdxInRange:
    def test_finds_largest_absolute_value(self):
        signal = np.array([10, -800, 50, 20], dtype=float)
        result = find_absolute_peak_idx_in_range(signal, np.array([0, 1, 2, 3]), 0)
        assert result[0] == 1  # index of -800

    def test_positive_extreme(self):
        signal = np.array([10, 50, 900, 20], dtype=float)
        result = find_absolute_peak_idx_in_range(signal, np.array([0, 1, 2, 3]), 0)
        assert result[0] == 2

    def test_empty_range_returns_empty(self):
        result = find_absolute_peak_idx_in_range(np.array([1.0, 2.0]), np.array([]), 0)
        assert result.size == 0


# ---------------------------------------------------------------------------
# _adjust_and_flatten_span_annotation
# ---------------------------------------------------------------------------


class TestAdjustAndFlattenSpanAnnotation:
    def test_none_data_falls_back_to_argmax(self):
        ds = _dataset()
        span_data = _span(1, 10, 3)
        span_data[0, 3:6, 1] = 1  # ALLELE
        result = ds._adjust_and_flatten_span_annotation(
            _profile(None), SpanAnnotation(span_data), 'top'
        )
        assert result.shape == (1, 10)
        # argmax: class 1 wins over class 0 in annotated region
        assert (result[0, 3:6] == 1).all()

    @pytest.mark.parametrize('adjustment_type', ['top', 'complete'])
    def test_allele_adjusted_to_peak(self, adjustment_type):
        ds = _dataset()
        scanpoints = 20
        signal = np.zeros((1, scanpoints))
        signal[0, 10] = 500  # clear peak
        span_data = _span(1, scanpoints, 2)
        span_data[0, 5:15, 1] = 1  # ALLELE span with peak at 10

        result = ds._adjust_and_flatten_span_annotation(
            _profile(signal), SpanAnnotation(span_data), adjustment_type
        )

        assert result[0, 10] == 1
        if adjustment_type == 'top':
            assert result[0, 5] == 0
            assert result[0, 14] == 0

    def test_two_disjoint_regions_both_adjusted(self):
        ds = _dataset()
        scanpoints = 50
        signal = np.zeros((1, scanpoints))
        signal[0, 5] = 300
        signal[0, 35] = 400
        span_data = _span(1, scanpoints, 2)
        span_data[0, 2:10, 1] = 1  # region A
        span_data[0, 30:40, 1] = 1  # region B

        result = ds._adjust_and_flatten_span_annotation(
            _profile(signal), SpanAnnotation(span_data), 'top', threshold=0
        )

        assert result[0, 5] == 1
        assert result[0, 35] == 1
        assert result.sum() == 2

    def test_bleed_through_uses_absolute_peak(self):
        ds = _dataset()
        scanpoints = 20
        signal = np.zeros((1, scanpoints))
        signal[0, 8] = -800  # strongest absolute value (negative)
        signal[0, 9] = 200

        num_classes = LabelCategory.BLEED_THROUGH.value  # index 4 → need 5 classes
        # LabelCategory members are 0-indexed, BLEED_THROUGH is index 4
        bt_idx = list(LabelCategory).index(LabelCategory.BLEED_THROUGH)
        num_classes = bt_idx + 1

        span_data = _span(1, scanpoints, num_classes)
        span_data[0, 5:12, bt_idx] = 1
        profile = _profile(signal)

        result = ds._adjust_and_flatten_span_annotation(
            profile, SpanAnnotation(span_data), 'top', threshold=0
        )

        assert result[0, 8] == bt_idx

    def test_no_peak_above_threshold_removes_annotation(self):
        ds = _dataset()
        scanpoints = 20
        signal = np.zeros((1, scanpoints))  # all zero → nothing above threshold=100
        span_data = _span(1, scanpoints, 2)
        span_data[0, 5:10, 1] = 1

        result = ds._adjust_and_flatten_span_annotation(
            _profile(signal), SpanAnnotation(span_data), 'top', threshold=100
        )

        assert (result == 0).all()

    def test_higher_class_wins_when_peaks_coincide(self):
        ds = _dataset()
        scanpoints = 20
        signal = np.zeros((1, scanpoints))
        signal[0, 10] = 500  # both ALLELE and STUTTER resolve to same peak
        num_classes = 3  # UNLABELED, ALLELE, STUTTER
        span_data = _span(1, scanpoints, num_classes)
        span_data[0, 7:14, 1] = 1  # ALLELE
        span_data[0, 7:14, 2] = 1  # STUTTER (same span)

        result = ds._adjust_and_flatten_span_annotation(
            _profile(signal), SpanAnnotation(span_data), 'top', threshold=0
        )

        assert result[0, 10] == 2  # STUTTER (2) wins over ALLELE (1)

    def test_background_class_never_adjusted(self):
        ds = _dataset()
        scanpoints = 10
        signal = np.ones((1, scanpoints), dtype=float) * 100
        span_data = _span(1, scanpoints, 2)
        # only background marked — no non-zero class to adjust
        span_data[0, :, 0] = 1

        result = ds._adjust_and_flatten_span_annotation(
            _profile(signal), SpanAnnotation(span_data), 'top', threshold=0
        )

        assert (result == 0).all()

    def test_multidye_adjusted_independently(self):
        ds = _dataset()
        scanpoints = 20
        signal = np.zeros((2, scanpoints))
        signal[0, 5] = 300  # dye 0 peak
        signal[1, 15] = 400  # dye 1 peak
        span_data = _span(2, scanpoints, 2)
        span_data[0, 2:10, 1] = 1
        span_data[1, 12:18, 1] = 1

        result = ds._adjust_and_flatten_span_annotation(
            _profile(signal), SpanAnnotation(span_data), 'top', threshold=0
        )

        assert result[0, 5] == 1
        assert result[1, 15] == 1
        assert result.sum() == 2


# ---------------------------------------------------------------------------
# ScanpointAnnotation fallback (one-hot conversion → adjust)
# ---------------------------------------------------------------------------


class TestScanpointAnnotationFallback:
    """Simulates the line-405 path: flat ScanpointAnnotation → one-hot → adjust."""

    def _convert_and_adjust(self, ds, flat, signal, adjustment_type='top', threshold=0):
        num_classes = max(int(flat.max()) + 1, 2)
        one_hot = np.zeros((*flat.shape, num_classes), dtype=np.int8)
        for c in range(num_classes):
            one_hot[..., c] = (flat == c).astype(np.int8)
        return ds._adjust_and_flatten_span_annotation(
            _profile(signal), SpanAnnotation(data=one_hot), adjustment_type, threshold
        )

    def test_binary_adjusted_to_peak(self):
        ds = _dataset()
        scanpoints = 20
        signal = np.zeros((1, scanpoints))
        signal[0, 10] = 500
        flat = np.zeros((1, scanpoints), dtype=np.int8)
        flat[0, 5:15] = 1

        result = self._convert_and_adjust(ds, flat, signal)

        assert result[0, 10] == 1
        assert result[0, 5] == 0
        assert result[0, 14] == 0

    def test_multiclass_each_class_adjusted(self):
        ds = _dataset()
        scanpoints = 40
        signal = np.zeros((1, scanpoints))
        signal[0, 5] = 300  # ALLELE peak
        signal[0, 25] = 400  # STUTTER peak
        flat = np.zeros((1, scanpoints), dtype=np.int8)
        flat[0, 2:10] = 1  # ALLELE region
        flat[0, 20:30] = 2  # STUTTER region

        result = self._convert_and_adjust(ds, flat, signal)

        assert result[0, 5] == 1
        assert result[0, 25] == 2
