"""Tests for LabelTool and related visualization helpers."""

import numpy as np
import pytest

from dnanet.core.constants import LabelCategory
from dnanet.tools.labeltool.tool import LabelTool, bp_to_scan, scan_to_bp
from dnanet.tools.labeltool.visualization import get_peaks, add_initial_spans


# ---------------------------------------------------------------------------
# scan_to_bp / bp_to_scan
# ---------------------------------------------------------------------------


class TestScanBpConversion:
    def test_scan_to_bp_at_zero(self):
        result = scan_to_bp(0)
        assert abs(result - 65.0) < 0.01

    def test_scan_to_bp_at_4096(self):
        result = scan_to_bp(4096)
        assert abs(result - 475.0) < 0.01

    def test_roundtrip(self):
        bp = 200.0
        assert abs(scan_to_bp(bp_to_scan(bp)) - bp) < 1e-10

    def test_bp_to_scan_array(self):
        bps = np.array([65, 270, 475])
        scans = bp_to_scan(bps)
        assert len(scans) == 3
        assert abs(scans[0] - 0) < 0.01
        assert abs(scans[2] - 4096) < 0.01


# ---------------------------------------------------------------------------
# get_peaks
# ---------------------------------------------------------------------------


class TestGetPeaks:
    def test_detects_single_peak(self):
        # Create signal with one clear peak
        signal = np.zeros((1, 100, 1), dtype=np.float32)
        signal[0, 40:60, 0] = np.concatenate(
            [
                np.linspace(0, 500, 10),
                np.linspace(500, 0, 10),
            ]
        )
        peaks = get_peaks(signal, min_rfu=100)
        assert len(peaks) == 1
        assert len(peaks[0]) >= 1

    def test_detects_multiple_peaks(self):
        signal = np.zeros((2, 200, 1), dtype=np.float32)
        # Peak in dye 0
        signal[0, 40:60, 0] = np.concatenate(
            [
                np.linspace(0, 500, 10),
                np.linspace(500, 0, 10),
            ]
        )
        # Peak in dye 1
        signal[1, 100:120, 0] = np.concatenate(
            [
                np.linspace(0, 300, 10),
                np.linspace(300, 0, 10),
            ]
        )
        peaks = get_peaks(signal, min_rfu=100)
        assert len(peaks) == 2
        assert len(peaks[0]) >= 1
        assert len(peaks[1]) >= 1

    def test_no_peaks_below_threshold(self):
        signal = np.zeros((1, 100, 1), dtype=np.float32)
        signal[0, 40:60, 0] = 50  # Below threshold
        peaks = get_peaks(signal, min_rfu=200)
        assert len(peaks[0]) == 0

    def test_2d_input(self):
        signal = np.zeros((1, 100), dtype=np.float32)
        signal[0, 50] = 500
        peaks = get_peaks(signal, min_rfu=100)
        assert len(peaks) == 1


# ---------------------------------------------------------------------------
# add_initial_spans
# ---------------------------------------------------------------------------


class TestAddInitialSpans:
    def test_creates_spans_from_peak_ranges(self):
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2)
        peak_ranges = [
            [(10, 20, 15), (50, 60, 55)],
            [(100, 110, 105)],
        ]
        spans = add_initial_spans(axs, peak_ranges)
        assert len(spans) == 3
        assert spans[0]['x0'] == 10
        assert spans[0]['x1'] == 20
        assert spans[0]['peak_idx'] == 15
        assert spans[0]['category'] is None
        plt.close(fig)

    def test_empty_peaks_returns_empty(self):
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2)
        spans = add_initial_spans(axs, [[], []])
        assert len(spans) == 0
        plt.close(fig)


# ---------------------------------------------------------------------------
# LabelTool category handling
# ---------------------------------------------------------------------------


class TestLabelToolCategories:
    @pytest.fixture
    def label_tool(self):
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(5)
        tool = LabelTool(fig, axs)
        yield tool
        plt.close(fig)

    def test_categories_populated(self, label_tool):
        # Keys are radio labels like "Unlabeled (Gray - 0)", "Allele (Green - 1)"
        assert LabelCategory.UNLABELED.radio_label in label_tool._categories
        assert LabelCategory.ALLELE.radio_label in label_tool._categories
        assert len(label_tool._categories) == len(LabelCategory)

    def test_default_category(self, label_tool):
        assert label_tool.current_category == LabelCategory.UNLABELED.radio_label

    def test_spans_initially_empty(self, label_tool):
        assert label_tool.spans == []
