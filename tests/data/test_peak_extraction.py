"""Tests for peak extraction functions."""

import numpy as np
import pytest

from dnanet.core.annotation import ScanpointAnnotation
from dnanet.core.panel import Panel
from dnanet.data.preprocessing.peak_extraction import (
    _build_peak_data,
    _slice_with_padding,
    extract_peaks_torch,
    setup_marker_to_idx,
    extract_peak_windows,
)
from tests.conftest import PANEL_PATH


class TestSliceWithPadding:
    """Tests for _slice_with_padding helper."""

    def test_normal_slice(self):
        arr = np.arange(50).reshape(5, 10)
        result = _slice_with_padding(arr, dye_index=0, start=2, length=5)
        np.testing.assert_array_equal(result, [2, 3, 4, 5, 6])

    def test_left_padding(self):
        arr = np.arange(50).reshape(5, 10)
        result = _slice_with_padding(arr, dye_index=0, start=-2, length=5)
        assert result[0] == 0.0  # padding
        assert result[1] == 0.0  # padding
        assert result[2] == 0  # arr[0, 0]

    def test_right_padding(self):
        arr = np.arange(50).reshape(5, 10)
        result = _slice_with_padding(arr, dye_index=0, start=8, length=5)
        assert result[0] == 8
        assert result[1] == 9
        assert result[2] == 0.0  # padding

    def test_3d_input(self):
        arr = np.arange(50).reshape(5, 10, 1)
        result = _slice_with_padding(arr, dye_index=1, start=0, length=3)
        np.testing.assert_array_equal(result, [10, 11, 12])

    def test_full_padding(self):
        arr = np.zeros((5, 10))
        result = _slice_with_padding(arr, dye_index=0, start=-10, length=5)
        np.testing.assert_array_equal(result, [0, 0, 0, 0, 0])


class TestBuildPeakData:
    """Tests for _build_peak_data helper."""

    def test_single_channel(self):
        arr = np.ones((5, 100))
        result = _build_peak_data(arr, dye_index=0, start=10, length=20, include_max_pool_dyes=False)
        assert result.shape == (20,)

    def test_with_max_pool(self):
        arr = np.ones((5, 100))
        arr[1, 10:30] = 5.0  # one dye is higher
        result = _build_peak_data(
            arr,
            dye_index=0,
            start=10,
            length=20,
            include_max_pool_dyes=True,
        )
        assert result.shape == (2, 20)
        # Second channel should contain max of other dyes (5.0)
        assert result[1, 0] == 5.0




class TestMarkerToIdx:
    """Tests for the marker index mapping."""

    @pytest.fixture
    def setup_mti(self, nfi_rnd_kit):
        mti, n_markers = setup_marker_to_idx(nfi_rnd_kit)
        return mti, n_markers

    def test_has_out_of_bin(self, setup_mti):
        assert setup_mti[0]['Out of Bin'] == 0

    def test_known_markers(self, setup_mti):
        assert 'D3S1358' in setup_mti[0]
        assert 'AMEL' in setup_mti[0]
        assert setup_mti[0]['D3S1358'] >= 1

    def test_n_markers(self, setup_mti):
        assert setup_mti[1] == 28


class TestExtractPeakWindows:
    """Tests for extract_peak_windows (requires mock HIDImage)."""

    class MockImage:
        """Minimal HIDImage-like object for testing."""

        def __init__(self, data, annotation_image=None):
            self._data = data
            self._panel = None
            if annotation_image is not None:
                self.annotation = ScanpointAnnotation(data=annotation_image)
            else:
                self.annotation = None
            self._scaler = np.arange(4096)

        @property
        def scaler(self):
            return self._scaler

        @property
        def data(self):
            return self._data

    def _make_profile_with_peaks(self, n_dyes=5, length=4096):
        """Create a profile with clear peaks for testing."""
        data = np.zeros((n_dyes, length))
        # Add a clear peak in dye 0 at position 2000
        for i in range(-20, 21):
            data[0, 2000 + i] = max(0, 500 - abs(i) * 20)
        # Add a peak in dye 1 at position 1000
        for i in range(-20, 21):
            data[1, 1000 + i] = max(0, 400 - abs(i) * 15)
        return data

    def test_extracts_peaks(self, nfi_rnd_kit):
        data = self._make_profile_with_peaks()
        image = self.MockImage(data)
        peaks = extract_peak_windows(image, threshold=100, window_size=120, scaling_strategy=nfi_rnd_kit)
        assert len(peaks) >= 2  # at least the two peaks we created

    def test_peak_properties(self, nfi_rnd_kit):
        data = self._make_profile_with_peaks()
        image = self.MockImage(data)
        peaks = extract_peak_windows(image, threshold=100, window_size=120, scaling_strategy=nfi_rnd_kit)
        for peak in peaks:
            assert peak.data.shape == (120,)
            assert peak.window_size == 120
            assert 0 <= peak.dye_index < 5

    def test_labels_from_annotation(self, nfi_rnd_kit):
        data = self._make_profile_with_peaks()
        ann = np.zeros_like(data)
        ann[0, 1990:2010] = 1  # annotate the peak in dye 0
        image = self.MockImage(data, annotation_image=ann)
        peaks = extract_peak_windows(
            image,
            threshold=100,
            window_size=120,
            scaling_strategy=nfi_rnd_kit,
        )
        dye0_peaks = [p for p in peaks if p.dye_index == 0]
        assert len(dye0_peaks) >= 1
        assert dye0_peaks[0].label == 'allele'

    def test_none_data_returns_empty(self, nfi_rnd_kit):
        image = self.MockImage(None)
        peaks = extract_peak_windows(image, threshold=100, window_size=120, scaling_strategy=nfi_rnd_kit)
        assert peaks == []

    def test_include_max_pool_dyes(self, nfi_rnd_kit):
        data = self._make_profile_with_peaks()
        image = self.MockImage(data)
        peaks = extract_peak_windows(
            image,
            threshold=100,
            window_size=120,
            scaling_strategy=nfi_rnd_kit,
            include_max_pool_dyes=True,
        )
        assert peaks[0].data.shape == (2, 120)

    def test_excludes_size_standard(self, nfi_rnd_kit):
        """Peaks in the 6th channel (size standard) should be excluded."""
        data = np.zeros((6, 4096))
        for i in range(-20, 21):
            data[5, 2000 + i] = max(0, 500 - abs(i) * 20)
        image = self.MockImage(data)
        peaks = extract_peak_windows(image, threshold=100, window_size=120, scaling_strategy=nfi_rnd_kit)
        assert all(p.dye_index < 5 for p in peaks)


class TestExtractPeaksTorch:
    """Tests for extract_peaks_torch."""

    class MockImage:
        def __init__(self, data):
            self._data = data
            self.adjusted_panel = Panel.from_xml(PANEL_PATH)
            self.annotation = None

        @property
        def data(self):
            return self._data

    def test_returns_correct_shapes(self, nfi_rnd_kit):
        data = np.zeros((5, 4096))
        for i in range(-20, 21):
            data[0, 2000 + i] = max(0, 500 - abs(i) * 20)
        image = self.MockImage(data)
        windows, markers, centers = extract_peaks_torch(
            image,
            scaling_strategy=nfi_rnd_kit,
            threshold=100,
            window_size=120,
        )
        assert windows.dim() == 3  # (P, C, W)
        assert windows.shape[1] == 1  # single channel
        assert windows.shape[2] == 120
        assert markers.dim() == 1  # (P,)
        assert centers.shape[1] == 2  # (P, 2)

    def test_empty_on_no_peaks(self, nfi_rnd_kit):
        data = np.zeros((5, 4096))
        image = self.MockImage(data)
        windows, markers, centers = extract_peaks_torch(
            image,
            scaling_strategy=nfi_rnd_kit,
            threshold=100,
            window_size=120,
        )
        assert windows.shape[0] == 0
        assert markers.shape[0] == 0
        assert centers.shape[0] == 0

    def test_none_data(self, nfi_rnd_kit):
        image = self.MockImage(None)
        with pytest.raises(ValueError):
            extract_peaks_torch(
                image,
                scaling_strategy=nfi_rnd_kit,
                threshold=100,
                window_size=120,
            )
