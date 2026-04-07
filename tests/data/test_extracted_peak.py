"""Tests for ExtractedPeak data model."""

import numpy as np
import pytest

from dnanet.data.extracted_peak import ExtractedPeak


class TestExtractedPeak:
    """Unit tests for the ExtractedPeak class."""

    def _make_peak(self, **kwargs) -> ExtractedPeak:
        defaults = dict(
            data=np.zeros((1, 120)),
            dye_index=0,
            peak_center=2048,
            window_size=120,
            peak_height=500.0,
            label="allele",
            marker_name="D3S1358",
            marker_index=1,
        )
        defaults.update(kwargs)
        return ExtractedPeak(**defaults)

    def test_basic_construction(self):
        peak = self._make_peak()
        assert peak.dye_index == 0
        assert peak.peak_center == 2048
        assert peak.window_size == 120
        assert peak.peak_height == 500.0
        assert peak.label == "allele"
        assert peak.marker_name == "D3S1358"
        assert peak.marker_index == 1

    def test_window_start_computed(self):
        peak = self._make_peak(peak_center=100, window_size=120)
        assert peak.window_start == 100 - 60

    def test_is_allele_true(self):
        peak = self._make_peak(label="allele")
        assert peak.is_allele is True

    def test_is_allele_false(self):
        peak = self._make_peak(label="noise")
        assert peak.is_allele is False

    def test_is_allele_none_label(self):
        peak = self._make_peak(label=None)
        assert peak.is_allele is False

    def test_equality(self):
        p1 = self._make_peak(dye_index=0, peak_center=100, window_size=120)
        p2 = self._make_peak(dye_index=0, peak_center=100, window_size=120,
                             label="noise")
        assert p1 == p2  # same dye, center, window → equal

    def test_inequality(self):
        p1 = self._make_peak(dye_index=0, peak_center=100)
        p2 = self._make_peak(dye_index=1, peak_center=100)
        assert p1 != p2

    def test_hash(self):
        p1 = self._make_peak(dye_index=0, peak_center=100, window_size=120)
        p2 = self._make_peak(dye_index=0, peak_center=100, window_size=120)
        assert hash(p1) == hash(p2)
        assert len({p1, p2}) == 1  # deduplication in set



    def test_data_shape_single_channel(self):
        data = np.random.rand(1, 120)
        peak = self._make_peak(data=data)
        assert peak.data.shape == (1, 120)

    def test_data_shape_two_channels(self):
        data = np.random.rand(2, 120)
        peak = self._make_peak(data=data)
        assert peak.data.shape == (2, 120)
