"""Tests for PeakWindowDataset."""

import numpy as np


from dnanet.core.annotation import ScanpointAnnotation
from dnanet.data.peak_dataset import PeakWindowDataset
from dnanet.data.extracted_peak import ExtractedPeak


class MockImage:
    """Minimal HIDImage-like object."""

    def __init__(self, data, annotation_image=None):
        self._raw_data = data
        self._panel = None
        if annotation_image is not None:
            self.annotation = ScanpointAnnotation(data=annotation_image)
        else:
            self.annotation = None
        self.scaler = np.arange(4096)

    @property
    def data(self):
        return self._raw_data


def _make_profile_with_peaks(n_dyes=5, length=4096, n_peaks=3):
    """Create profile with known peaks."""
    data = np.zeros((n_dyes, length))
    centers = np.linspace(500, 3500, n_peaks, dtype=int)
    for c in centers:
        for i in range(-20, 21):
            data[0, c + i] = max(0, 500 - abs(i) * 20)
    return data, centers


class TestPeakWindowDataset:

    def _make_base_dataset(self, n_images=3, n_peaks=3):
        """Create a SimpleDataset of mock images."""
        ## TODO
        ...

    def test_extracts_peaks_from_images(self, nfi_rnd_kit):
        base = self._make_base_dataset(n_images=2, n_peaks=3)
        ds = PeakWindowDataset(
            base_dataset=base,
            threshold=100,
            window_size=120,
            preprocess=False,
        )
        # Each image has 3 peaks in dye 0
        assert len(ds) >= 6

    def test_items_are_extracted_peaks(self, nfi_rnd_kit):
        base = self._make_base_dataset(n_images=1)
        ds = PeakWindowDataset(
            base_dataset=base, threshold=100, window_size=120, preprocess=False,
        )
        for peak in ds:
            assert isinstance(peak, ExtractedPeak)
            assert peak.data.shape == (1, 120)

    def test_label_mapping(self, nfi_rnd_kit):
        base = self._make_base_dataset(n_images=1)
        ds = PeakWindowDataset(
            base_dataset=base, threshold=100, window_size=120,
            preprocess=False,
        )
        assert ds.label_to_idx["noise"] == 0
        assert ds.label_to_idx["allele"] == 1
        assert ds.idx_to_label[0] == "noise"

    def test_preprocessing_changes_data(self, nfi_rnd_kit):
        base = self._make_base_dataset(n_images=1)
        ds_raw = PeakWindowDataset(
            base_dataset=base, threshold=100, window_size=120, preprocess=False,
        )
        ds_prep = PeakWindowDataset(
            base_dataset=base, threshold=100, window_size=120,
            preprocess=True, log_scale=True,
        )
        # Preprocessed data should be different (scaled)
        raw_max = max(p.data.max() for p in ds_raw)
        prep_max = max(p.data.max() for p in ds_prep)
        assert prep_max < raw_max  # log scaling reduces magnitude

    def test_split(self):
        base = self._make_base_dataset(n_images=5, n_peaks=2)
        ds = PeakWindowDataset(
            base_dataset=base, threshold=100, window_size=120, preprocess=False,
        )
        total = len(ds)
        train, val = ds.split(fraction=0.8, seed=42)
        assert len(train) + len(val) == total



    def test_include_max_pool_dyes(self, nfi_rnd_kit):
        base = self._make_base_dataset(n_images=1)
        ds = PeakWindowDataset(
            base_dataset=base, threshold=100, window_size=120,
            include_max_pool_dyes=True, preprocess=False,
        )
        assert ds[0].data.shape == (2, 120)
