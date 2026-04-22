"""Tests for PeakWindowDataset."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dnanet.core.annotation import ScanpointAnnotation
from dnanet.data.peak_dataset import PeakWindowDataset
from dnanet.data.extracted_peak import ExtractedPeak


class MockImage:
    """Minimal HIDImage-like object."""

    def __init__(self, data, scaling_strategy, annotation_image=None, name="mock"):
        self._raw_data = data
        self._panel = None
        self.path = Path(f"{name}.hid")
        self.scaling_strategy = scaling_strategy
        if annotation_image is not None:
            self.annotation = ScanpointAnnotation(data=annotation_image)
        else:
            self.annotation = None
        self.scaler = np.arange(4096)

    @property
    def data(self):
        return self._raw_data


class MockBaseDataset:
    """Minimal HIDDataset-like object for PeakWindowDataset tests."""

    def __init__(self, images, scaling_strategy, dataset_strategy):
        self._images = images
        self._scaling = scaling_strategy
        self._dataset_strategy = dataset_strategy
        self.transform = None

    @property
    def images(self):
        return self._images

    @property
    def dataset_strategy(self):
        return self._dataset_strategy


def _make_profile_with_peaks(n_dyes=5, length=4096, n_peaks=3):
    """Create profile with known peaks."""
    data = np.zeros((n_dyes, length))
    centers = np.linspace(500, 3500, n_peaks, dtype=int)
    for c in centers:
        for i in range(-20, 21):
            data[0, c + i] = max(0, 500 - abs(i) * 20)
    return data, centers


class TestPeakWindowDataset:

    def _make_base_dataset(self, scaling_strategy, dataset_strategy, n_images=3, n_peaks=3):
        """Create a SimpleDataset of mock images."""
        images = []
        for idx in range(n_images):
            data, centers = _make_profile_with_peaks(n_peaks=n_peaks)
            ann = np.zeros_like(data)
            ann[0, centers[0] - 2 : centers[0] + 3] = 1
            images.append(
                MockImage(
                    data,
                    scaling_strategy,
                    annotation_image=ann,
                    name=f"mock_{idx}",
                )
            )
        return MockBaseDataset(images, scaling_strategy, dataset_strategy)

    def test_extracts_peaks_from_images(self, nfi_rnd_kit, nfi_rnd_dataset):
        base = self._make_base_dataset(nfi_rnd_kit, nfi_rnd_dataset, n_images=2, n_peaks=3)
        ds = PeakWindowDataset(
            images=base.images,
            dataset_strategy=base.dataset_strategy,
            threshold=100,
            window_size=120,
            preprocess=False,
        )
        # Each image has 3 peaks in dye 0
        assert len(list(ds)) >= 6

    def test_items_are_extracted_peaks(self, nfi_rnd_kit, nfi_rnd_dataset):
        base = self._make_base_dataset(nfi_rnd_kit, nfi_rnd_dataset, n_images=1)
        ds = PeakWindowDataset(
            images=base.images,
            dataset_strategy=base.dataset_strategy,
            threshold=100,
            window_size=120,
            preprocess=False,
        )
        for peak in ds:
            assert isinstance(peak, ExtractedPeak)
            assert peak.data.shape == (1, 120)

    def test_label_mapping(self, nfi_rnd_kit, nfi_rnd_dataset):
        base = self._make_base_dataset(nfi_rnd_kit, nfi_rnd_dataset, n_images=1)
        ds = PeakWindowDataset(
            images=base.images,
            dataset_strategy=base.dataset_strategy,
            threshold=100,
            window_size=120,
            preprocess=False,
        )
        assert ds.label_to_idx["noise"] == 0
        assert ds.label_to_idx["allele"] == 1
        assert ds.idx_to_label[0] == "noise"

    def test_preprocessing_changes_data(self, nfi_rnd_kit, nfi_rnd_dataset):
        base = self._make_base_dataset(nfi_rnd_kit, nfi_rnd_dataset, n_images=1)
        ds_raw = PeakWindowDataset(
            images=base.images,
            dataset_strategy=base.dataset_strategy,
            threshold=100,
            window_size=120,
            preprocess=False,
        )
        ds_prep = PeakWindowDataset(
            images=base.images,
            dataset_strategy=base.dataset_strategy,
            threshold=100,
            window_size=120,
            preprocess=True, log_scale=True,
        )
        # Preprocessed data should be different (scaled)
        raw_max = max(p.data.max() for p in ds_raw)
        prep_max = max(p.data.max() for p in ds_prep)
        assert prep_max < raw_max  # log scaling reduces magnitude



    def test_include_max_pool_dyes(self, nfi_rnd_kit, nfi_rnd_dataset):
        base = self._make_base_dataset(nfi_rnd_kit, nfi_rnd_dataset, n_images=1)
        ds = PeakWindowDataset(
            images=base.images,
            dataset_strategy=base.dataset_strategy,
            threshold=100,
            window_size=120,
            include_max_pool_dyes=True, preprocess=False,
        )
        assert next(iter(ds)).data.shape == (2, 120)

    def test_worker_images_returns_all_images_without_worker_info(
        self,
        nfi_rnd_kit,
        nfi_rnd_dataset,
        monkeypatch,
    ):
        base = self._make_base_dataset(nfi_rnd_kit, nfi_rnd_dataset, n_images=4)
        ds = PeakWindowDataset(
            images=base.images,
            dataset_strategy=base.dataset_strategy,
            preprocess=False,
        )

        monkeypatch.setattr("dnanet.data.peak_dataset.get_worker_info", lambda: None)

        assert ds._worker_images() == base.images

    def test_worker_images_are_sharded_across_workers(
        self,
        nfi_rnd_kit,
        nfi_rnd_dataset,
        monkeypatch,
    ):
        base = self._make_base_dataset(nfi_rnd_kit, nfi_rnd_dataset, n_images=7)
        ds = PeakWindowDataset(
            images=base.images,
            dataset_strategy=base.dataset_strategy,
            preprocess=False,
        )

        seen = []
        for worker_id in range(3):
            monkeypatch.setattr(
                "dnanet.data.peak_dataset.get_worker_info",
                lambda worker_id=worker_id: SimpleNamespace(id=worker_id, num_workers=3),
            )
            worker_images = ds._worker_images()
            assert worker_images == base.images[worker_id::3]
            seen.extend(worker_images)

        counts = Counter(map(id, seen))
        assert len(seen) == len(base.images)
        assert set(counts) == {id(image) for image in base.images}
        assert all(count == 1 for count in counts.values())

    def test_subset_iteration_uses_worker_shard_before_transform(
        self,
        nfi_rnd_kit,
        nfi_rnd_dataset,
        monkeypatch,
    ):
        base = self._make_base_dataset(nfi_rnd_kit, nfi_rnd_dataset, n_images=5)
        ds = PeakWindowDataset(
            images=base.images,
            dataset_strategy=base.dataset_strategy,
            transform=lambda peak: f"transformed:{peak}",
            preprocess=False,
        )
        subset = ds.subset([1, 2, 4])

        monkeypatch.setattr(
            "dnanet.data.peak_dataset.get_worker_info",
            lambda: SimpleNamespace(id=1, num_workers=2),
        )

        extracted_from = []

        def fake_extract_peak_windows(image, **_kwargs):
            extracted_from.append(image.path.stem)
            return [image.path.stem]

        monkeypatch.setattr(
            "dnanet.data.peak_dataset.extract_peak_windows",
            fake_extract_peak_windows,
        )

        assert list(subset) == ["transformed:mock_2"]
        assert extracted_from == ["mock_2"]
