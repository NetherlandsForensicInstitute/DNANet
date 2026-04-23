"""Tests for PeakWindowDataset."""

import numpy as np

from dnanet.core.annotation import ScanpointAnnotation
from dnanet.data.peak_dataset import PeakWindowDataset
from dnanet.data.extracted_peak import ExtractedPeak


class MockImage:
    """Minimal HIDImage-like object."""

    def __init__(self, data, scaling_strategy, annotation_image=None):
        self._raw_data = data
        self._panel = None
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


class StubOnlyBaseDataset:
    """HIDDataset-like double with stub metadata and lazy image materialization."""

    def __init__(self, scaling_strategy, dataset_strategy):
        self._dataset_strategy = dataset_strategy
        self.transform = None
        self._scaling = scaling_strategy
        self.materialized_indices: list[int] = []
        self._stubs = []
        self._materialized = []

        for idx in range(2):
            stub = MockImage(
                data=np.zeros((5, 4096)),
                scaling_strategy=self._scaling,
                annotation_image=None,
                name=f"cached_{idx}",
            )
            stub.adjusted_panel = None
            self._stubs.append(stub)

            data, centers = _make_profile_with_peaks(n_peaks=1)
            ann = np.zeros_like(data)
            ann[0, centers[0] - 2 : centers[0] + 3] = 1
            image = MockImage(
                data,
                scaling_strategy=self._scaling,
                annotation_image=ann,
                name=f"cached_{idx}",
            )
            image.adjusted_panel = object()
            self._materialized.append(image)

    def __len__(self) -> int:
        return len(self._stubs)

    @property
    def images(self):
        return self._stubs

    @property
    def dataset_strategy(self):
        return self._dataset_strategy

    def get_stub_image(self, index: int):
        return self._stubs[index]

    def get_image(self, index: int):
        self.materialized_indices.append(index)
        return self._materialized[index]


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
        for _ in range(n_images):
            data, centers = _make_profile_with_peaks(n_peaks=n_peaks)
            ann = np.zeros_like(data)
            ann[0, centers[0] - 2 : centers[0] + 3] = 1
            images.append(MockImage(data, scaling_strategy, annotation_image=ann))
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

    def test_from_hid_dataset_defers_materialization_until_iteration(
        self,
        nfi_rnd_kit,
        nfi_rnd_dataset,
        monkeypatch,
    ):
        base = StubOnlyBaseDataset(nfi_rnd_kit, nfi_rnd_dataset)

        ds = PeakWindowDataset.from_hid_dataset(
            base,
            threshold=100,
            window_size=120,
            preprocess=False,
        )

        assert [image.path.stem for image in ds.images] == ["cached_0", "cached_1"]
        assert base.materialized_indices == []

        monkeypatch.setattr(
            "dnanet.data.peak_dataset.get_worker_info",
            lambda: SimpleNamespace(id=1, num_workers=2),
        )

        extracted_from = []

        def fake_extract_peak_windows(image, **_kwargs):
            assert image.annotation is not None
            assert image.adjusted_panel is not None
            extracted_from.append(image.path.stem)
            return [image.path.stem]

        monkeypatch.setattr(
            "dnanet.data.peak_dataset.extract_peak_windows",
            fake_extract_peak_windows,
        )

        assert list(ds) == ["cached_1"]
        assert extracted_from == ["cached_1"]
        assert base.materialized_indices == [1]

    def test_load_in_memory_materializes_all_peaks_once(
        self,
        nfi_rnd_kit,
        nfi_rnd_dataset,
        monkeypatch,
    ):
        base = StubOnlyBaseDataset(nfi_rnd_kit, nfi_rnd_dataset)
        extracted_from = []

        def fake_extract_peak_windows(image, **_kwargs):
            extracted_from.append(image.path.stem)
            return [image.path.stem]

        monkeypatch.setattr(
            "dnanet.data.peak_dataset.extract_peak_windows",
            fake_extract_peak_windows,
        )

        ds = PeakWindowDataset.from_hid_dataset(
            base,
            threshold=100,
            window_size=120,
            preprocess=False,
            load_in_memory=True,
        )

        assert extracted_from == ["cached_0", "cached_1"]
        assert base.materialized_indices == [0, 1]

        monkeypatch.setattr(
            "dnanet.data.peak_dataset.get_worker_info",
            lambda: SimpleNamespace(id=1, num_workers=2),
        )

        assert list(ds) == ["cached_1"]
        assert extracted_from == ["cached_0", "cached_1"]
        assert base.materialized_indices == [0, 1]

    def test_load_in_memory_preprocesses_during_materialization(
        self,
        nfi_rnd_kit,
        nfi_rnd_dataset,
        monkeypatch,
    ):
        base = self._make_base_dataset(nfi_rnd_kit, nfi_rnd_dataset, n_images=2, n_peaks=2)
        preprocess_calls = []
        original_preprocess_peak = PeakWindowDataset._preprocess_peak

        def counting_preprocess_peak(self, peak):
            preprocess_calls.append((peak.dye_index, peak.peak_center))
            original_preprocess_peak(self, peak)

        monkeypatch.setattr(
            PeakWindowDataset,
            "_preprocess_peak",
            counting_preprocess_peak,
        )

        ds = PeakWindowDataset(
            images=base.images,
            dataset_strategy=base.dataset_strategy,
            threshold=100,
            window_size=120,
            preprocess=True,
            log_scale=True,
            load_in_memory=True,
        )

        cached_peak_count = sum(len(peak_list) for peak_list in ds._cached_peak_lists)
        assert len(preprocess_calls) == cached_peak_count

        preprocess_calls.clear()
        list(ds)
        assert preprocess_calls == []

    def test_subset_reuses_eager_peak_cache(
        self,
        nfi_rnd_kit,
        nfi_rnd_dataset,
        monkeypatch,
    ):
        base = StubOnlyBaseDataset(nfi_rnd_kit, nfi_rnd_dataset)
        extracted_from = []

        def fake_extract_peak_windows(image, **_kwargs):
            extracted_from.append(image.path.stem)
            return [image.path.stem]

        monkeypatch.setattr(
            "dnanet.data.peak_dataset.extract_peak_windows",
            fake_extract_peak_windows,
        )

        ds = PeakWindowDataset.from_hid_dataset(
            base,
            threshold=100,
            window_size=120,
            preprocess=False,
            load_in_memory=True,
        )

        subset = ds.subset([1])

        assert list(subset) == ["cached_1"]
        assert extracted_from == ["cached_0", "cached_1"]
        assert base.materialized_indices == [0, 1]

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
            assert peak.data.shape == (120,)

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
