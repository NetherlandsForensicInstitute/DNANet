"""Tests for HIDImage — lazy loading, segmentation masks, annotation adjustment."""

import numpy as np
import pytest

from dnanet.data.image import HIDImage
from dnanet.data.strategies import PowerPlexFusion6CStrategy
from tests.conftest import RD_DIR


@pytest.fixture
def ppf6c():
    return PowerPlexFusion6CStrategy()


# ---------------------------------------------------------------------------
# Init / defaults
# ---------------------------------------------------------------------------

class TestHIDImageInit:
    def test_default_attributes(self, ppf6c, nfi_rnd_dataset):
        img = HIDImage(path="fake.hid", scaling_strategy=ppf6c, dataset_strategy=nfi_rnd_dataset)
        assert img.path.name == "fake.hid"
        assert img.load_in_memory is True
        assert img.data_loading_strategy == "superior"
        assert img.rfu_threshold == 40
        assert img._data is None

    def test_meta_defaults_to_empty_dict(self, ppf6c, nfi_rnd_dataset):
        img = HIDImage(path="fake.hid", scaling_strategy=ppf6c, dataset_strategy=nfi_rnd_dataset)
        assert img.meta == {}

    def test_custom_meta_preserved(self, ppf6c, nfi_rnd_dataset):
        img = HIDImage(path="fake.hid", scaling_strategy=ppf6c, dataset_strategy=nfi_rnd_dataset, meta={"noc": "2p", "extra": 42})
        assert img.meta["noc"] == "2p"
        assert img.meta["extra"] == 42

    def test_include_size_standard_default(self, ppf6c, nfi_rnd_dataset):
        img = HIDImage(path="fake.hid", scaling_strategy=ppf6c, dataset_strategy=nfi_rnd_dataset)
        assert img.include_size_standard is False

    def test_repr_contains_filename(self, ppf6c, nfi_rnd_dataset):
        img = HIDImage(path="/some/path/sample.hid", scaling_strategy=ppf6c, dataset_strategy=nfi_rnd_dataset)
        assert "sample.hid" in repr(img)


# ---------------------------------------------------------------------------
# Lazy loading with real data
# ---------------------------------------------------------------------------

class TestHIDImageLazyLoading:
    @pytest.fixture
    def sample_path(self):
        return RD_DIR / "1A2_A01_01.hid"

    def test_data_not_loaded_on_init(self, sample_path, ppf6c, nfi_rnd_dataset):
        img = HIDImage(path=sample_path, scaling_strategy=ppf6c, dataset_strategy=nfi_rnd_dataset)
        assert img._data is None

    def test_data_loaded_on_first_access(self, sample_path, ppf6c, nfi_rnd_dataset):
        img = HIDImage(path=sample_path, scaling_strategy=ppf6c, dataset_strategy=nfi_rnd_dataset)
        data = img.data
        assert data is not None
        assert data.shape == (5, 4096)

    def test_data_cached_after_first_load(self, sample_path, ppf6c, nfi_rnd_dataset):
        img = HIDImage(path=sample_path, scaling_strategy=ppf6c, dataset_strategy=nfi_rnd_dataset)
        d1 = img.data
        d2 = img.data
        assert d1 is d2  # same object

    def test_data_not_cached_when_disabled(self, sample_path, ppf6c, nfi_rnd_dataset):
        img = HIDImage(path=sample_path, scaling_strategy=ppf6c, load_in_memory=False, dataset_strategy=nfi_rnd_dataset)
        _ = img.data
        assert img._data is None  # not stored

    def test_dimensions_property(self, sample_path, ppf6c, nfi_rnd_dataset):
        img = HIDImage(path=sample_path, scaling_strategy=ppf6c, dataset_strategy=nfi_rnd_dataset)
        assert img.dimensions == (5, 4096)

    def test_scaler_shape(self, sample_path, ppf6c,nfi_rnd_dataset):
        img = HIDImage(path=sample_path, scaling_strategy=ppf6c, dataset_strategy=nfi_rnd_dataset)
        assert img.scaler.shape == (4096,)

    def test_scaler_monotonic_nondecreasing(self, sample_path, ppf6c, nfi_rnd_dataset):
        img = HIDImage(path=sample_path, scaling_strategy=ppf6c, dataset_strategy=nfi_rnd_dataset)
        scaler = img.scaler.flatten()
        nonzero = scaler[scaler > 0]
        assert np.all(np.diff(nonzero) >= 0)

    def test_file_not_found_raises(self, ppf6c, nfi_rnd_dataset):
        img = HIDImage(path="/nonexistent/path/fake.hid", scaling_strategy=ppf6c, dataset_strategy=nfi_rnd_dataset)
        with pytest.raises(FileNotFoundError):
            _ = img.data

    def test_data_loading_strategy_raw(self, sample_path, ppf6c, nfi_rnd_dataset):
        img = HIDImage(path=sample_path, scaling_strategy=ppf6c, data_loading_strategy="raw", dataset_strategy=nfi_rnd_dataset)
        assert img.data is not None
        assert img.data.shape == (5, 4096)
