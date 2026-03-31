"""Tests for Arrow-based caching of HIDImage datasets."""

import json

import numpy as np
import pytest

from dnanet.core.panel import Panel
from dnanet.data.image import HIDImage
from dnanet.core.allele import Allele
from dnanet.core.marker import Marker
from dnanet.data.caching import write_to_cache, load_from_cache, _reconstruct_image
from dnanet.core.annotation import ScanpointAnnotation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cached_image(
    name: str = "test.hid",
    num_dyes: int = 5,
    signal_length: int = 100,
) -> HIDImage:
    """Build an HIDImage with all fields needed for caching."""
    adjusted_panel = Panel(markers=[
        Marker(name="D3S1358", dye_row=0, alleles=frozenset(
            [Allele(name="12", base_pair=120.0, left_bin=0.4, right_bin=0.4),]
        )),
    ])
    called = [
        Marker(name="D3S1358", dye_row=0, alleles=frozenset(
            [Allele(name="12", base_pair=120.0, left_bin=0.4, right_bin=0.4, height=1500.0),]
        )),
    ]
    mask = np.zeros((num_dyes, signal_length, 1), dtype=np.int8)
    mask[0, 10:20, 0] = 1

    img = HIDImage(path=name, adjusted_panel=adjusted_panel, load_in_memory=True)
    img._data = (np.random.rand(num_dyes, signal_length, 1) * 1000).astype(np.int16)
    img._scaler = np.linspace(60, 480, signal_length)
    img._annotation = ScanpointAnnotation(data=mask)
    img._meta["called_alleles"] = called
    return img


@pytest.fixture
def cached_images():
    return [_make_cached_image(f"sample_{i}.hid") for i in range(5)]


# ---------------------------------------------------------------------------
# write_to_cache
# ---------------------------------------------------------------------------

class TestWriteToCache:
    def test_writes_arrow_files(self, cached_images, tmp_path):
        write_to_cache(tmp_path / "cache", cached_images)
        arrow_files = list((tmp_path / "cache").rglob("*.arrow"))
        assert len(arrow_files) > 0

    def test_empty_images_returns(self, tmp_path):
        """Empty list should return without error."""
        write_to_cache(tmp_path / "cache", [])
        # Directory may or may not be created

    def test_first_image_no_data_raises(self, tmp_path):
        img = HIDImage(path="bad.hid", load_in_memory=True)
        img._data = None  # Explicitly set to None (bypass lazy load)
        # Patch the data property to return None without trying to load
        with pytest.raises((ValueError, FileNotFoundError)):
            write_to_cache(tmp_path / "cache", [img])


# ---------------------------------------------------------------------------
# load_from_cache (roundtrip)
# ---------------------------------------------------------------------------

class TestLoadFromCache:
    def test_roundtrip_image_count(self, cached_images, tmp_path):
        cache = tmp_path / "cache"
        write_to_cache(cache, cached_images)
        loaded = load_from_cache(cache)
        assert len(loaded) == len(cached_images)

    def test_roundtrip_data_preserved(self, cached_images, tmp_path):
        cache = tmp_path / "cache"
        write_to_cache(cache, cached_images)
        loaded = load_from_cache(cache)
        np.testing.assert_array_equal(loaded[0].data, cached_images[0].data)

    def test_roundtrip_annotation_preserved(self, cached_images, tmp_path):
        cache = tmp_path / "cache"
        write_to_cache(cache, cached_images)
        loaded = load_from_cache(cache)
        np.testing.assert_array_equal(
            loaded[0].annotation.data, cached_images[0].annotation.data
        )

    def test_roundtrip_scaler_preserved(self, cached_images, tmp_path):
        cache = tmp_path / "cache"
        write_to_cache(cache, cached_images)
        loaded = load_from_cache(cache)
        np.testing.assert_array_almost_equal(
            loaded[0]._scaler, cached_images[0]._scaler
        )

    def test_roundtrip_panel_preserved(self, cached_images, tmp_path):
        cache = tmp_path / "cache"
        write_to_cache(cache, cached_images)
        loaded = load_from_cache(cache)
        assert loaded[0].adjusted_panel is not None
        assert loaded[0].adjusted_panel.markers[0].name == "D3S1358"

    def test_roundtrip_called_alleles_preserved(self, cached_images, tmp_path):
        cache = tmp_path / "cache"
        write_to_cache(cache, cached_images)
        loaded = load_from_cache(cache)
        called = loaded[0].meta.get("called_alleles", [])
        assert len(called) == 1
        assert called[0].name == "D3S1358"

    def test_limit_parameter(self, cached_images, tmp_path):
        cache = tmp_path / "cache"
        write_to_cache(cache, cached_images)
        loaded = load_from_cache(cache, limit=2)
        assert len(loaded) == 2

    def test_hid_path_preserved(self, cached_images, tmp_path):
        cache = tmp_path / "cache"
        write_to_cache(cache, cached_images)
        loaded = load_from_cache(cache)
        assert loaded[0].path == cached_images[0].path


# ---------------------------------------------------------------------------
# _reconstruct_image
# ---------------------------------------------------------------------------

class TestReconstructImage:
    def test_basic_reconstruction(self):
        img_data = np.zeros((5, 100, 1), dtype=np.int16)
        ann_data = np.zeros((5, 100, 1), dtype=np.int8)
        scaler = np.linspace(60, 480, 100).tolist()
        panel_json = json.dumps([{
            "name": "D3S1358", "dye_row": 0,
            "alleles": [{"name": "12", "base_pair": 120.0, "left_bin": 0.4, "right_bin": 0.4}],
        }])
        alleles_json = "[]"

        img = _reconstruct_image(img_data, ann_data, "test.hid", scaler, panel_json, alleles_json, False)
        assert isinstance(img, HIDImage)
        assert img.data is not None
        assert img.adjusted_panel is not None

    def test_empty_panel_json(self):
        img = _reconstruct_image(
            np.zeros((5, 100, 1), dtype=np.int16),
            np.zeros((5, 100, 1), dtype=np.int8),
            "test.hid",
            np.linspace(60, 480, 100).tolist(),
            "[]",
            "[]",
            False,
        )
        assert img.adjusted_panel is None

    def test_empty_alleles_json(self):
        img = _reconstruct_image(
            np.zeros((5, 100, 1), dtype=np.int16),
            np.zeros((5, 100, 1), dtype=np.int8),
            "test.hid",
            np.linspace(60, 480, 100).tolist(),
            "[]",
            "[]",
            False,
        )
        assert img.meta.get("called_alleles") is None
