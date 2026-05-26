"""Tests for HIDDataset — file discovery, annotation mapping, and loading."""

import pytest

from tests.conftest import RD_DIR
from dnanet.data.strategies import NFIRnDStrategy, PowerPlexFusion6CStrategy
from dnanet.data.hid_dataset import HIDDataset


def _make_hid_dataset(**kwargs) -> HIDDataset:
    """Build an HIDDataset using the current constructor contract."""
    return HIDDataset(
        root=RD_DIR,
        cache_dir='/tmp/var/hiddataset-tests/',
        scaling_strategy=PowerPlexFusion6CStrategy(),
        dataset_strategy=NFIRnDStrategy('DTH'),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Ladder path loading (static method)
# ---------------------------------------------------------------------------

# class TestLadderPathLoading:
#     def test_loads_real_csv(self, nfi_rnd_kit):
#         mapping = HIDDataset._load_ladder_paths(RD_DIR / "best_ladder_paths.csv")
#         assert len(mapping) > 0

#     def test_empty_rows_skipped(self, nfi_rnd_kit, tmp_path):
#         csv_path = tmp_path / "ladders.csv"
#         csv_path.write_text("image_path,ladder_path\n,\nsample1,ladder1\n")
#         mapping = HIDDataset._load_ladder_paths(csv_path)
#         assert len(mapping) == 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestHIDDatasetValidation:
    def test_invalid_adjustment_raises(self, nfi_rnd_kit):
        with pytest.raises(ValueError, match='adjustment_of_annotations'):
            _make_hid_dataset(adjustment_of_annotations='bad')

    def test_empty_root_raises(self, nfi_rnd_kit, tmp_path):
        empty_dir = tmp_path / 'empty'
        empty_dir.mkdir()
        with pytest.raises(ValueError, match='Path does not contain the necessary ladder mapping'):
            HIDDataset(
                root=empty_dir,
                scaling_strategy=PowerPlexFusion6CStrategy(),
                dataset_strategy=NFIRnDStrategy('DTH'),
                cache_dir='/tmp/var/hiddataset-tests/',
            )


# ---------------------------------------------------------------------------
# Integration: loading from test resources
# ---------------------------------------------------------------------------


class TestHIDDatasetIntegration:
    def test_load_from_rd_dir(self, nfi_rnd_kit):
        """Load from test resources — should find 2 sample HID files."""
        ds = _make_hid_dataset()
        assert len(ds) == 2

    def test_images_have_correct_shape(self, nfi_rnd_kit):
        ds = _make_hid_dataset()
        for img in ds:
            assert img.data is not None
            assert img.data.shape == (5, 4096)

    def test_limit_parameter(self, nfi_rnd_kit):
        ds = _make_hid_dataset(limit=1)
        assert len(ds) == 1

    def test_repr(self, nfi_rnd_kit):
        ds = _make_hid_dataset()
        r = repr(ds)
        assert 'HIDDataset' in r
        assert 'RD' in r

    def test_indexing_returns_loaded_image(self, nfi_rnd_kit):
        ds = _make_hid_dataset()
        img = ds[0]

        assert img.data is not None
        assert img.data.shape == (5, 4096)

    def test_annotations_populated(self, nfi_rnd_kit):
        """Images with annotation mapping should have non-None annotations."""
        ds = _make_hid_dataset()
        annotated = [img for img in ds if img.annotation is not None]
        assert len(annotated) > 0, 'Expected at least one image with annotation'

    def test_adjustment_top(self, nfi_rnd_kit):
        """Loading with annotation adjustment should not crash."""
        ds = _make_hid_dataset(adjustment_of_annotations='top')
        assert len(ds) >= 1
