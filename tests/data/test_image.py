"""Tests for HIDImage — lazy loading, segmentation masks, annotation adjustment."""

import numpy as np
import pytest

from tests.conftest import RD_DIR
from dnanet.data.image import HIDImage
from dnanet.core.allele import Allele
from dnanet.core.marker import Marker
from dnanet.core.annotation import Annotation
from dnanet.data.strategies.registry import StrategyRegistry


# ---------------------------------------------------------------------------
# Init / defaults
# ---------------------------------------------------------------------------

class TestHIDImageInit:
    def test_default_attributes(self):
        img = HIDImage(path="fake.hid")
        assert img.path.name == "fake.hid"
        assert img.use_cache is True
        assert img.data_loading_strategy == "superior"
        assert img.rfu_threshold == 40
        assert img._data is None

    def test_meta_defaults_to_empty_dict(self):
        img = HIDImage(path="fake.hid")
        assert img.meta == {}

    def test_custom_meta_preserved(self):
        img = HIDImage(path="fake.hid", meta={"noc": "2p", "extra": 42})
        assert img.meta["noc"] == "2p"
        assert img.meta["extra"] == 42

    def test_include_size_standard_default(self):
        img = HIDImage(path="fake.hid")
        assert img.include_size_standard is False

    def test_repr_contains_filename(self):
        img = HIDImage(path="/some/path/sample.hid")
        assert "sample.hid" in repr(img)


# ---------------------------------------------------------------------------
# Lazy loading with real data
# ---------------------------------------------------------------------------

class TestHIDImageLazyLoading:
    @pytest.fixture(autouse=True)
    def setup_kit(self):
        StrategyRegistry.configure_kit("PPF6C")
        yield
        StrategyRegistry.reset()

    @pytest.fixture
    def sample_path(self):
        return RD_DIR / "1A2_A01_01.hid"

    def test_data_not_loaded_on_init(self, sample_path):
        img = HIDImage(path=sample_path)
        assert img._data is None

    def test_data_loaded_on_first_access(self, sample_path):
        img = HIDImage(path=sample_path)
        data = img.data
        assert data is not None
        assert data.shape == (5, 4096, 1)

    def test_data_cached_after_first_load(self, sample_path):
        img = HIDImage(path=sample_path)
        d1 = img.data
        d2 = img.data
        assert d1 is d2  # same object

    def test_data_not_cached_when_disabled(self, sample_path):
        img = HIDImage(path=sample_path, use_cache=False)
        _ = img.data
        assert img._data is None  # not stored

    def test_dimensions_property(self, sample_path):
        img = HIDImage(path=sample_path)
        assert img.dimensions == (5, 4096)

    def test_scaler_shape(self, sample_path):
        img = HIDImage(path=sample_path)
        assert img.scaler.shape == (1, 4096)

    def test_scaler_monotonic_nondecreasing(self, sample_path):
        img = HIDImage(path=sample_path)
        scaler = img.scaler.flatten()
        nonzero = scaler[scaler > 0]
        assert np.all(np.diff(nonzero) >= 0)

    def test_file_not_found_raises(self):
        img = HIDImage(path="/nonexistent/path/fake.hid")
        with pytest.raises(FileNotFoundError):
            _ = img.data

    def test_data_loading_strategy_raw(self, sample_path):
        img = HIDImage(path=sample_path, data_loading_strategy="raw")
        assert img.data is not None
        assert img.data.shape == (5, 4096, 1)


# ---------------------------------------------------------------------------
# _build_segmentation (pure numpy, no I/O)
# ---------------------------------------------------------------------------

class TestBuildSegmentation:
    def test_simple_mask(self):
        # Scaler: linear from 60 to 480 over 4096 points
        scaler = np.linspace(60, 480, 4096)[np.newaxis, :]
        marker = Marker(
            name="D3S1358",
            dye_row=0,
            alleles=(
                Allele(name="12", base_pair=120.0, left_bin=2.0, right_bin=2.0),
            ),
        )
        shape = (5, 4096, 1)
        mask = HIDImage._build_segmentation(scaler, [marker], shape)
        assert mask.shape == shape
        # Dye 0 should have 1s around bp=120
        assert mask[0, :, 0].sum() > 0
        # Other dyes should be 0
        assert mask[1:, :, 0].sum() == 0

    def test_multiple_dye_rows(self):
        scaler = np.linspace(60, 480, 4096)[np.newaxis, :]
        m1 = Marker(
            name="D3S1358", dye_row=0,
            alleles=(Allele(name="12", base_pair=120.0, left_bin=2.0, right_bin=2.0),),
        )
        m2 = Marker(
            name="TH01", dye_row=2,
            alleles=(Allele(name="9", base_pair=200.0, left_bin=2.0, right_bin=2.0),),
        )
        shape = (5, 4096, 1)
        mask = HIDImage._build_segmentation(scaler, [m1, m2], shape)
        assert mask[0, :, 0].sum() > 0
        assert mask[2, :, 0].sum() > 0
        assert mask[1, :, 0].sum() == 0

    def test_skips_alleles_without_bp(self):
        scaler = np.linspace(60, 480, 4096)[np.newaxis, :]
        marker = Marker(
            name="D3S1358", dye_row=0,
            alleles=(Allele(name="OL", base_pair=None, left_bin=0.4, right_bin=0.4),),
        )
        shape = (5, 4096, 1)
        mask = HIDImage._build_segmentation(scaler, [marker], shape)
        assert mask.sum() == 0

    def test_mask_dtype(self):
        scaler = np.linspace(60, 480, 4096)[np.newaxis, :]
        marker = Marker(
            name="D3S1358", dye_row=0,
            alleles=(Allele(name="12", base_pair=120.0, left_bin=2.0, right_bin=2.0),),
        )
        mask = HIDImage._build_segmentation(scaler, [marker], (5, 4096, 1))
        assert mask.dtype == np.int8


# ---------------------------------------------------------------------------
# Annotation adjustment
# ---------------------------------------------------------------------------

# TODO: Do the adjustment with the new appropriate translation
class TestAnnotationAdjustment:
    @pytest.fixture(autouse=True)
    def setup_kit(self):
        StrategyRegistry.configure_kit("PPF6C")
        yield
        StrategyRegistry.reset()

    def test_no_annotation_returns_self(self):
        img = HIDImage(path="fake.hid")
        img._data = np.random.rand(5, 100, 1).astype(np.float32)
        result = img.adjust_annotations("top")
        assert result is img

    def test_invalid_method_raises(self):
        img = HIDImage(path="fake.hid")
        # Create data with a clear peak in the annotated region
        data = np.zeros((5, 100, 1), dtype=np.float32)
        data[0, 40:50, 0] = np.array([10, 30, 60, 100, 200, 200, 100, 60, 30, 10], dtype=np.float32)
        img._data = data
        mask = np.zeros((5, 100, 1), dtype=np.int8)
        mask[0, 40:50, 0] = 1
        img._annotation = Annotation(image=mask)

        with pytest.raises(ValueError, match="Unknown adjustment method"):
            img.adjust_annotations("bad_method")

    def test_adjust_top_with_real_data(self, nfi_rnd_kit):
        """Load real image with annotation, adjust with 'top' method."""
        path = RD_DIR / "1A2_A01_01.hid"
        panel_path = "resources/kit_panels/SGPanel_PPF6C.xml"
        from dnanet.core.panel import Panel
        panel = Panel.from_xml(panel_path)
        ann_path = RD_DIR / "Dataset 1 DTH_AlleleReport.txt"

        img = HIDImage(
            path=path,
            adjusted_panel=panel,
            annotations_file=ann_path,
            meta={"annotations_name": "1_11148_1A2"},
        )
        # Trigger load
        assert img.data is not None
        if img.annotation is not None:
            original_sum = img.annotation.image.sum()
            result = img.adjust_annotations("top")
            assert result is img
            # After top adjustment, the mask should be sparser (only peak tops)
            assert img.annotation.image.sum() <= original_sum

    def test_adjust_complete_with_real_data(self, nfi_rnd_kit):
        """Load real image with annotation, adjust with 'complete' method."""
        path = RD_DIR / "1A2_A01_01.hid"
        from dnanet.core.panel import Panel
        panel = Panel.from_xml("resources/kit_panels/SGPanel_PPF6C.xml")
        ann_path = RD_DIR / "Dataset 1 DTH_AlleleReport.txt"

        img = HIDImage(
            path=path,
            adjusted_panel=panel,
            annotations_file=ann_path,
            meta={"annotations_name": "1_11148_1A2"},
        )
        assert img.data is not None
        if img.annotation is not None:
            result = img.adjust_annotations("complete")
            assert result is img
            # Complete should have broader regions than top
            assert img.annotation.image.sum() > 0
