import numpy as np
import pytest

from DNAnet.data.data_models.old_hid_image import OldHIDImage as HIDImage
from DNAnet.data.data_models.hid_image import HIDImage as ProvedItHIDImage
from DNAnet.data.data_models import Panel


@pytest.fixture
def legacy_hid_image(hid_dataset_rd):
    """One legacy HIDImage from the RD fixture to compare against."""
    return hid_dataset_rd[0]


def test_provedit_matches_legacy_data_and_annotations():
    """Same legacy-style args should yield identical data/annotations/meta."""
    panel = Panel(pytest.PANEL_PATH)
    shared_kwargs = dict(
        path=pytest.RESOURCES_DIR / "profiles" / "RD" / "1A2_A01_01.hid",
        annotations_file=pytest.RESOURCES_DIR / "profiles" / "RD" / "Dataset 1 DTH_AlleleReport.txt",
        panel=panel,
        meta={"annotations_name": "1_11148_1A2"},
    )

    legacy = HIDImage(**shared_kwargs)
    proved = ProvedItHIDImage(**shared_kwargs)

    np.testing.assert_array_equal(proved.data, legacy.data)
    np.testing.assert_array_equal(proved.annotation.image, legacy.annotation.image)
    assert proved.meta["called_alleles"] == legacy.meta["called_alleles"]
    np.testing.assert_allclose(proved.scaler, legacy.scaler)
    assert proved.dimensions == legacy.dimensions
    assert proved.hash == legacy.hash


def test_backwards_compatible_constructor():
    """
    Construct ProvedItHIDImage using the same arguments as HIDImage (plus the
    required dataset_strategy) and assert parity.
    """
    panel = Panel(pytest.PANEL_PATH)

    shared_kwargs = dict(
        path=pytest.RESOURCES_DIR / "profiles" / "RD" / "1A2_A01_01.hid",
        annotations_file=pytest.RESOURCES_DIR / "profiles" / "RD" / "Dataset 1 DTH_AlleleReport.txt",
        panel=panel,
        meta={"annotations_name": "1_11148_1A2"},
    )

    legacy = HIDImage(**shared_kwargs)
    proved = ProvedItHIDImage(**shared_kwargs)

    np.testing.assert_array_equal(proved.data, legacy.data)
    np.testing.assert_array_equal(proved.annotation.image, legacy.annotation.image)
    assert proved.meta["called_alleles"] == legacy.meta["called_alleles"]


def test_provedit_adjust_annotations_matches_legacy(legacy_hid_image):
    """Annotation adjustments should align with legacy behavior."""
    panel = Panel(pytest.PANEL_PATH)

    legacy = HIDImage(
        path=legacy_hid_image.path,
        annotations_file=legacy_hid_image.annotations_file,
        panel=panel,
        meta=dict(legacy_hid_image.meta),
    )
    proved = ProvedItHIDImage(
        path=legacy_hid_image.path,
        annotations_file=legacy_hid_image.annotations_file,
        panel=panel,
        meta=dict(legacy_hid_image.meta),
    )

    legacy_top = legacy.adjust_annotations("top").annotation.image.copy()
    proved_top = proved.adjust_annotations("top").annotation.image.copy()
    np.testing.assert_array_equal(proved_top, legacy_top)

    legacy_complete = legacy.adjust_annotations("complete").annotation.image.copy()
    proved_complete = proved.adjust_annotations("complete").annotation.image.copy()
    np.testing.assert_array_equal(proved_complete, legacy_complete)
