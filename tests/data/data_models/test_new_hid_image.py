# import numpy as np
# import pytest

# from DNAnet.data.data_models.old_deprecated_hid_image import OldHIDImage
# from DNAnet.data.data_models.hid_image import HIDImage
# from DNAnet.data.data_models import Panel


# @pytest.fixture
# def legacy_hid_image(hid_dataset_rd):
#     """Legacy (old) HIDImage to compare against the new implementation."""
#     panel = Panel(pytest.PANEL_PATH)
#     return OldHIDImage(
#         path=pytest.RESOURCES_DIR / "profiles" / "RD" / "1A2_A01_01.hid",
#         annotations_file=pytest.RESOURCES_DIR / "profiles" / "RD" / "Dataset 1 DTH_AlleleReport.txt",
#         panel=panel,
#         meta={"annotations_name": "1_11148_1A2"},
#     )


# def test_hid_matches_old_data_and_annotations():
#     """Same legacy-style args should yield identical data/annotations/meta."""
#     panel = Panel(pytest.PANEL_PATH)
#     shared_kwargs = dict(
#         path=pytest.RESOURCES_DIR / "profiles" / "RD" / "1A2_A01_01.hid",
#         annotations_file=pytest.RESOURCES_DIR / "profiles" / "RD" / "Dataset 1 DTH_AlleleReport.txt",
#         panel=panel,
#         meta={"annotations_name": "1_11148_1A2"},
#     )

#     legacy = OldHIDImage(**shared_kwargs)
#     current = HIDImage(**shared_kwargs)

#     np.testing.assert_array_equal(current.data, legacy.data)
#     np.testing.assert_array_equal(current.annotation.image, legacy.annotation.image)
#     assert current.meta["called_alleles"] == legacy.meta["called_alleles"]
#     np.testing.assert_allclose(current.scaler, legacy.scaler)
#     assert current.dimensions == legacy.dimensions
#     assert current.hash == legacy.hash


# def test_backwards_compatible_constructor():
#     """
#     Construct HIDImage using the same arguments as OldHIDImage and assert parity.
#     """
#     panel = Panel(pytest.PANEL_PATH)

#     shared_kwargs = dict(
#         path=pytest.RESOURCES_DIR / "profiles" / "RD" / "1A2_A01_01.hid",
#         annotations_file=pytest.RESOURCES_DIR / "profiles" / "RD" / "Dataset 1 DTH_AlleleReport.txt",
#         panel=panel,
#         meta={"annotations_name": "1_11148_1A2"},
#     )

#     legacy = OldHIDImage(**shared_kwargs)
#     current = HIDImage(**shared_kwargs)

#     np.testing.assert_array_equal(current.data, legacy.data)
#     np.testing.assert_array_equal(current.annotation.image, legacy.annotation.image)
#     assert current.meta["called_alleles"] == legacy.meta["called_alleles"]


# def test_hid_adjust_annotations_matches_old(legacy_hid_image):
#     """Annotation adjustments should align with legacy behavior."""
#     panel = Panel(pytest.PANEL_PATH)

#     legacy = OldHIDImage(
#         path=legacy_hid_image.path,
#         annotations_file=legacy_hid_image.annotations_file,
#         panel=panel,
#         meta=dict(legacy_hid_image.meta),
#     )
#     current = HIDImage(
#         path=legacy_hid_image.path,
#         annotations_file=legacy_hid_image.annotations_file,
#         panel=panel,
#         meta=dict(legacy_hid_image.meta),
#     )

#     legacy_top = legacy.adjust_annotations("top").annotation.image.copy()
#     current_top = current.adjust_annotations("top").annotation.image.copy()
#     np.testing.assert_array_equal(current_top, legacy_top)

#     legacy_complete = legacy.adjust_annotations("complete").annotation.image.copy()
#     current_complete = current.adjust_annotations("complete").annotation.image.copy()
#     np.testing.assert_array_equal(current_complete, legacy_complete)
