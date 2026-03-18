"""Tests for domain constants."""

import pytest

from dnanet.core.constants import DyeIndex, LabelCategory


class TestLabelCategory:
    """Test suite for the LabelCategory enum."""

    def test_all_categories_have_unique_shortcuts(self):
        shortcuts = [cat.shortcut for cat in LabelCategory]
        assert len(shortcuts) == len(set(shortcuts))

    def test_label_names_match_original(self):
        """Ensure backward compatibility with the original LABEL_CATEGORIES_STR."""
        expected = [
            "", "Allele", "Stutter", "PullUp", "BleedThrough", "Spike",
            "DyeBlob", "Artefact", "Unclear", "Shoulder", "ForeignDna",
            "OverloadingArtefact",
        ]
        actual = LabelCategory.label_names()
        assert len(actual) == len(expected)
        # Check that non-empty names are present (case-insensitive)
        for name in expected:
            if name:
                assert name.lower() in [a.lower() for a in actual]

    def test_from_index(self):
        assert LabelCategory.from_index(0) is LabelCategory.UNLABELED
        assert LabelCategory.from_index(1) is LabelCategory.ALLELE

    def test_unlabeled_has_empty_name(self):
        assert LabelCategory.UNLABELED.label_name == ""


class TestDyeIndex:
    """Test suite for the DyeIndex enum."""

    def test_from_hid_index_valid(self):
        assert DyeIndex.from_hid_index(1) == DyeIndex.BLUE
        assert DyeIndex.from_hid_index(6) == DyeIndex.ORANGE

    def test_from_hid_index_invalid(self):
        with pytest.raises(ValueError, match="Unknown HID dye index"):
            DyeIndex.from_hid_index(5)

    def test_values_are_zero_based(self):
        assert DyeIndex.BLUE == 0
        assert DyeIndex.ORANGE == 4
