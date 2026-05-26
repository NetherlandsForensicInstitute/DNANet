"""Tests for domain constants."""

from dnanet.core.constants import LabelCategory


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
