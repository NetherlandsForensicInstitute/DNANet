"""Tests for the Annotation value object."""

import numpy as np
import pytest

from dnanet.core import Annotation


class TestAnnotation:
    """Test suite for Annotation."""

    def test_from_labels_single(self):
        ann = Annotation.from_labels("Allele")
        assert ann.label == "Allele"

    def test_from_labels_multiple(self):
        ann = Annotation.from_labels(["Allele", "Stutter"])
        assert ann.labels == frozenset({"Allele", "Stutter"})

    def test_single_label_fails_with_multiple(self):
        ann = Annotation.from_labels(["Allele", "Stutter"])
        with pytest.raises(ValueError, match="Multiple labels"):
            _ = ann.label

    def test_single_label_fails_with_none(self):
        ann = Annotation()
        with pytest.raises(ValueError, match="No labels"):
            _ = ann.label

    def test_from_mask(self, segmentation_mask):
        ann = Annotation.from_mask(segmentation_mask, annotator="alice")
        assert ann.image is not None
        assert ann.meta["annotator"] == "alice"

    def test_equality_with_image(self):
        img = np.ones((5, 100), dtype=np.float32)
        a = Annotation(image=img)
        b = Annotation(image=img.copy())
        assert a == b

    def test_immutability(self):
        ann = Annotation.from_labels("Allele")
        with pytest.raises(AttributeError):
            ann.labels = frozenset({"Stutter"})
