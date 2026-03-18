"""Tests for the Prediction value object."""

import numpy as np
import pytest

from dnanet.core import Allele, Marker, Prediction


class TestPrediction:
    """Test suite for Prediction."""

    def test_for_segmentation(self, segmentation_mask):
        pred = Prediction.for_segmentation(
            image=segmentation_mask,
            source_path="/tmp/test.hid",
        )
        assert pred.image is not None
        assert pred.original_image_path.name == "test.hid"

    def test_for_classification(self):
        pred = Prediction.for_classification(
            scores={"Allele": 0.9, "Stutter": 0.1},
            source_path="/tmp/test.hid",
        )
        assert pred.label == "Allele"
        assert pred.confidence == 0.9

    def test_label_without_classification_fails(self):
        pred = Prediction()
        with pytest.raises(ValueError, match="No classification"):
            _ = pred.label

    def test_roundtrip_serialization(self, segmentation_mask):
        marker = Marker(
            name="D3S1358",
            dye_row=0,
            alleles=(Allele(name="12", base_pair=120.0),),
        )
        pred = Prediction.for_segmentation(
            image=segmentation_mask,
            source_path="/tmp/test.hid",
            called_alleles=[marker],
        )
        restored = Prediction.from_dict(pred.to_dict())
        assert np.array_equal(restored.image, pred.image)
        assert restored.called_alleles[0].name == "D3S1358"

    def test_immutability(self, segmentation_mask):
        pred = Prediction.for_segmentation(
            image=segmentation_mask,
            source_path="/tmp/test.hid",
        )
        with pytest.raises(AttributeError):
            pred.image = None
