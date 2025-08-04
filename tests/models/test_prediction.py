import numpy as np
from pathlib import Path

from DNAnet.data.data_models import Allele, Marker
from DNAnet.models.prediction import Prediction


def test_prediction_roundtrip_with_alleles():
    classification = {"pos": 0.9, "neg": 0.1}
    image = np.array([[0.1, 0.2], [0.3, 0.4]])
    called_alleles = [
        Marker(dye_row=0, name="AMEL", alleles=[Allele("X"), Allele("Y")])
    ]
    original_path = Path("/tmp/sample.hid")

    pred = Prediction(
        classification=classification,
        image=image,
        original_image_path=original_path,
        called_alleles=called_alleles,
    )

    pred_dict = pred.to_dict()
    expected_dict = {
        "classification": classification,
        "image": image.tolist(),
        "original_image_path": str(original_path),
        "called_alleles": [
            {
                "dye_row": 0,
                "name": "AMEL",
                "alleles": [
                    {"name": "X", "base_pair": None, "left_bin": None, "right_bin": None, "height": None},
                    {"name": "Y", "base_pair": None, "left_bin": None, "right_bin": None, "height": None},
                ],
            }
        ],
    }
    assert pred_dict == expected_dict

    restored = Prediction.from_dict(pred_dict)
    assert restored.classification == classification
    np.testing.assert_equal(restored.image, image)
    assert Path(restored.original_image_path) == original_path
    assert restored.called_alleles == called_alleles


def test_prediction_roundtrip_without_called_alleles():
    classification = {"pos": 1.0}
    image = np.zeros((1, 2))
    original_path = Path("/tmp/img.hid")
    pred = Prediction(
        classification=classification,
        image=image,
        original_image_path=original_path,
    )

    pred_dict = pred.to_dict()
    assert pred_dict["called_alleles"] is None

    restored = Prediction.from_dict(pred_dict)
    assert restored.called_alleles is None
    assert restored.classification == classification
    np.testing.assert_equal(restored.image, image)
    assert Path(restored.original_image_path) == original_path
