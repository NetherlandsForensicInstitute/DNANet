import os

import numpy as np
import pytest

from DNAnet.data.data_models import Annotation
from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.evaluation import average_binary_iou, pixel_f1_score, pixel_precision, pixel_recall
from DNAnet.models.prediction import Prediction


@pytest.fixture
def hid_annotation():
    return np.load(
        os.path.join(pytest.RESOURCES_DIR, "profiles", "RD", "1A2_A01_01_annotation.npy")
    )


@pytest.fixture
def hid_image():
    return HIDImage(
        path=os.path.join(pytest.RESOURCES_DIR, "profiles", "RD", "1A2_A01_01.hid"),
        annotation=Annotation(
            image=np.load(
                os.path.join(pytest.RESOURCES_DIR, "profiles", "RD", "1A2_A01_01_annotation.npy")
            )
        ),
    )


def test_segmentation_metrics_dummy_image():
    annotation = Annotation(image=np.array([[1, 1, 1], [0, 1, 0], [1, 0, 0]]))
    image = HIDImage(path="/dummypath", annotation=annotation)
    prediction = Prediction(image=np.array([[0.6, 0.8, 0.1], [0.2, 0.7, 0.6], [0.9, 0.2, 0.7]]))
    tp, _, fn, fp = 4, 2, 1, 2

    expected_precision = tp / (tp + fp)
    expected_recall = tp / (tp + fn)
    expected_f1 = 2 * (
        (expected_precision * expected_recall) / (expected_precision + expected_recall)
    )
    expected_iou = tp / (tp + fn + fp)

    assert pixel_precision([image], [prediction]) == expected_precision
    assert pixel_recall([image], [prediction]) == expected_recall
    assert pixel_f1_score([image], [prediction]) == expected_f1
    assert average_binary_iou([image], [prediction]) == expected_iou


def test_precision_segmentation_hid_image(hid_image, hid_annotation):
    assert pixel_precision([hid_image], [Prediction(image=hid_annotation)]) == 1.0

    # Add 100 false positives, so precision will drop
    hid_annotation[0, :100, 0] = 1
    assert pixel_precision(
        [hid_image], [Prediction(image=hid_annotation)]
    ) == pytest.approx(expected=0.865, rel=0.001)


def test_recall_segmentation_hid_image(hid_image, hid_annotation):
    assert pixel_recall([hid_image], [Prediction(image=hid_annotation)]) == 1.0

    # Change 100 true positives to 0, so recall will drop
    indices_one = np.where(hid_annotation[0, :, 0] == 1)[0]
    indices_to_zero = np.random.choice(indices_one, size=100, replace=False)
    hid_annotation[0, indices_to_zero, 0] = 0
    assert pixel_recall(
        [hid_image], [Prediction(image=hid_annotation)]
    ) == pytest.approx(expected=0.844, rel=0.001)


def test_f1score_segmentation_hid_image(hid_image, hid_annotation):
    assert pixel_f1_score([hid_image], [Prediction(image=hid_annotation)]) == 1.0

    # Add 100 false positives, so f1_score will drop
    hid_annotation[0, :100, 0] = 1
    assert pixel_f1_score(
        [hid_image], [Prediction(image=hid_annotation)]
    ) == pytest.approx(expected=0.927, rel=0.001)


def test_average_binary_iou_segmentation_hid_image(hid_image, hid_annotation):
    assert average_binary_iou([hid_image], [Prediction(image=hid_annotation)]) == 1.0

    # Add 100 false positives, so average_binary_iou will drop
    hid_annotation[0, :100, 0] = 1
    assert average_binary_iou(
        [hid_image], [Prediction(image=hid_annotation)]
    ) == pytest.approx(expected=0.865, rel=0.001)
