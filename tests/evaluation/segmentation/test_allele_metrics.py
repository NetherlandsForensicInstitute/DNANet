import os

import numpy as np
import pytest

from DNAnet.data.data_models import Allele, Annotation, Marker
from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.evaluation import allele_f1_score, allele_precision, allele_recall
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


def test_allele_metrics():
    marker_1 = Marker(name="AMEL", dye_row=0, alleles=[Allele("13"), Allele("15.1")])
    marker_2 = Marker(name="Penta E", dye_row=0, alleles=[Allele("14"), Allele("15.1")])
    markers = [marker_1, marker_2]
    image = HIDImage(path="/dummypath", meta={"called_alleles": markers})
    prediction = Prediction(meta={"called_alleles": [marker_1]})

    tp, fn, fp = 2, 2, 0

    expected_precision = tp / (tp + fp)
    expected_recall = tp / (tp + fn)
    expected_f1 = 2 * (
        (expected_precision * expected_recall) / (expected_precision + expected_recall)
    )

    assert allele_precision([image], [prediction]) == expected_precision
    assert allele_recall([image], [prediction]) == expected_recall
    assert allele_f1_score([image], [prediction]) == expected_f1
