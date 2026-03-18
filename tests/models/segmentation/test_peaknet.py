import pytest
import torch
from DNAnet.models.segmentation.combined_peaknet import CombinedPeakNet
from DNAnet.models.classification.peak_classification import PeakClassification
from tests.conftest import SKIP_MODELS

# Skip all these tests if an environment variable tells us to.
pytestmark = pytest.mark.skipif(**SKIP_MODELS)

def test_peaknet_train_inference(hid_dataset_rd):
    # hid_dataset_rd is an HIDDataset which yields HIDImage objects
    assert len(hid_dataset_rd) > 0

    # PeakNet requires a peak_classifier.
    # We can pass it as a config mapping to be loaded by load_model inside PeakNet.__init__
    peak_classifier_config = {
        "name": "peak_classification",
        "labels": ["allele", "noise"],
        "window_size": 120,
        "channels": [8, 16]
    }

    # We test PeakNet without autoencoder first as it's simpler
    model = CombinedPeakNet(
        peak_classifier=peak_classifier_config,
        hidden_dims=[32, 16],
        threshold=100
    )

    # Test training (one epoch for speed)
    model.fit(hid_dataset_rd, batch_size=1, num_epochs=1)

    # Test inference on a batch
    predictions = model.predict_batch(hid_dataset_rd)
    assert len(predictions) == len(hid_dataset_rd)

    # Check prediction format
    for pred in predictions:
        assert pred.image is not None
        # PeakNet output image should have shape (C, 4096, 1) or (C, 4096)
        # Based on create_predictions, it seems to be (C, 4096, 1)
        assert pred.image.shape[0] == 5 # num dyes
        assert pred.image.shape[1] == 4096 # signal length
