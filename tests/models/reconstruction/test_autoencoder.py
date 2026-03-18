import pytest
import torch
import numpy as np
from DNAnet.models.reconstruction.autoencoder import Autoencoder
from tests.conftest import SKIP_MODELS

# Skip all these tests if an environment variable tells us to.
pytestmark = pytest.mark.skipif(**SKIP_MODELS)

def test_hid_autoencoder_train_inference(hid_dataset_rd):
    # hid_dataset_rd is an HIDDataset which yields HIDImage objects
    assert len(hid_dataset_rd) > 0

    # We use small hyperparameters for the test to be fast
    model = Autoencoder(
        input_dyes=5,
        signal_length=4096,
        architecture="cnn_per_dye",
        compression=8,
        hidden_dims=8,
        depth=3
    )

    # Test training (one epoch for speed)
    model.fit(hid_dataset_rd, batch_size=1, num_epochs=1)

    # Test inference on a batch
    predictions = model.predict_batch(hid_dataset_rd)
    assert len(predictions) == len(hid_dataset_rd)

    # Check prediction format
    for pred_list in predictions:
        # Note: Autoencoder's create_predictions returns a list of Predictions per image in batch
        assert isinstance(pred_list, list)
        for pred in pred_list:
            assert pred.image is not None
            assert pred.image.shape == (5, 4096)
            assert not np.isnan(pred.image).any()
