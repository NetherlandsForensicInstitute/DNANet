import pytest
import torch
from DNAnet.models.classification.peak_classification import PeakClassification
from tests.conftest import SKIP_MODELS

# Skip all these tests if an environment variable tells us to.
pytestmark = pytest.mark.skipif(**SKIP_MODELS)

def test_peak_classification_train_inference(peak_dataset):
    # peak_dataset is a PeakWindowDataset which yields ExtractedPeak objects
    assert len(peak_dataset) > 0

    labels = ["allele", "noise"]
    window_size = peak_dataset.window_size

    # Initialize the model
    model = PeakClassification(
        labels=labels,
        window_size=window_size,
        include_marker=True,
        channels=[8, 16] # Small channels for fast test
    )

    # Test training (one epoch for speed)
    model.fit(peak_dataset, batch_size=2, num_epochs=1)

    # Test inference on a batch
    predictions = model.predict_batch(peak_dataset)
    assert len(predictions) == len(peak_dataset)

    # Check prediction format
    for pred in predictions:
        assert "classification" in pred.__dict__
        for label in labels:
            assert label in pred.classification
            assert 0 <= pred.classification[label] <= 1
