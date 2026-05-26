import pytest

from DNAnet.evaluation.visualizations import plot_profile
from DNAnet.models.segmentation.trainable_unet import DNANet_UNet
from tests.conftest import SKIP_MODELS


# Skip all these tests if an environment variable tells us to.
pytestmark = pytest.mark.skipif(**SKIP_MODELS)


def test_dnanet_unet(hid_dataset_rd):
    assert len(hid_dataset_rd) == 2

    model = DNANet_UNet(4, (1, 3))
    model.fit(hid_dataset_rd, batch_size=1, num_epochs=5)
    predictions = model.predict_batch(hid_dataset_rd)

    plot_profile(hid_dataset_rd, predictions, prediction_as_mask=False)
