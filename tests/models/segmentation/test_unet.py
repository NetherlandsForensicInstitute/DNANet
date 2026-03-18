from copy import deepcopy

import pytest
from torch.nn import CrossEntropyLoss

from DNAnet.data.data_models import Annotation
from DNAnet.data.data_models.base import SimpleDataset
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


def test_dnanet_unet_multiclass(hid_dataset_rd):
    multiclass_images = []
    for image in hid_dataset_rd:
        multiclass_image = deepcopy(image)
        annotation = multiclass_image.annotation.image.astype("int64").squeeze(-1)
        annotation[:, 100:200] = 2
        multiclass_image._annotation = Annotation(image=annotation)
        multiclass_images.append(multiclass_image)

    dataset = SimpleDataset(multiclass_images, shuffle=False)

    model = DNANet_UNet(4, (1, 3), device="cpu", apply_allele_caller=False, num_classes=3)

    assert isinstance(model.loss_fn, CrossEntropyLoss)

    model.fit(dataset, batch_size=1, num_epochs=1, use_evaluation_metric=False)
    predictions = model.predict_batch(dataset)

    assert len(predictions) == len(dataset)
    assert predictions[0].image.shape == (*dataset[0].annotation.image.shape, 3)
