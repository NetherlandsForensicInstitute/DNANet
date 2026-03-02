from typing import Sequence

from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.models.prediction import Prediction
import numpy as np


def reconstruction_mse(
        images: Sequence[HIDImage],
        predictions: Sequence[Prediction],
) -> float:
    total_mse = 0.0
    if len(predictions) == 0:
        return 0.0
    for image, prediction in zip(images, predictions):
        replicated_signal = prediction.image.squeeze()
        true_signal = image.data.squeeze()
        mse = np.mean((replicated_signal - true_signal) ** 2)
        total_mse += mse
    return total_mse / len(predictions)
