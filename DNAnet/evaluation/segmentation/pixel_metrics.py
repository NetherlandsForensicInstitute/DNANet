from typing import Sequence

import numpy as np

from DNAnet.data.data_models.base import Image
from DNAnet.models.prediction import Prediction


def pixel_precision(
        images: Sequence[Image],
        predictions: Sequence[Prediction],
) -> float:
    true_positives = 0
    false_positives = 0
    for image, prediction in zip(images, predictions):
        true_positives += np.sum(
            (prediction.image >= 0.5) & (image.annotation.image == 1)
        )
        false_positives += np.sum(
            (prediction.image >= 0.5) & (image.annotation.image == 0)
        )
    return true_positives / (true_positives + false_positives)


def pixel_recall(
        images: Sequence[Image],
        predictions: Sequence[Prediction],
) -> float:
    true_positives = 0
    false_negatives = 0
    for image, prediction in zip(images, predictions):
        true_positives += np.sum(
            (prediction.image >= 0.5) & (image.annotation.image == 1)
        )
        false_negatives += np.sum(
            (prediction.image < 0.5) & (image.annotation.image == 1)
        )
    return true_positives / (true_positives + false_negatives)


def pixel_f1_score(
        images: Sequence[Image],
        predictions: Sequence[Prediction]
) -> float:
    p = pixel_precision(images, predictions)
    r = pixel_recall(images, predictions)
    return 2 * p * r / (p + r) if p + r else 0


def average_binary_iou(
        images: Sequence[Image],
        predictions: Sequence[Prediction],
        threshold: float = 0.5,
) -> float:
    """
    Computes the average intersection over union over all the segmentation
    predictions. Set a threshold to consider above which probability a classification
    is deemed positive.
    """

    total = 0
    n = 0
    for image, prediction in zip(images, predictions):
        y_true = image.annotation.image
        y_pred = np.where((prediction.image > threshold), 1.0, 0.0)

        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred) - intersection
        if union > 0:
            total += intersection / union
        n += 1
    iou = total / n

    return iou
