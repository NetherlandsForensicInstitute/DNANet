"""Pixel-level segmentation metrics.

These metrics compare predicted segmentation masks against ground-truth
annotation masks at the individual scan-point (pixel) level.

All functions accept sequences of numpy arrays directly, keeping them
decoupled from specific domain objects.

Design pattern: **Pure Functions**
    Stateless, side-effect-free functions that depend only on their inputs.
    This makes them easy to test, compose, and parallelize.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def pixel_precision(
    ground_truths: Sequence[np.ndarray],
    predictions: Sequence[np.ndarray],
    threshold: float = 0.5,
) -> float:
    """Compute micro-averaged pixel precision across all samples.

    Args:
        ground_truths: Binary annotation masks (0/1).
        predictions: Predicted probability/logit masks.
        threshold: Threshold above which a prediction is positive.

    Returns:
        Precision = TP / (TP + FP). Returns 0 if no positives predicted.
    """
    tp, fp = 0, 0
    for gt, pred in zip(ground_truths, predictions):
        pred_pos = pred >= threshold
        tp += np.sum(pred_pos & (gt == 1))
        fp += np.sum(pred_pos & (gt == 0))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def pixel_recall(
    ground_truths: Sequence[np.ndarray],
    predictions: Sequence[np.ndarray],
    threshold: float = 0.5,
) -> float:
    """Compute micro-averaged pixel recall across all samples.

    Args:
        ground_truths: Binary annotation masks (0/1).
        predictions: Predicted probability/logit masks.
        threshold: Threshold above which a prediction is positive.

    Returns:
        Recall = TP / (TP + FN). Returns 0 if no positives in ground truth.
    """
    tp, fn = 0, 0
    for gt, pred in zip(ground_truths, predictions):
        pred_pos = pred >= threshold
        tp += np.sum(pred_pos & (gt == 1))
        fn += np.sum(~pred_pos & (gt == 1))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def pixel_f1_score(
    ground_truths: Sequence[np.ndarray],
    predictions: Sequence[np.ndarray],
    threshold: float = 0.5,
) -> float:
    """Compute pixel-level F1 score (harmonic mean of precision and recall).

    Args:
        ground_truths: Binary annotation masks (0/1).
        predictions: Predicted probability/logit masks.
        threshold: Threshold above which a prediction is positive.

    Returns:
        F1 = 2 * P * R / (P + R). Returns 0 if both are 0.
    """
    p = pixel_precision(ground_truths, predictions, threshold)
    r = pixel_recall(ground_truths, predictions, threshold)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def average_binary_iou(
    ground_truths: Sequence[np.ndarray],
    predictions: Sequence[np.ndarray],
    threshold: float = 0.5,
) -> float:
    """Compute mean binary Intersection over Union across samples.

    For each sample, computes IoU = intersection / union of the positive
    class, then averages across all samples.

    Args:
        ground_truths: Binary annotation masks (0/1).
        predictions: Predicted probability/logit masks.
        threshold: Threshold above which a prediction is positive.

    Returns:
        Mean IoU across all samples. Samples with union=0 contribute 0.
    """
    if not ground_truths:
        return 0.0

    total_iou = 0.0
    for gt, pred in zip(ground_truths, predictions):
        y_pred = (pred >= threshold).astype(float)
        intersection = np.sum(gt * y_pred)
        union = np.sum(gt) + np.sum(y_pred) - intersection
        if union > 0:
            total_iou += intersection / union

    return total_iou / len(ground_truths)
