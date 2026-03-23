"""Allele-level segmentation metrics.

These metrics compare predicted allele calls against ground-truth allele
calls at the allele granularity (not pixel-level). An allele is correctly
predicted if the same ``"MarkerName_AlleleName"`` string appears in both
the prediction and the ground truth.

Design pattern: **Pure Functions**
    Like the pixel metrics, these are stateless functions that accept
    pre-flattened allele name sets or Marker sequences.
"""

from __future__ import annotations

from typing import Sequence

from dnanet.core.marker import Marker
from dnanet.evaluation.utils import flatten_markers_to_allele_names


def allele_precision(
    ground_truth_markers: Sequence[Sequence[Marker]],
    predicted_markers: Sequence[Sequence[Marker]],
    *,
    locus: str | None = None,
    min_rfu: int | None = None,
    low_or_high: str | None = None,
    gt_min_rfu: int | None = None,
    gt_low_or_high: str | None = None,
) -> float:
    """Compute micro-averaged allele-level precision.

    Args:
        ground_truth_markers: Per-sample ground truth marker sequences.
        predicted_markers: Per-sample predicted marker sequences.
        locus: Filter to a specific locus.
        min_rfu: RFU threshold for predictions.
        low_or_high: Kit threshold mode for predictions.
        gt_min_rfu: RFU threshold for ground truth (None = same as predictions,
            or no threshold if annotations lack heights).
        gt_low_or_high: Kit threshold mode for ground truth.

    Returns:
        Precision = TP / (TP + FP). Returns 0 if no positives predicted.
    """
    tp, fp = 0, 0
    for gt_markers, pred_markers in zip(ground_truth_markers, predicted_markers):
        predicted = flatten_markers_to_allele_names(
            pred_markers, locus=locus, min_rfu=min_rfu, low_or_high=low_or_high,
        )
        annotated = flatten_markers_to_allele_names(
            gt_markers, locus=locus,
            min_rfu=gt_min_rfu if gt_min_rfu is not None else min_rfu,
            low_or_high=gt_low_or_high if gt_low_or_high is not None else low_or_high,
        )
        matched = len(predicted & annotated)
        tp += matched
        fp += len(predicted) - matched

    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def allele_recall(
    ground_truth_markers: Sequence[Sequence[Marker]],
    predicted_markers: Sequence[Sequence[Marker]],
    *,
    locus: str | None = None,
    min_rfu: int | None = None,
    low_or_high: str | None = None,
    gt_min_rfu: int | None = None,
    gt_low_or_high: str | None = None,
) -> float:
    """Compute micro-averaged allele-level recall.

    Args:
        ground_truth_markers: Per-sample ground truth marker sequences.
        predicted_markers: Per-sample predicted marker sequences.
        locus: Filter to a specific locus.
        min_rfu: RFU threshold for predictions.
        low_or_high: Kit threshold mode for predictions.
        gt_min_rfu: RFU threshold for ground truth.
        gt_low_or_high: Kit threshold mode for ground truth.

    Returns:
        Recall = TP / (TP + FN). Returns 0 if no positives in ground truth.
    """
    tp, fn = 0, 0
    for gt_markers, pred_markers in zip(ground_truth_markers, predicted_markers):
        predicted = flatten_markers_to_allele_names(
            pred_markers, locus=locus, min_rfu=min_rfu, low_or_high=low_or_high,
        )
        annotated = flatten_markers_to_allele_names(
            gt_markers, locus=locus,
            min_rfu=gt_min_rfu if gt_min_rfu is not None else min_rfu,
            low_or_high=gt_low_or_high if gt_low_or_high is not None else low_or_high,
        )
        matched = len(predicted & annotated)
        tp += matched
        fn += len(annotated) - matched

    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def allele_f1_score(
    ground_truth_markers: Sequence[Sequence[Marker]],
    predicted_markers: Sequence[Sequence[Marker]],
    *,
    locus: str | None = None,
    min_rfu: int | None = None,
    low_or_high: str | None = None,
    gt_min_rfu: int | None = None,
    gt_low_or_high: str | None = None,
) -> float:
    """Compute allele-level F1 score.

    Returns:
        F1 = 2 * P * R / (P + R). Returns 0 if both are 0.
    """
    p = allele_precision(
        ground_truth_markers, predicted_markers,
        locus=locus, min_rfu=min_rfu, low_or_high=low_or_high,
        gt_min_rfu=gt_min_rfu, gt_low_or_high=gt_low_or_high,
    )
    r = allele_recall(
        ground_truth_markers, predicted_markers,
        locus=locus, min_rfu=min_rfu, low_or_high=low_or_high,
        gt_min_rfu=gt_min_rfu, gt_low_or_high=gt_low_or_high,
    )
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
