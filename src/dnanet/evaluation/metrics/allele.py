"""TorchMetrics for allele-level segmentation evaluation.

These metrics compare predicted allele calls against ground-truth allele
calls at the allele granularity (not pixel-level). An allele is correctly
predicted if the same ``"MarkerName_AlleleName"`` string appears in both
the prediction and the ground truth.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Sequence

import torch
from torch import Tensor
from torchmetrics import Metric

from dnanet.evaluation.utils import flatten_markers_to_allele_names


if TYPE_CHECKING:
    from dnanet.core.marker import Marker


__all__ = [
    "AlleleMetric",
    "AllelePrecision",
    "AlleleRecall",
    "AlleleF1Score",
]

class AlleleMetric(Metric, abc.ABC):
    """Base class for micro-averaged allele call metrics.

    ``update`` accepts per-sample ground-truth and predicted marker sequences.
    The metric stores only aggregate true-positive, false-positive, and
    false-negative counts, so it is cheap to sync across distributed workers.
    """

    full_state_update = False
    higher_is_better = True
    is_differentiable = False

    def __init__(
        self,
        *,
        locus: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize the metric with optional locus filtering."""
        super().__init__(**kwargs)
        self.locus = locus
        self.add_state("tp", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(
        self,
        ground_truth_markers: Sequence[Sequence[Marker]],
        predicted_markers: Sequence[Sequence[Marker]],
    ) -> None:
        """Update true-positive, false-positive, and false-negative counts."""
        tp, fp, fn = _count_allele_matches(
            ground_truth_markers,
            predicted_markers,
            locus=self.locus,
        )
        self.tp += torch.as_tensor(tp, dtype=self.tp.dtype, device=self.tp.device)
        self.fp += torch.as_tensor(fp, dtype=self.fp.dtype, device=self.fp.device)
        self.fn += torch.as_tensor(fn, dtype=self.fn.dtype, device=self.fn.device)

    @abc.abstractmethod
    def compute(self) -> Tensor:
        """Compute the metric value from accumulated counts."""


class AllelePrecision(AlleleMetric):
    """Micro-averaged precision for allele calls."""

    def compute(self) -> Tensor:
        """Compute precision from accumulated counts."""
        return _safe_divide(self.tp, self.tp + self.fp)


class AlleleRecall(AlleleMetric):
    """Micro-averaged recall for allele calls."""

    def compute(self) -> Tensor:
        """Compute recall from accumulated counts."""
        return _safe_divide(self.tp, self.tp + self.fn)


class AlleleF1Score(AlleleMetric):
    """Micro-averaged F1 score for allele calls."""

    def compute(self) -> Tensor:
        """Compute F1 score from accumulated counts."""
        precision = _safe_divide(self.tp, self.tp + self.fp)
        recall = _safe_divide(self.tp, self.tp + self.fn)
        return _safe_divide(2 * precision * recall, precision + recall)


def _count_allele_matches(
    ground_truth_markers: Sequence[Sequence[Marker]],
    predicted_markers: Sequence[Sequence[Marker]],
    *,
    locus: str | None = None,
) -> tuple[int, int, int]:
    """Count true positives, false positives, and false negatives."""
    if len(ground_truth_markers) != len(predicted_markers):
        raise ValueError(
            "ground_truth_markers and predicted_markers must contain the same "
            f"number of samples, got {len(ground_truth_markers)} and "
            f"{len(predicted_markers)}."
        )

    tp, fp = 0, 0
    fn = 0
    for gt_markers, pred_markers in zip(
        ground_truth_markers, predicted_markers, strict=True,
    ):
        predicted = flatten_markers_to_allele_names(pred_markers, locus=locus)
        annotated = flatten_markers_to_allele_names(gt_markers, locus=locus)
        matched = len(predicted & annotated)
        tp += matched
        fp += len(predicted) - matched
        fn += len(annotated) - matched

    return tp, fp, fn


def _safe_divide(numerator: Tensor, denominator: Tensor) -> Tensor:
    value = numerator.float() / denominator.float().clamp_min(1)
    return torch.where(denominator > 0, value, torch.zeros_like(value))
