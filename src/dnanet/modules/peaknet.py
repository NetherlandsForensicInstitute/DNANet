"""Lightning module for combined PeakNet training.

Wraps :class:`~dnanet.models.peaknet.CombinedClassifier` (or
:class:`~dnanet.models.peaknet.PeakOnlyClassifier`) with per-position
classification training logic, metrics, and optimizer configuration.

The key difference from :class:`ClassificationModule` is that PeakNet
operates on **per-scan-point** classification of full profiles, not on
individual peak windows. The model receives both full images and
extracted peak windows, and produces logits of shape ``(N, K, C, L)``.

Design pattern: **Mediator**
    Coordinates model, loss, optimizer, scheduler, and metrics for the
    combined PeakNet training scenario.
"""

from __future__ import annotations

from typing import Any

import torch
import torchmetrics
from torch import Tensor, nn

from dnanet.modules.base import BaseTaskModule


class PeakNetModule(BaseTaskModule):
    """PyTorch Lightning module for combined PeakNet.

    Handles:
        - Forward pass with dual-input (full images + peak windows)
        - Per-position cross-entropy loss
        - Multi-class metric tracking (accuracy, F1)
        - Optimizer + LR scheduler configuration

    Args:
        model: Combined classifier (CombinedClassifier or PeakOnlyClassifier).
        loss_fn: Loss function (e.g. ``nn.CrossEntropyLoss``).
        num_classes: Number of output classes.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization.
        scheduler_gamma: Exponential LR decay. Set to 1.0 to disable.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        num_classes: int = 2,
        learning_rate: float = 1e-4,
        weight_decay: float = 5e-4,
        scheduler_gamma: float = 1.0,
    ) -> None:
        super().__init__(model=model, loss_fn=loss_fn)
        self.save_hyperparameters({
            "num_classes": num_classes,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "scheduler_gamma": scheduler_gamma,
        })
        self.initialize_metrics()

    def build_metrics(self) -> torchmetrics.MetricCollection:
        num_classes = int(self.hparams.num_classes)
        return torchmetrics.MetricCollection(
            {
                'accuracy': torchmetrics.classification.MulticlassAccuracy(
                    num_classes=num_classes,
                    average='micro',
                ),
                'f1': torchmetrics.classification.MulticlassF1Score(
                    num_classes=num_classes,
                    average='macro',
                ),
            }
        )

    def compute_step_outputs(
        self,
        batch: tuple[Tensor, ...],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute per-position loss and metric inputs.

        Accepts either the nested output of
        :meth:`dnanet.data.transformer.CombinedTransformer.collate_fn`,
        ``((full_images, peak_windows, marker_idxs, peak_centers, peak_counts), targets)``,
        or the equivalent flattened 6-item tuple.
        """
        (
            full_images,
            peak_windows,
            marker_idxs,
            peak_centers,
            peak_counts,
        ), targets = self._split_batch(batch, require_targets=True)

        # Forward: (N, K, C, L)
        logits = self.model(
            full_images,
            peak_windows,
            marker_idxs,
            peak_centers,
            peak_counts,
        )

        # Loss: CrossEntropyLoss expects (N, K, ...) logits and (N, ...) targets
        # logits: (N, K, C, L) → reshape to (N*C*L, K)
        # targets: (N, C, L) → reshape to (N*C*L,)
        num_classes = logits.shape[1]
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, num_classes)
        targets_flat = targets.reshape(-1)

        loss = self.loss_fn(logits_flat, targets_flat)
        preds_flat = logits_flat.argmax(dim=1).detach()
        return loss, preds_flat, targets_flat

    @staticmethod
    def _split_batch(
        batch: Any,
        *,
        require_targets: bool,
    ) -> tuple[tuple[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor | None]:
        if not isinstance(batch, (tuple, list)):
            raise TypeError("PeakNetModule expects batches to be tuples or lists.")

        if len(batch) == 2 and isinstance(batch[0], (tuple, list)):
            inputs = tuple(batch[0])
            targets = batch[1]
        elif len(batch) == 6:
            inputs = tuple(batch[:5])
            targets = batch[5]
        elif len(batch) == 5:
            inputs = tuple(batch)
            targets = None
        else:
            raise ValueError(
                "PeakNetModule expects a nested (inputs, targets) batch, a flat "
                "6-item batch, or a 5-item input-only batch.",
            )

        if len(inputs) != 5:
            raise ValueError("PeakNetModule requires five input tensors per batch.")
        if require_targets and targets is None:
            raise ValueError("PeakNetModule training and validation batches must include targets.")

        full_images, peak_windows, marker_idxs, peak_centers, peak_counts = inputs
        return (
            full_images,
            peak_windows,
            marker_idxs,
            peak_centers,
            peak_counts,
        ), targets

    def predict_step(self, batch: Any, batch_idx: int) -> Tensor:
        del batch_idx
        (
            full_images,
            peak_windows,
            marker_idxs,
            peak_centers,
            peak_counts,
        ), _targets = self._split_batch(batch, require_targets=False)
        logits = self.model(
            full_images,
            peak_windows,
            marker_idxs,
            peak_centers,
            peak_counts,
        )
        return torch.softmax(logits, dim=1)
