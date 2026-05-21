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

from typing import TYPE_CHECKING, Any

import torch

from dnanet.modules.base import BaseTaskModule


if TYPE_CHECKING:
    from torch import Tensor, nn
    from torchmetrics import MetricCollection

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
        optimizer: Optimizer instance for training.
        num_classes: Number of output classes.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization.
        lr_scheduler: Optional learning-rate scheduler.
        metrics: Metric collection used for train/validation logging.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        num_classes: int = 2,
        learning_rate: float = 1e-4,
        weight_decay: float = 5e-4,
        allele_class_index: int = 1,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        metrics: MetricCollection | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        batch_size: int | None = None,
    ) -> None:
        if lr_scheduler is None:
            lr_scheduler = scheduler

        super().__init__(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            lr_scheduler=lr_scheduler,
            batch_size=batch_size,
        )
        self.allele_class_index = allele_class_index
        self.save_hyperparameters({
            "num_classes": num_classes,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "allele_class_index": allele_class_index,
        })

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
        logits, targets = self._compute_logits_and_targets(batch)
        return self._compute_loss_and_metric_inputs(logits, targets)

    def compute_test_step_outputs(
        self,
        batch: tuple[Tensor, ...],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        logits, targets = self._compute_logits_and_targets(batch)
        loss, preds_flat, targets_flat = self._compute_loss_and_metric_inputs(logits, targets)
        return loss, preds_flat, targets_flat, self._allele_probabilities(logits)

    def compute_validation_step_outputs(
        self,
        batch: tuple[Tensor, ...],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.compute_test_step_outputs(batch)

    def _compute_logits_and_targets(
        self,
        batch: Any,
    ) -> tuple[Tensor, Tensor]:
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
        return logits, targets

    def _compute_loss_and_metric_inputs(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # logits: (N, K, C, L)
        # targets: (N, C, L)

        loss = self.loss_fn(logits, targets)
        preds = logits.argmax(dim=1)  # (N, C, L)
        return loss, preds, targets

    def _allele_probabilities(self, logits: Tensor) -> Tensor:
        if logits.shape[1] <= self.allele_class_index:
            raise ValueError(
                "PeakNet allele_class_index must refer to an output class, got "
                f"{self.allele_class_index} for logits with {logits.shape[1]} classes."
            )
        return torch.softmax(logits.detach(), dim=1)[:, self.allele_class_index]

    @staticmethod
    def _split_batch(
        batch: Any,
        *,
        require_targets: bool,
    ) -> tuple[tuple[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor | None]:
        if not isinstance(batch, (tuple, list)):
            raise TypeError("PeakNetModule expects batches to be tuples or lists.")

        if len(batch) == 3 and isinstance(batch[0], (tuple, list)):
            inputs = tuple(batch[0])
            targets = batch[1]
        elif len(batch) == 2 and isinstance(batch[0], (tuple, list)):
            inputs = tuple(batch[0])
            targets = batch[1]
        elif len(batch) == 7:
            inputs = tuple(batch[:5])
            targets = batch[5]
        elif len(batch) == 6:
            inputs = tuple(batch[:5])
            targets = batch[5]
        elif len(batch) == 5:
            inputs = tuple(batch)
            targets = None
        else:
            raise ValueError(
                "PeakNetModule expects a nested (inputs, targets) batch, a flat "
                "6-item batch, a metadata-augmented batch, or a 5-item input-only batch."
                f" Got {len(batch)}.",
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
