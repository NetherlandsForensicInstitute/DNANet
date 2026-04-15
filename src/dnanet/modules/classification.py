"""Lightning module for peak classification.

Wraps a :class:`~dnanet.models.peak_classifier.PeakClassificationModel`
with multi-class training logic, metrics, and optimizer configuration.

Design pattern: **Mediator**
    Coordinates model, loss, optimizer, scheduler, and metrics — each
    independently configurable via Hydra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from dnanet.modules.base import BaseTaskModule


if TYPE_CHECKING:
    from torchmetrics import MetricCollection

class ClassificationModule(BaseTaskModule):
    """PyTorch Lightning module for peak classification.

    Handles:
        - Forward pass + loss computation (cross-entropy, focal, or KL-div)
        - Multi-class metric tracking (accuracy, precision, recall, F1)
        - Optimizer + LR scheduler configuration

    Args:
        model: Peak classification network.
        loss_fn: Loss function (e.g. ``nn.CrossEntropyLoss``).
        num_classes: Number of output classes.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization.
        optimizer: Optimizer instance for training.
        lr_scheduler: Optional learning-rate scheduler.
        metrics: Metric collection used for train/validation logging.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        num_classes: int = 2,
        learning_rate: float = 1e-4,
        weight_decay: float = 5e-4,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        metrics: MetricCollection | None = None,
    ) -> None:
        super().__init__(model=model, loss_fn=loss_fn, metrics=metrics, optimizer=optimizer, lr_scheduler=lr_scheduler)
        self.save_hyperparameters({
            "num_classes": num_classes,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        })

    def compute_step_outputs(
        self, batch: tuple[Tensor, ...],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute loss and metric inputs.

        Expects batch to be ``(peak_data, marker_idx, targets)`` or
        ``(peak_data, targets)`` (without marker embeddings).
        """
        if len(batch) == 3:
            peak_data, marker_idx, targets = batch
            logits = self.model((peak_data, marker_idx))
        else:
            peak_data, targets = batch
            logits = self.model(peak_data)

        loss = self.loss_fn(logits, targets)
        preds = logits.argmax(dim=1).detach()
        return loss, preds, targets

    def _extract_predict_inputs(self, batch: Any) -> Tensor | tuple[Tensor, Tensor]:
        if not isinstance(batch, (tuple, list)):
            return batch
        if len(batch) == 0:
            raise ValueError("ClassificationModule.predict_step received an empty batch.")
        if len(batch) == 3:
            peak_data, marker_idx, _ = batch
            return peak_data, marker_idx

        first = batch[0]
        if isinstance(first, (tuple, list)):
            return first[0], first[1]

        if len(batch) == 2 and getattr(self.model, "use_embedding", False):
            peak_data, marker_idx = batch
            return peak_data, marker_idx

        return first

    def predict_step(self, batch: Any, batch_idx: int) -> Tensor:
        del batch_idx
        logits = self.model(self._extract_predict_inputs(batch))
        return torch.softmax(logits, dim=1)
