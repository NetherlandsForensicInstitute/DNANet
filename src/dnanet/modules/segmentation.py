"""Lightning module for EPG segmentation.

Design pattern: **Mediator**
    The ``SegmentationModule`` mediates between the model architecture
    (``nn.Module``), the loss function, the optimizer, the LR scheduler,
    and the metrics — each of which is independently configurable through
    Hydra. This replaces the original ~500-line ``BaseModel`` class that
    tangled all these concerns together.

Design pattern: **Template Method**
    Lightning's ``training_step`` / ``validation_step`` / ``configure_optimizers``
    define the training skeleton. We fill in the domain-specific pieces
    (Dice loss, sigmoid thresholding, EPG-specific metrics) without
    reimplementing the training loop.

Usage:
    Instantiated by Hydra (via ``_target_``) or manually::

        module = SegmentationModule(
            model=UNet(depth=4, kernel_size=(3, 5), num_filters=32),
            loss_fn=DiceLoss(),
            learning_rate=1e-4,
        )
        trainer = L.Trainer(max_epochs=15)
        trainer.fit(module, datamodule=dm)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from dnanet.modules.base import BaseTaskModule


if TYPE_CHECKING:
    from torchmetrics import MetricCollection
    

class SegmentationModule(BaseTaskModule):
    """PyTorch Lightning module for binary EPG segmentation.

    This module wraps any segmentation model (e.g. UNet) and handles:
    - Forward pass + loss computation
    - Metric tracking (accuracy, precision, recall, F1, IoU)
    - Optimizer + LR scheduler configuration
    - Logging to any Lightning logger (MLflow, TensorBoard, etc.)

    Args:
        model: The segmentation network (e.g. ``UNet``).
        loss_fn: Loss function (e.g. ``DiceLoss``).
        optimizer: Optimizer instance for training.
        learning_rate: Initial learning rate for Adam.
        weight_decay: L2 regularization strength.
        lr_scheduler: Optional learning-rate scheduler.
        threshold: Sigmoid threshold for converting logits to binary
            predictions (used for metric computation, not loss).
        metrics: Metric collection used for train/validation logging.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        learning_rate: float = 1e-4,
        weight_decay: float = 5e-4,
        threshold: float = 0.5,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        metrics: MetricCollection | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> None:
        if lr_scheduler is None:
            lr_scheduler = scheduler

        super().__init__(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            lr_scheduler=lr_scheduler,
        )
        self.save_hyperparameters({
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "threshold": threshold,
        })

    def compute_step_outputs(
        self, batch: tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Any],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute loss and metric inputs for a single batch.

        Args:
            batch: ``(input_tensor, target_tensor)`` from DataLoader.

        Returns:
            Scalar loss plus flattened prediction/target tensors for metrics.
        """
        loss, preds, y = self._compute_loss_and_probabilities(batch)
        return loss, preds.reshape(-1), y.reshape(-1).int()

    def compute_test_step_outputs(
        self,
        batch: tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Any],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        loss, preds, y = self._compute_loss_and_probabilities(batch)
        return loss, preds.reshape(-1), y.reshape(-1).int(), preds

    @staticmethod
    def _split_batch(batch: tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Any]) -> tuple[Tensor, Tensor]:
        if len(batch) == 3:
            x, y, _metadata = batch
            return x, y
        x, y = batch
        return x, y

    def _compute_loss_and_probabilities(
        self,
        batch: tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Any],
    ) -> tuple[Tensor, Tensor, Tensor]:
        x, y = self._split_batch(batch)
        logits = self(x)
        loss = self.loss_fn(logits, y)
        return loss, torch.sigmoid(logits).detach(), y

    def predict_step(self, batch: Any, batch_idx: int) -> Tensor:
        del batch_idx
        """Return sigmoid probabilities for prediction."""
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        logits = self(x)
        return torch.sigmoid(logits)
