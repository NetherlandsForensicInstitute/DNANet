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

from typing import Any

import torch
from torch import Tensor, nn

from dnanet.modules.base import BaseTaskModule


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
        learning_rate: Initial learning rate for Adam.
        weight_decay: L2 regularization strength.
        scheduler_gamma: Multiplicative LR decay factor per epoch.
            Set to ``1.0`` to disable scheduling.
        threshold: Sigmoid threshold for converting logits to binary
            predictions (used for metric computation, not loss).
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 5e-4,
        scheduler_gamma: float = 0.8,
        threshold: float = 0.5,
        metrics_cfg: Any = None,
    ) -> None:
        super().__init__(model=model, loss_fn=loss_fn, metrics_cfg=metrics_cfg)
        self.save_hyperparameters({
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "scheduler_gamma": scheduler_gamma,
            "threshold": threshold,
        })

    def compute_step_outputs(
        self, batch: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute loss and metric inputs for a single batch.

        Args:
            batch: ``(input_tensor, target_tensor)`` from DataLoader.

        Returns:
            Scalar loss plus flattened prediction/target tensors for metrics.
        """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Update metrics with flattened predictions
        preds = torch.sigmoid(logits).detach()
        return loss, preds.reshape(-1), y.reshape(-1).int()

    def predict_step(self, batch: Any, batch_idx: int) -> Tensor:
        del batch_idx
        """Return sigmoid probabilities for prediction."""
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        logits = self(x)
        return torch.sigmoid(logits)
