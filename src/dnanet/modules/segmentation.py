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

import lightning as L
import torch
import torchmetrics
from torch import Tensor, nn

from loguru import logger


class SegmentationModule(L.LightningModule):
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
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss_fn"])

        self.model = model
        self.loss_fn = loss_fn

        # Metrics — separate instances for train and val to avoid state leakage
        metrics = torchmetrics.MetricCollection({
            "accuracy": torchmetrics.classification.BinaryAccuracy(threshold=threshold),
            "precision": torchmetrics.classification.BinaryPrecision(threshold=threshold),
            "recall": torchmetrics.classification.BinaryRecall(threshold=threshold),
            "f1": torchmetrics.classification.BinaryF1Score(threshold=threshold),
            "iou": torchmetrics.classification.BinaryJaccardIndex(threshold=threshold),
        })
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass — returns raw logits."""
        return self.model(x)

    def _shared_step(
        self, batch: tuple[Tensor, Tensor], stage: str
    ) -> Tensor:
        """Compute loss and update metrics for a single batch.

        Args:
            batch: ``(input_tensor, target_tensor)`` from DataLoader.
            stage: ``"train"`` or ``"val"``.

        Returns:
            Scalar loss.
        """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Log loss
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # Update metrics with flattened predictions
        preds = torch.sigmoid(logits).detach()
        metrics = self.train_metrics if stage == "train" else self.val_metrics
        metrics.update(preds.reshape(-1), y.reshape(-1).int())

        return loss

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute(), prog_bar=False)
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute(), prog_bar=False)
        self.val_metrics.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure Adam optimizer with ExponentialLR scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        config: dict[str, Any] = {"optimizer": optimizer}

        if self.hparams.scheduler_gamma < 1.0:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.hparams.scheduler_gamma
            )
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
            }

        return config

    def predict_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Return sigmoid probabilities for prediction."""
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        logits = self(x)
        return torch.sigmoid(logits)
