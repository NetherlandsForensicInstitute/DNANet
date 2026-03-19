"""Lightning module for peak classification.

Wraps a :class:`~dnanet.models.peak_classifier.PeakClassificationModel`
with multi-class training logic, metrics, and optimizer configuration.

Design pattern: **Mediator**
    Coordinates model, loss, optimizer, scheduler, and metrics — each
    independently configurable via Hydra.
"""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torchmetrics
from torch import Tensor, nn

from loguru import logger


class ClassificationModule(L.LightningModule):
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
        scheduler_gamma: Exponential LR decay factor. Set to 1.0 to disable.
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
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss_fn"])

        self.model = model
        self.loss_fn = loss_fn

        metrics = torchmetrics.MetricCollection({
            "accuracy": torchmetrics.classification.MulticlassAccuracy(
                num_classes=num_classes, average="micro",
            ),
            "precision": torchmetrics.classification.MulticlassPrecision(
                num_classes=num_classes, average="macro",
            ),
            "recall": torchmetrics.classification.MulticlassRecall(
                num_classes=num_classes, average="macro",
            ),
            "f1": torchmetrics.classification.MulticlassF1Score(
                num_classes=num_classes, average="macro",
            ),
        })
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")

    def forward(self, x: Tensor | tuple[Tensor, ...]) -> Tensor:
        return self.model(x)

    def _shared_step(
        self, batch: tuple[Tensor, ...], stage: str,
    ) -> Tensor:
        """Compute loss and update metrics.

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
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        preds = logits.argmax(dim=1).detach()
        metrics = self.train_metrics if stage == "train" else self.val_metrics
        metrics.update(preds, targets)

        return loss

    def training_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute(), prog_bar=False)
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute(), prog_bar=False)
        self.val_metrics.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        config: dict[str, Any] = {"optimizer": optimizer}

        if self.hparams.scheduler_gamma < 1.0:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.hparams.scheduler_gamma,
            )
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
            }

        return config

    def predict_step(self, batch: Any, batch_idx: int) -> Tensor:
        if isinstance(batch, (tuple, list)):
            if len(batch) == 3:
                peak_data, marker_idx, _ = batch
                logits = self.model((peak_data, marker_idx))
            else:
                logits = self.model(batch[0])
        else:
            logits = self.model(batch)
        return torch.softmax(logits, dim=1)
