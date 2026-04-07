"""Shared Lightning module behavior for DNANet task modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import lightning as L
import torch
import torchmetrics
from lightning import Callback
from loguru import logger
from torch import Tensor, nn


class BaseTaskModule(L.LightningModule, ABC):
    """Common Lightning scaffolding for DNANet training tasks.

    Subclasses provide task-specific metrics, batch handling, and prediction
    behavior while this base class centralizes the train/validation lifecycle
    and optimizer configuration.
    """

    def __init__(self, model: nn.Module, loss_fn: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    @abstractmethod
    def build_metrics(self) -> torchmetrics.MetricCollection:
        """Return the unprefixed metric collection for this task."""

    def initialize_metrics(self) -> None:
        metrics = self.build_metrics()
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")

    @abstractmethod
    def compute_step_outputs(
        self, batch: Any,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return loss, metric predictions, and metric targets for a batch."""

    def _metrics_for_stage(self, stage: str) -> torchmetrics.MetricCollection:
        if stage == "train":
            return self.train_metrics
        if stage == "val":
            return self.val_metrics
        raise ValueError(f"Unsupported stage: {stage}")

    def _shared_step(self, batch: Any, stage: str) -> Tensor:
        loss, preds, targets = self.compute_step_outputs(batch)

        metrics = self._metrics_for_stage(stage)
        metrics.update(preds, targets)

        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log_dict(
            metrics,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        del batch_idx
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        del batch_idx
        self._shared_step(batch, "val")



    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        config: dict[str, Any] = {"optimizer": optimizer}

        if self.hparams.scheduler_gamma < 1.0:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.hparams.scheduler_gamma,
            )
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
            }

        return config



class EpochConsoleLogger(Callback):
    @staticmethod
    def _format(metrics: dict[str, object]) -> str:
        parts = []
        for key, value in sorted(metrics.items()):
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, float):
                parts.append(f"{key}={value:.4f}")
            else:
                parts.append(f"{key}={value}")
        return ", ".join(parts)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        metrics = {
            k: v for k, v in trainer.callback_metrics.items()
            if k.startswith("train/")
        }
        if metrics:
            logger.info(
                "epoch={} train {}",
                trainer.current_epoch,
                self._format(metrics),
            )

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        metrics = {
            k: v for k, v in trainer.callback_metrics.items()
            if k.startswith("val/")
        }
        if metrics:
            logger.info(
                "epoch={} val {}",
                trainer.current_epoch,
                self._format(metrics),
            )
