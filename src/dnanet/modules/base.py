"""Shared Lightning module behavior for DNANet task modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
from collections.abc import Mapping, Sequence

import lightning as L
import torchmetrics
from loguru import logger
from lightning import Callback
from torchmetrics import MetricCollection


if TYPE_CHECKING:
    import torch
    from torch import Tensor, nn


class BaseTaskModule(L.LightningModule, ABC):
    """Common Lightning scaffolding for DNANet training tasks.

    Subclasses provide task-specific metrics, batch handling, and prediction
    behavior while this base class centralizes the train/validation lifecycle
    and optimizer configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        metrics: torchmetrics.MetricCollection | None = None,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        batch_size: int | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.batch_size = batch_size
        if metrics is None:
            metrics = MetricCollection([])
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')
        self.enable_validation_callback_preds = False

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    @abstractmethod
    def compute_step_outputs(
        self,
        batch: Any,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return loss, metric predictions, and metric targets for a batch."""

    def compute_test_step_outputs(
        self,
        batch: Any,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Return test loss, metric inputs, and optional callback predictions."""
        loss, preds, targets = self.compute_step_outputs(batch)
        return loss, preds, targets, None

    def compute_validation_step_outputs(
        self,
        batch: Any,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Return validation loss, metric inputs, and optional callback predictions."""
        loss, preds, targets = self.compute_step_outputs(batch)
        return loss, preds, targets, None

    def _metrics_for_stage(self, stage: str) -> torchmetrics.MetricCollection:
        if stage == 'train':
            return self.train_metrics
        if stage == 'val':
            return self.val_metrics
        if stage == 'test':
            return self.test_metrics
        raise ValueError(f'Unsupported stage: {stage}')

    def _shared_step(self, batch: Any, stage: str) -> Tensor:
        loss, preds, targets = self.compute_step_outputs(batch)
        self._log_step_outputs(loss, preds, targets, stage)
        return loss

    def _log_step_outputs(
        self,
        loss: Tensor,
        preds: Tensor,
        targets: Tensor,
        stage: str,
    ) -> None:
        metrics = self._metrics_for_stage(stage)
        if len(metrics) > 0:
            metrics.update(preds, targets)

        self.log(
            f'{stage}/loss',
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=self.batch_size,
        )
        if len(metrics) > 0:
            self.log_dict(
                metrics,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=self.batch_size,
            )

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        del batch_idx
        return self._shared_step(batch, 'train')

    @staticmethod
    def _callback_metric_outputs(preds: Tensor, targets: Tensor) -> dict[str, Tensor]:
        return {
            'metric_preds': preds.detach(),
            'targets': targets.detach(),
        }

    def validation_step(self, batch: Any, batch_idx: int) -> dict[str, Tensor]:
        del batch_idx
        if self.enable_validation_callback_preds:
            loss, preds, targets, callback_preds = self.compute_validation_step_outputs(batch)
        else:
            loss, preds, targets = self.compute_step_outputs(batch)
            callback_preds = None
        self._log_step_outputs(loss, preds, targets, 'val')
        outputs = self._callback_metric_outputs(preds, targets)
        if callback_preds is not None:
            outputs['preds'] = callback_preds.detach()
        return outputs

    def test_step(self, batch: Any, batch_idx: int) -> dict[str, Tensor] | None:
        del batch_idx
        loss, preds, targets, callback_preds = self.compute_test_step_outputs(batch)
        self._log_step_outputs(loss, preds, targets, 'test')
        outputs = self._callback_metric_outputs(preds, targets)
        if callback_preds is not None:
            outputs['preds'] = callback_preds.detach()
        return outputs

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        if self._is_metadata_batch(batch):
            data_batch = (batch[0], batch[1])
            moved_data_batch = super().transfer_batch_to_device(data_batch, device, dataloader_idx)
            return moved_data_batch[0], moved_data_batch[1], batch[2]

        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    @classmethod
    def _is_metadata_batch(cls, batch: Any) -> bool:
        return (
            isinstance(batch, (tuple, list))
            and len(batch) == 3
            and cls._is_metadata_sequence(batch[2])
        )

    @staticmethod
    def _is_metadata_sequence(metadata: Any) -> bool:
        return (
            isinstance(metadata, Sequence)
            and not isinstance(metadata, (str, bytes))
            and all(isinstance(sample_metadata, Mapping) for sample_metadata in metadata)
        )

    def configure_optimizers(self) -> dict[str, Any]:
        if self.optimizer is None:
            raise ValueError('Optimizer must be provided to configure_optimizers')

        config: dict[str, Any] = {'optimizer': self.optimizer}

        if self.lr_scheduler is not None:
            config['lr_scheduler'] = {'scheduler': self.lr_scheduler, 'interval': 'epoch'}

        return config
