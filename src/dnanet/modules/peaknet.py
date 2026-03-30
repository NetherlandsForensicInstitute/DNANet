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
import lightning as L
import torchmetrics
from torch import Tensor, nn


class PeakNetModule(L.LightningModule):
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
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'loss_fn'])

        self.model = model
        self.loss_fn = loss_fn

        metrics = torchmetrics.MetricCollection(
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
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')

    def forward(
        self,
        full_image: Tensor,
        peak_windows: Tensor,
        marker_idxs: Tensor,
        peak_centers: Tensor,
        peak_counts: Tensor,
    ) -> Tensor:
        return self.model(
            full_image,
            peak_windows,
            marker_idxs,
            peak_centers,
            peak_counts,
        )

    def _shared_step(
        self,
        batch: tuple[Tensor, ...],
        stage: str,
    ) -> Tensor:
        """Compute per-position loss and update metrics.

        Expects batch from :func:`~dnanet.data.datamodule.peaknet_collate_fn`:
        ``(full_images, peak_windows, marker_idxs, peak_centers, peak_counts, targets)``
        """
        full_images, peak_windows, marker_idxs, peak_centers, peak_counts, targets = batch

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
        N, K, C, L_dim = logits.shape
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, K)
        targets_flat = targets.reshape(-1)

        loss = self.loss_fn(logits_flat, targets_flat)
        self.log(f'{stage}/loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        # Metrics on argmax predictions
        preds_flat = logits_flat.argmax(dim=1).detach()
        metrics = self.train_metrics if stage == 'train' else self.val_metrics
        metrics.update(preds_flat, targets_flat)

        return loss

    def training_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> Tensor:
        return self._shared_step(batch, 'train')

    def validation_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> None:
        self._shared_step(batch, 'val')

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

        config: dict[str, Any] = {'optimizer': optimizer}

        if self.hparams.scheduler_gamma < 1.0:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.hparams.scheduler_gamma,
            )
            config['lr_scheduler'] = {
                'scheduler': scheduler,
                'interval': 'epoch',
            }

        return config

    def predict_step(self, batch: Any, batch_idx: int) -> Tensor:
        full_images, peak_windows, marker_idxs, peak_centers, peak_counts, _ = batch
        logits = self.model(
            full_images,
            peak_windows,
            marker_idxs,
            peak_centers,
            peak_counts,
        )
        return torch.softmax(logits, dim=1)
