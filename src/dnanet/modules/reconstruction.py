"""Lightning module for autoencoder reconstruction.

Wraps autoencoder architectures with MSE-based training, optional
RFU-space evaluation, and optimizer configuration.

The autoencoder is trained in preprocessed space (log-scaled,
normalized) but metrics can be computed in original RFU space for
interpretability.

Design pattern: **Mediator**
    Coordinates model, loss, preprocessing, and metrics.
"""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torchmetrics
from torch import Tensor, nn

from loguru import logger


class ReconstructionModule(L.LightningModule):
    """PyTorch Lightning module for autoencoder reconstruction.

    Args:
        model: Autoencoder network (must have ``encode``/``decode``).
        loss_fn: Loss function (default: ``nn.MSELoss``).
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization.
        scheduler_gamma: Exponential LR decay. Set to 1.0 to disable.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        scheduler_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss_fn"])

        self.model = model
        self.loss_fn = loss_fn or nn.MSELoss()

        metrics = torchmetrics.MetricCollection({
            "mse": torchmetrics.regression.MeanSquaredError(),
        })
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _shared_step(self, batch: tuple[Tensor, ...], stage: str) -> Tensor:
        """Compute reconstruction loss.

        Expects batch as ``(input_tensor,)`` or ``(input_tensor, target_tensor)``.
        For autoencoders the target is typically the input itself.
        """
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x, target = batch[0], batch[1]
        else:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            target = x

        reconstruction = self.model(x)

        # Handle shape mismatches from trailing singleton dims
        if reconstruction.dim() == 4 and reconstruction.shape[-1] == 1:
            reconstruction = reconstruction.squeeze(-1)
        if target.dim() == 4 and target.shape[-1] == 1:
            target = target.squeeze(-1)

        loss = self.loss_fn(reconstruction, target)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # Update metrics
        metrics = self.train_metrics if stage == "train" else self.val_metrics
        metrics.update(reconstruction.detach().reshape(-1), target.reshape(-1))

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
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        return self.model(x)
