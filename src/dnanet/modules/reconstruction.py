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

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from dnanet.modules.base import BaseTaskModule
from dnanet.data.preprocessing.scaling import inverse_scale_rfu_torch


if TYPE_CHECKING:
    from torchmetrics import MetricCollection


class ReconstructionModule(BaseTaskModule):
    """PyTorch Lightning module for autoencoder reconstruction.

    Args:
        model: Autoencoder network (must have ``encode``/``decode``).
        loss_fn: Loss function (default: ``nn.MSELoss``).
        optimizer: Optimizer instance for training.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization.
        lr_scheduler: Optional learning-rate scheduler.
        metrics: Metric collection used for train/validation logging.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module | None = None,
        *,
        optimizer: torch.optim.Optimizer,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        metrics: MetricCollection | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        autoencoder_log_scale: bool = True,
        autoencoder_max_rfu: int | None = None,
    ) -> None:
        if lr_scheduler is None:
            lr_scheduler = scheduler

        super().__init__(
            model=model,
            loss_fn=loss_fn or nn.MSELoss(),
            optimizer=optimizer,
            metrics=metrics,
            lr_scheduler=lr_scheduler,
        )
        self.save_hyperparameters({
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        })

        self.autoencoder_log_scale = autoencoder_log_scale
        self.autoencoder_max_rfu = autoencoder_max_rfu

    def compute_step_outputs(
        self, batch: Tensor | tuple[Tensor, ...],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute reconstruction loss and metric inputs.

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

        # denormalize
        reconstruction = inverse_scale_rfu_torch(
            reconstruction,
            self.autoencoder_log_scale,
            self.autoencoder_max_rfu
        )

        loss = self.loss_fn(reconstruction, target)
        return loss, reconstruction.detach().reshape(-1), target.reshape(-1)

    def predict_step(self, batch: Any, batch_idx: int) -> Tensor:
        del batch_idx
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        return self.model(x)
