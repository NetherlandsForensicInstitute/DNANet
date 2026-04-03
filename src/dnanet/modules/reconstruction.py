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

import torchmetrics
from torch import Tensor, nn

from dnanet.data.preprocessing.scaling import inverse_scale_rfu_torch
from dnanet.modules.base import BaseTaskModule


class ReconstructionModule(BaseTaskModule):
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
        super().__init__(model=model, loss_fn=loss_fn or nn.MSELoss())
        self.save_hyperparameters({
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "scheduler_gamma": scheduler_gamma,
        })
        self.initialize_metrics()

    def build_metrics(self) -> torchmetrics.MetricCollection:
        return torchmetrics.MetricCollection({
            "mse": torchmetrics.regression.MeanSquaredError(),
        })

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
        log_scale = True
        max_rfu = 33000 ## TODO do not hardcode values
        reconstruction = inverse_scale_rfu_torch(reconstruction, log_scale, max_rfu)

        loss = self.loss_fn(reconstruction, target)
        return loss, reconstruction.detach().reshape(-1), target.reshape(-1)

    def predict_step(self, batch: Any, batch_idx: int) -> Tensor:
        del batch_idx
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        return self.model(x)
