"""Loss functions for DNA profile analysis.

Design pattern: **Strategy**
    Each loss function is a self-contained ``nn.Module`` that can be
    swapped in via Hydra config (``model.loss._target_``). The training
    module doesn't need to know which loss it's using.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DiceLoss(nn.Module):
    """Sørensen–Dice loss for binary segmentation.

    Computes ``1 - Dice coefficient`` where:

        Dice = (2 × |A ∩ B| + smooth) / (|A| + |B| + smooth)

    The input logits are passed through sigmoid before computing the
    coefficient. This loss is particularly effective for imbalanced
    segmentation tasks (e.g. allele peaks occupy a small fraction of
    the electropherogram).

    Args:
        smooth: Laplace smoothing term to avoid division by zero.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute Dice loss.

        Args:
            inputs: Raw logits of any shape.
            targets: Binary ground truth with the same shape.

        Returns:
            Scalar loss value in [0, 1].
        """
        inputs = torch.sigmoid(inputs)

        inputs_flat = inputs.reshape(-1)
        targets_flat = targets.reshape(-1)

        intersection = (inputs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            inputs_flat.sum() + targets_flat.sum() + self.smooth
        )

        return 1.0 - dice


class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced classification.

    Reduces the contribution of easy examples so that the model focuses
    on hard, misclassified samples.

    Args:
        alpha: Weighting factor for the positive class.
        gamma: Focusing parameter — higher values down-weight easy examples more.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute focal loss.

        Args:
            inputs: Raw logits of shape ``(B, C)`` or ``(B, C, ...)``.
            targets: Class indices of shape ``(B,)`` or ``(B, ...)``.

        Returns:
            Scalar loss value.
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
