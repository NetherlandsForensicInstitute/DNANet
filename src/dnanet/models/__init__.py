"""Neural network architectures for DNA profile analysis.

This package contains the model architectures (pure ``nn.Module`` classes)
used by DNANet. These are *stateless* network definitions — training logic
lives in the ``dnanet.modules`` package (Lightning modules).

Available architectures:

Segmentation:
    - :class:`~dnanet.models.unet.UNet` — U-Net for binary segmentation

Reconstruction:
    - :class:`~dnanet.models.autoencoder.Conv1dAutoencoder`
    - :class:`~dnanet.models.autoencoder.PerDyeConv1dAutoencoder`
    - :class:`~dnanet.models.autoencoder.SharedWeightPerDyeConv1dAutoencoder`
    - :class:`~dnanet.models.autoencoder.UNet2DAutoEncoder`
    - :class:`~dnanet.models.autoencoder.FourierAutoencoder`

Classification:
    - :class:`~dnanet.models.peak_classifier.PeakClassificationModel`

Combined:
    - :class:`~dnanet.models.peaknet.CombinedClassifier`
    - :class:`~dnanet.models.peaknet.PeakOnlyClassifier`

Loss functions:
    - :class:`~dnanet.models.loss.DiceLoss` — Dice loss for segmentation
    - :class:`~dnanet.models.loss.FocalLoss` — Focal loss for classification
"""

from dnanet.models.unet import UNet  # noqa
from dnanet.models.autoencoder import (
    Conv1dAutoencoder,
    FourierAutoencoder,
    PerDyeConv1dAutoencoder,
    SharedWeightPerDyeConv1dAutoencoder,
)
from dnanet.models.loss import DiceLoss, FocalLoss
from dnanet.models.peak_classifier import PeakClassificationModel
from dnanet.models.peaknet import CombinedClassifier, PeakOnlyClassifier


__all__ = [
    # Segmentation
    'UNet',
    # Reconstruction
    'Conv1dAutoencoder',
    'PerDyeConv1dAutoencoder',
    'SharedWeightPerDyeConv1dAutoencoder',
    'FourierAutoencoder',
    # Classification
    'PeakClassificationModel',
    # Combined
    'CombinedClassifier',
    'PeakOnlyClassifier',
    # Loss
    'DiceLoss',
    'FocalLoss',
]
