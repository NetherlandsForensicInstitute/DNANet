"""PyTorch Lightning modules for DNANet.

This package contains Lightning modules that wrap model architectures
with training logic (loss, optimizer, metrics, logging). These replace
the original monolithic ``BaseModel`` class.

Available modules:
    - :class:`~dnanet.modules.base.BaseTaskModule`
      — Shared superclass for DNANet Lightning task modules
    - :class:`~dnanet.modules.segmentation.SegmentationModule`
      — Binary segmentation of EPG signals (e.g. with U-Net + Dice loss)
    - :class:`~dnanet.modules.classification.ClassificationModule`
      — Multi-class peak classification (e.g. allele vs noise)
    - :class:`~dnanet.modules.reconstruction.ReconstructionModule`
      — Autoencoder reconstruction of EPG profiles
    - :class:`~dnanet.modules.peaknet.PeakNetModule`
      — Combined PeakNet per-position classification
"""

from dnanet.modules.base import BaseTaskModule
from dnanet.modules.peaknet import PeakNetModule
from dnanet.modules.segmentation import SegmentationModule
from dnanet.modules.classification import ClassificationModule
from dnanet.modules.reconstruction import ReconstructionModule


__all__ = [
    'BaseTaskModule',
    'SegmentationModule',
    'ClassificationModule',
    'ReconstructionModule',
    'PeakNetModule',
]
