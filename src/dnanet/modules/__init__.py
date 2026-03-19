"""PyTorch Lightning modules for DNANet.

This package contains Lightning modules that wrap model architectures
with training logic (loss, optimizer, metrics, logging). These replace
the original monolithic ``BaseModel`` class.

Available modules:
    - :class:`~dnanet.modules.segmentation.SegmentationModule`
      — Binary segmentation of EPG signals (e.g. with U-Net + Dice loss)
    - :class:`~dnanet.modules.classification.ClassificationModule`
      — Multi-class peak classification (e.g. allele vs noise)
    - :class:`~dnanet.modules.reconstruction.ReconstructionModule`
      — Autoencoder reconstruction of EPG profiles
"""

from dnanet.modules.classification import ClassificationModule
from dnanet.modules.reconstruction import ReconstructionModule
from dnanet.modules.segmentation import SegmentationModule

__all__ = [
    "SegmentationModule",
    "ClassificationModule",
    "ReconstructionModule",
]
