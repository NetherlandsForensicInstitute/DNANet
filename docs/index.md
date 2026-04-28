# DNANet Documentation

**DNANet** is a deep-learning framework for forensic DNA electropherogram (EPG)
analysis, developed at the Netherlands Forensic Institute (NFI). It provides
end-to-end pipelines for segmentation, classification, and reconstruction of
Short Tandem Repeat (STR) profiles from `.hid` files produced by capillary
electrophoresis instruments.

## Key Features

- **Modular architecture** — Clean separation between domain models, data
  loading, neural networks, training logic, and evaluation.
- **Multiple model types** — U-Net segmentation, peak classification,
  autoencoders, and combined classifiers.
- **Kit-agnostic** — Strategy pattern supports PowerPlex Fusion 6C (PPF6C) and
  GlobalFiler kits; new kits can be added by implementing a scaling strategy.
- **Hydra configuration** — All parameters (model, training, data, logging) are
  composed from YAML config groups with full CLI override support.
- **PyTorch Lightning** — Training, evaluation, and cross-validation are
  orchestrated by Lightning, providing automatic GPU support, checkpointing,
  early stopping, and logging.
- **Comprehensive evaluation** — Pixel-level and allele-level metrics, allele
  calling strategies, and EPG visualisation.

## Quick Start

```bash
# Install (editable, with dev dependencies)
pip install -e ".[dev]"

# Train a U-Net on the NFI R&D dataset
dnanet task=train data=dnanet_rd model=unet training=segmentation

# Train on ProvedIt dataset
dnanet task=train data=provedit model=unet training=segmentation

# Evaluate a checkpoint
dnanet task=evaluate data=dnanet_rd model=unet checkpoint=outputs/.../checkpoints/best.ckpt

# 5-fold cross-validation
dnanet task=cross_validate data=dnanet_rd model=unet training=segmentation
```

```{toctree}
:maxdepth: 2
:caption: User Guide

guides/installation
guides/quickstart
guides/configuration
guides/datasets
```

```{toctree}
:maxdepth: 2
:caption: Architecture

architecture/overview
architecture/design-patterns
architecture/data-pipeline
architecture/caching
architecture/models
architecture/training
architecture/evaluation
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/core
api/data
api/models
api/modules
api/evaluation
api/tasks
```
