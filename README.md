# DNANet

Deep learning framework for forensic DNA electropherogram (EPG) analysis.

Developed at the Netherlands Forensic Institute (NFI).

## Capabilities

End-to-end pipelines for STR profile analysis from capillary electrophoresis `.hid` files:

- **Segmentation** — Binary pixel-level labeling of EPG signals (U-Net + Dice loss)
- **Classification** — Multi-class peak classification (allele vs stutter vs noise)
- **Reconstruction** — Autoencoder-based EPG profile reconstruction
- **Combined PeakNet** — Per-position classification with integrated classification head

## Architecture

```
src/dnanet/
├── cli.py                  # Hydra entry point (train / evaluate / cross_validate)
├── core/                   # Domain primitives (allele, marker, panel, constants, annotation)
├── data/                   # Data loading & preprocessing
│   ├── datamodule.py       # Lightning DataModule
│   ├── splitting.py        # Train/val/test splits, cross-validation
│   ├── preprocessing/      # Peak extraction, baseline correction, scaling
│   ├── strategies/         # Dataset strategies, scaling strategies (PPF6C, GlobalFiler, Y23)
│   ├── cache/              # Cached data pipeline
│   ├── ladders/            # Allele ladder catalogs
│   └── parsing/            # .hid file parser
├── models/                 # Neural network architectures (nn.Module)
│   ├── unet.py             # U-Net segmentation
│   ├── autoencoder.py      # 1D autoencoders (standard, per-dye, shared-weight, Fourier)
│   ├── peak_classifier.py  # Peak classification network
│   ├── peaknet.py          # Combined classifier, peak-only classifier
│   └── loss.py             # DiceLoss, FocalLoss
├── modules/                # Lightning modules (training logic + metrics)
│   ├── base.py             # BaseTaskModule
│   ├── segmentation.py     # SegmentationModule
│   ├── classification.py   # ClassificationModule
│   ├── reconstruction.py   # ReconstructionModule
│   └── peaknet.py          # PeakNetModule
├── tasks/                  # Task runners (dispatched by CLI)
│   ├── train.py
│   ├── evaluate.py
│   └── cross_validate.py
├── evaluation/             # Metrics & visualization
│   ├── metrics/            # Allele-level, per-RFU metrics
│   ├── allele_caller.py    # Allele calling strategies
│   ├── callbacks.py        # Lightning callbacks for evaluation
│   └── visualization.py    # EPG plotting
├── tools/                  # CLI tools
│   └── labeltool/          # Interactive annotation tool (dnanet-label)
└── logging.py              # Loguru configuration
```

### Design patterns

- **Command** — CLI dispatches to task functions, each receives full Hydra config
- **Composition over Inheritance** — Hydra composes YAML config groups (data, model, training, logging) at runtime
- **Strategy** — Kit-agnostic scaling via strategy pattern (PowerPlex Fusion 6C, GlobalFiler, PowerPlex Y23)

## Tech stack

- **Deep learning**: PyTorch 2.5+, Lightning 2.4+, torchmetrics
- **Configuration**: Hydra + OmegaConf (YAML config groups, CLI overrides)
- **Data**: NumPy, SciPy, construct
- **Tracking**: MLflow (experiment tracking)
- **Logging**: Loguru
- **Quality**: Ruff, mypy, pytest + coverage
- **Build**: hatchling

## Install

```bash
pip install -e ".[tools]"
```

Requires Python 3.12–3.14.

## Usage

```bash
# Train U-Net segmentation on NFI R&D dataset
dnanet task=train data=dnanet_rd model=unet

# Train on ProvedIT dataset
dnanet task=train data=provedit model=unet

# Evaluate a checkpoint
dnanet task=evaluate data=dnanet_rd model=unet checkpoint=outputs/.../best.ckpt

# 5-fold cross-validation
dnanet task=cross_validate data=dnanet_rd model=unet

# Override any config from CLI
dnanet task=train data=dnanet_rd model=unet training.learning_rate=0.0001 training.batch_size=32

# Interactive annotation tool
dnanet-label
```

## Configuration

All parameters composed from `conf/` YAML groups:

| Group | Path | Purpose |
|-------|------|---------|
| data | `conf/data/*.yaml` | Dataset selection & loading |
| model | `conf/model/*.yaml` | Architecture hyperparameters |
| training | `conf/train/*.yaml` | Optimizer, LR, epochs, early stopping |
| evaluation | `conf/evaluate/*.yaml` | Eval metrics & allele calling |
| splitting | `conf/splitting/*.yaml` | Train/val/test ratios, CV folds |
| logging | `conf/logging/*.yaml` | CSV logger, TensorBoard, MLflow |
| metrics | `conf/metrics/*.yaml` | Binary, multi-class, MSE metrics |

Master config: `conf/config.yaml` — Hydra merges selected groups at runtime.

## Datasets

| Config key | Description |
|------------|-------------|
| `dnanet_rd` | NFI R&D internal dataset |
| `provedit` | ProvedIT benchmark dataset |
| `peaks_rd` | Pre-extracted peaks |

## Evaluation

Metrics computed per task:

- **Segmentation**: Dice score, pixel accuracy, precision, recall
- **Classification**: Per-class precision, recall, F1; allele-level metrics
- **Reconstruction**: MSE, per-RFU error analysis
- **Allele calling**: Genotype Concordance Rate (GCR), stutter detection

## Tests & quality

```bash
# Run tests with coverage
pdm run pytest

# Lint
pdm run ruff check

# Type check
pdm run mypy
```

## License

Apache-2.0
