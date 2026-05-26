# DNANet
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FNetherlandsForensicInstitute%2FDNANet%2Frefs%2Fheads%2Fmain%2Fpyproject.toml&logo=python&label=Python&color=bright-green)
![GitHub License](https://img.shields.io/github/license/NetherlandsForensicInstitute/DNANet)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/NetherlandsForensicInstitute/DNANet)
[![pdm-managed](https://img.shields.io/endpoint?url=https%3A%2F%2Fcdn.jsdelivr.net%2Fgh%2Fpdm-project%2F.github%2Fbadge.json)](https://pdm-project.org)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


Deep learning framework for forensic DNA electropherogram (EPG) analysis.
Developed at the Netherlands Forensic Institute (NFI).

This a Python repository that can be used to analyze DNA profiles using deep learning. It contains functionality to parse .hid files and train and 
evaluate models. The pre-trained U-Net provided can be used to call alleles in a DNA profile.

If you find this repository useful, please cite
```bibtex
@ARTICLE{Benschop2019,
      title     = "An assessment of the performance of the probabilistic genotyping
                   software {EuroForMix}: Trends in likelihood ratios and analysis
                   of Type {I} \& {II} errors",
      author    = "Benschop, Corina C G and Nijveld, Alwart and Duijs, Francisca E
                   and Sijen, Titia",
      journal   = "Forensic Sci. Int. Genet.",
      volume    =  42,
      pages     = "31--38",
      year      =  2019,
    }
```
for the data, and 
```bibtex
@ARTICLE{de-Wit2025,
    title = {Making AI accessible for forensic DNA profile analysis},
    journal = {Forensic Science International: Genetics},
    volume = {81},
    pages = {103345},
    year = {2026},
    issn = {1872-4973},
    doi = {https://doi.org/10.1016/j.fsigen.2025.103345},
    url = {https://www.sciencedirect.com/science/article/pii/S1872497325001255},
    author = {
        Abel K.J.G. de Wit and Claire D. Wagenaar and Nathalie A.C. Janssen and Brechtje Hoegen 
        and Judith van de Wetering and Huub Hoofs and Simone Ariëns and Corina C.G. Benschop 
        and Rolf J.F. Ypma
        }
}
```
for the code and model.

For work related to the Data synthetization please cite the following:
```
@ARTICLE{Taylor2025,
    title = {Simulating realistic short tandem repeat capillary electrophoretic signal using a generative adversarial network},
    journal = {Expert Systems with Applications},
    volume = {280},
    pages = {127536},
    year = {2025},
    doi = {https://doi.org/10.1016/j.eswa.2025.127536},
    author = {D. A. Taylor and M. Humphries}
}
```

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

## Pre-commit hooks

Install pre-commit to validate code before each commit and push:

```bash
pdm run pip install pre-commit
pre-commit install
pre-commit install --hook-type pre-push
```

This runs **ruff lint + format** on every commit and **pytest** before each push.

To run all hooks manually:

```bash
pre-commit run --all-files
```

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
