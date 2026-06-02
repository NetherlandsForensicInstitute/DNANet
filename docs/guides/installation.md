# Installation

## Requirements

- Python 3.12–3.14
- PyTorch 2.0+ (with MPS/CUDA support recommended)

## Install from Source

```bash
git clone <repo-url> DNANet-clean
cd DNANet-clean

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with development dependencies
pip install -e ".[dev]"
```

## Verify Installation

```bash
# Check CLI is available
dnanet --help

# Run the test suite
pytest tests/ -q
```

## Data Setup

DNANet expects forensic HID data to be placed under `data/` relative to the
project root. Two datasets are supported out of the box:

### NFI R&D Dataset (2p/5p)

Place the dataset so the directory structure looks like:

```
data/
  2p_5p_Dataset_NFI/
    Raw data .HID files/
      Mixture dataset 1/...
      Mixture dataset 2/...
      ...
    txt_annotations_2024/
      Dataset 1 DTH_AlleleReport.txt
      ...
    2p_5p_hid_to_annotation.csv
    best_ladder_paths_DTH.csv
    ladder_alleles.csv
```

### PROVEDIt Dataset

```
data/
  PROVEDIt/
    5 sec/
      injection_dir_1/
        *.hid
      ...
    PROVEDIt_RD14-0003 GF Known Genotypes.xlsx
```

## Optional Dependencies

| Package | Purpose |
|---------|---------|
| `mlflow` | Experiment tracking (default logger) |
| `tensorboard` | Alternative experiment tracking |
| `openpyxl` | Reading ProvedIt XLSX genotype files |
| `mkdocs` | Building documentation with Zensical |
