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

The dataset is hosted on Hugging Face:
[`NetherlandsForensicInstitute/DNANet_2p5pMixture_PPF6C_2024`](https://huggingface.co/datasets/NetherlandsForensicInstitute/DNANet_2p5pMixture_PPF6C_2024).

**Automatic download.** With the optional `hub` extra installed, `data=dnanet_rd`
downloads the dataset into `data/2p_5p_Dataset_NFI` the first time it is
missing or empty:

```bash
pip install 'dnanet[hub]'   # or: pdm install -G hub
dnanet task=train data=dnanet_rd model=unet training=segmentation
```

Disable it with `data.dataset.hf_repo_id=null`, or pin a version with
`data.dataset.hf_revision=<tag-or-commit>`.

**Manual download.**

```bash
huggingface-cli download NetherlandsForensicInstitute/DNANet_2p5pMixture_PPF6C_2024 \
    --repo-type dataset --local-dir data
```

Either way, the directory structure should look as follows:

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
| `huggingface-hub` | Auto-downloading the NFI R&D dataset (`hub` extra) |
| `mkdocs` | Building documentation with Zensical |
