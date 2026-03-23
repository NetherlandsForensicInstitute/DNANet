# Quick Start

This guide walks through training your first model with DNANet.

## 1. Train a Segmentation Model

The simplest command trains a U-Net on the NFI R&D dataset:

```bash
dnanet task=train data=dnanet_rd model=unet training=segmentation
```

This will:
1. Load all HID files from `data/2p_5p_Dataset_NFI/`
2. Parse size standards and rescale profiles to 4096 scan points
3. Build segmentation masks from called allele annotations
4. Split data 80/20 into train/validation sets
5. Train for 15 epochs with early stopping (patience=5)
6. Save checkpoints to `outputs/<timestamp>/checkpoints/`
7. Log metrics to MLflow (`mlruns/`)

## 2. Override Parameters from the CLI

Hydra allows overriding any config parameter from the command line:

```bash
# Limit dataset size for quick testing
dnanet task=train data=dnanet_rd model=unet training=segmentation data.limit=20

# Change hyperparameters
dnanet task=train data=dnanet_rd model=unet training=segmentation \
    training.learning_rate=0.001 \
    training.max_epochs=50 \
    training.batch_size=8

# Use a different logger
dnanet task=train data=dnanet_rd model=unet training=segmentation \
    logging=tensorboard

# Use ProvedIt dataset with GlobalFiler kit
dnanet task=train data=provedit model=unet training=segmentation
```

## 3. Evaluate a Trained Model

```bash
dnanet task=evaluate \
    data=dnanet_rd model=unet \
    checkpoint=outputs/2024-01-01_12-00-00/checkpoints/best-epoch=10-val_loss=0.1234.ckpt \
    evaluation=segmentation
```

## 4. Cross-Validation

```bash
dnanet task=cross_validate \
    data=dnanet_rd model=unet training=segmentation \
    training.n_folds=5
```

This trains 5 models (one per fold), evaluates each on its held-out fold,
and reports aggregate metrics with mean and standard deviation.

## 5. Programmatic Usage

For notebooks or custom scripts:

```python
from hydra import compose, initialize
from dnanet.data.loading import load_dataset
from dnanet.tasks.train import run_with_data

# Compose config
with initialize(config_path="../conf"):
    cfg = compose(config_name="config", overrides=[
        "data=dnanet_rd",
        "model=unet",
        "training=segmentation",
        "data.limit=50",
    ])

# Load dataset
dataset = load_dataset(cfg.data)
print(f"Loaded {len(dataset)} images")
print(f"Shape: {dataset[0].data.shape}")  # (5, 4096, 1)

# Train
trainer, module = run_with_data(cfg, dataset)
```
