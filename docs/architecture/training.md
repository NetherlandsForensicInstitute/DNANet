from dnanet.data import HIDDataset

# Training

Training is orchestrated by PyTorch Lightning via the `dnanet/tasks/train.py`
facade and `dnanet/modules/` Lightning modules.

## Lightning Modules

Each training type has a dedicated Lightning module that encapsulates:
- Forward pass
- Loss computation
- Metric logging
- Optimizer and scheduler configuration

### SegmentationModule

Binary segmentation of EPG signals. Wraps any model that produces a
`(B, 1, D, L)` mask.

**Metrics logged:** Accuracy, Precision, Recall, F1, IoU (train and val)
**Default loss:** DiceLoss
**Default threshold:** 0.5

### ClassificationModule

Multi-class peak classification.

**Metrics logged:** Accuracy, Precision, Recall, F1 (train and val)
**Default loss:** CrossEntropyLoss or FocalLoss

### ReconstructionModule

Autoencoder reconstruction of EPG profiles.

**Metrics logged:** MSE (train and val)
**Default loss:** MSELoss

## Training Pipeline

The `train.run(cfg)` function:

1. **Seed** — `L.seed_everything(cfg.seed)` for reproducibility
2. **Model** — Instantiate architecture and loss via Hydra `_target_`
3. **Module** — Select Lightning module by `cfg.training.type`
4. **Data** — Load dataset from config, create `DNANetDataModule`
5. **Callbacks** — Build early stopping and checkpointing
6. **Logger** — Build MLflow, TensorBoard, or CSV logger
7. **Trainer** — Create `L.Trainer` with all components
8. **Fit** — `trainer.fit(module, datamodule=datamodule)`

### Callbacks

**Early Stopping:**
Monitors `val/loss` and stops training after `patience` epochs without
improvement. Prevents overfitting on small forensic datasets.

**Model Checkpointing:**
Saves the best model (by `val/loss`) to `outputs/<timestamp>/checkpoints/`.
Filename includes epoch and loss: `best-epoch=10-val_loss=0.1234.ckpt`.

### Learning Rate Scheduling

By default, an exponential scheduler decays the learning rate by `gamma`
(0.8) each epoch:

```
lr_epoch = lr_initial × gamma^epoch
```

## Cross-Validation

The `cross_validate.run(cfg)` function extends training with k-fold
evaluation:

1. Split dataset into k folds (stratified by number of contributors if possible)
2. For each fold:
   - Train on k-1 folds
   - Evaluate on the held-out fold
   - Save per-fold metrics
3. Aggregate metrics (mean ± std) across all folds
4. Save results to `aggregate_metrics.json`

## Programmatic API

```python

from dnanet.tasks.train import run

# Load data
dataset = HIDDataset(...)

# Train (returns trainer + module for further use)
trainer, module = run(cfg, dataset)

# Access best checkpoint
best_ckpt = trainer.checkpoint_callback.best_model_path
```

## Resuming Training

Pass a checkpoint path to resume from:

```bash
dnanet task=train data=dnanet_rd model=unet training=segmentation \
    checkpoint=outputs/.../checkpoints/best.ckpt
```
