# Configuration Reference

DNANet uses [Hydra](https://hydra.cc/) for hierarchical configuration
composition. The master config at `conf/config.yaml` merges independent
config groups (data, model, training, evaluation, logging) into a single
resolved config object.

## Master Config

```yaml
# conf/config.yaml
defaults:
  - data: dnanet_rd
  - model: unet
  - training: segmentation
  - evaluation: segmentation
  - logging: mlflow
  - _self_

task: train              # One of: train, evaluate, cross_validate
seed: 42                 # Random seed for reproducibility
output_dir: outputs/${now:%Y-%m-%d_%H-%M-%S}
verbosity: INFO          # DEBUG, INFO, WARNING, ERROR
checkpoint: null         # Path to checkpoint (for evaluate/resume)
```

## Config Groups

### Data (`conf/data/`)

Controls dataset loading. Each YAML file maps to a supported dataset.

| Key | Type | Description |
|-----|------|-------------|
| `name` | str | Dataset identifier |
| `kit` | str | Kit/scaling strategy (`PPF6C`, `GLOBALFILER`) |
| `dataset_strategy` | str | File handling strategy (`NFI_RND`, `PROVEDIT`) |
| `root` | str | Path to HID files directory |
| `annotations_path` | str/null | Path to annotation files |
| `hid_to_annotations_path` | str/null | CSV mapping HID filenames to annotations |
| `best_ladder_paths_csv` | str/null | CSV mapping samples to ladder files |
| `ladder_alleles_csv` | str/null | CSV with expected ladder alleles |
| `analysis_threshold_type` | str | `DTH` (high) or `DTL` (low) threshold |
| `data_loading_strategy` | str | `raw`, `analyzed`, or `superior` |
| `adjustment_of_annotations` | str/null | `top`, `complete`, or null |
| `limit` | int/null | Max images to load |
| `skip_if_invalid_ladder` | bool | Skip images with bad ladders |
| `include_size_standard` | bool | Include 6th dye (size standard) |

#### Available Data Configs

**`dnanet_rd`** — NFI R&D 2p/5p mixture dataset (PPF6C kit, 350 samples)

**`provedit`** — PROVEDIt court validation dataset (GlobalFiler kit, ~750 samples)

### Model (`conf/model/`)

Defines the neural network architecture and loss function using Hydra's
`_target_` instantiation.

#### `unet` — U-Net Segmentation

```yaml
architecture:
  _target_: dnanet.models.unet.UNet
  depth: 4
  kernel_size: [3, 5]     # (height=dyes, width=signal)
  num_filters: 32
loss:
  _target_: dnanet.models.loss.DiceLoss
```

#### `autoencoder` — Convolutional Autoencoder

```yaml
architecture:
  _target_: dnanet.models.autoencoder.Conv1dAutoencoder
  num_dyes: 5
  signal_length: 4096
  depth: 5
  compression: 16
loss:
  _target_: torch.nn.MSELoss
```

#### `peak_classifier` — Peak Classification

```yaml
architecture:
  _target_: dnanet.models.peak_classifier.PeakClassificationModel
  input_size: 100
  hidden_size: 64
  num_classes: 5
  num_layers: 3
loss:
  _target_: dnanet.models.loss.FocalLoss
```

#### `peaknet` — Combined Classifier

```yaml
architecture:
  _target_: dnanet.models.peaknet.CombinedClassifier
  segmentation_model: ...
  classification_model: ...
  combiner_type: film     # mlp, film, or cross_attention
loss:
  _target_: dnanet.models.loss.FocalLoss
```

### Training (`conf/training/`)

Controls the training loop, optimizer, scheduler, and callbacks.

| Key | Type | Description |
|-----|------|-------------|
| `type` | str | `segmentation`, `classification`, `reconstruction` |
| `max_epochs` | int | Maximum training epochs |
| `batch_size` | int | Batch size |
| `learning_rate` | float | Initial learning rate |
| `weight_decay` | float | L2 regularization |
| `val_fraction` | float | Fraction of data for validation (0.0–1.0) |
| `num_workers` | int | DataLoader workers |
| `threshold` | float | Prediction threshold (segmentation only) |
| `n_folds` | int | Number of cross-validation folds |

#### Early Stopping

```yaml
early_stopping:
  monitor: val/loss
  patience: 5
  min_delta: 0.01
  mode: min
```

#### Checkpointing

```yaml
checkpoint:
  monitor: val/loss
  save_top_k: 1
  mode: min
```

#### Scheduler

```yaml
scheduler:
  _target_: torch.optim.lr_scheduler.ExponentialLR
  gamma: 0.8
```

### Evaluation (`conf/evaluation/`)

```yaml
metrics:
  - pixel_precision
  - pixel_recall
  - pixel_f1_score
  - average_binary_iou
save_predictions: false
predictions_dir: predictions
```

### Logging (`conf/logging/`)

**`mlflow`** — MLflow experiment tracking (default)
```yaml
logger_type: mlflow
mlflow:
  tracking_uri: mlruns
  experiment_name: dnanet
```

**`tensorboard`** — TensorBoard logging

**`csvlogger`** — CSV file logging

## CLI Override Examples

```bash
# Override nested keys with dot notation
dnanet training.learning_rate=0.001 training.max_epochs=100

# Override list values
dnanet model.architecture.kernel_size=[3,7]

# Multi-run (sweep)
dnanet -m training.learning_rate=0.001,0.0001,0.00001
```
