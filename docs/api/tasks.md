# Task Runners

Task runners are Facade functions dispatched by the CLI. Each provides
both a config-driven entry point (`run(cfg)`) and a programmatic entry
point (`run_with_data(cfg, dataset)`).

## Train

```python
from dnanet.tasks.train import run, run_with_data
```

**`run(cfg)`** — Full training pipeline from Hydra config. Loads data,
builds model, trains, saves checkpoint.

**`run_with_data(cfg, dataset)`** — Training with a pre-loaded dataset.
Useful for notebooks and testing.

**Returns:** `(trainer, module)` tuple for programmatic access.

## Evaluate

```python
from dnanet.tasks.evaluate import run, run_with_data
```

**`run(cfg)`** — Load checkpoint, run predictions, compute metrics, save
results.

**`run_with_data(cfg, dataset)`** — Evaluation with a pre-loaded dataset.

**Returns:** `dict[str, float]` — metric name → value.

## Cross-Validate

```python
from dnanet.tasks.cross_validate import run, run_with_data
```

**`run(cfg)`** — K-fold cross-validation from config.

**`run_with_data(cfg, dataset)`** — Cross-validation with a pre-loaded
dataset.

**Returns:** `{"per_fold": [...], "aggregate": {...}}` with mean ± std
for each metric.

## Infer

```python
from dnanet.tasks.infer import run
```

**`run(cfg)`** — Run allele calling on HID profiles from a trained model.
Loads checkpoint, parses HID files, runs inference, calls alleles, and
saves results (JSON + optional plots/predictions).

**Returns:** `None` — results are saved to `output_dir`.

**Config keys:**

| Key | Type | Default | Description |
|---|---|---|---|
| `checkpoint` | str | *required* | Path to trained model checkpoint |
| `hid_profiles` | list or str | *required* | HID file paths (with optional ladder paths) |
| `kit` | str | `PPF6C` | Kit name: `PPF6C`, `GF`, `PY23` |
| `scaling_strategy` | str | — | Alternative: direct strategy class name |
| `caller` | str | `nearest` | Allele caller strategy name |
| `prediction_threshold` | float | `0.5` | Min prediction probability for allele call |
| `confidence_threshold` | float | `None` | Min confidence to include allele |
| `save_predictions` | bool | `False` | Save raw prediction arrays |
| `save_plots` | bool | `False` | Save EPG plots |
| `output_dir` | str | — | Directory for outputs |
| `save_json` | bool | `True` | Save `inference_results.json` |
| `device` | str | `auto` | Device: `cuda`, `cpu`, or `auto` |

```bash
dnanet task=infer checkpoint=/path/to/best.ckpt kit=PPF6C hid_profiles='["sample1.HID"]'
dnanet task=infer checkpoint=best.ckpt kit=GF hid_profiles='[["sample.GF", "ladder.GF"]]' save_plots=true output_dir=outputs/
```

## CLI

```python
from dnanet.cli import main
```

Hydra-decorated main function. Dispatches to the appropriate task runner
based on `cfg.task`.

```bash
dnanet task=train data=dnanet_rd model=unet training=segmentation
dnanet task=evaluate checkpoint=best.ckpt
dnanet task=cross_validate training.n_folds=5
dnanet task=infer checkpoint=best.ckpt kit=PPF6C hid_profiles='["sample.HID"]'
```
