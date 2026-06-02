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
```
