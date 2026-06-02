# Base Module

The `dnanet.modules.base` module provides shared Lightning module scaffolding
for DNANet training tasks.

## BaseTaskModule

Common Lightning scaffolding for DNANet training tasks. Subclasses provide
task-specific metrics, batch handling, and prediction behavior while this
base class centralizes the train/validation lifecycle and optimizer
configuration.

```python
from dnanet.modules.base import BaseTaskModule
```

**Constructor args:**
- `model` — Any `nn.Module` for the neural network
- `loss_fn` — Loss function module
- `optimizer` — Optimizer instance (required for `configure_optimizers`)
- `metrics` — `torchmetrics.MetricCollection` (default: empty)
- `lr_scheduler` — Optional learning-rate scheduler
- `batch_size` — Batch size for logging

**Key methods:**
- `forward(*args, **kwargs)` — Delegates to `self.model`
- `compute_step_outputs(batch)` — Abstract: return `(loss, preds, targets)`
- `compute_test_step_outputs(batch)` — Return `(loss, preds, targets, callback_preds)`
- `compute_validation_step_outputs(batch)` — Return `(loss, preds, targets, callback_preds)`
- `training_step(batch, batch_idx)` — Shared training step
- `validation_step(batch, batch_idx)` — Validation step with optional callback preds
- `test_step(batch, batch_idx)` — Test step with optional callback preds
- `configure_optimizers()` — Optimizer + scheduler configuration
- `transfer_batch_to_device(batch, device, dataloader_idx)` — Handles metadata batches

**Metrics:** Maintains separate `MetricCollection` instances for train, val,
and test stages, prefixed with `train/`, `val/`, and `test/` respectively.
