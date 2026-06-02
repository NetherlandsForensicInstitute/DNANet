# Per-RFU Outcome Metrics

The `dnanet.evaluation.metrics.per_RFU` module provides RFU outcome collection
and binned F1 computation for precision/recall/F1 analysis by RFU threshold.

## PerRFUOutcomeMetric

Collect RFU values for true positives, false positives, and false negatives.

```python
from dnanet.evaluation.metrics.per_RFU import PerRFUOutcomeMetric
```

The metric intentionally does not bin RFU values during evaluation. It stores
only the RFU values needed to compute precision, recall, and F1 later with
arbitrary bin edges.

**Constructor args:**
- `threshold` — Prediction threshold (default: 0.5)

**State:**
- `tp_rfus` — RFU values for true positives
- `fp_rfus` — RFU values for false positives
- `fn_rfus` — RFU values for false negatives

**Methods:**
- `update(preds, targets, rfu_values)` — Collect RFU values for a batch
- `compute()` → `dict[str, Tensor]` — Return accumulated RFU values
- `compute_outcomes()` → `dict[str, Tensor]` — Stable one-dimensional tensors

## RFU I/O Functions

```python
from dnanet.evaluation.metrics.per_RFU import (
    write_rfu_outcome_file,
    write_rfu_outcome_npz,
    write_rfu_outcome_csv,
    load_rfu_outcome_npz,
    load_rfu_outcome_csv,
)
```

- `write_rfu_outcome_file(path, outcomes)` — Write based on file extension (`.npz` or `.csv`)
- `write_rfu_outcome_npz(path, outcomes)` — Write as compressed NPZ archive
- `write_rfu_outcome_csv(path, outcomes)` — Write as compact `outcome,rfu` CSV rows
- `load_rfu_outcome_npz(path)` → `dict[str, np.ndarray]` — Load NPZ file
- `load_rfu_outcome_csv(path)` → `dict[str, np.ndarray]` — Load CSV file

## Binned F1 Computation

```python
from dnanet.evaluation.metrics.per_RFU import (
    compute_binned_f1,
    compute_binned_f1_from_npz,
    compute_binned_f1_from_csv,
)
```

- `compute_binned_f1(outcomes, bin_edges)` → `list[dict]` — Compute precision, recall, and F1 for arbitrary RFU bins
- `compute_binned_f1_from_npz(path, bin_edges)` — Load NPZ and compute binned F1
- `compute_binned_f1_from_csv(path, bin_edges)` — Load CSV and compute binned F1

Each returned dict contains: `bin_left`, `bin_right`, `tp`, `fp`, `fn`,
`support`, `precision`, `recall`, `f1`.

## Constants

- `RFU_OUTCOMES` = `('tp', 'fp', 'fn')`
