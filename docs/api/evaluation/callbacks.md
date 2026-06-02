# Lightning Evaluation Callbacks

The `dnanet.evaluation.callbacks` package provides Lightning `Callback`
implementations for evaluation-time domain metrics.

## AlleleMetricsCallback

Compute allele-level metrics (precision, recall, F1) during Lightning
validation and test runs.

```python
from dnanet.evaluation.callbacks.allele_metrics import AlleleMetricsCallback
```

**Constructor args:**
- `allele_caller` — An `AlleleCaller` instance for translating predictions
- `precision` / `recall` / `f1` — Metric instances (auto-created if not provided)
- `skip_missing_annotations` — Skip samples without allele annotations (default: True)

**Lifecycle hooks:** `on_validation_epoch_start`, `on_validation_batch_end`,
`on_validation_epoch_end`, `on_test_epoch_start`, `on_test_batch_end`,
`on_test_epoch_end`.

## ConfusionMatrixCallback

Collect and save multiclass confusion matrices for validation/test runs.

Outputs per stage:
- `confusion_matrix_counts.csv` — Raw counts
- `confusion_matrix_normalized.csv` — Row-normalized values
- `confusion_matrix.png` — Heatmap plot (seaborn)

```python
from dnanet.evaluation.callbacks.confusion_matrix import ConfusionMatrixCallback
```

**Constructor args:**
- `num_classes` — Number of output classes (minimum: 2)
- `class_names` — Class name labels (auto-derived from `LabelCategory` if not provided)
- `output_dir` — Output directory (default: "confusion_matrix")
- `stages` — Which stages to collect ("val", "test"; default: both)
- `ignore_index` — Label index to ignore
- `normalize` — Output normalized values in plot (default: True)
- `threshold` — Binary classification threshold (default: 0.5)
- `annot` — Annotate heatmap cells (default: True)
- `cmap` — Colormap for heatmap (default: "Blues")
- `filename_prefix` — Prefix for output files

## PerRFUOutcomeCallback

Collect TP/FP/FN RFU values during Lightning test runs.

Outputs an NPZ or CSV file with RFU values grouped by outcome type.

```python
from dnanet.evaluation.callbacks.per_rfu_outcome import PerRFUOutcomeCallback
```

**Constructor args:**
- `threshold` — Prediction threshold (default: 0.5)
- `filename` — Output filename (default: "per_rfu_outcomes.npz")
- `metric` — `PerRFUOutcomeMetric` instance (auto-created if not provided)

## ProfilePlotCallback

Save a limited number of evaluation profile plots during test runs.

Uses `plot_profile()` from `dnanet.evaluation.visualization` to generate
multi-panel EPG plots with optional annotation and prediction overlays.

```python
from dnanet.evaluation.callbacks.profile_plot import ProfilePlotCallback
```

**Constructor args:**
- `include_annotations` — Show ground-truth annotations (default: True)
- `include_predictions` — Show model predictions (default: True)
- `num_profiles` — Maximum number of profiles to save (default: 5)

Outputs to `{root_dir}/plots/profile_XXXX_{sample_name}.png`
