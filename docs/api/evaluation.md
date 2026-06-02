# Evaluation

## Allele Metrics

Metrics that evaluate allele-level accuracy after allele calling.

```python
from dnanet.evaluation.metrics.allele import AllelePrecision, AlleleRecall, AlleleF1Score
```

- `AllelePrecision().update(gt_markers, pred_markers)` / `compute()` → tensor
- `AlleleRecall().update(gt_markers, pred_markers)` / `compute()` → tensor
- `AlleleF1Score().update(gt_markers, pred_markers)` / `compute()` → tensor

## Allele Caller

Translates pixel-level predictions into discrete allele calls.

```python
from dnanet.evaluation.allele_caller import AlleleCaller, NearestBasePairCaller
```

## Visualization

```python
from dnanet.evaluation.visualization import plot_profile, plot_profile_marker
```

`plot_profile(signal, annotation, prediction, title)` — Plot a full DNA
profile with optional multiclass annotations.

`plot_profile_marker(signal, scaler, marker_bp_range, dye_row, annotation, prediction, title)` — Plot a single marker region of an EPG profile.
