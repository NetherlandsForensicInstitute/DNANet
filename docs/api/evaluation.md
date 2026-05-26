# Evaluation

```{eval-rst}
.. automodule:: dnanet.evaluation
```

## Pixel Metrics

```{eval-rst}
.. automodule:: dnanet.evaluation.metrics.pixel
   :members:
```

Functions that compare predicted and ground-truth binary masks at the
scan-point level.

- `pixel_precision(gt_list, pred_list)` → float
- `pixel_recall(gt_list, pred_list)` → float
- `pixel_f1_score(gt_list, pred_list)` → float
- `average_binary_iou(gt_list, pred_list)` → float

All functions accept lists of numpy arrays (one per sample).

## Allele Metrics

```{eval-rst}
.. automodule:: dnanet.evaluation.metrics.allele
   :members:
```

Metrics that evaluate allele-level accuracy after allele calling.

- `AllelePrecision().update(gt_markers, pred_markers)` / `compute()` → tensor
- `AlleleRecall().update(gt_markers, pred_markers)` / `compute()` → tensor
- `AlleleF1Score().update(gt_markers, pred_markers)` / `compute()` → tensor

## Allele Caller

```{eval-rst}
.. autoclass:: dnanet.evaluation.allele_caller.AlleleCaller
   :members:
.. autoclass:: dnanet.evaluation.allele_caller.NearestBasePairCaller
   :members:
```

Translates pixel-level predictions into discrete allele calls.

## Visualization

```{eval-rst}
.. automodule:: dnanet.evaluation.visualization
   :members:
```

`plot_epg(image, prediction, title)` — Multi-panel EPG plot with optional
prediction overlay.
