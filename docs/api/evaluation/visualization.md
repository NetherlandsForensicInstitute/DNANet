# Visualization

The `dnanet.evaluation.visualization` module provides matplotlib-based
plotting functions for DNA electropherogram (EPG) profiles.

## Functions

```python
from dnanet.evaluation.visualization import (
    plot_profile,
    plot_profile_marker,
    coerce_class_map,
)
```

### plot_profile

`plot_profile(signal, *, annotation=None, prediction=None, title=None, dye_colors=None, figsize=(20, 20))` — Plot a full DNA profile with optional multiclass annotations.

**Args:**
- `signal` — `(C, L)` EPG signal data (one row per dye channel)
- `annotation` — `(C, L)` class-index ground-truth annotation map
- `prediction` — `(C, L)` class-index prediction map
- `title` — Optional figure title
- `dye_colors` — Colors for each dye channel (defaults to standard forensic dye colors)
- `figsize` — Figure size in inches

**Returns:** The matplotlib `Figure`.

**Layout:** Multi-panel with signal, annotation (optional), and prediction
(optional) tracks for each dye channel.

### plot_profile_marker

`plot_profile_marker(signal, scaler, marker_bp_range, dye_row, *, annotation=None, prediction=None, title=None, zoom=None)` — Plot a single marker region of an EPG profile.

**Args:**
- `signal` — `(C, L)` full EPG signal
- `scaler` — `(L,)` scan-to-base-pair mapping array
- `marker_bp_range` — `(left_bp, right_bp)` bin range of the marker
- `dye_row` — Dye channel index for this marker
- `annotation` — `(C, L)` ground-truth annotation mask
- `prediction` — `(C, L)` predicted mask
- `title` — Plot title (usually the marker name)
- `zoom` — Optional `(start, end)` scan indices to zoom into

**Returns:** The matplotlib `Figure`.

### coerce_class_map

`coerce_class_map(data, *, signal_shape, source)` — Normalize plotting labels to a 2-D class-index map.

Supports `(C, L)`, `(K, C, L)`, and `(C, L, K)` shaped arrays. Binary
probability maps are thresholded at 0.5 for predictions; multiclass arrays
are reduced with `argmax`.

### Private Helpers

The following private functions support the public API:

- `_coerce_signal(signal)` — Normalize signal to `(num_dyes, scanpoints)`
- `_plot_lines(axs, data, color, scale_to, alpha)` — Plot dye channel lines
- `_plot_class_tracks(axs, class_map, lane_label)` — Plot per-class mini-lanes
- `_iter_class_spans(class_row)` — Return contiguous non-zero class spans
- `_plot_segmentation_mask(axs, mask, color, alpha, y_range, min_width)` — Plot mask as colored tracks
- `_iter_mask_spans(mask_row, min_width)` — Return contiguous positive mask spans
