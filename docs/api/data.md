# Data Layer

The `dnanet.data` package handles everything from raw HID files to
PyTorch-ready tensors.

## HIDImage

The central data container. Wraps a path to a HID file with lazy loading.
`HIDImage` receives the kit scaling strategy directly.

```python
from dnanet.data.image import HIDImage
```

**Properties:**
- `data` → `np.ndarray | None` — Shape `(num_dyes, signal_length, 1)`. Triggers load on first access.
- `annotation` → `ScanpointAnnotation | None` — Ground-truth segmentation mask
- `scaler` → `np.ndarray` — Shape `(1, signal_length)`. Maps pixel → base pair.
- `adjusted_panel` → `Panel | None` — Reference panel
- `dimensions` → `tuple[int, int]` — Data array shape `(num_dyes, signal_length)`
- `meta` → `dict` — Metadata (NOC, ladder path, etc.)

**Methods:**
- `adjust_annotations(method)` — Snap annotation mask to actual peaks
  - `"top"` — Label only peak apex
  - `"complete"` — Label entire peak boundary-to-boundary

## Datasets

### TransformableDataset

Abstract base class for datasets of `HIDImage` objects. Provides properties
for `images`, `transform`, and `dataset_strategy`.

```python
from dnanet.data.dataset import TransformableDataset
```

### HIDDataset

Loads HID files from a directory. Implements `TransformableDataset` and
`torch.utils.data.Dataset`.

```python
from dnanet.data.hid_dataset import HIDDataset
```

**Constructor args:** See {doc}`/guides/datasets` for details.

## DataModule

### DNANetDataModule

Lightning DataModule bridging `TransformableDataset` → PyTorch DataLoaders.

```python
from dnanet.data.datamodule import DNANetDataModule
```

**Args:**
- `dataset` — A `TransformableDataset` (e.g. `HIDDataset`)
- `batch_size` — Batch size
- `num_workers` — DataLoader workers
- `shuffle_train` — Whether to shuffle the training DataLoader
- `**split_kwargs` — Passed to `dataset_splitter()`: `val_fraction`, `test_fraction`, `seed`, etc.

## Parsing

```python
from dnanet.data.parsing.hid import get_peak_data
```

### HID Parsing

`get_peak_data(path, strategy, data_loading_strategy)` — Parse a HID file
and return raw/analyzed data as a numpy array.

## Preprocessing

### Peak Detection

```python
from dnanet.data.preprocessing.peaks import (
    find_peaks_above_threshold,
    find_peak_boundary,
    find_peak_near_idx,
    find_peak_idx_near_or_in_range,
    find_valley_idx_in_range,
    find_absolute_peak_idx_in_range,
)
```

- `find_peaks_above_threshold(signal, threshold)` — Detect peaks including
  flat-top peaks
- `find_peak_boundary(signal, peak_idx, threshold)` — Walk left/right to
  find peak start and end
- `find_peak_near_idx(signal, idx)` — Find nearest peak at least as high
- `find_peak_idx_near_or_in_range(signal, index_range, threshold)` — Find
  dominant peak within or near an index range
- `find_valley_idx_in_range(signal, index_range, threshold)` — Find signal
  minimum within an index range
- `find_absolute_peak_idx_in_range(signal, index_range, threshold)` — Find
  peak by absolute value within an index range

### Baseline Estimation

```python
from dnanet.data.preprocessing.baseline import (
    baseline_superior,
    baseline_classic,
    baseline_enhanced,
)
```

- `baseline_superior(signal)` — DNANet's recommended baseline method
  (100-pt window, 20th percentile)
- `baseline_classic(signal)` — Traditional rolling-minimum approach
  (551-pt window, 20th percentile)
- `baseline_enhanced(signal)` — Improved classic with piecewise weighted
  linear fits and Savitzky-Golay smoothing

## Strategies

### ScalingStrategy

Abstract base for kit-specific base-pair calibration.

```python
from dnanet.data.strategies.scaling import ScalingStrategy, PowerPlexFusion6CStrategy, GlobalFilerStrategy
```

**Concrete implementations:**
- `PowerPlexFusion6CStrategy` — PPF6C kit with WEN ILS (bp range 65–475)
- `GlobalFilerStrategy` — GlobalFiler kit with GeneScan 600 LIZ (bp range 60–480)

### DatasetStrategy

Abstract base for dataset-specific file handling.

```python
from dnanet.data.strategies.datasets import DatasetStrategy, NFIRnDStrategy, ProvedItStrategy
```

**Concrete implementations:**
- `NFIRnDStrategy` — NFI R&D 2p/5p dataset
- `ProvedItStrategy` — PROVEDIt court validation dataset

Dataset strategies are instantiated from config and passed directly to
`HIDDataset`, `HIDImage`, parsing helpers, and transformers.
