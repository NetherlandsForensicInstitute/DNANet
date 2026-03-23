# Data Layer

The `dnanet.data` package handles everything from raw HID files to
PyTorch-ready tensors.

## HIDImage

```{eval-rst}
.. autoclass:: dnanet.data.image.HIDImage
   :members:
```

The central data container. Wraps a path to a HID file with lazy loading.

**Properties:**
- `data` → `np.ndarray | None` — Shape `(num_dyes, signal_length, 1)`. Triggers load on first access.
- `annotation` → `Annotation | None` — Ground-truth segmentation mask
- `scaler` → `np.ndarray` — Shape `(1, signal_length)`. Maps pixel → base pair.
- `panel` → `Panel | None` — Reference panel
- `dimensions` → `(height, width)` — Data array shape
- `meta` → `dict` — Metadata (NOC, ladder path, etc.)

**Methods:**
- `adjust_annotations(method)` — Snap annotation mask to actual peaks
  - `"top"` — Label only peak apex
  - `"complete"` — Label entire peak boundary-to-boundary

## Datasets

```{eval-rst}
.. autoclass:: dnanet.data.dataset.InMemoryDataset
   :members:
.. autoclass:: dnanet.data.dataset.SimpleDataset
   :members:
.. autoclass:: dnanet.data.hid_dataset.HIDDataset
   :members:
```

### InMemoryDataset

Abstract base for in-memory datasets. Implements `Sequence[HIDImage]`.

**Key methods:**
- `split(val_fraction, seed)` → `(train, val)` — Random split
- `split_k_fold(n_folds, seed)` → `list[SimpleDataset]` — K-fold split
- `__len__()`, `__getitem__()`, `__iter__()`

### HIDDataset

Loads HID files from a directory. Extends `InMemoryDataset`.

**Constructor args:** See {doc}`/guides/datasets` for details.

### SimpleDataset

Lightweight wrapper around a list of `HIDImage` objects. Returned by
`split()` and `split_k_fold()`.

## DataModule

```{eval-rst}
.. autoclass:: dnanet.data.datamodule.DNANetDataModule
   :members:
.. autoclass:: dnanet.data.datamodule.HIDTorchDataset
   :members:
```

### DNANetDataModule

Lightning DataModule bridging `InMemoryDataset` → PyTorch DataLoaders.

**Args:**
- `dataset` — An `InMemoryDataset`
- `batch_size` — Batch size
- `val_fraction` — Train/val split ratio
- `num_workers` — DataLoader workers
- `seed` — Random seed for reproducible splits

### HIDTorchDataset

Adapts `list[HIDImage]` to PyTorch `Dataset[tuple[Tensor, Tensor]]`.

Transposes data from `(D, L, 1)` to `(1, D, L)` for Conv2d compatibility.

## Parsing

```{eval-rst}
.. automodule:: dnanet.data.parsing.hid
   :members:
.. automodule:: dnanet.data.parsing.annotations
   :members:
```

### HID Parsing

`get_peak_data(path, strategy)` — Parse a HID file and return raw/analyzed
data as a numpy array.

### Annotation Parsing

`parse_called_alleles(annotation_file, panel, sample_name)` — Parse an
AlleleReport TXT file and return called alleles for a specific sample.

## Preprocessing

```{eval-rst}
.. automodule:: dnanet.data.preprocessing.peaks
   :members:
.. automodule:: dnanet.data.preprocessing.baseline
   :members:
```

### Peak Detection

- `find_peaks_above_threshold(signal, threshold)` — Detect peaks including
  flat-top peaks
- `find_peak_boundary(signal, peak_idx, threshold)` — Walk left/right to
  find peak start and end
- `find_peak_near_idx(signal, idx)` — Find nearest peak at least as high
- `find_peak_idx_near_or_in_range(signal, range, threshold)` — Find dominant
  peak within or near an index range

### Baseline Estimation

- `superior_baseline(signal)` — DNANet's recommended baseline method
- `classic_baseline(signal)` — Traditional rolling-minimum approach
- `enhanced_baseline(signal)` — Improved classic with smoothing

## Strategies

```{eval-rst}
.. autoclass:: dnanet.data.strategies.scaling.ScalingStrategy
   :members:
.. autoclass:: dnanet.data.strategies.scaling.PowerPlexFusion6CStrategy
   :members:
.. autoclass:: dnanet.data.strategies.scaling.GlobalFilerStrategy
   :members:
.. autoclass:: dnanet.data.strategies.dataset.DatasetStrategy
   :members:
.. autoclass:: dnanet.data.strategies.registry.StrategyRegistry
   :members:
```

### ScalingStrategy

Abstract base for kit-specific base-pair calibration.

**Concrete implementations:**
- `PowerPlexFusion6CStrategy` — PPF6C kit with WEN ILS (bp range 65–475)
- `GlobalFilerStrategy` — GlobalFiler kit with GeneScan 600 LIZ (bp range 60–480)

### DatasetStrategy

Abstract base for dataset-specific file handling.

**Concrete implementations:**
- `NFIRnDStrategy` — NFI R&D 2p/5p dataset
- `ProvedItStrategy` — PROVEDIt court validation dataset

### StrategyRegistry

Singleton holding the active kit and dataset strategies.

```python
StrategyRegistry.configure_kit("PPF6C")
StrategyRegistry.configure_dataset("NFI_RND")
scaling = StrategyRegistry.get_scaling_strategy()
```

## Convenience

```{eval-rst}
.. autofunction:: dnanet.data.loading.load_dataset
```

`load_dataset(data_cfg)` — One-line dataset loading from Hydra config.
Handles strategy configuration and `HIDDataset` construction.
