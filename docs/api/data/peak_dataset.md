# PeakWindowDataset

The `dnanet.data.peak_dataset` package provides a dataset that produces
extracted peak windows from full DNA profiles, used for training the standalone
peak classifier.

## PeakWindowDataset

An `IterableDataset` that wraps a base `HIDDataset`, extracts peaks from every
loaded profile, and presents them as a flat sequence of
`ExtractedPeak` objects.

**Design pattern: Decorator** — Wraps an existing `HIDDataset` and transforms
its items from full profiles to extracted peak windows, adding peak-specific
preprocessing (optional smoothing, log-scale normalization) on top.

```python
from dnanet.data.peak_dataset import PeakWindowDataset
```

**Constructor args:**
- `dataset_strategy` — Dataset strategy for annotation classes and splitting
- `base_dataset` — Source `HIDDataset` of full DNA profiles
- `images` — Alternative: list of `HIDImage` objects
- `threshold` — Minimum RFU height for peak detection (default: 40)
- `window_size` — Width of extraction window in scan points (default: 120)
- `preprocess` — Apply preprocessing (smoothing + scaling) to peaks (default: True)
- `smooth_keep_factor` — FFT smoothing keep fraction (default: 0.4)
- `log_scale` — Apply log1p scaling during preprocessing (default: True)
- `max_rfu_value` — Max RFU for normalization (default: 10000)
- `load_in_memory` — Eagerly extract and cache all peaks at init time
- `include_max_pool_dyes` — Include max-pooled other-dye channel (default: False)

**Factory:** `PeakWindowDataset.from_hid_dataset(base_dataset, **kwargs)`

**Worker support:** Automatically shards work across DataLoader workers using
`get_worker_info()` stride slicing.

**Properties:**
- `images` — List of `HIDImage` objects
- `transform` — Optional data transform
- `dataset_strategy` — The dataset strategy
- `labels` / `label_to_idx` / `idx_to_label` — Class label mappings

**Methods:**
- `subset(indices)` — Create a subset with only indicated indices
