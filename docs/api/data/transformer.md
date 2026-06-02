# Data Transformers

The `dnanet.data.transformer` package provides callable transformers that
convert `HIDImage` or `ExtractedPeak` objects into PyTorch tensor inputs
and targets for different training tasks.

## Base Class

Abstract base class for all transformers. Implements `__call__` (image →
`(inputs, target)` tuple) and a `collate_fn` for batch assembly.

```python
from dnanet.data.transformer import TransformDataCallable
```

## Segmentation Transformer

Produces `(data, target)` where both are `(D, L)` tensors. The target is a
scanpoint-indexed label mask.

```python
from dnanet.data.transformer import SegmentationTransformer
```

## Allele Metadata Transformer

Wraps another transformer and appends allele metadata (panel, scaler, signal,
allele annotation) to each output. Enables allele-level metric computation
during evaluation.

**Output:** `(inputs, target, metadata_dict)`

```python
from dnanet.data.transformer import AlleleMetadataTransformer
```

## Combined Transformer

Transformer for the combined PeakNet model. Produces a tuple of:
`(full_image, peak_windows, marker_idxs, peak_centers, n_peaks)` and a
`(D, L)` target mask.

**Constructor args:**
- `threshold` — Minimum RFU for peak detection (default: 25)
- `window_size` — Peak extraction window in scan points (default: 120)
- `include_max_pool_dyes` — Include max-pooled other-dye channel
- `autoencoder_log_scale` — Apply log scaling to full image
- `autoencoder_max_rfu` — Max RFU for normalization

```python
from dnanet.data.transformer import CombinedTransformer
```

## Reconstruction Transformer

For autoencoder reconstruction tasks. Input is preprocessed (log-scaled),
target is raw data so the loss is computed in non-preprocessed space.

**Constructor args:**
- `n_dyes` — Number of dye channels to include (default: 5)
- `autoencoder_log_scale` — Apply log scaling (default: True)
- `autoencoder_max_rfu` — Max RFU for normalization

```python
from dnanet.data.transformer import ReconstructionTransformer
```

## Peak Classification Transformer

For the PeakNet classifier. Converts `ExtractedPeak` objects into
`(peak_tensor, marker_tensor)` inputs and class-index targets.

**Constructor args:**
- `include_marker` — Include marker index tensor (default: True)

```python
from dnanet.data.transformer import PeakClassificationTransformer
```
