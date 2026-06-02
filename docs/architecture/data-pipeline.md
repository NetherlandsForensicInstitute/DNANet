# Data Pipeline

This page describes the complete data loading pipeline from HID files on disk
to PyTorch tensors fed into the training loop.

## HID File Format

HID files are XML-based containers produced by Applied Biosystems capillary
electrophoresis instruments (e.g., 3500 Genetic Analyzer). Each file contains:

- **Raw data** — Unprocessed fluorescence intensity per scan point per dye
- **Analyzed data** — Instrument-processed data (baseline-subtracted, smoothed)
- **Size standard channel** — The last dye channel (e.g., WEN ILS or LIZ)
- **Metadata** — Sample name, instrument settings, run date

DNANet supports three loading strategies:
- `raw` — Use only the raw data
- `analyzed` — Use the instrument-analyzed data
- `superior` — Use raw data with DNANet's own baseline estimation (recommended)

## Size Standard Calibration

The size standard channel maps scan-point positions to base-pair (bp) values.
This calibration is critical for allele identification.

### PPF6C (WEN ILS)
1. Detect peaks above 180 RFU (adaptive: 120 RFU for tail peaks)
2. Remove duplicates within 15 scan points
3. Take the last 19 peaks (excluding the final)
4. Validate: pixel-per-bp ratio must be 7–13 for all adjacent pairs
5. Fit cubic spline through (peak_idx, expected_bp) pairs
6. Rescale profile to 4096 scan points over 65–475 bp

### GlobalFiler (GeneScan 600 LIZ)
1. Detect peaks above 300 RFU
2. Remove duplicates within 15 scan points
3. Iterative polynomial fitting (degree 2):
   - Take last N peaks matching expected bp count
   - Fit quadratic polynomial
   - If max deviation > 5.0 bp, trim last expected bp and retry
   - Up to 10 shrinkage iterations
4. Fit cubic spline and rescale to 4096 scan points over 60–480 bp

## Annotation Pipeline

For the NFI R&D dataset, annotations come from AlleleReport TXT files:

1. **Mapping CSV** — `2p_5p_hid_to_annotation.csv` maps each HID filename
   to an annotation name in the AlleleReport
2. **AlleleReport parsing** — Tab-separated file with columns for each marker;
   rows contain allele calls per sample
3. **Panel lookup** — Called allele names are resolved to base-pair positions
   and bin widths from the SGPanel XML
4. **Mask construction** — For each allele, pixels between `bp ± bin_width`
   are set to 1 in the binary mask
5. **Adjustment** (optional) — Masks are refined by finding actual peaks:
   - `top` — Set only the peak apex to 1
   - `complete` — Set the entire peak (boundary to boundary) to 1

## Ladder-Based Panel Adjustment

Ladder files contain known alleles at fixed positions. By comparing detected
ladder peaks to expected positions, the panel can be adjusted for
run-specific instrument drift:

1. Parse ladder HID file and calibrate its size standard
2. For each marker, detect the allele peak and compute its actual bp position
3. Update the panel's bin positions to match the ladder measurements

This produces an **adjusted panel** that is more accurate than the default
XML panel for each specific electrophoresis run.

## Persistent cache

Parsed and pre-processed profiles are persisted to an on-disk memmap-backed
cache so that subsequent runs only open a small index and stream single rows
on demand. See [Dataset Caching](caching.md) for the layout, fingerprint
invalidation, deduped sidecars, RAM guard, and the `cache-inspect` tool.

## HIDDataset Construction

`HIDDataset` orchestrates the full loading pipeline:

```python
dataset = HIDDataset(
    root="data/2p_5p_Dataset_NFI/Raw data .HID files",
    scaling_strategy=scaling_strategy,
    dataset_strategy=dataset_strategy,
    cache_dir="data/cache/dnanet_rd",
    adjustment_of_annotations="complete",
)
```

### Steps:
1. **Inject strategies** — pass a kit scaling strategy and dataset strategy
2. **Load mappings** — Annotation mapping CSV, ladder paths CSV, ladder catalog
3. **Collect files** — Walk root recursively, filter via `DatasetStrategy.categorize_file()`
4. **Load images** — For each sample:
   - Build adjusted panel from its ladder (cached per ladder file)
   - Create `HIDImage` with panel, annotation file, and metadata
   - Trigger lazy load and validate (skip if data is None)
5. **Adjust annotations** — Optionally snap masks to actual peaks

## DNANetDataModule

The Lightning DataModule bridges the domain dataset to PyTorch:

```python
datamodule = DNANetDataModule(
    dataset=dataset,           # TransformableDataset (e.g. HIDDataset)
    batch_size=16,
    num_workers=0,
    shuffle_train=True,
    val_fraction=0.2,          # 80/20 train/val split
    seed=42,                   # via **split_kwargs → dataset_splitter()
)
```

Internally:
1. Calls the dataset strategy split helper when available, otherwise `dataset.split(...)`
2. Applies the dataset transformer and collate function
3. Creates `DataLoader` instances for Lightning's `fit()` / `validate()` / `predict()`
