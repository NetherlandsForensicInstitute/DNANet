# Datasets

DNANet supports loading forensic DNA profiles from HID files produced by
capillary electrophoresis instruments (e.g., Applied Biosystems 3500).

## Supported Datasets

### NFI R&D Dataset (`data=dnanet_rd`)

The Netherlands Forensic Institute Research & Development dataset contains
mixture profiles from 2 to 5 contributors, amplified with the PowerPlex
Fusion 6C (PPF6C) kit.

**Statistics:**
- 350 sample HID files (87×2p, 87×3p, 88×4p, 86×5p contributors)
- 72 ladder files
- 6 mixture dataset directories (different electrophoresis runs)
- Annotation format: AlleleReport TXT files (tab-separated)
- Size standard: WEN ILS (19 peaks)

**Required files:**

| File | Description |
|------|-------------|
| `Raw data .HID files/` | Root directory with HID files |
| `txt_annotations_2024/` | AlleleReport TXT annotation files |
| `2p_5p_hid_to_annotation.csv` | Maps HID filenames → annotation names |
| `best_ladder_paths_DTH.csv` | Maps samples → best ladder file |
| `ladder_alleles.csv` | Expected ladder alleles per marker |

**Annotation pipeline:**
1. CSV maps each HID stem to an annotation name in the AlleleReport file
2. Called alleles are parsed from the AlleleReport (marker → allele list)
3. Alleles are mapped to pixel positions using the base-pair scaler
4. Binary segmentation masks are built from allele bin ranges
5. Optionally, masks are refined by snapping to actual peak positions
   (`adjustment_of_annotations: top` or `complete`)

### PROVEDIt Dataset (`data=provedit`)

The Project on Validation of Evidence Data Interpretation Tools (PROVEDIt)
is a publicly available court validation dataset using the GlobalFiler kit.

**Statistics:**
- ~750 sample HID files across multiple injection times
- 85 ladder files
- Size standard: GeneScan 600 LIZ (34 peaks, iterative fit)

**Notes:**
- ProvedIt currently loads without allele annotations (segmentation masks
  are all-zero). Full genotype annotation loading from the XLSX file is
  planned.
- The GlobalFiler size standard parser uses an iterative shrinking fit.
  Files with poor size standard quality are still loaded with a warning
  (best-effort fit), matching the original implementation.
- Ladders are auto-discovered by matching the well prefix (e.g., sample
  `B03_RD14-...` matches ladder `B03_Ladder-GF_...` in the same
  directory).

## Data Pipeline

The data loading pipeline follows this flow:

```
HID files on disk
    │
    ▼
HIDDataset.__init__()
    ├─ StrategyRegistry.configure_kit(kit)
    ├─ StrategyRegistry.configure_dataset(dataset_strategy)
    ├─ _create_annotation_mapping()    ← CSV lookup
    ├─ _load_ladder_paths()            ← CSV lookup
    ├─ _collect_files()                ← walk + filter via strategy
    └─ _load_images()                  ← create HIDImage instances
         │
         ▼
    HIDImage._load()  (per file, lazy)
         ├─ get_peak_data(path, strategy)     ← parse HID XML
         ├─ scaling.parse_size_standard(lane)  ← validate & fit
         ├─ profile[:, rescaled_indices]       ← rescale to bp grid
         ├─ parse_called_alleles(...)          ← load annotation
         └─ _build_segmentation(...)           ← binary mask
              │
              ▼
         HIDImage(data, annotation, scaler, meta)
              │
              ▼
    adjust_annotations(method)  ← optional: snap to peaks
              │
              ▼
    InMemoryDataset (list of HIDImages)
              │
              ▼
    DNANetDataModule
         ├─ split(val_fraction) → train/val
         └─ HIDTorchDataset    → (x, y) tensors
              │
              ▼
         DataLoader → Lightning Trainer
```

## Data Shapes

| Stage | Shape | Description |
|-------|-------|-------------|
| Raw HID | `(num_dyes+1, ~10000)` | Raw scan points per dye |
| After rescaling | `(num_dyes, 4096, 1)` | Uniform base-pair grid |
| Segmentation mask | `(num_dyes, 4096, 1)` | Binary annotation |
| Torch input (x) | `(1, num_dyes, 4096)` | Transposed for Conv2d |
| Torch target (y) | `(1, num_dyes, 4096)` | Transposed for Conv2d |

Where `num_dyes = 5` (analysis channels; size standard excluded by default).

## Adding a New Dataset

1. **Create a dataset strategy** in `src/dnanet/data/strategies/datasets/`:

```python
class MyDatasetStrategy(DatasetStrategy):
    @classmethod
    def categorize_file(cls, file_name: str) -> FileCategory:
        # Classify as "sample", "ladder", "control", or "unknown"
        ...

    @classmethod
    def get_contributors(cls, file_name: str) -> str | None:
        # Extract NOC from filename (e.g., "2p")
        ...

    @classmethod
    def get_sample_id(cls, file_name: str) -> str:
        # Extract unique sample ID from filename
        ...

    # ... implement remaining abstract methods
```

2. **Register** the strategy in `datasets/__init__.py`:

```python
DATASET_STRATEGIES["MY_DATASET"] = MyDatasetStrategy
```

3. **Create a config** in `conf/data/my_dataset.yaml`:

```yaml
name: my_dataset
kit: PPF6C           # or GLOBALFILER
dataset_strategy: MY_DATASET
root: data/my_dataset/
# ... other parameters
```

4. **Run**: `dnanet task=train data=my_dataset model=unet training=segmentation`
