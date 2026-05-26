# Core Domain Models

The `dnanet.core` package contains pure domain models with no ML framework
dependencies. These are the fundamental data structures of forensic DNA
analysis.

## Allele

```{eval-rst}
.. automodule:: dnanet.core.allele
   :members:
   :undoc-members:
```

A single measured allele within a DNA marker (locus).

**Fields:**
- `name` (str) — Allele designation (e.g., "12", "14.3", "OL")
- `base_pair` (float | None) — Position in base pairs
- `left_bin` (float) — Left bin boundary width (bp)
- `right_bin` (float) — Right bin boundary width (bp)
- `height` (float | None) — Peak height in RFU

## Marker

```{eval-rst}
.. automodule:: dnanet.core.marker
   :members:
   :undoc-members:
```

A DNA marker (locus) containing one or more alleles.

**Fields:**
- `name` (str) — Marker name (e.g., "D3S1358", "AMEL", "vWA")
- `dye_row` (int) — 0-based dye channel index
- `alleles` (list[Allele]) — Called alleles at this locus

**Properties:**
- `is_autosomal` — True if not a sex or Y-chromosome marker
- `allele_names` — List of allele name strings

## Panel

```{eval-rst}
.. automodule:: dnanet.core.panel
   :members:
   :undoc-members:
```

Reference panel of markers and alleles for a forensic DNA kit.

**Factory:** `Panel.from_xml(path, hid_dye_mapping)` — Parses GeneMarker
SGPanel XML format.

**Key Methods:**
- `fill_allele_bins(allele_name, marker_name)` — Look up bin position for an allele
- `get_markers_for_dye(dye_idx)` — Get all markers on a specific dye channel
- `adjust(offsets)` — Create a copy with adjusted bin positions

## Annotation

```{eval-rst}
.. automodule:: dnanet.core.annotation
   :members:
   :undoc-members:
```

Ground-truth annotation for a DNA profile.

**Fields:**
- `labels` (frozenset[LabelCategory]) — Categorical labels
- `image` (np.ndarray | None) — Pixel-level mask `(dyes, signal_length, 1)`
- `meta` (dict) — Arbitrary metadata

## Prediction

```{eval-rst}
.. automodule:: dnanet.core.prediction
   :members:
   :undoc-members:
```

Model prediction container.

**Fields:**
- `classification` (dict[LabelCategory, float]) — Label → confidence
- `image` (np.ndarray | None) — Predicted pixel mask
- `called_alleles` (list[Marker]) — Called alleles (after allele calling)

**Factory:** `Prediction.for_segmentation(mask, panel, scaler, caller)` —
Build from a segmentation mask, running allele calling automatically.
