# Core Domain Models

The `dnanet.core` package contains pure domain models with no ML framework
dependencies. These are the fundamental data structures of forensic DNA
analysis.

## Allele

A single measured allele within a DNA marker (locus).

```python
from dnanet.core.allele import Allele
```

**Fields:**
- `name` (str) — Allele designation (e.g., "12", "14.3", "OL")
- `base_pair` (float | None) — Position in base pairs
- `left_bin` (float) — Left bin boundary width (bp)
- `right_bin` (float) — Right bin boundary width (bp)
- `height` (float | None) — Peak height in RFU

## Marker

A DNA marker (locus) containing one or more alleles.

```python
from dnanet.core.marker import Marker
```

**Fields:**
- `name` (str) — Marker name (e.g., "D3S1358", "AMEL", "vWA")
- `dye_row` (int) — 0-based dye channel index
- `alleles` (list of Allele) — Called alleles at this locus

**Properties:**
- `is_autosomal` — True if not a sex or Y-chromosome marker
- `allele_names` — List of allele name strings

## Panel

Reference panel of markers and alleles for a forensic DNA kit.

```python
from dnanet.core.panel import Panel
```

**Factory:** `Panel.from_xml(path, hid_dye_mapping)` — Parses GeneMarker
SGPanel XML format.

**Key Methods:**
- `fill_allele_bins(allele_name, marker_name)` — Look up bin position for an allele
- `get_markers_for_dye(dye_idx)` — Get all markers on a specific dye channel
- `adjust(offsets)` — Create a copy with adjusted bin positions

## Annotation

Ground-truth annotation for a DNA profile.

```python
from dnanet.core.annotation import Annotation
```

**Fields:**
- `labels` (frozenset of LabelCategory) — Categorical labels
- `image` (np.ndarray | None) — Pixel-level mask `(dyes, signal_length, 1)`
- `meta` (dict) — Arbitrary metadata
