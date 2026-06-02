# ExtractedPeak

The `dnanet.data.extracted_peak` module defines the `ExtractedPeak` data model
for a single peak window.

## ExtractedPeak

A single extracted peak window from a DNA profile. Represents a fixed-width
window of signal data centered on a detected fluorescence peak.

Each peak carries:
- The signal window (1 or 2 channels × `window_size` scan points)
- Its dye channel index and scan-point position
- The marker it falls within (if any)
- An annotation label (`"allele"` or `"noise"`)

```python
from dnanet.data.extracted_peak import ExtractedPeak
```

**Design pattern: Value Object** — Lightweight, immutable-ish data carriers
produced in bulk by `extract_peak_windows` and consumed by the peak
classification data module.

**Constructor args:**
- `data` — Peak signal array of shape `(C, W)` where *C* is 1 or 2, and *W* is the window width
- `dye_index` — Index of the dye channel
- `peak_center` — Scan-point position of the peak apex
- `window_size` — Width of the extraction window
- `peak_height` — RFU amplitude at the peak apex
- `label` — Annotation label (e.g. `"allele"`, `"noise"`)
- `annotation_idx` — Integer class index
- `marker_name` — STR marker name (e.g. `"D3S1358"`), or `None`
- `marker_index` — Integer index of the marker in the panel, or -1

**Properties:**
- `data` — The signal array
- `annotation` — Computed `ClassAnnotation` from label, or `None`
- `is_allele` — Whether this peak is labeled as an allele
- `peak_basepair` — Base-pair position of the peak apex
- `window_start` — Start index of the window
