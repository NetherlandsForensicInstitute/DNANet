# Evaluation Utilities

The `dnanet.evaluation.utils` module provides utility functions for
allele-level metric computation.

## Functions

```python
from dnanet.evaluation.utils import flatten_markers_to_allele_names
```

`flatten_markers_to_allele_names(markers, *, locus=None)` — Flatten a
sequence of `Marker` objects to a set of `"MarkerName_AlleleName"` strings.

Optionally filters by locus name.

**Example:**

```python
>>> markers = [Marker("D5S818", 0, frozenset(Allele("13", height=500), Allele("15", height=200)))]
>>> flatten_markers_to_allele_names(markers)
frozenset({'D5S818_13', 'D5S818_15'})
```

**Args:**
- `markers` — Sequence of `Marker` objects to flatten
- `locus` — If provided, only include alleles from this locus

**Returns:** Frozen set of `"marker_allele"` strings

**Raises:** `ValueError` — If an RFU threshold is set but allele height is `None`
