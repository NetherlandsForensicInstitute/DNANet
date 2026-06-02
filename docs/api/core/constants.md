# LabelCategory Constants

The `dnanet.core.constants` module defines domain constants for forensic DNA
analysis.

## LabelCategory Enum

Peak label categories used in interactive annotation and classification.

```python
from dnanet.core.constants import LabelCategory
```

Each member stores its display metadata as a tuple: `(color, alpha, shortcut_key)`.

**Members:**

| Member | Color | Shortcut | Display Name |
|--------|-------|----------|--------------|
| `UNLABELED` | gray | 0 | Unlabeled |
| `ALLELE` | green | 1 | Allele |
| `STUTTER` | yellow | 2 | Stutter |
| `PULL_UP` | blue | 3 | Pull Up |
| `BLEED_THROUGH` | red | 4 | Bleed Through |
| `SPIKE` | cyan | 5 | Spike |
| `DYE_BLOB` | purple | 6 | Dye Blob |
| `ARTEFACT` | pink | 7 | Artefact |
| `UNCLEAR` | tab:orange | 8 | Unclear |
| `SHOULDER` | tab:olive | 9 | Shoulder |
| `FOREIGN_DNA` | tab:brown | f | Foreign DNA |
| `OVERLOADING_ARTEFACT` | lime | o | Overloading artefact |

**Properties:**
- `label_name` — Human-readable name (e.g., "Allele", "PullUp")
- `display_name` — Name with spaces (e.g., "Pull Up", "Foreign DNA")
- `color_display` — Human-readable color name (e.g., "Green", "Orange")
- `radio_label` — Full radio-button label (e.g., "Allele (Green - 1)")

**Class methods:**
- `from_index(index)` — Look up a category by integer position (0-based)
- `label_names()` — Return all label names in order
- `display_name_to_index(search_name)` — Look up by any name variant (case-insensitive)

## Other Constants

- `DEFAULT_SIGNAL_LENGTH` = 4096 — Default number of scan points per dye channel
- `LABELTOOL_VERSION` = '1.0' — Label tool format version
- `NON_AUTOSOMAL_PREFIXES` = ('DYS',) — Prefixes for non-autosomal markers
- `NON_AUTOSOMAL_MARKERS` = frozenset({'AMEL'}) — Set of non-autosomal marker names
