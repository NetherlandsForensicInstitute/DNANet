# Ladder Handling

The `dnanet.data.ladders` package handles base-pair calibration via ladder
profiles. It uses detected peaks in a ladder HID file to create an adjusted
panel where allele base-pair positions reflect the actual electrophoresis
conditions of the run.

## Ladder

A `Ladder` wraps a ladder HID file and uses it to create an *adjusted panel*.
The calibration process:

1. Detects peaks in each dye channel of the ladder profile
2. Matches detected peaks to known alleles (by count per dye)
3. Maps peak pixel positions to base-pair positions via the size-standard scaler
4. Inter/extrapolates missing alleles from neighboring alleles
5. Returns a `Panel` with adjusted base-pair values

```python
from dnanet.data.ladders.ladder import Ladder
```

**Factory:** `Ladder.create_adjusted_panel()` — reads a ladder HID file and
returns an adjusted `Panel`, or `None` if peak detection fails.

## Ladder Allele Catalog

A frozen data catalog of alleles expected in a ladder, organized by dye row.
Loaded from a CSV file with columns: `Marker`, `Allele`, `Dye`.

```python
from dnanet.data.ladders.ladder_allele_catalog import LadderAlleleCatalog
```

**Factory:** `LadderAlleleCatalog.from_panel(panel)` — creates a catalog from
a reference `Panel`.
