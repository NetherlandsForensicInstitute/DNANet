# Allele Caller

The `dnanet.evaluation.allele_caller` module implements allele calling
strategies that translate pixel-level model outputs into discrete allele calls.

## Design Pattern: Strategy

`AlleleCaller` defines the interface; concrete implementations provide
different allele-calling algorithms.

## Classes

```python
from dnanet.evaluation.allele_caller import (
    AlleleCaller,
    FromSegmentationImageCaller,
    NearestBasePairCaller,
    ExactBasePairCaller,
)
```

### NearestBasePairCaller

Assigns each predicted region to the nearest allele by base-pair distance.
The allele with the smallest absolute difference to the predicted base-pair
position is selected.

### ExactBasePairCaller

Assigns each predicted region to the allele that has no more than 0.3 base-pair
distance from the predicted position. Peaks that fall outside any bin by more
than 0.3 bp are marked as "Out of Bin".

Both classes inherit from `FromSegmentationImageCaller` which handles:
- Normalizing prediction images (binary or multiclass mode)
- Finding connected components of positive predictions
- Translating pixel positions to allele calls via the reference panel

**Common constructor args:**
- `threshold` — Probability threshold for positive predictions (default: 0.5)
- `exclude_non_autosomal` — Filter out non-autosomal markers (default: False)
- `prediction_mode` — `'auto'`, `'binary'`, or `'multiclass_labels'` (default: `'auto'`)
