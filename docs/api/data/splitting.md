# Data Splitting Utilities

The `dnanet.data.splitting` package provides unified splitting logic for
fractional and k-fold dataset splits, with support for both single and
concatenated datasets.

## Main Entry Point

Dispatches splitting to the appropriate helper based on keyword arguments:

- **Fractional split** (`val_fraction` required): Returns `(train, val, test)`
  where `test` is `None` when `test_fraction` is 0.
- **K-fold split** (`k_folds` required): Returns `(folds, test)` where folds
  is a list of `(train, val)` pairs.
- **No kwargs**: Returns `(dataset, None, None)`.

**Validation:** Raises `ValueError` if `val_fraction + test_fraction >= 1.0`,
if fractions are negative, or if an unrecognized combination of kwargs is
provided.

```python
from dnanet.data.splitting import dataset_splitter
```

## Type Aliases

- `FractionalSplitResult` = `tuple[Dataset, Dataset | None, Dataset | None]`
- `KFoldSplitResult` = `tuple[list[tuple[Dataset, Dataset]], Dataset | None]`
- `AnyDataset` = `Dataset | TransformableDataset`
