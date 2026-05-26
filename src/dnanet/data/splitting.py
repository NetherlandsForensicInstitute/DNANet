from __future__ import annotations

from typing import Any, cast

from torch.utils.data import ConcatDataset, Dataset, Subset

from dnanet.data.dataset import TransformableDataset
from dnanet.data.strategies.datasets.dataset import DatasetStrategy


AnyDataset = Dataset | TransformableDataset

FractionalSplitResult = tuple[Dataset, Dataset | None, Dataset | None]
KFoldSplitResult = tuple[list[tuple[Dataset, Dataset]], Dataset | None]


def dataset_splitter(dataset: AnyDataset, **split_kwargs: Any) -> FractionalSplitResult:
    """Dispatch splitting to the right helper based on kwargs.

    Fractional split (val_fraction required):
        Returns (train, val, test) — test is None when test_fraction is 0.
    K-Fold split (k_folds required, val_fraction must be absent):
        Returns (folds, test) where folds is a list of (train, val) pairs.
    No kwargs:
        Returns (dataset, None, None).
    """
    val_fraction = split_kwargs.get("val_fraction")
    k_folds = split_kwargs.get("k_folds")

    # Normalize missing test_fraction to 0.0 for fractional splits
    if "test_fraction" not in split_kwargs and val_fraction is not None:
        split_kwargs = {**split_kwargs, "test_fraction": 0.0}
    test_fraction = split_kwargs.get("test_fraction")

    if val_fraction is None and test_fraction is None and k_folds is None:
        return cast(Dataset, dataset), None, None  # type: ignore[return-value]

    match (val_fraction, test_fraction, k_folds):
        case (float(), float(), None) if val_fraction + test_fraction >= 1.0:
            raise ValueError(
                f'val_fraction ({val_fraction}) + test_fraction ({test_fraction}) must be < 1.0'
            )
        case (float(), _, _) if val_fraction < .0:
            raise ValueError(f'val_fraction must be > 0, got {val_fraction}')
        case (_, float(), _) if test_fraction < .0:
            raise ValueError(f'test_fraction must be >= 0, got {test_fraction}')

        case (float(), float(), None) if isinstance(dataset, ConcatDataset):
            return _apply_concatenated_dataset_splitting(dataset, **split_kwargs)
        case (float(), float(), None):
            return _apply_single_dataset_splitting(dataset, **split_kwargs)

        case (None, float(), int()) if isinstance(dataset, ConcatDataset):
            return _apply_concatenated_dataset_kfold_splitting(dataset, **split_kwargs)  # type: ignore[return-value]
        case (None, float(), int()):
            return _apply_single_dataset_kfold_splitting(dataset, **split_kwargs)  # type: ignore[return-value]

    raise ValueError(
        f'Unrecognised split_kwargs combination: {split_kwargs}. '
        'Provide val_fraction for fractional splits or k_folds (without val_fraction) for k-fold.'
    )


def _apply_single_dataset_splitting(
    dataset: AnyDataset, **split_kwargs: Any
) -> tuple[Subset, Subset | None, Subset | None]:
    """Apply fractional splitting for a single dataset using its strategy."""
    val_fraction: float = split_kwargs.get('val_fraction', 0.0)
    test_fraction: float = split_kwargs.get('test_fraction', 0.0)
    train_fraction = 1.0 - val_fraction - test_fraction

    strategy: DatasetStrategy = dataset.dataset_strategy  # type: ignore[union-attr]

    val_data: Subset | None = None
    test_data: Subset | None = None

    if test_fraction > 0.0:
        train_data, val_data, test_data = cast(
            tuple[Subset, Subset, Subset],
            strategy.split(dataset, fraction=train_fraction, **split_kwargs),
        )
    elif val_fraction > 0.0:
        train_data, val_data = cast(
            tuple[Subset, Subset],
            strategy.split(dataset, fraction=train_fraction, **split_kwargs),
        )
    else:
        train_data = cast(Subset, dataset)

    return train_data, val_data, test_data


def _apply_concatenated_dataset_splitting(
    dataset: ConcatDataset, **split_kwargs: Any
) -> tuple[ConcatDataset, ConcatDataset | None, ConcatDataset | None]:
    """Split each sub-dataset independently and recombine."""
    subsets_train: list[Dataset] = []
    subsets_val: list[Dataset] = []
    subsets_test: list[Dataset] = []

    for ds in dataset.datasets:
        train_data, val_data, test_data = _apply_single_dataset_splitting(ds, **split_kwargs)
        subsets_train.append(train_data)
        if val_data is not None:
            subsets_val.append(val_data)
        if test_data is not None:
            subsets_test.append(test_data)

    train = ConcatDataset(subsets_train)
    val: ConcatDataset | None = ConcatDataset(subsets_val) if subsets_val else None
    test: ConcatDataset | None = ConcatDataset(subsets_test) if subsets_test else None
    return train, val, test


def _apply_single_dataset_kfold_splitting(
    dataset: AnyDataset, **split_kwargs: Any
) -> KFoldSplitResult:
    split_parameters = dict(split_kwargs)
    k_folds: int = split_parameters.pop('k_folds')
    test_fraction: float | None = split_parameters.pop('test_fraction', None)

    strategy: DatasetStrategy = dataset.dataset_strategy  # type: ignore[union-attr]

    if test_fraction:
        # Split off test set first, then k-fold the remainder
        k_fold_set, test_set = cast(
            tuple[Subset, Subset],
            strategy.split(dataset=dataset, fraction=1.0 - test_fraction, **split_parameters),
        )
        folds = cast(
            list[tuple[Dataset, Dataset]],
            strategy.split(dataset=k_fold_set, k_folds=k_folds, **split_parameters),
        )
        return folds, test_set

    folds = cast(
        list[tuple[Dataset, Dataset]],
        strategy.split(dataset=dataset, k_folds=k_folds, **split_parameters),
    )
    return folds, None


def _apply_concatenated_dataset_kfold_splitting(
    dataset: ConcatDataset, **split_kwargs: Any
) -> KFoldSplitResult:
    test_subsets: list[Dataset] = []
    k_folds: int = split_kwargs['k_folds']
    fold_groups: list[list[tuple[Dataset, Dataset]]] = [[] for _ in range(k_folds)]

    for ds in dataset.datasets:
        folds, test_set = _apply_single_dataset_kfold_splitting(ds, **split_kwargs)
        for idx, fold in enumerate(folds):
            fold_groups[idx].append(fold)
        if test_set is not None:
            test_subsets.append(test_set)

    fold_datasets: list[tuple[Dataset, Dataset]] = [
        (ConcatDataset([t for t, _ in group]), ConcatDataset([v for _, v in group]))
        for group in fold_groups
    ]
    return fold_datasets, ConcatDataset(test_subsets) if test_subsets else None
