"""Tests for dataset_splitter and its helpers.

Uses lightweight fake datasets — no real HID files required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from torch.utils.data import ConcatDataset, Subset

from dnanet.data.splitting import (
    dataset_splitter,
    _apply_single_dataset_splitting,
    _apply_concatenated_dataset_splitting,
    _apply_single_dataset_kfold_splitting,
    _apply_concatenated_dataset_kfold_splitting,
)
from dnanet.data.strategies.datasets.nfi_rnd import NFIRnDStrategy


# ---------------------------------------------------------------------------
# Fake dataset helpers (reuse pattern from test_nfi_rnd_splitting)
# ---------------------------------------------------------------------------

def _fake_img(stem: str):
    img = MagicMock()
    img.path.stem = stem.split('/')[-1]
    img.path.absolute = lambda: stem
    return img


STEMS = [
    'Mixture dataset 1/Injx/1A2_A01_01',
    'Mixture dataset 1/Injx/1A2_A01_02',
    'Mixture dataset 1/Injx/1A2_A02_01',
    'Mixture dataset 1/Injx/2A2_A01_01',
    'Mixture dataset 2/Injx/2A2_A01_02',
    'Mixture dataset 2/Injx/2A2_A02_01',
    'Mixture dataset 2/Injx/3A2_A01_01',
    'Mixture dataset 2/Injx/3A2_A01_02',
    'Mixture dataset 3/Injx/3A2_A02_01',
    'Mixture dataset 3/Injx/4A2_A01_01',
    'Mixture dataset 3/Injx/4A2_A01_02',
    'Mixture dataset 3/Injx/4A2_A02_01',
    'Mixture dataset 4/Injx/1B5_A01_01',
    'Mixture dataset 4/Injx/1B5_A01_02',
    'Mixture dataset 4/Injx/1B5_A02_01',
    'Mixture dataset 4/Injx/2B5_A01_01',
    'Mixture dataset 5/Injx/2B5_A01_02',
    'Mixture dataset 5/Injx/2B5_A02_01',
    'Mixture dataset 5/Injx/3B5_A01_01',
    'Mixture dataset 5/Injx/3B5_A01_02',
    'Mixture dataset 6/Injx/3B5_A02_01',
    'Mixture dataset 6/Injx/4B5_A01_01',
    'Mixture dataset 6/Injx/4B5_A01_02',
    'Mixture dataset 6/Injx/4B5_A02_01',
]

STRATEGY = NFIRnDStrategy(annotation_type='DTH')


def make_dataset(stems: list[str] = STEMS):
    ds = MagicMock()
    ds.images = [_fake_img(s) for s in stems]
    ds.__len__ = MagicMock(return_value=len(stems))
    ds.dataset_strategy = STRATEGY
    return ds


def make_concat_dataset(stems_per_sub: list[list[str]] = None):
    if stems_per_sub is None:
        stems_per_sub = [STEMS[:12], STEMS[12:]]
    subs = [make_dataset(s) for s in stems_per_sub]
    return ConcatDataset(subs)


# ---------------------------------------------------------------------------
# dataset_splitter — no split
# ---------------------------------------------------------------------------


class TestDatasetSplitterNoSplit:
    def test_no_kwargs_returns_dataset_none_none(self):
        ds = make_dataset()
        train, val, test = dataset_splitter(ds)
        assert train is ds
        assert val is None
        assert test is None

    def test_only_seed_returns_dataset_none_none(self):
        ds = make_dataset()
        train, val, test = dataset_splitter(ds, seed=42)
        assert train is ds
        assert val is None
        assert test is None


# ---------------------------------------------------------------------------
# dataset_splitter — validation errors
# ---------------------------------------------------------------------------


class TestDatasetSplitterValidation:
    def test_val_plus_test_ge_1_raises(self):
        with pytest.raises(ValueError, match='must be < 1.0'):
            dataset_splitter(make_dataset(), val_fraction=0.7, test_fraction=0.3)

    def test_val_plus_test_eq_1_raises(self):
        with pytest.raises(ValueError, match='must be < 1.0'):
            dataset_splitter(make_dataset(), val_fraction=0.5, test_fraction=0.5)

    def test_negative_val_fraction_raises(self):
        with pytest.raises(ValueError, match='val_fraction must be > 0'):
            dataset_splitter(make_dataset(), val_fraction=-0.1, test_fraction=0.1)

    def test_negative_test_fraction_raises(self):
        with pytest.raises(ValueError, match='test_fraction must be >= 0'):
            dataset_splitter(make_dataset(), val_fraction=0.2, test_fraction=-0.1)


# ---------------------------------------------------------------------------
# dataset_splitter — fractional, single dataset
# ---------------------------------------------------------------------------


class TestDatasetSplitterFractional:
    def test_val_only_returns_three_tuple(self):
        train, val, test = dataset_splitter(make_dataset(), val_fraction=0.2, seed=42)
        assert isinstance(train, Subset)
        assert isinstance(val, Subset)
        assert test is None

    def test_val_only_covers_all_samples(self):
        train, val, _ = dataset_splitter(make_dataset(), val_fraction=0.2, seed=42)
        assert set(train.indices) | set(val.indices) == set(range(len(STEMS)))

    def test_val_only_no_overlap(self):
        train, val, _ = dataset_splitter(make_dataset(), val_fraction=0.2, seed=42)
        assert set(train.indices).isdisjoint(set(val.indices))

    def test_val_and_test_covers_all_samples(self):
        train, val, test = dataset_splitter(
            make_dataset(), val_fraction=0.2, test_fraction=0.1, seed=42
        )
        all_idx = set(train.indices) | set(val.indices) | set(test.indices)
        assert all_idx == set(range(len(STEMS)))

    def test_val_and_test_no_overlap(self):
        train, val, test = dataset_splitter(
            make_dataset(), val_fraction=0.2, test_fraction=0.1, seed=42
        )
        assert set(train.indices).isdisjoint(set(val.indices))
        assert set(train.indices).isdisjoint(set(test.indices))
        assert set(val.indices).isdisjoint(set(test.indices))

    def test_deterministic_with_seed(self):
        t1, v1, _ = dataset_splitter(make_dataset(), val_fraction=0.2, seed=0, genotype_aware=False)
        t2, v2, _ = dataset_splitter(make_dataset(), val_fraction=0.2, seed=0, genotype_aware=False)
        assert t1.indices == t2.indices
        assert v1.indices == v2.indices

    def test_different_seeds_differ(self):
        t1, _, _ = dataset_splitter(make_dataset(), val_fraction=0.2, seed=0, genotype_aware=False)
        t2, _, _ = dataset_splitter(make_dataset(), val_fraction=0.2, seed=99, genotype_aware=False)
        assert t1.indices != t2.indices

    def test_test_fraction_zero_returns_none_test(self):
        train, val, test = dataset_splitter(
            make_dataset(), val_fraction=0.2, test_fraction=0.0, seed=42
        )
        assert test is None


# ---------------------------------------------------------------------------
# dataset_splitter — fractional, ConcatDataset
# ---------------------------------------------------------------------------


class TestDatasetSplitterConcatFractional:
    def test_returns_concat_train_val(self):
        train, val, test = dataset_splitter(
            make_concat_dataset(), val_fraction=0.2, seed=42
        )
        assert isinstance(train, ConcatDataset)
        assert isinstance(val, ConcatDataset)
        assert test is None

    def test_covers_all_samples(self):
        concat = make_concat_dataset()
        train, val, _ = dataset_splitter(concat, val_fraction=0.2, seed=42)
        total = sum(len(ds) for ds in train.datasets) + sum(len(ds) for ds in val.datasets)
        assert total == len(concat)

    def test_with_test_fraction(self):
        train, val, test = dataset_splitter(
            make_concat_dataset(), val_fraction=0.2, test_fraction=0.1, seed=42
        )
        assert isinstance(test, ConcatDataset)


# ---------------------------------------------------------------------------
# _apply_single_dataset_splitting
# ---------------------------------------------------------------------------


class TestApplySingleDatasetSplitting:
    def test_val_only_two_subsets(self):
        train, val, test = _apply_single_dataset_splitting(
            make_dataset(), val_fraction=0.2, seed=42
        )
        assert isinstance(train, Subset)
        assert isinstance(val, Subset)
        assert test is None

    def test_with_test_three_subsets(self):
        train, val, test = _apply_single_dataset_splitting(
            make_dataset(), val_fraction=0.2, test_fraction=0.1, seed=42
        )
        assert all(isinstance(s, Subset) for s in (train, val, test))

    def test_sizes_sum_to_total(self):
        train, val, test = _apply_single_dataset_splitting(
            make_dataset(), val_fraction=0.2, test_fraction=0.1, seed=42
        )
        assert len(train) + len(val) + len(test) == len(STEMS)


# ---------------------------------------------------------------------------
# _apply_single_dataset_kfold_splitting
# ---------------------------------------------------------------------------


class TestApplySingleDatasetKFoldSplitting:
    def test_returns_folds_and_none_test(self):
        folds, test = _apply_single_dataset_kfold_splitting(
            make_dataset(), k_folds=3, seed=42
        )
        assert isinstance(folds, list) and len(folds) == 3
        assert test is None

    @pytest.mark.xfail(
        reason="NFIRnDStrategy._kfold_split requires .images, not available on Subset after test split",
        strict=True,
    )
    def test_returns_folds_and_test_subset(self):
        folds, test = _apply_single_dataset_kfold_splitting(
            make_dataset(), k_folds=3, test_fraction=0.2, seed=42
        )
        assert isinstance(folds, list) and len(folds) == 3
        assert isinstance(test, Subset)

    def test_val_indices_partition_all_samples(self):
        folds, _ = _apply_single_dataset_kfold_splitting(
            make_dataset(), k_folds=3, seed=42
        )
        val_counts = [0] * len(STEMS)
        for _, val in folds:
            assert isinstance(val, Subset)
            for i in val.indices:
                val_counts[i] += 1
        assert all(c == 1 for c in val_counts)

    def test_each_fold_no_overlap(self):
        folds, _ = _apply_single_dataset_kfold_splitting(
            make_dataset(), k_folds=3, seed=42
        )
        for train, val in folds:
            assert isinstance(train, Subset) and isinstance(val, Subset)
            assert set(train.indices).isdisjoint(set(val.indices))


# ---------------------------------------------------------------------------
# _apply_concatenated_dataset_kfold_splitting
# ---------------------------------------------------------------------------


class TestApplyConcatenatedDatasetKFoldSplitting:
    @pytest.mark.xfail(
        reason="NFIRnDStrategy._kfold_split requires .images, not available on Subset after test split",
        strict=True,
    )
    def test_returns_k_fold_concat_pairs(self):
        fold_datasets, test = _apply_concatenated_dataset_kfold_splitting(
            make_concat_dataset(), k_folds=3, test_fraction=0.2, seed=42
        )
        assert len(fold_datasets) == 3
        assert all(
            isinstance(t, ConcatDataset) and isinstance(v, ConcatDataset)
            for t, v in fold_datasets
        )

    @pytest.mark.xfail(
        reason="NFIRnDStrategy._kfold_split requires .images, not available on Subset after test split",
        strict=True,
    )
    def test_test_dataset_is_concat(self):
        _, test = _apply_concatenated_dataset_kfold_splitting(
            make_concat_dataset(), k_folds=3, test_fraction=0.2, seed=42
        )
        assert isinstance(test, ConcatDataset)
