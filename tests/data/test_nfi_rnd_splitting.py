"""Tests for NFIRnDStrategy splitting logic.

Covers _build_replica_map, _replica_noc_labels, _subsets, and the public
split() entry point for both fractional and k-fold modes (grouped/ungrouped).
All tests use a lightweight fake dataset — no real HID files required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from torch.utils.data import Subset

import dnanet.data.strategies.datasets.nfi_rnd as nfi_rnd_module
from dnanet.data.strategies.datasets.nfi_rnd import NFIRnDStrategy


# ---------------------------------------------------------------------------
# Fake dataset helpers
# ---------------------------------------------------------------------------


def _fake_img(stem: str):
    img = MagicMock()
    img.path.stem = stem.split('/')[-1]
    img.path.absolute = lambda: stem
    return img


def make_dataset(stems: list[str]):
    """Minimal fake dataset: exposes .images (list of images) and __len__."""
    ds = MagicMock()
    ds.images = [_fake_img(s) for s in stems]
    ds.__len__ = MagicMock(return_value=len(stems))
    return ds


# 8 replicas × 3 samples each, 24 samples total, 2 NoC values (2 and 5).
# Using 2 classes lets StratifiedGroupKFold/StratifiedKFold succeed even with
# small val folds (e.g. fraction=0.75 → n_splits=4 → 2 replicas in val ≥ 2 classes).
STEMS = [
    'Mixture dataset 1/Injx/1A2_A01_01',
    'Mixture dataset 1/Injx/1A2_A01_02',
    'Mixture dataset 1/Injx/1A2_A02_01',  # replica 1A2, NoC=2
    'Mixture dataset 1/Injx/2A2_A01_01',
    'Mixture dataset 2/Injx/2A2_A01_02',
    'Mixture dataset 2/Injx/2A2_A02_01',  # replica 2A2, NoC=2
    'Mixture dataset 2/Injx/3A2_A01_01',
    'Mixture dataset 2/Injx/3A2_A01_02',
    'Mixture dataset 3/Injx/3A2_A02_01',  # replica 3A2, NoC=2
    'Mixture dataset 3/Injx/4A2_A01_01',
    'Mixture dataset 3/Injx/4A2_A01_02',
    'Mixture dataset 3/Injx/4A2_A02_01',  # replica 4A2, NoC=2
    'Mixture dataset 4/Injx/1B5_A01_01',
    'Mixture dataset 4/Injx/1B5_A01_02',
    'Mixture dataset 4/Injx/1B5_A02_01',  # replica 1B5, NoC=5
    'Mixture dataset 4/Injx/2B5_A01_01',
    'Mixture dataset 5/Injx/2B5_A01_02',
    'Mixture dataset 5/Injx/2B5_A02_01',  # replica 2B5, NoC=5
    'Mixture dataset 5/Injx/3B5_A01_01',
    'Mixture dataset 5/Injx/3B5_A01_02',
    'Mixture dataset 6/Injx/3B5_A02_01',  # replica 3B5, NoC=5
    'Mixture dataset 6/Injx/4B5_A01_01',
    'Mixture dataset 6/Injx/4B5_A01_02',
    'Mixture dataset 6/Injx/4B5_A02_01',  # replica 4B5, NoC=5
]


# ---------------------------------------------------------------------------
# split() dispatch — negative flows
# ---------------------------------------------------------------------------


class TestSplitDispatch:
    def test_both_fraction_and_kfolds_raises(self):
        with pytest.raises(ValueError):
            NFIRnDStrategy.split(make_dataset(STEMS), fraction=0.8, k_folds=3)

    def test_neither_raises(self):
        with pytest.raises(ValueError):
            NFIRnDStrategy.split(make_dataset(STEMS))

    def test_fraction_zero_raises(self):
        with pytest.raises(ValueError):
            NFIRnDStrategy.split(make_dataset(STEMS), fraction=0.0)

    def test_fraction_one_raises(self):
        with pytest.raises(ValueError):
            NFIRnDStrategy.split(make_dataset(STEMS), fraction=1.0)

    def test_kfolds_one_raises(self):
        with pytest.raises(ValueError):
            NFIRnDStrategy.split(make_dataset(STEMS), k_folds=1)

    def test_kfolds_more_folds_than_len(self):
        with pytest.raises(ValueError):
            NFIRnDStrategy.split(make_dataset(STEMS), k_folds=25)


# ---------------------------------------------------------------------------
# Fractional split — grouped (default)
# ---------------------------------------------------------------------------


class TestFractionalSplitGrouped:
    def test_returns_two_subsets(self):
        train, val = NFIRnDStrategy.split(make_dataset(STEMS), fraction=0.75, seed=42)
        assert isinstance(train, Subset) and isinstance(val, Subset)

    def test_all_samples_covered(self):
        train, val = NFIRnDStrategy.split(make_dataset(STEMS), fraction=0.75, seed=42)
        assert set(train.indices) | set(val.indices) == set(range(len(STEMS)))

    def test_deterministic_with_seed(self):
        t1, v1 = NFIRnDStrategy.split(make_dataset(STEMS), fraction=0.75, seed=0)
        t2, v2 = NFIRnDStrategy.split(make_dataset(STEMS), fraction=0.75, seed=0)
        assert t1.indices == t2.indices
        assert v1.indices == v2.indices


# ---------------------------------------------------------------------------
# Fractional split — ungrouped
# ---------------------------------------------------------------------------


class TestFractionalSplitUngrouped:
    def test_returns_two_subsets(self):
        train, val = NFIRnDStrategy.split(
            make_dataset(STEMS), fraction=0.75, seed=42, genotype_aware=False
        )
        assert isinstance(train, Subset) and isinstance(val, Subset)

    def test_sizes_approximately_correct(self):
        train, val = NFIRnDStrategy.split(
            make_dataset(STEMS), fraction=0.75, seed=42, genotype_aware=False
        )
        n = len(STEMS)
        assert len(train) + len(val) == n
        assert abs(len(train) - round(n * 0.75)) <= 1

    def test_no_index_overlap(self):
        train, val = NFIRnDStrategy.split(
            make_dataset(STEMS), fraction=0.75, seed=42, genotype_aware=False
        )
        assert set(train.indices).isdisjoint(set(val.indices))


# ---------------------------------------------------------------------------
# K-fold split — grouped (default)
# ---------------------------------------------------------------------------


class TestKFoldSplitGrouped:
    def test_returns_k_fold_pairs(self):
        folds = NFIRnDStrategy.split(make_dataset(STEMS), k_folds=4, seed=42)
        assert isinstance(folds, list) and len(folds) == 4
        assert all(isinstance(t, Subset) and isinstance(v, Subset) for t, v in folds)

    def test_each_fold_covers_all_samples(self):
        folds = NFIRnDStrategy.split(make_dataset(STEMS), k_folds=4, seed=42)
        all_idx = set(range(len(STEMS)))
        for train, val in folds:
            assert set(train.indices) | set(val.indices) == all_idx

    @pytest.mark.parametrize(
        'k_folds, warning',
        [
            (2, False),
            (3, False),
            (4, True),
            (5, True),
            (6, False),
        ],
    )
    def test_different_fold_numbers(self, monkeypatch, k_folds: int, warning: bool):
        warning_messages: list[str] = []
        monkeypatch.setattr(
            nfi_rnd_module.logger,
            'warning',
            lambda message, *args, **kwargs: warning_messages.append(str(message)),
        )

        folds = NFIRnDStrategy.split(make_dataset(STEMS), k_folds=k_folds, seed=42)
        if warning:
            assert warning_messages == [
                f'Splitting the NFI R&D into {k_folds} folds results in uneven splits (2, 3, or 6 will)'
            ]
        else:
            assert warning_messages == []

    def test_val_indices_partition_all_samples(self):
        """Each sample appears in exactly one val fold."""
        folds = NFIRnDStrategy.split(make_dataset(STEMS), k_folds=4, seed=42)
        n = len(STEMS)
        val_counts = [0] * n
        for _, val in folds:
            for i in val.indices:
                val_counts[i] += 1
        assert all(c == 1 for c in val_counts)

    def test_deterministic_with_seed(self):
        f1 = NFIRnDStrategy.split(make_dataset(STEMS), k_folds=4, seed=7)
        f2 = NFIRnDStrategy.split(make_dataset(STEMS), k_folds=4, seed=7)
        for (t1, v1), (t2, v2) in zip(f1, f2, strict=True):
            assert t1.indices == t2.indices
            assert v1.indices == v2.indices


# ---------------------------------------------------------------------------
# K-fold split — ungrouped
# ---------------------------------------------------------------------------


class TestKFoldSplitUngrouped:
    def test_returns_k_tuples(self):
        folds = NFIRnDStrategy.split(make_dataset(STEMS), k_folds=3, seed=42, genotype_aware=False)
        assert len(folds) == 3

    def test_all_samples_seen_across_val_folds(self):
        folds = NFIRnDStrategy.split(make_dataset(STEMS), k_folds=3, seed=42, genotype_aware=False)
        seen = set()
        for _, val in folds:
            seen |= set(val.indices)
        assert seen == set(range(len(STEMS)))

    def test_k_fold_numbers(self):
        NFIRnDStrategy.split(make_dataset(STEMS), k_folds=2, seed=42, genotype_aware=False)
        NFIRnDStrategy.split(make_dataset(STEMS), k_folds=3, seed=42, genotype_aware=False)
        NFIRnDStrategy.split(make_dataset(STEMS), k_folds=4, seed=42, genotype_aware=False)
        NFIRnDStrategy.split(make_dataset(STEMS), k_folds=5, seed=42, genotype_aware=False)
        NFIRnDStrategy.split(make_dataset(STEMS), k_folds=6, seed=42, genotype_aware=False)

        with pytest.raises(ValueError, match='Provide either a fraction'):
            NFIRnDStrategy.split(make_dataset(STEMS), k_folds=7, seed=42, genotype_aware=False)
