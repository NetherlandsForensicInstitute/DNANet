"""Tests for NFIRnDStrategy splitting logic.

Covers _build_replica_map, _replica_noc_labels, _subsets, and the public
split() entry point for both fractional and k-fold modes (grouped/ungrouped).
All tests use a lightweight fake dataset — no real HID files required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from torch.utils.data import Subset

from dnanet.data.strategies.datasets.nfi_rnd import NFIRnDStrategy


# ---------------------------------------------------------------------------
# Fake dataset helpers
# ---------------------------------------------------------------------------


def _fake_img(stem: str):
    img = MagicMock()
    img.path.stem = stem
    return img


def make_dataset(stems: list[str]):
    """Minimal fake dataset: exposes .data (list of images) and __len__."""
    ds = MagicMock()
    ds.data = [_fake_img(s) for s in stems]
    ds.__len__ = MagicMock(return_value=len(stems))
    return ds


# 8 replicas × 3 samples each, 24 samples total, 2 NoC values (2 and 5).
# Using 2 classes lets StratifiedGroupKFold/StratifiedKFold succeed even with
# small val folds (e.g. fraction=0.75 → n_splits=4 → 2 replicas in val ≥ 2 classes).
STEMS = [
    '1A2_A01_01',
    '1A2_A01_02',
    '1A2_A02_01',  # replica 1A2, NoC=2
    '2A2_A01_01',
    '2A2_A01_02',
    '2A2_A02_01',  # replica 2A2, NoC=2
    '3A2_A01_01',
    '3A2_A01_02',
    '3A2_A02_01',  # replica 3A2, NoC=2
    '4A2_A01_01',
    '4A2_A01_02',
    '4A2_A02_01',  # replica 4A2, NoC=2
    '1B5_A01_01',
    '1B5_A01_02',
    '1B5_A02_01',  # replica 1B5, NoC=5
    '2B5_A01_01',
    '2B5_A01_02',
    '2B5_A02_01',  # replica 2B5, NoC=5
    '3B5_A01_01',
    '3B5_A01_02',
    '3B5_A02_01',  # replica 3B5, NoC=5
    '4B5_A01_01',
    '4B5_A01_02',
    '4B5_A02_01',  # replica 4B5, NoC=5
]


# ---------------------------------------------------------------------------
# _build_replica_map
# ---------------------------------------------------------------------------


class TestBuildReplicaMap:
    def test_groups_indices_by_prefix(self):
        ds = make_dataset(STEMS)
        replica_map = NFIRnDStrategy._build_replica_map(ds)
        assert set(replica_map.keys()) == {'1A2', '2A2', '3A2', '4A2', '1B5', '2B5', '3B5', '4B5'}
        assert replica_map['1A2'] == [0, 1, 2]
        assert replica_map['1B5'] == [12, 13, 14]

    def test_single_replica(self):
        ds = make_dataset(['1A2_A01_01', '1A2_A02_01'])
        replica_map = NFIRnDStrategy._build_replica_map(ds)
        assert list(replica_map.keys()) == ['1A2']
        assert replica_map['1A2'] == [0, 1]


# ---------------------------------------------------------------------------
# _replica_noc_labels
# ---------------------------------------------------------------------------


class TestReplicaNocLabels:
    def test_returns_one_label_per_replica(self):
        ds = make_dataset(STEMS)
        replica_map = NFIRnDStrategy._build_replica_map(ds)
        labels = NFIRnDStrategy._replica_noc_labels(ds, replica_map)
        assert len(labels) == 8

    def test_correct_noc_per_replica(self):
        ds = make_dataset(['1A2_A01_01', '1A2_A01_02'])
        replica_map = NFIRnDStrategy._build_replica_map(ds)
        labels = NFIRnDStrategy._replica_noc_labels(ds, replica_map)
        assert labels == [2]

    def test_raises_when_noc_not_inferable(self, monkeypatch):
        ds = make_dataset(['1A2_A01_01'])
        replica_map = {'1A2': [0]}
        monkeypatch.setattr(
            NFIRnDStrategy,
            'get_number_of_contributors',
            classmethod(lambda cls, file_name: None),
        )
        with pytest.raises(ValueError, match='NoC'):
            NFIRnDStrategy._replica_noc_labels(ds, replica_map)


# ---------------------------------------------------------------------------
# _subsets
# ---------------------------------------------------------------------------


class TestSubsets:
    def test_expands_positions_to_flat_indices(self):
        ds = make_dataset(STEMS)
        replica_map = NFIRnDStrategy._build_replica_map(ds)
        # positions 0,1 → replicas 1A2, 2A2; position 2 → 3A2
        train, val = NFIRnDStrategy._subsets(ds, replica_map, [0, 1], [2])
        assert isinstance(train, Subset)
        assert isinstance(val, Subset)
        assert set(train.indices) == {0, 1, 2, 3, 4, 5}
        assert set(val.indices) == {6, 7, 8}

    def test_no_index_overlap(self):
        ds = make_dataset(STEMS)
        replica_map = NFIRnDStrategy._build_replica_map(ds)
        train, val = NFIRnDStrategy._subsets(ds, replica_map, [0, 2], [1, 3])
        assert set(train.indices).isdisjoint(set(val.indices))


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

    def test_no_replica_split_across_sets(self):
        ds = make_dataset(STEMS)
        train, val = NFIRnDStrategy.split(ds, fraction=0.75, seed=42)
        replica_map = NFIRnDStrategy._build_replica_map(ds)
        train_set, val_set = set(train.indices), set(val.indices)
        for indices in replica_map.values():
            in_train = any(i in train_set for i in indices)
            in_val = any(i in val_set for i in indices)
            assert not (in_train and in_val), 'Replica was split across train and val'

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
            make_dataset(STEMS), fraction=0.75, seed=42, group_by_replica=False
        )
        assert isinstance(train, Subset) and isinstance(val, Subset)

    def test_sizes_approximately_correct(self):
        train, val = NFIRnDStrategy.split(
            make_dataset(STEMS), fraction=0.75, seed=42, group_by_replica=False
        )
        n = len(STEMS)
        assert len(train) + len(val) == n
        assert abs(len(train) - round(n * 0.75)) <= 1

    def test_no_index_overlap(self):
        train, val = NFIRnDStrategy.split(
            make_dataset(STEMS), fraction=0.75, seed=42, group_by_replica=False
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

    def test_no_replica_split_within_fold(self):
        ds = make_dataset(STEMS)
        folds = NFIRnDStrategy.split(ds, k_folds=4, seed=42)
        replica_map = NFIRnDStrategy._build_replica_map(ds)
        for fold_idx, (train, val) in enumerate(folds):
            train_set, val_set = set(train.indices), set(val.indices)
            for indices in replica_map.values():
                in_train = any(i in train_set for i in indices)
                in_val = any(i in val_set for i in indices)
                assert not (in_train and in_val), f'Fold {fold_idx}: replica split'

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
        folds = NFIRnDStrategy.split(make_dataset(STEMS), k_folds=3, seed=42, group_by_replica=False)
        assert len(folds) == 3

    def test_all_samples_seen_across_val_folds(self):
        folds = NFIRnDStrategy.split(make_dataset(STEMS), k_folds=3, seed=42, group_by_replica=False)
        seen = set()
        for _, val in folds:
            seen |= set(val.indices)
        assert seen == set(range(len(STEMS)))


# ---------------------------------------------------------------------------
# stratify_noc error handling
# ---------------------------------------------------------------------------


class TestStratifyNocErrors:
    def test_replica_noc_labels_raises_on_missing_noc(self, monkeypatch):
        ds = make_dataset(['1A2_A01_01', '1A2_A01_02'])
        replica_map = NFIRnDStrategy._build_replica_map(ds)
        monkeypatch.setattr(
            NFIRnDStrategy,
            'get_number_of_contributors',
            classmethod(lambda cls, file_name: None),
        )
        with pytest.raises(ValueError, match='NoC'):
            NFIRnDStrategy._replica_noc_labels(ds, replica_map)
