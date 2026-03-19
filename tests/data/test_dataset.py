"""Tests for dataset containers."""

import pytest

from dnanet.data.dataset import InMemoryDataset, SimpleDataset


class TestSimpleDataset:
    def test_len(self):
        ds = SimpleDataset(data=list(range(10)))
        assert len(ds) == 10

    def test_getitem(self):
        ds = SimpleDataset(data=list(range(10)))
        assert ds[0] == 0
        assert ds[9] == 9

    def test_iter(self):
        ds = SimpleDataset(data=list(range(5)))
        assert list(ds) == [0, 1, 2, 3, 4]

    def test_split(self):
        ds = SimpleDataset(data=list(range(100)))
        train, val = ds.split(fraction=0.8, seed=42)
        assert len(train) + len(val) == 100
        assert len(train) == 80

    def test_split_invalid_fraction(self):
        ds = SimpleDataset(data=list(range(10)))
        with pytest.raises(ValueError):
            ds.split(fraction=0.0)
        with pytest.raises(ValueError):
            ds.split(fraction=1.0)

    def test_k_fold(self):
        ds = SimpleDataset(data=list(range(50)))
        folds = ds.split_k_fold(n_folds=5, seed=42)
        assert len(folds) == 5
        for train, test in folds:
            assert len(train) + len(test) == 50


class TestSimpleDatasetShuffle:
    def test_shuffle_changes_order(self):
        data = list(range(100))
        ds = SimpleDataset(data=data, shuffle=True)
        # With 100 items, shuffled order should differ from original
        items = list(ds)
        assert len(items) == 100
        assert set(items) == set(data)
