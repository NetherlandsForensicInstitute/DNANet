"""Tests for data splitting utilities."""

import pytest

from dnanet.data.splitting import split_data_in_k_folds


class TestSplitKFolds:
    def test_correct_number_of_folds(self):
        data = list(range(100))
        folds = split_data_in_k_folds(data, n_folds=5, seed=42)
        assert len(folds) == 5

    def test_all_items_present(self):
        data = list(range(100))
        folds = split_data_in_k_folds(data, n_folds=5, seed=42)
        all_items = sorted([item for fold in folds for item in fold])
        assert all_items == data

    def test_no_duplicates(self):
        data = list(range(100))
        folds = split_data_in_k_folds(data, n_folds=5, seed=42)
        all_items = [item for fold in folds for item in fold]
        assert len(all_items) == len(set(all_items))

    def test_approximately_equal_sizes(self):
        data = list(range(103))  # not evenly divisible by 5
        folds = split_data_in_k_folds(data, n_folds=5, seed=42)
        sizes = [len(f) for f in folds]
        assert max(sizes) - min(sizes) <= 1

    def test_deterministic_with_seed(self):
        data = list(range(50))
        folds_a = split_data_in_k_folds(data, n_folds=3, seed=123)
        folds_b = split_data_in_k_folds(data, n_folds=3, seed=123)
        for a, b in zip(folds_a, folds_b):
            assert a == b
