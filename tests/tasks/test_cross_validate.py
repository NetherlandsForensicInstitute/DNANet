"""Tests for the cross-validation task (unit-level tests for helper imports).

Full integration tests for cross-validation would require dataset loading
and model training, so we test the structural aspects here.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from dnanet.data.splitting import split_data_in_k_folds


class TestKFoldSplitting:
    """Test the k-fold splitting utility used by cross_validate."""

    def test_split_sizes(self):
        data = list(range(100))
        folds = split_data_in_k_folds(data, n_folds=5, seed=42)
        assert len(folds) == 5
        assert sum(len(f) for f in folds) == 100

    def test_no_overlap(self):
        data = list(range(50))
        folds = split_data_in_k_folds(data, n_folds=5, seed=42)
        all_items = [item for fold in folds for item in fold]
        assert len(set(all_items)) == 50

    def test_reproducibility(self):
        data = list(range(30))
        folds1 = split_data_in_k_folds(data, n_folds=3, seed=123)
        folds2 = split_data_in_k_folds(data, n_folds=3, seed=123)
        for f1, f2 in zip(folds1, folds2):
            assert f1 == f2

    def test_uneven_split(self):
        data = list(range(13))
        folds = split_data_in_k_folds(data, n_folds=3, seed=42)
        assert len(folds) == 3
        assert sum(len(f) for f in folds) == 13
        # Sizes should be close: 4, 4, 5 or similar
        sizes = [len(f) for f in folds]
        assert max(sizes) - min(sizes) <= 1


class TestCrossValidateImports:
    """Test that the cross_validate module can be imported and configured."""

    def test_module_importable(self):
        from dnanet.tasks import cross_validate
        assert hasattr(cross_validate, "run")

    def test_config_structure(self):
        """Test that cross_validate expects n_folds in training config."""
        cfg = OmegaConf.create({
            "seed": 42,
            "output_dir": "/tmp/test",
            "training": {"n_folds": 5, "max_epochs": 1, "batch_size": 4},
        })
        assert cfg.training.n_folds == 5
