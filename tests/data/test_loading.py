"""Tests for load_dataset() — Hydra config to HIDDataset bridge."""

import pytest
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from tests.conftest import RD_DIR, LADDER_ALLELES_CSV
from dnanet.data.dataset import InMemoryDataset
from dnanet.data.loading import load_dataset
from dnanet.data.strategies.registry import StrategyRegistry


@pytest.fixture(autouse=True)
def reset_registry():
    yield
    StrategyRegistry.reset()


class TestLoadDataset:
    def _make_cfg(self, **overrides):
        """Build a minimal data config DictConfig."""
        base = {
            "kit": "PPF6C",
            "dataset_strategy": "NFI_RND",
            "root": str(RD_DIR),
            "annotations_path": str(RD_DIR),
            "hid_to_annotations_path": str(RD_DIR / "2p_5p_hid_to_annotation.csv"),
            "best_ladder_paths_csv": str(RD_DIR / "best_ladder_paths.csv"),
            "ladder_alleles_csv": str(LADDER_ALLELES_CSV),
            "analysis_threshold_type": "DTH",
            "data_loading_strategy": "superior",
            "adjustment_of_annotations": None,
            "limit": None,
            "skip_if_invalid_ladder": False,
            "include_size_standard": False,
        }
        base.update(overrides)
        return OmegaConf.create(base)

    def test_returns_pytorch_dataset(self):
        cfg = self._make_cfg()
        ds = load_dataset(cfg)
        assert isinstance(ds, Dataset)

    def test_configures_registry(self):
        cfg = self._make_cfg()
        load_dataset(cfg)
        # Registry should now have a scaling strategy configured
        scaling = StrategyRegistry.get_scaling_strategy()
        assert scaling is not None
        assert scaling.kit.name == "PPF6C"

    def test_limit_parameter(self):
        cfg = self._make_cfg(limit=1)
        ds = load_dataset(cfg)
        assert len(ds) == 1

    def test_nonexistent_root_raises(self, tmp_path):
        cfg = self._make_cfg(root=str(tmp_path / "nonexistent"))
        with pytest.raises(Exception):
            load_dataset(cfg)

    def test_globalfiler_kit(self):
        """Loading with GlobalFiler kit should configure the correct strategy."""
        cfg = OmegaConf.create({
            "kit": "GLOBALFILER",
            "dataset_strategy": "PROVEDIT",
            "root": str(RD_DIR),  # Reuse RD dir (will find 0 samples but that's fine)
            "annotations_path": None,
            "hid_to_annotations_path": None,
            "best_ladder_paths_csv": None,
            "ladder_alleles_csv": None,
            "analysis_threshold_type": "DTH",
            "data_loading_strategy": "superior",
            "adjustment_of_annotations": None,
            "limit": None,
            "skip_if_invalid_ladder": False,
            "include_size_standard": False,
        })
        # ProvedIt strategy won't find samples in RD dir, so this may raise
        # Just verify that the registry was configured correctly
        try:
            load_dataset(cfg)
        except ValueError:
            pass  # Expected: no valid images
        scaling = StrategyRegistry.get_scaling_strategy()
        assert scaling.kit.name == "GlobalFiler"
