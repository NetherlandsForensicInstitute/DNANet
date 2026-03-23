"""Tests for the training task."""

from __future__ import annotations

import pytest
from omegaconf import DictConfig, OmegaConf
from unittest.mock import MagicMock, patch

from dnanet.tasks.train import (
    _build_callbacks,
    _build_logger,
    _build_module,
    _resolve_module_class,
    _save_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_cfg(**overrides) -> DictConfig:
    """Create a minimal Hydra-like config for testing."""
    base = {
        "seed": 42,
        "output_dir": "/tmp/dnanet_test",
        "task": "train",
        "model": {
            "architecture": {
                "_target_": "dnanet.models.unet.UNet",
                "depth": 2,
                "kernel_size": [3],
                "num_filters": 8,
            },
            "loss": {
                "_target_": "dnanet.models.loss.DiceLoss",
            },
        },
        "training": {
            "type": "segmentation",
            "max_epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "scheduler": {"gamma": 1.0},
        },
    }
    base.update(overrides)
    return OmegaConf.create(base)


# ---------------------------------------------------------------------------
# _resolve_module_class
# ---------------------------------------------------------------------------

class TestResolveModuleClass:
    def test_segmentation(self):
        cls = _resolve_module_class("segmentation")
        assert cls.__name__ == "SegmentationModule"

    def test_classification(self):
        cls = _resolve_module_class("classification")
        assert cls.__name__ == "ClassificationModule"

    def test_reconstruction(self):
        cls = _resolve_module_class("reconstruction")
        assert cls.__name__ == "ReconstructionModule"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown training type"):
            _resolve_module_class("unknown")


# ---------------------------------------------------------------------------
# _build_callbacks
# ---------------------------------------------------------------------------

class TestBuildCallbacks:
    def test_no_callbacks(self):
        cfg = _make_cfg()
        callbacks = _build_callbacks(cfg)
        assert len(callbacks) == 0

    def test_early_stopping(self):
        cfg = _make_cfg()
        cfg.training.early_stopping = {
            "monitor": "val/loss",
            "patience": 3,
            "min_delta": 0.01,
            "mode": "min",
        }
        callbacks = _build_callbacks(cfg)
        assert len(callbacks) == 1
        assert type(callbacks[0]).__name__ == "EarlyStopping"

    def test_checkpoint(self):
        cfg = _make_cfg()
        cfg.training.checkpoint = {
            "monitor": "val/loss",
            "save_top_k": 1,
            "mode": "min",
        }
        callbacks = _build_callbacks(cfg)
        assert len(callbacks) == 1
        assert type(callbacks[0]).__name__ == "ModelCheckpoint"

    def test_both_callbacks(self):
        cfg = _make_cfg()
        cfg.training.early_stopping = {
            "monitor": "val/loss",
            "patience": 3,
        }
        cfg.training.checkpoint = {
            "monitor": "val/loss",
        }
        callbacks = _build_callbacks(cfg)
        assert len(callbacks) == 2


# ---------------------------------------------------------------------------
# _build_logger
# ---------------------------------------------------------------------------

class TestBuildLogger:
    def test_no_logging_config(self):
        cfg = _make_cfg()
        assert _build_logger(cfg) is None

    def test_csv_logger(self):
        cfg = _make_cfg()
        cfg.logging = {"logger_type": "csv"}
        lgr = _build_logger(cfg)
        assert lgr is not None
        assert type(lgr).__name__ == "CSVLogger"

    def test_tensorboard_logger(self):
        pytest.importorskip("tensorboard", reason="tensorboard not installed")
        cfg = _make_cfg()
        cfg.logging = {"logger_type": "tensorboard"}
        lgr = _build_logger(cfg)
        assert lgr is not None
        assert type(lgr).__name__ == "TensorBoardLogger"


# ---------------------------------------------------------------------------
# _build_module
# ---------------------------------------------------------------------------

class TestBuildModule:
    def test_segmentation_module(self):
        cfg = _make_cfg()
        from dnanet.models.unet import UNet
        from dnanet.models.loss import DiceLoss
        network = UNet(depth=2, kernel_size=(3, 5), num_filters=8)
        loss = DiceLoss()
        module = _build_module(cfg, network, loss)
        assert type(module).__name__ == "SegmentationModule"

    def test_classification_module(self):
        cfg = _make_cfg()
        cfg.training.type = "classification"
        from dnanet.models.peak_classifier import PeakClassificationModel
        from torch import nn
        network = PeakClassificationModel(num_classes=3, width=120, embedding_dim=0, hidden_channels=[16])
        loss = nn.CrossEntropyLoss()
        module = _build_module(cfg, network, loss)
        assert type(module).__name__ == "ClassificationModule"

    def test_reconstruction_module(self):
        cfg = _make_cfg()
        cfg.training.type = "reconstruction"
        from dnanet.models.autoencoder import Conv1dAutoencoder
        from torch import nn
        network = Conv1dAutoencoder(in_channels=1, hidden_channels=8, depth=3,
                                     input_length=4096, kernel_size=7, compression=8)
        loss = nn.MSELoss()
        module = _build_module(cfg, network, loss)
        assert type(module).__name__ == "ReconstructionModule"


# ---------------------------------------------------------------------------
# _save_config
# ---------------------------------------------------------------------------

class TestSaveConfig:
    def test_saves_yaml(self, tmp_path):
        cfg = _make_cfg(output_dir=str(tmp_path))
        _save_config(cfg, str(tmp_path))
        config_file = tmp_path / "config.yaml"
        assert config_file.exists()
        content = config_file.read_text()
        assert "seed: 42" in content
