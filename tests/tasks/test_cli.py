"""Tests for the CLI entry point."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf


class TestCLIImport:
    def test_main_importable(self):
        from dnanet.cli import main
        assert callable(main)

    def test_task_dispatch_train(self):
        """Verify the train task module is importable."""
        from dnanet.tasks.train import run
        assert callable(run)

    def test_task_dispatch_evaluate(self):
        """Verify the evaluate task module is importable."""
        from dnanet.tasks.evaluate import run
        assert callable(run)

    def test_task_dispatch_cross_validate(self):
        """Verify the cross_validate task module is importable."""
        from dnanet.tasks.cross_validate import run
        assert callable(run)


class TestConfigComposition:
    """Test that Hydra config groups compose correctly."""

    def test_master_config_defaults(self):
        """Verify the master config has expected defaults."""
        cfg = OmegaConf.load(
            "conf/config.yaml"
        )
        assert cfg.task == "train"
        assert cfg.seed == 42
        assert cfg.verbosity == "INFO"
        assert cfg.checkpoint is None

    def test_training_configs_have_type(self):
        """All training configs should have a 'type' field."""
        for name in ("segmentation", "classification", "reconstruction"):
            cfg = OmegaConf.load(f"conf/training/{name}.yaml")
            assert cfg.type == name

    def test_model_configs_have_architecture(self):
        """All model configs should have architecture._target_."""
        for name in ("unet", "autoencoder", "peak_classifier", "peaknet"):
            cfg = OmegaConf.load(f"conf/model/{name}.yaml")
            assert "architecture" in cfg
            assert "_target_" in cfg.architecture
