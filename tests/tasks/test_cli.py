"""Tests for the CLI entry point."""

from __future__ import annotations

from pathlib import Path

from collections.abc import Mapping

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import dnanet


def _compose(*overrides: str):
    workspace = Path(dnanet.__file__).parents[2]
    conf_dir = workspace / 'conf'

    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        return compose(config_name='config', overrides=list(overrides))


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
        cfg = OmegaConf.load('conf/config.yaml')
        task_defaults = [
            d for d in cfg.defaults if isinstance(d, Mapping) and d.get('task') == 'train'
        ]
        assert len(task_defaults) == 1
        assert cfg.seed == 42
        assert cfg.verbosity == 'INFO'
        assert cfg.checkpoint is None

    def test_training_configs_have_type(self):
        """All training configs should have a 'type' field."""
        for name in ('segmentation', 'classification', 'reconstruction'):
            cfg = OmegaConf.load(f'conf/train/{name}.yaml')
            assert cfg.type == name

    def test_model_configs_have_architecture(self):
        """All model configs should have architecture._target_."""
        for name in ('unet', 'autoencoder', 'peak_classifier', 'peaknet'):
            cfg = OmegaConf.load(f'conf/model/{name}.yaml')
            assert 'architecture' in cfg
            assert '_target_' in cfg.architecture
