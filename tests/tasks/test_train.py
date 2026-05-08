"""Tests for the training task."""

from __future__ import annotations

from unittest.mock import ANY, MagicMock, call, patch

import pytest
from torch import nn
from omegaconf import OmegaConf, DictConfig

from dnanet.tasks.train import run, _save_config, _build_logger, _build_callbacks


def _make_cfg(**overrides) -> DictConfig:
    """Create a minimal Hydra-like config for training task tests."""
    base = {
        'seed': 42,
        'output_dir': '/tmp/dnanet_test',
        'task': 'train',
        'checkpoint': None,
        'model': {
            'architecture': {
                '_target_': 'dnanet.models.unet.UNet',
                'depth': 2,
                'kernel_size': [3],
                'num_filters': 8,
            },
        },
        'training': {
            'type': 'segmentation',
            'max_epochs': 2,
            'batch_size': 4,
            'learning_rate': 0.001,
            'weight_decay': 0.0,
            'optimizer': {
                '_target_': 'torch.optim.AdamW',
                'lr': '${training.learning_rate}',
                'weight_decay': '${training.weight_decay}',
            },
            'scheduler': {
                '_target_': 'torch.optim.lr_scheduler.ExponentialLR',
                'gamma': 0.8,
                'optimizer': '${training.optimizer}',
            },
            'lightning_module': {
                '_target_': 'dnanet.modules.segmentation.SegmentationModule',
            },
            'data_module': {
                '_target_': 'dnanet.data.datamodule.DNANetDataModule',
            },
        },
    }
    return OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(overrides))


class TestBuildCallbacks:
    def test_no_callbacks(self):
        cfg = _make_cfg()
        callbacks = _build_callbacks(cfg)
        assert callbacks == []

    def test_early_stopping(self):
        cfg = _make_cfg(
            training={
                'early_stopping': {
                    'monitor': 'val/loss',
                    'patience': 3,
                    'min_delta': 0.01,
                    'mode': 'min',
                },
            },
        )
        callbacks = _build_callbacks(cfg)
        assert len(callbacks) == 1
        assert type(callbacks[0]).__name__ == 'EarlyStopping'

    def test_checkpoint(self):
        cfg = _make_cfg(
            training={
                'checkpoint': {
                    'monitor': 'val/loss',
                    'save_top_k': 1,
                    'mode': 'min',
                },
            },
        )
        callbacks = _build_callbacks(cfg)
        assert len(callbacks) == 1
        assert type(callbacks[0]).__name__ == 'ModelCheckpoint'

    def test_both_callbacks(self):
        cfg = _make_cfg(
            training={
                'early_stopping': {
                    'monitor': 'val/loss',
                    'patience': 3,
                },
                'checkpoint': {
                    'monitor': 'val/loss',
                },
            },
        )
        callbacks = _build_callbacks(cfg)
        assert len(callbacks) == 2

    def test_configured_callbacks(self):
        cfg = _make_cfg(
            training={
                'callbacks': {
                    'confusion_matrix': {
                        '_target_': 'dnanet.evaluation.callbacks.ConfusionMatrixCallback',
                        'num_classes': 3,
                    },
                },
            },
        )

        callbacks = _build_callbacks(cfg)

        assert len(callbacks) == 1
        assert type(callbacks[0]).__name__ == 'ConfusionMatrixCallback'


class TestBuildLogger:
    def test_no_logging_config(self):
        cfg = _make_cfg()
        assert _build_logger(cfg) is None

    def test_csv_logger(self):
        cfg = _make_cfg(logging={'logger_type': 'csv'})
        logger = _build_logger(cfg)
        assert logger is not None
        assert type(logger).__name__ == 'CSVLogger'

    def test_tensorboard_logger(self):
        pytest.importorskip('tensorboard', reason='tensorboard not installed')
        cfg = _make_cfg(logging={'logger_type': 'tensorboard'})
        logger = _build_logger(cfg)
        assert logger is not None
        assert type(logger).__name__ == 'TensorBoardLogger'


class TestTrainingConfigFormat:
    @pytest.mark.parametrize(
        ('name', 'module_target'),
        [
            ('segmentation', 'dnanet.modules.segmentation.SegmentationModule'),
            ('reconstruction', 'dnanet.modules.reconstruction.ReconstructionModule'),
            ('peaknet', 'dnanet.modules.peaknet.PeakNetModule'),
        ],
    )
    def test_updated_training_configs_have_new_sections(self, name: str, module_target: str):
        cfg = OmegaConf.load(f'conf/training/{name}.yaml')

        for key in ('metrics', 'optimizer', 'scheduler', 'lightning_module', 'data_module'):
            assert key in cfg

        assert cfg.seed == 42
        assert cfg.optimizer._target_ == 'torch.optim.AdamW'
        assert cfg.scheduler._target_ == 'torch.optim.lr_scheduler.ExponentialLR'
        assert cfg.lightning_module._target_ == module_target
        assert cfg.data_module._target_ == 'dnanet.data.datamodule.DNANetDataModule'

    def test_updated_training_configs_match_classification_sections(self):
        reference = OmegaConf.load('conf/training/classification.yaml')
        expected_sections = {'metrics', 'optimizer', 'scheduler', 'lightning_module', 'data_module'}

        assert expected_sections.issubset(reference.keys())

        for name in ('segmentation', 'reconstruction', 'peaknet'):
            cfg = OmegaConf.load(f'conf/training/{name}.yaml')
            assert expected_sections.issubset(cfg.keys())


class TestRun:
    def test_run_uses_configured_optimizer_scheduler_module_and_datamodule(self):
        cfg = _make_cfg()
        dataset = MagicMock(name='dataset')
        network = nn.Linear(4, 2)
        optimizer = MagicMock(name='optimizer')
        scheduler = MagicMock(name='scheduler')
        module = MagicMock(name='module')
        datamodule = MagicMock(name='datamodule')
        trainer_instance = MagicMock(name='trainer')

        with (
            patch(
                'dnanet.tasks.train.instantiate',
                side_effect=[network, optimizer, scheduler, module, datamodule],
            ) as instantiate_mock,
            patch('dnanet.tasks.train._load_pretrained_weights') as load_weights_mock,
            patch('dnanet.tasks.train._build_callbacks', return_value=[]) as callbacks_mock,
            patch('dnanet.tasks.train._build_logger', return_value=None) as build_logger_mock,
            patch('dnanet.tasks.train._save_config') as save_config_mock,
            patch('dnanet.tasks.train.L.seed_everything') as seed_mock,
            patch('dnanet.tasks.train.L.Trainer', return_value=trainer_instance) as trainer_cls,
        ):
            trainer, built_module = run(cfg, dataset=dataset)

        seed_mock.assert_called_once_with(cfg.seed, workers=True)
        load_weights_mock.assert_called_once_with(network, cfg)
        callbacks_mock.assert_called_once_with(cfg)
        build_logger_mock.assert_called_once_with(cfg)

        assert instantiate_mock.call_args_list == [
            call(cfg.model.architecture),
            call(cfg.training.optimizer, params=ANY),
            call(cfg.training.scheduler, optimizer=optimizer),
            call(
                cfg.training.lightning_module,
                model=network,
                optimizer=optimizer,
                scheduler=scheduler,
                _convert_='partial',
            ),
            call(cfg.training.data_module, dataset=dataset),
        ]
        assert [id(param) for param in instantiate_mock.call_args_list[1].kwargs['params']] == [
            id(param) for param in network.parameters()
        ]

        trainer_cls.assert_called_once_with(
            max_epochs=cfg.training.max_epochs,
            callbacks=[],
            logger=None,
            default_root_dir=cfg.output_dir,
            deterministic=False,
            enable_progress_bar=True,
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
        )
        trainer_instance.fit.assert_called_once_with(module, datamodule=datamodule, ckpt_path=None)
        save_config_mock.assert_called_once_with(cfg, cfg.output_dir)
        assert trainer is trainer_instance
        assert built_module is module

    def test_run_skips_scheduler_instantiation_when_scheduler_missing(self):
        cfg = _make_cfg(training={'scheduler': None})
        dataset = MagicMock(name='dataset')
        network = nn.Linear(4, 2)
        optimizer = MagicMock(name='optimizer')
        module = MagicMock(name='module')
        datamodule = MagicMock(name='datamodule')
        trainer_instance = MagicMock(name='trainer')

        with (
            patch(
                'dnanet.tasks.train.instantiate',
                side_effect=[network, optimizer, module, datamodule],
            ) as instantiate_mock,
            patch('dnanet.tasks.train._load_pretrained_weights'),
            patch('dnanet.tasks.train._build_callbacks', return_value=[]),
            patch('dnanet.tasks.train._build_logger', return_value=None),
            patch('dnanet.tasks.train._save_config'),
            patch('dnanet.tasks.train.L.seed_everything'),
            patch('dnanet.tasks.train.L.Trainer', return_value=trainer_instance),
        ):
            run(cfg, dataset=dataset)

        assert instantiate_mock.call_args_list == [
            call(cfg.model.architecture),
            call(cfg.training.optimizer, params=ANY),
            call(
                cfg.training.lightning_module,
                model=network,
                optimizer=optimizer,
                scheduler=None,
                _convert_='partial',
            ),
            call(cfg.training.data_module, dataset=dataset),
        ]


class TestSaveConfig:
    def test_saves_yaml(self, tmp_path):
        cfg = _make_cfg(output_dir=str(tmp_path))
        _save_config(cfg, str(tmp_path))
        config_file = tmp_path / 'config.yaml'
        assert config_file.exists()
        content = config_file.read_text()
        assert 'seed: 42' in content
