"""Tests for the cross_validate task."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf, DictConfig


def _make_cfg(**overrides) -> DictConfig:
    base = {
        'seed': 42,
        'output_dir': '/tmp/dnanet_cv_test',
        'task': 'cross_validate',
        'model': {
            'architecture': {
                '_target_': 'dnanet.models.unet.UNet',
                'depth': 2,
                'kernel_size': [3],
                'num_filters': 8,
            },
        },
        'train': {
            'max_epochs': 1,
            'batch_size': 4,
            'num_workers': 0,
            'optimizer': {
                '_target_': 'torch.optim.AdamW',
                'lr': 0.001,
            },
            'lightning_module': {
                '_target_': 'dnanet.modules.segmentation.SegmentationModule',
            },
        },
        'splitting': {
            'k_folds': 3,
            'test_fraction': 0.0,
            'seed': 42,
        },
    }
    return OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(overrides))


def _mock_dataset(size: int = 10) -> MagicMock:
    ds = MagicMock()
    ds.__len__ = MagicMock(return_value=size)
    return ds


def _mock_network() -> MagicMock:
    net = MagicMock(name='network')
    net.parameters.return_value = iter([])
    return net


def _make_trainer(metrics: dict | None = None) -> MagicMock:
    t = MagicMock(name='trainer')
    t.callback_metrics = metrics or {}
    return t


class TestRun:
    def test_raises_when_no_data_config(self):
        from dnanet.tasks.cross_validate import run

        cfg = _make_cfg()
        with (
            patch('dnanet.tasks.cross_validate.L.seed_everything'),
            pytest.raises(ValueError, match='requires a dataset'),
        ):
            run(cfg)

    def test_raises_when_no_k_folds(self):
        from dnanet.tasks.cross_validate import run

        cfg = _make_cfg(splitting={'k_folds': 0})
        with (
            patch('dnanet.tasks.cross_validate.L.seed_everything'),
            pytest.raises(ValueError, match='requires k-fold splitting'),
        ):
            run(cfg, dataset=MagicMock())


class TestRunWithData:
    def _run(self, cfg, folds, trainers, n_instantiate_per_fold=3):
        """Helper: mock all I/O, run run, return result."""
        instantiate_side = []
        for _ in range(len(folds)):
            instantiate_side += [_mock_network(), MagicMock(), MagicMock()]

        with (
            patch('dnanet.tasks.cross_validate.L.seed_everything'),
            patch('dnanet.tasks.cross_validate.dataset_splitter', return_value=(folds, None)),
            patch('dnanet.tasks.cross_validate.instantiate', side_effect=instantiate_side),
            patch('dnanet.tasks.cross_validate.DataLoader', return_value=MagicMock()),
            patch('dnanet.tasks.cross_validate.L.Trainer', side_effect=trainers),
            patch('dnanet.tasks.cross_validate._build_callbacks', return_value=[]),
            patch('dnanet.tasks.cross_validate._build_logger', return_value=None),
            patch('dnanet.tasks.cross_validate._build_fold_logger', return_value=None),
        ):
            from dnanet.tasks.cross_validate import run

            return run(cfg, MagicMock())

    def test_trainer_fit_called_once_per_fold(self, tmp_path):
        cfg = _make_cfg(
            output_dir=str(tmp_path), splitting={'k_folds': 2, 'test_fraction': 0.0, 'seed': 42}
        )
        folds = [(_mock_dataset(), _mock_dataset()) for _ in range(2)]
        trainer = _make_trainer()

        self._run(cfg, folds, [trainer, trainer])

        assert trainer.fit.call_count == 2

    def test_returns_per_fold_metrics(self, tmp_path):
        cfg = _make_cfg(
            output_dir=str(tmp_path), splitting={'k_folds': 2, 'test_fraction': 0.0, 'seed': 42}
        )
        folds = [(_mock_dataset(), _mock_dataset()) for _ in range(2)]
        t1 = _make_trainer({'val/loss': 0.4})
        t2 = _make_trainer({'val/loss': 0.6})

        result = self._run(cfg, folds, [t1, t2])

        assert result['per_fold'] == [{'val/loss': 0.4}, {'val/loss': 0.6}]

    def test_aggregates_mean_and_std(self, tmp_path):
        cfg = _make_cfg(
            output_dir=str(tmp_path), splitting={'k_folds': 2, 'test_fraction': 0.0, 'seed': 42}
        )
        folds = [(_mock_dataset(), _mock_dataset()) for _ in range(2)]
        t1 = _make_trainer({'val/loss': 0.4})
        t2 = _make_trainer({'val/loss': 0.6})

        result = self._run(cfg, folds, [t1, t2])

        assert result['aggregate']['val/loss_mean'] == pytest.approx(0.5)
        assert result['aggregate']['val/loss_std'] == pytest.approx(0.1)

    def test_saves_aggregate_metrics_json(self, tmp_path):
        cfg = _make_cfg(
            output_dir=str(tmp_path), splitting={'k_folds': 1, 'test_fraction': 0.0, 'seed': 42}
        )
        folds = [(_mock_dataset(), _mock_dataset())]
        trainer = _make_trainer({'val/loss': 0.3})

        self._run(cfg, folds, [trainer])

        agg_path = tmp_path / 'aggregate_metrics.json'
        assert agg_path.exists()
        data = json.loads(agg_path.read_text())
        assert 'val/loss_mean' in data

    def test_saves_per_fold_metrics_json(self, tmp_path):
        cfg = _make_cfg(
            output_dir=str(tmp_path), splitting={'k_folds': 2, 'test_fraction': 0.0, 'seed': 42}
        )
        folds = [(_mock_dataset(), _mock_dataset()) for _ in range(2)]
        trainer = _make_trainer({'val/loss': 0.5})

        self._run(cfg, folds, [trainer, trainer])

        assert (tmp_path / 'fold_1' / 'metrics.json').exists()
        assert (tmp_path / 'fold_2' / 'metrics.json').exists()

    def test_saves_config_yaml(self, tmp_path):
        cfg = _make_cfg(
            output_dir=str(tmp_path), splitting={'k_folds': 1, 'test_fraction': 0.0, 'seed': 42}
        )
        folds = [(_mock_dataset(), _mock_dataset())]
        trainer = _make_trainer()

        self._run(cfg, folds, [trainer])

        assert (tmp_path / 'config.yaml').exists()

    def test_keyboard_interrupt_stops_loop_early(self, tmp_path):
        cfg = _make_cfg(
            output_dir=str(tmp_path), splitting={'k_folds': 3, 'test_fraction': 0.0, 'seed': 42}
        )
        folds = [(_mock_dataset(), _mock_dataset()) for _ in range(3)]

        call_idx = [0]

        def fit_side_effect(*args, **kwargs):
            call_idx[0] += 1
            if call_idx[0] == 2:
                raise KeyboardInterrupt()

        trainer = _make_trainer()
        trainer.fit.side_effect = fit_side_effect

        instantiate_side = [_mock_network(), MagicMock(), MagicMock()] * 3
        with (
            patch('dnanet.tasks.cross_validate.L.seed_everything'),
            patch('dnanet.tasks.cross_validate.dataset_splitter', return_value=(folds, None)),
            patch('dnanet.tasks.cross_validate.instantiate', side_effect=instantiate_side),
            patch('dnanet.tasks.cross_validate.DataLoader', return_value=MagicMock()),
            patch('dnanet.tasks.cross_validate.L.Trainer', return_value=trainer),
            patch('dnanet.tasks.cross_validate._build_callbacks', return_value=[]),
            patch('dnanet.tasks.cross_validate._build_logger', return_value=None),
            patch('dnanet.tasks.cross_validate._build_fold_logger', return_value=None),
        ):
            from dnanet.tasks.cross_validate import run

            result = run(cfg, MagicMock())

        assert trainer.fit.call_count == 2
        # per_fold has 2 elements because fold 2 completes before KeyboardInterrupt is raised
        assert len(result['per_fold']) == 2

    def test_skips_scheduler_instantiation_when_not_configured(self, tmp_path):
        cfg = _make_cfg(
            output_dir=str(tmp_path),
            splitting={'k_folds': 1, 'test_fraction': 0.0, 'seed': 42},
            train={'scheduler': None},
        )
        folds = [(_mock_dataset(), _mock_dataset())]
        trainer = _make_trainer()

        with (
            patch('dnanet.tasks.cross_validate.L.seed_everything'),
            patch('dnanet.tasks.cross_validate.dataset_splitter', return_value=(folds, None)),
            patch(
                'dnanet.tasks.cross_validate.instantiate',
                side_effect=[_mock_network(), MagicMock(), MagicMock()],
            ) as inst_mock,
            patch('dnanet.tasks.cross_validate.DataLoader', return_value=MagicMock()),
            patch('dnanet.tasks.cross_validate.L.Trainer', return_value=trainer),
            patch('dnanet.tasks.cross_validate._build_callbacks', return_value=[]),
            patch('dnanet.tasks.cross_validate._build_logger', return_value=None),
            patch('dnanet.tasks.cross_validate._build_fold_logger', return_value=None),
        ):
            from dnanet.tasks.cross_validate import run

            run(cfg, MagicMock())

        # Should have 3 instantiate calls per fold: network, optimizer, module
        # No scheduler, so only 3 total
        assert inst_mock.call_count == 3

    def test_instantiates_scheduler_when_configured(self, tmp_path):
        cfg = _make_cfg(
            output_dir=str(tmp_path),
            splitting={'k_folds': 1, 'test_fraction': 0.0, 'seed': 42},
            train={
                'scheduler': {'_target_': 'torch.optim.lr_scheduler.ExponentialLR', 'gamma': 0.8}
            },
        )
        folds = [(_mock_dataset(), _mock_dataset())]
        trainer = _make_trainer()

        with (
            patch('dnanet.tasks.cross_validate.L.seed_everything'),
            patch('dnanet.tasks.cross_validate.dataset_splitter', return_value=(folds, None)),
            patch(
                'dnanet.tasks.cross_validate.instantiate',
                side_effect=[_mock_network(), MagicMock(), MagicMock(), MagicMock()],
            ) as inst_mock,
            patch('dnanet.tasks.cross_validate.DataLoader', return_value=MagicMock()),
            patch('dnanet.tasks.cross_validate.L.Trainer', return_value=trainer),
            patch('dnanet.tasks.cross_validate._build_callbacks', return_value=[]),
            patch('dnanet.tasks.cross_validate._build_logger', return_value=None),
            patch('dnanet.tasks.cross_validate._build_fold_logger', return_value=None),
        ):
            from dnanet.tasks.cross_validate import run

            run(cfg, MagicMock())

        # Should have 4 instantiate calls per fold: network, optimizer, scheduler, module
        assert inst_mock.call_count == 4

    def test_metric_tensors_converted_to_float(self, tmp_path):
        cfg = _make_cfg(
            output_dir=str(tmp_path), splitting={'k_folds': 1, 'test_fraction': 0.0, 'seed': 42}
        )
        folds = [(_mock_dataset(), _mock_dataset())]
        trainer = _make_trainer({'val/loss': 0.3})

        result = self._run(cfg, folds, [trainer])

        assert isinstance(result['per_fold'][0]['val/loss'], float)
