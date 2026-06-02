"""Tests for the CLI infer task module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dnanet.tasks.infer import run, _resolve_strategy, _parse_hid_profiles


class TestParseHidProfiles:
    """Tests for hid_profiles parsing from Hydra config."""

    def test_empty_list(self):
        """Empty list returns empty."""
        cfg = MagicMock()
        cfg.get.return_value = []
        assert _parse_hid_profiles(cfg) == []

    def test_none_returns_empty(self):
        """None returns empty."""
        cfg = MagicMock()
        cfg.get.return_value = None
        assert _parse_hid_profiles(cfg) == []

    def test_single_string_no_ladder(self):
        """Single string returns list with no ladder."""
        cfg = MagicMock()
        cfg.get.return_value = 'sample1.HID'
        result = _parse_hid_profiles(cfg)
        assert result == [('sample1.HID', None)]

    def test_json_string_list(self):
        """JSON string list is parsed correctly."""
        cfg = MagicMock()
        cfg.get.return_value = '[["sample1.HID", "ladder1.HID"], ["sample2.HID"]]'
        result = _parse_hid_profiles(cfg)
        assert result == [('sample1.HID', 'ladder1.HID'), ('sample2.HID', None)]

    def test_config_list(self):
        """Config list is parsed correctly."""
        cfg = MagicMock()
        cfg.get.return_value = ['sample1.HID', ('sample2.HID', 'ladder2.HID')]
        result = _parse_hid_profiles(cfg)
        assert result == [('sample1.HID', None), ('sample2.HID', 'ladder2.HID')]


class TestResolveStrategy:
    """Tests for scaling strategy resolution."""

    @pytest.fixture
    def mock_get_class(self):
        """Mock hydra.utils.get_class."""
        with patch('hydra.utils.get_class') as mock:
            mock.return_value = MagicMock
            yield mock

    def test_kit_ppf6c(self, mock_get_class):
        """PPF6C kit resolves to powerplex_fusion_6c strategy."""
        cfg = MagicMock()
        cfg.get.side_effect = lambda key, default=None: {'kit': 'PPF6C'}.get(key, default)
        _resolve_strategy(cfg)
        mock_get_class.assert_called_with('dnanet.data.strategies.scaling.PowerPlexFusion6CStrategy')

    def test_kit_gf(self, mock_get_class):
        """GF kit resolves to globalfiler strategy."""
        cfg = MagicMock()
        cfg.get.side_effect = lambda key, default=None: {'kit': 'GF'}.get(key, default)
        _resolve_strategy(cfg)
        mock_get_class.assert_called_with('dnanet.data.strategies.scaling.GlobalFilerStrategy')

    def test_kit_py23(self, mock_get_class):
        """PY23 kit resolves to powerplex_y23 strategy."""
        cfg = MagicMock()
        cfg.get.side_effect = lambda key, default=None: {'kit': 'PY23'}.get(key, default)
        _resolve_strategy(cfg)
        mock_get_class.assert_called_with('dnanet.data.strategies.scaling.PowerplexY23')

    def test_unknown_kit_raises(self):
        """Unknown kit raises ValueError."""
        cfg = MagicMock()
        cfg.get.side_effect = lambda key, default=None: {'kit': 'UNKNOWN'}.get(key, default)
        with pytest.raises(ValueError, match="Unknown kit: 'UNKNOWN'"):
            _resolve_strategy(cfg)

    def test_default_strategy(self, mock_get_class):
        """No kit or strategy defaults to powerplex_fusion_6c."""
        cfg = MagicMock()
        cfg.get.return_value = None
        _resolve_strategy(cfg)
        mock_get_class.assert_called_with('dnanet.data.strategies.scaling.PowerPlexFusion6CStrategy')

    def test_scaling_strategy_name(self, mock_get_class):
        """Direct scaling_strategy name is used."""
        cfg = MagicMock()
        cfg.get.side_effect = lambda key, default=None: {'scaling_strategy': 'globalfiler'}.get(
            key, default
        )
        _resolve_strategy(cfg)
        mock_get_class.assert_called_with('dnanet.data.strategies.scaling.GlobalFilerStrategy')


class TestRun:
    """Tests for the run function."""

    def test_missing_checkpoint_raises(self):
        """Missing checkpoint raises ValueError."""
        cfg = MagicMock()
        cfg.get.return_value = None
        with pytest.raises(ValueError, match='requires a checkpoint path'):
            run(cfg)

    def test_no_profiles_logs_warning(self):
        """No profiles logs warning and returns."""
        cfg = MagicMock()
        cfg.get.side_effect = lambda key, default=None: {
            'checkpoint': '/fake.ckpt',
            'hid_profiles': [],
            'kit': 'PPF6C',
            'caller': 'nearest',
            'prediction_threshold': 0.5,
            'confidence_threshold': None,
            'batch_size': 1,
            'num_workers': 0,
            'save_predictions': False,
            'save_plots': False,
            'output_dir': None,
            'save_json': True,
            'device': None,
        }.get(key, default)

        with patch('dnanet.tasks.infer._resolve_strategy') as mock_strategy:
            with patch('dnanet.tasks.infer._parse_hid_profiles') as mock_profiles:
                mock_profiles.return_value = []
                mock_strategy.return_value = MagicMock()
                # Should not raise, just log warning
                run(cfg)

    def test_run_success(self):
        """Successful run calls DNANetInfer.run."""
        result_mock = MagicMock()
        result_mock.total_profiles = 1
        result_mock.total_alleles = 10
        result_mock.total_markers_called = 5
        result_mock.save_json.return_value = Path('/tmp/result.json')

        cfg = MagicMock()
        cfg.get.side_effect = lambda key, default=None: {
            'checkpoint': '/fake.ckpt',
            'hid_profiles': '[["sample1.HID"]]',
            'kit': 'PPF6C',
            'caller': 'nearest',
            'prediction_threshold': 0.5,
            'confidence_threshold': None,
            'batch_size': 1,
            'num_workers': 0,
            'save_predictions': False,
            'save_plots': False,
            'output_dir': '/tmp/output',
            'save_json': True,
            'device': None,
        }.get(key, default)

        with patch('dnanet.tasks.infer._resolve_strategy') as mock_strategy:
            with patch('dnanet.tasks.infer._parse_hid_profiles') as mock_profiles:
                with patch('dnanet.tasks.infer.DNANetInfer.run', return_value=result_mock):
                    mock_strategy.return_value = MagicMock()
                    mock_profiles.return_value = [('sample1.HID', None)]
                    run(cfg)

                    result_mock.save_json.assert_called_once()
