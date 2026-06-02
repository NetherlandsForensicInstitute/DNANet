"""Tests for the inference pipeline and DNANetInfer API."""

from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

from dnanet.infer import DNANetInfer, InferencePipeline
from dnanet.infer.output import (
    AlleleCall,
    MarkerResult,
    ProfileResult,
    InferenceResult,
    save_epg_plot,
)
from dnanet.data.strategies.scaling import PowerPlexFusion6CStrategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scaling_strategy():
    """Default scaling strategy for tests."""
    return PowerPlexFusion6CStrategy()


@pytest.fixture
def mock_pipeline(scaling_strategy, tmp_path):
    """Create a mock InferencePipeline with mocked model loading."""
    with patch.object(InferencePipeline, '__init__', return_value=None) as mock_init:
        pipeline = InferencePipeline.__new__(InferencePipeline)
        pipeline.checkpoint = tmp_path / 'best.ckpt'
        pipeline.scaling_strategy = scaling_strategy
        pipeline.device = 'cpu'
        pipeline._module = MagicMock()
        pipeline._model_type = 'segmentation'
        mock_init.return_value = None
        yield pipeline


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInferencePipeline:
    """Tests for InferencePipeline class."""

    def test_init_missing_config(self, tmp_path, scaling_strategy):
        """Pipeline raises if config.yaml is missing from checkpoint directory."""
        ckpt_dir = tmp_path / 'outputs' / 'no_config' / 'checkpoints'
        ckpt_dir.mkdir(parents=True)
        ckpt_path = ckpt_dir / 'missing.ckpt'
        ckpt_path.touch()

        with pytest.raises(FileNotFoundError, match='Config not found'):
            InferencePipeline(checkpoint=ckpt_path, scaling_strategy=scaling_strategy)

    def test_model_type_property(self, mock_pipeline):
        """Model type property returns correct value."""
        mock_pipeline._model_type = 'segmentation'
        assert mock_pipeline.model_type == 'segmentation'

    def test_model_property(self, mock_pipeline):
        """Model property returns the model."""
        mock_model = MagicMock()
        mock_pipeline._module = MagicMock()
        mock_pipeline._module.model = mock_model
        assert mock_pipeline.model is mock_model


class TestDNANetInfer:
    """Tests for DNANetInfer high-level API."""

    def test_run_classmethod_exists(self):
        """DNANetInfer.run() classmethod exists and is callable."""
        assert callable(DNANetInfer.run)

    def test_instance_api(self):
        """DNANetInfer instance creation works."""
        infer = DNANetInfer()
        assert infer is not None

    def test_run_signature(self):
        """DNANetInfer.run() accepts expected parameters."""
        import inspect

        sig = inspect.signature(DNANetInfer.run)
        params = list(sig.parameters.keys())
        assert 'checkpoint' in params
        assert 'hid_profiles' in params
        assert 'scaling_strategy' in params
        assert 'confidence_threshold' in params
        assert 'save_plots' in params
        assert 'output_dir' in params


class TestAlleleCalling:
    """Tests for allele calling from predictions."""

    def test_confidence_extraction(self, mock_pipeline):
        """Confidence scores are extracted from connected components."""
        # Create synthetic prediction data
        pred = np.zeros((4, 100), dtype=np.float32)
        # Create a clear peak at positions 20-25
        pred[0, 20:26] = np.array([0.7, 0.8, 0.9, 0.95, 0.85, 0.6], dtype=np.float32)

        components = mock_pipeline._find_connected_components(pred)
        assert 0 in components
        assert (0, 20, 25) == (0, components[0][0][0], components[0][0][1])

    def test_confidence_with_threshold(self, mock_pipeline):
        """Alleles below confidence threshold are filtered."""
        # Create prediction with low confidence
        pred = np.zeros((4, 100), dtype=np.float32)
        pred[0, 20:26] = np.array([0.1, 0.2, 0.15, 0.1, 0.2, 0.1], dtype=np.float32)

        components = mock_pipeline._find_connected_components(pred)
        # No components should be found since all values < 0.5
        assert 0 not in components

    def test_extract_confidence_closest_position(self, mock_pipeline):
        """Confidence is extracted at the position closest to base pair."""
        pred = np.zeros((4, 100), dtype=np.float32)
        pred[0, 20:26] = np.array([0.7, 0.8, 0.9, 0.95, 0.85, 0.6], dtype=np.float32)

        scaler = np.linspace(0, 100, 100)  # Simple linear scaler

        # Find confidence at base_pair=23 (should map to scanpoint ~23)
        components = mock_pipeline._find_connected_components(pred)
        confidence = mock_pipeline._extract_confidence(
            components=components,
            dye_row=0,
            base_pair=23.0,
            scaler=scaler,
            prediction_image=pred,
        )
        assert confidence > 0  # Should find a component

    def test_extract_confidence_no_component(self, mock_pipeline):
        """Returns 0.0 when no component found."""
        pred = np.zeros((4, 100), dtype=np.float32)
        components = mock_pipeline._find_connected_components(pred)

        confidence = mock_pipeline._extract_confidence(
            components=components,
            dye_row=0,
            base_pair=50.0,
            scaler=np.linspace(0, 100, 100),
            prediction_image=pred,
        )
        assert confidence == 0.0


class TestResultSerialization:
    """Tests for result serialization."""

    def test_profile_to_json(self, tmp_path):
        """ProfileResult serializes to JSON correctly via InferenceResult."""
        import json

        profile = ProfileResult(
            sample='test_sample',
            hid_path='/data/test.hid',
            ladder_path='/data/ladder.hid',
            markers=[
                MarkerResult(
                    name='D3S1358',
                    dye_row=0,
                    alleles=[
                        AlleleCall(name='12', base_pair=120.0, height=1500.0, confidence=0.95),
                        AlleleCall(name='15', base_pair=132.0, height=2200.0, confidence=0.88),
                    ],
                ),
            ],
            warnings=['Low signal in dye 4'],
        )

        result = InferenceResult(
            checkpoint='test.ckpt',
            kit='PPF6C',
            profiles=[profile],
        )

        output_path = tmp_path / 'result.json'
        saved_path = result.save_json(output_path)
        assert saved_path.exists()

        content = json.loads(saved_path.read_text())
        assert content['profiles'][0]['sample'] == 'test_sample'
        assert content['profiles'][0]['hid_path'] == '/data/test.hid'
        assert content['profiles'][0]['ladder_path'] == '/data/ladder.hid'
        assert len(content['profiles'][0]['markers']) == 1
        assert content['profiles'][0]['markers'][0]['name'] == 'D3S1358'
        assert len(content['profiles'][0]['markers'][0]['alleles']) == 2
        assert content['profiles'][0]['markers'][0]['alleles'][0]['name'] == '12'
        assert content['profiles'][0]['markers'][0]['alleles'][0]['confidence'] == 0.95
        assert content['profiles'][0]['warnings'] == ['Low signal in dye 4']

    def test_inference_result_to_json(self, tmp_path):
        """InferenceResult serializes to JSON correctly."""
        import json

        result = InferenceResult(
            checkpoint='best.ckpt',
            kit='PPF6C',
            profiles=[
                ProfileResult(
                    sample='s1',
                    hid_path='/data/s1.hid',
                    markers=[
                        MarkerResult(
                            name='D3S1358',
                            dye_row=0,
                            alleles=[
                                AlleleCall(
                                    name='12', base_pair=120.0, height=1500.0, confidence=0.95
                                )
                            ],
                        ),
                    ],
                ),
            ],
            timing={'s1.hid': 0.5},
        )

        output_path = tmp_path / 'inference.json'
        saved_path = result.save_json(output_path)
        assert saved_path.exists()

        content = json.loads(saved_path.read_text())
        assert content['checkpoint'] == 'best.ckpt'
        assert content['kit'] == 'PPF6C'
        assert content['total_profiles'] == 1
        assert content['total_alleles'] == 1
        assert content['timing']['s1.hid'] == 0.5


class TestEPGPlotSaving:
    """Tests for EPG plot saving."""

    def test_save_epg_plot(self, tmp_path):
        """EPG plot is saved correctly."""
        rng = np.random.default_rng()
        signal = rng.random((4, 4096)).astype(np.float32) * 1000
        prediction = rng.random((4, 4096)).astype(np.float32)

        output_path = tmp_path / 'epg.png'
        result = save_epg_plot(
            signal=signal.tolist(),
            prediction=prediction.tolist(),
            title='test_profile',
            output_path=output_path,
        )
        assert result.exists()
        assert result.stat().st_size > 1000

    def test_save_epg_plot_no_prediction(self, tmp_path):
        """EPG plot saved without prediction overlay."""
        rng = np.random.default_rng()
        signal = rng.random((4, 4096)).astype(np.float32) * 1000

        output_path = tmp_path / 'epg_no_pred.png'
        result = save_epg_plot(
            signal=signal.tolist(),
            title='test_profile',
            output_path=output_path,
        )
        assert result.exists()
