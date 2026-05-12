"""Tests for peak classification architecture."""

from __future__ import annotations

import torch
import pytest

from dnanet.models.peak_classifier import BackboneModule, PeakClassificationModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def peak_batch() -> torch.Tensor:
    """Synthetic peak windows: (B=4, C=1, W=120)."""
    return torch.randn(4, 1, 120)


@pytest.fixture
def peak_batch_2ch() -> torch.Tensor:
    """Peak windows with max-pooled dyes: (B=4, C=2, W=120)."""
    return torch.randn(4, 2, 120)


@pytest.fixture
def marker_idxs() -> torch.Tensor:
    """Marker indices for 4 peaks."""
    return torch.randint(0, 28, (4,))


# ---------------------------------------------------------------------------
# BackboneModule ABC
# ---------------------------------------------------------------------------


class TestBackboneModuleABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BackboneModule()


# ---------------------------------------------------------------------------
# PeakClassificationModel — basic
# ---------------------------------------------------------------------------


class TestPeakClassificationModel:
    def test_forward_shape(self, peak_batch, marker_idxs):
        model = PeakClassificationModel(
            num_classes=3,
            width=120,
            n_markers=28,
            embedding_dim=8,
        )
        logits = model((peak_batch, marker_idxs))
        assert logits.shape == (4, 3)

    def test_forward_without_marker(self, peak_batch):
        model = PeakClassificationModel(
            num_classes=2,
            width=120,
            embedding_dim=0,
            use_embedding=False,
        )
        logits = model(peak_batch)
        assert logits.shape == (4, 2)

    def test_backbone_out_features(self):
        model = PeakClassificationModel(
            width=120,
            embedding_dim=8,
            hidden_channels=[16, 32],
        )
        out = model.backbone_out_features()
        assert isinstance(out, int) and out > 0

    def test_backbone_returns_features(self, peak_batch, marker_idxs):
        model = PeakClassificationModel(
            width=120,
            embedding_dim=8,
            hidden_channels=[16],
        )
        features = model.backbone((peak_batch, marker_idxs))
        assert features.shape == (4, model.backbone_out_features())

    def test_gradient_flows(self, peak_batch, marker_idxs):
        model = PeakClassificationModel(
            num_classes=2,
            width=120,
            embedding_dim=8,
        )
        logits = model((peak_batch, marker_idxs))
        logits.sum().backward()
        for p in model.parameters():
            assert p.grad is not None


# ---------------------------------------------------------------------------
# Pooling strategies
# ---------------------------------------------------------------------------


class TestPoolingStrategies:
    @pytest.mark.parametrize('pooling', ['flat', 'avg', 'attn'])
    def test_pooling_forward(self, peak_batch, pooling):
        model = PeakClassificationModel(
            num_classes=2,
            width=120,
            embedding_dim=0,
            use_embedding=False,
            pooling=pooling,
        )
        logits = model(peak_batch)
        assert logits.shape == (4, 2)

    def test_invalid_pooling_rejected(self):
        with pytest.raises(ValueError, match='pooling must be'):
            PeakClassificationModel(pooling='invalid')


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------


class TestActivations:
    @pytest.mark.parametrize('activation', ['relu', 'tanh', 'gelu'])
    def test_activation_forward(self, peak_batch, activation):
        model = PeakClassificationModel(
            num_classes=2,
            width=120,
            embedding_dim=0,
            use_embedding=False,
            activation=activation,
        )
        logits = model(peak_batch)
        assert logits.shape == (4, 2)

    def test_invalid_activation_rejected(self):
        with pytest.raises(ValueError, match='activation must be'):
            PeakClassificationModel(activation='silu')


# ---------------------------------------------------------------------------
# Downsample strategies
# ---------------------------------------------------------------------------


class TestDownsampleStrategies:
    @pytest.mark.parametrize('downsample', ['maxpool', 'conv'])
    def test_downsample_forward(self, peak_batch, downsample):
        model = PeakClassificationModel(
            num_classes=2,
            width=120,
            embedding_dim=0,
            use_embedding=False,
            downsample=downsample,
        )
        logits = model(peak_batch)
        assert logits.shape == (4, 2)

    def test_invalid_downsample_rejected(self):
        with pytest.raises(ValueError, match='downsample must be'):
            PeakClassificationModel(downsample='stride')


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


class TestOptions:
    def test_max_pool_dyes(self, peak_batch_2ch, marker_idxs):
        model = PeakClassificationModel(
            num_classes=2,
            width=120,
            include_max_pool_dyes=True,
            embedding_dim=8,
        )
        logits = model((peak_batch_2ch, marker_idxs))
        assert logits.shape == (4, 2)

    def test_batchnorm(self, peak_batch):
        model = PeakClassificationModel(
            num_classes=2,
            width=120,
            embedding_dim=0,
            use_embedding=False,
            use_batchnorm=True,
        )
        logits = model(peak_batch)
        assert logits.shape == (4, 2)

    def test_dropout(self, peak_batch):
        model = PeakClassificationModel(
            num_classes=2,
            width=120,
            embedding_dim=0,
            use_embedding=False,
            conv_dropout_p=0.5,
            head_dropout_p=0.3,
        )
        model.eval()
        logits = model(peak_batch)
        assert logits.shape == (4, 2)

    def test_custom_hidden_channels(self, peak_batch):
        model = PeakClassificationModel(
            num_classes=2,
            width=120,
            embedding_dim=0,
            use_embedding=False,
            hidden_channels=[8, 16, 32],
        )
        logits = model(peak_batch)
        assert logits.shape == (4, 2)
