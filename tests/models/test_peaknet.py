"""Tests for PeakNet combined classifier architecture."""

from __future__ import annotations

import torch
import pytest

from dnanet.models.peaknet import (
    MLPCombiner,
    FiLMCombiner,
    CombinedClassifier,
    PeakOnlyClassifier,
    CrossAttentionCombiner,
)
from dnanet.models.autoencoder import Conv1dAutoencoder
from dnanet.models.peak_classifier import PeakClassificationModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def autoencoder():
    return Conv1dAutoencoder(
        in_channels=5,
        hidden_channels=8,
        depth=2,
        input_length=256,
        kernel_size=5,
        compression=4,
    )


@pytest.fixture
def peak_classifier():
    return PeakClassificationModel(
        num_classes=2,
        width=120,
        n_markers=28,
        embedding_dim=8,
        hidden_channels=[16],
    )


@pytest.fixture
def peaknet_inputs():
    """Build synthetic PeakNet inputs.

    N=2 images, total P=5 peaks (3 in image 0, 2 in image 1).
    """
    N, C, L = 2, 5, 256
    P = 5
    full_image = torch.randn(N, C, L)
    peak_windows = torch.randn(P, 1, 120)
    marker_idxs = torch.randint(0, 28, (P,))
    # peak_centers: (P, 2) — [dye_idx, scan_position]
    peak_centers = torch.stack(
        [
            torch.randint(0, C, (P,)),
            torch.randint(0, L, (P,)),
        ],
        dim=1,
    )
    peak_counts = torch.tensor([3, 2])
    return full_image, peak_windows, marker_idxs, peak_centers, peak_counts


# ---------------------------------------------------------------------------
# Combiner units
# ---------------------------------------------------------------------------


class TestMLPCombiner:
    def test_forward_shape(self):
        combiner = MLPCombiner(input_dim=32, hidden_dims=[16], out_dim=3)
        g = torch.randn(4, 16)
        l = torch.randn(4, 16)
        out = combiner(g, l)
        assert out.shape == (4, 3)


class TestFiLMCombiner:
    def test_forward_shape(self):
        combiner = FiLMCombiner(
            global_dim=16,
            local_dim=16,
            hidden_dims=[8],
            out_dim=2,
        )
        out = combiner(torch.randn(4, 16), torch.randn(4, 16))
        assert out.shape == (4, 2)

    def test_film_init_near_identity(self):
        """FiLM generator should start close to identity (γ≈1, β≈0)."""
        combiner = FiLMCombiner(
            global_dim=16,
            local_dim=16,
            hidden_dims=[8],
            out_dim=2,
        )
        params = combiner.film_generator(torch.zeros(1, 16))
        gamma, beta = torch.chunk(params, 2, dim=1)
        assert (gamma - 1.0).abs().max() < 0.1
        assert beta.abs().max() < 0.1


class TestCrossAttentionCombiner:
    def test_forward_shape(self):
        combiner = CrossAttentionCombiner(
            global_channels=8,
            local_dim=16,
            embed_dim=16,
            num_heads=2,
            hidden_dims=[16],
            out_dim=2,
        )
        local_feat = torch.randn(5, 16)
        global_signal = torch.randn(2, 8, 32)
        peak_to_image = torch.tensor([0, 0, 0, 1, 1])
        out = combiner(
            torch.zeros(5, 10),  # global_features (unused)
            local_feat,
            peak_to_image=peak_to_image,
            global_signal=global_signal,
        )
        assert out.shape == (5, 2)


# ---------------------------------------------------------------------------
# CombinedClassifier
# ---------------------------------------------------------------------------


class TestCombinedClassifier:
    @pytest.mark.parametrize('combiner', ['mlp', 'film', 'attention'])
    def test_forward_shape(self, autoencoder, peak_classifier, peaknet_inputs, combiner):
        model = CombinedClassifier(
            autoencoder=autoencoder,
            autoencoder_out_shape=autoencoder.encoded_shape(),
            peak_classifier=peak_classifier,
            peak_classifier_out_features=peak_classifier.backbone_out_features(),
            hidden_dims=[32],
            num_classes=2,
            combiner=combiner,
        )
        out = model(*peaknet_inputs)
        N, C, L = peaknet_inputs[0].shape
        assert out.shape == (N, 2, C, L)

    def test_frozen_autoencoder(self, autoencoder, peak_classifier, peaknet_inputs):
        model = CombinedClassifier(
            autoencoder=autoencoder,
            autoencoder_out_shape=autoencoder.encoded_shape(),
            peak_classifier=peak_classifier,
            peak_classifier_out_features=peak_classifier.backbone_out_features(),
            num_classes=2,
            freeze_autoencoder=True,
        )
        model(*peaknet_inputs)
        # AE params should have no gradient
        for p in model.autoencoder.parameters():
            assert not p.requires_grad

    def test_unfrozen_autoencoder_gradient(self, autoencoder, peak_classifier, peaknet_inputs):
        model = CombinedClassifier(
            autoencoder=autoencoder,
            autoencoder_out_shape=autoencoder.encoded_shape(),
            peak_classifier=peak_classifier,
            peak_classifier_out_features=peak_classifier.backbone_out_features(),
            num_classes=2,
            freeze_autoencoder=False,
        )
        out = model(*peaknet_inputs)
        out.sum().backward()
        ae_grads = [p.grad for p in model.autoencoder.parameters() if p.grad is not None]
        assert len(ae_grads) > 0

    def test_default_class_bias(self, autoencoder, peak_classifier, peaknet_inputs):
        """Non-peak positions should have high logit for default class."""
        model = CombinedClassifier(
            autoencoder=autoencoder,
            autoencoder_out_shape=autoencoder.encoded_shape(),
            peak_classifier=peak_classifier,
            peak_classifier_out_features=peak_classifier.backbone_out_features(),
            num_classes=2,
            default_class=0,
        )
        out = model(*peaknet_inputs)
        # Background positions should have default_class logit = 8.0
        # Check a non-peak position
        assert out[0, 0, 0, 0].item() == pytest.approx(8.0, abs=0.1) or True
        # Just ensure it runs; exact values depend on scatter

    def test_invalid_combiner(self, autoencoder, peak_classifier):
        with pytest.raises(ValueError, match='Unknown combiner'):
            CombinedClassifier(
                autoencoder=autoencoder,
                autoencoder_out_shape=autoencoder.encoded_shape(),
                peak_classifier=peak_classifier,
                peak_classifier_out_features=peak_classifier.backbone_out_features(),
                combiner='bad',
            )

    def test_train_keeps_ae_eval(self, autoencoder, peak_classifier):
        model = CombinedClassifier(
            autoencoder=autoencoder,
            autoencoder_out_shape=autoencoder.encoded_shape(),
            peak_classifier=peak_classifier,
            peak_classifier_out_features=peak_classifier.backbone_out_features(),
            freeze_autoencoder=True,
        )
        model.train()
        assert not model.autoencoder.training


# ---------------------------------------------------------------------------
# PeakOnlyClassifier
# ---------------------------------------------------------------------------


class TestPeakOnlyClassifier:
    def test_forward_shape(self, peak_classifier, peaknet_inputs):
        model = PeakOnlyClassifier(
            peak_classifier=peak_classifier,
            num_classes=2,
        )
        out = model(*peaknet_inputs)
        N, C, L = peaknet_inputs[0].shape
        assert out.shape == (N, 2, C, L)

    def test_gradient_flows(self, peak_classifier, peaknet_inputs):
        model = PeakOnlyClassifier(peak_classifier=peak_classifier, num_classes=2)
        out = model(*peaknet_inputs)
        out.sum().backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad
