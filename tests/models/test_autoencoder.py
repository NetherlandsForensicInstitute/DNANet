"""Tests for autoencoder architectures."""

from __future__ import annotations

import pytest
import torch

from dnanet.models.autoencoder import (
    Conv1dAutoencoder,
    FourierAutoencoder,
    PerDyeConv1dAutoencoder,
    SharedWeightPerDyeConv1dAutoencoder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def batch() -> torch.Tensor:
    """Synthetic batch: (B=2, C=5, L=4096)."""
    return torch.randn(2, 5, 4096)


@pytest.fixture
def single_channel_batch() -> torch.Tensor:
    """Single-channel batch: (B=2, C=1, L=4096)."""
    return torch.randn(2, 1, 4096)


# ---------------------------------------------------------------------------
# Conv1dAutoencoder
# ---------------------------------------------------------------------------

class TestConv1dAutoencoder:
    def test_forward_shape(self, single_channel_batch):
        model = Conv1dAutoencoder(in_channels=1, hidden_channels=16,
                                   depth=3, input_length=4096,
                                   kernel_size=7, compression=8)
        out = model(single_channel_batch)
        assert out.shape == single_channel_batch.shape

    def test_encode_decode_roundtrip(self, single_channel_batch):
        model = Conv1dAutoencoder(in_channels=1, hidden_channels=16,
                                   depth=3, input_length=4096,
                                   kernel_size=7, compression=8)
        z = model.encode(single_channel_batch)
        reconstructed = model.decode(z)
        assert reconstructed.shape == single_channel_batch.shape

    def test_encoded_shape(self):
        model = Conv1dAutoencoder(in_channels=1, hidden_channels=16,
                                   depth=3, input_length=4096,
                                   kernel_size=7, compression=8)
        shape = model.encoded_shape()
        assert len(shape) == 2
        # Latent size should be input_length / compression = 512
        assert shape[0] * shape[1] == 4096 // 8

    def test_multichannel(self, batch):
        model = Conv1dAutoencoder(in_channels=5, hidden_channels=16,
                                   depth=3, input_length=4096,
                                   kernel_size=7, compression=8)
        out = model(batch)
        assert out.shape == batch.shape

    def test_invalid_depth(self):
        with pytest.raises(ValueError, match="depth must be at least 1"):
            Conv1dAutoencoder(depth=0)

    def test_even_kernel_rejected(self):
        with pytest.raises(ValueError, match="kernel_size must be odd"):
            Conv1dAutoencoder(kernel_size=4)

    def test_no_sigmoid(self, single_channel_batch):
        model = Conv1dAutoencoder(in_channels=1, hidden_channels=16,
                                   depth=3, input_length=4096,
                                   kernel_size=7, compression=8,
                                   use_sigmoid=False)
        out = model(single_channel_batch)
        # Without sigmoid, output can be negative
        assert out.shape == single_channel_batch.shape

    def test_with_batchnorm(self, single_channel_batch):
        model = Conv1dAutoencoder(in_channels=1, hidden_channels=16,
                                   depth=3, input_length=4096,
                                   kernel_size=7, compression=8,
                                   use_batchnorm=True)
        out = model(single_channel_batch)
        assert out.shape == single_channel_batch.shape

    def test_gradient_flows(self, single_channel_batch):
        model = Conv1dAutoencoder(in_channels=1, hidden_channels=16,
                                   depth=3, input_length=4096,
                                   kernel_size=7, compression=8)
        out = model(single_channel_batch)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


# ---------------------------------------------------------------------------
# PerDyeConv1dAutoencoder
# ---------------------------------------------------------------------------

class TestPerDyeAutoencoder:
    def test_forward_shape(self, batch):
        model = PerDyeConv1dAutoencoder(
            in_channels=5, hidden_channels=16, depth=3,
            input_length=4096, kernel_size=7, compression=8,
        )
        out = model(batch)
        assert out.shape == batch.shape

    def test_encoded_shape(self):
        model = PerDyeConv1dAutoencoder(
            in_channels=5, hidden_channels=16, depth=3,
            input_length=4096, kernel_size=7, compression=8,
        )
        shape = model.encoded_shape()
        assert len(shape) == 2
        # 5 channels × sub-AE encoded channels
        sub_shape = model.per_channel_nets[0].encoded_shape()
        assert shape[0] == 5 * sub_shape[0]
        assert shape[1] == sub_shape[1]

    def test_independent_weights(self):
        """Each dye channel should have its own weights."""
        model = PerDyeConv1dAutoencoder(in_channels=3, hidden_channels=8,
                                         depth=3, input_length=4096,
                                         kernel_size=7, compression=8)
        # Weights for channel 0 and channel 1 should differ
        w0 = list(model.per_channel_nets[0].parameters())[0]
        w1 = list(model.per_channel_nets[1].parameters())[0]
        assert not torch.equal(w0, w1)

    def test_handles_4d_input(self, batch):
        """Should squeeze trailing singleton dim."""
        model = PerDyeConv1dAutoencoder(
            in_channels=5, hidden_channels=16, depth=3,
            input_length=4096, kernel_size=7, compression=8,
        )
        x_4d = batch.unsqueeze(-1)  # (B, C, L, 1)
        z = model.encode(x_4d)
        assert z.dim() == 3


# ---------------------------------------------------------------------------
# SharedWeightPerDyeConv1dAutoencoder
# ---------------------------------------------------------------------------

class TestSharedWeightAutoencoder:
    def test_forward_shape(self, batch):
        model = SharedWeightPerDyeConv1dAutoencoder(
            in_channels=5, hidden_channels=16, depth=3,
            input_length=4096, kernel_size=7, compression=8,
        )
        out = model(batch)
        assert out.shape == batch.shape

    def test_shared_weights(self):
        """All channels should share the same parameters."""
        model = SharedWeightPerDyeConv1dAutoencoder(
            in_channels=3, hidden_channels=8, depth=3,
            input_length=4096, kernel_size=7, compression=8,
        )
        # All per_channel_nets should be the same object
        assert model.per_channel_nets[0] is model.per_channel_nets[1]
        assert model.per_channel_nets[1] is model.per_channel_nets[2]


# ---------------------------------------------------------------------------
# FourierAutoencoder
# ---------------------------------------------------------------------------

class TestFourierAutoencoder:
    def test_forward_shape(self, batch):
        model = FourierAutoencoder(
            in_channels=5, signal_length=4096, latent_coeffs=256,
        )
        out = model(batch)
        assert out.shape == batch.shape

    def test_encoded_shape(self):
        model = FourierAutoencoder(
            in_channels=5, signal_length=4096, latent_coeffs=256,
        )
        assert model.encoded_shape() == (5, 256, 2)

    def test_no_trainable_parameters(self):
        model = FourierAutoencoder(in_channels=5, signal_length=4096)
        assert sum(p.numel() for p in model.parameters()) == 0

    def test_reconstruction_quality(self):
        """Low-frequency signal should be reconstructed well."""
        model = FourierAutoencoder(
            in_channels=1, signal_length=1024, latent_coeffs=64,
        )
        # Create a low-frequency sinusoid
        t = torch.linspace(0, 2 * torch.pi, 1024).unsqueeze(0).unsqueeze(0)
        signal = torch.sin(t) + 0.5 * torch.sin(3 * t)
        reconstructed = model(signal)
        error = (signal - reconstructed).abs().max()
        assert error < 0.01, f"Max reconstruction error {error} too high"

    def test_invalid_latent_coeffs(self):
        with pytest.raises(ValueError, match="exceeds max rFFT length"):
            FourierAutoencoder(signal_length=100, latent_coeffs=200)

    def test_handles_4d_input(self, batch):
        model = FourierAutoencoder(in_channels=5, signal_length=4096)
        x_4d = batch.unsqueeze(-1)
        z = model.encode(x_4d)
        assert z.dim() == 4  # (B, C, K, 2)
