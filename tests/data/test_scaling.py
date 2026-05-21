"""Tests for RFU scaling and normalization utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dnanet.data.preprocessing.scaling import (
    RFU_MAX_VALUE,
    inverse_scale_rfu_torch,
    scale_rfu_numpy,
    scale_rfu_torch,
)


# ---------------------------------------------------------------------------
# scale_rfu_torch
# ---------------------------------------------------------------------------


class TestScaleRfuTorch:
    def test_log_clamp_output_range(self):
        """log + max_rfu should produce values in [0, 1]."""
        x = torch.tensor([0.0, 100.0, 1000.0, 32768.0, 50000.0])
        scaled = scale_rfu_torch(x, log_scale=True, max_rfu=RFU_MAX_VALUE)
        assert scaled.min() >= 0.0
        assert scaled.max() <= 1.0

    def test_log_only(self):
        x = torch.tensor([0.0, 100.0])
        scaled = scale_rfu_torch(x, log_scale=True, max_rfu=None)
        expected = torch.log1p(x.clamp(0.0))
        assert torch.allclose(scaled, expected)

    def test_linear_normalization(self):
        x = torch.tensor([0.0, 16384.0, 32768.0])
        scaled = scale_rfu_torch(x, log_scale=False, max_rfu=RFU_MAX_VALUE)
        assert scaled[0].item() == pytest.approx(0.0)
        assert scaled[1].item() == pytest.approx(0.5)
        assert scaled[2].item() == pytest.approx(1.0)

    def test_identity_mode(self):
        x = torch.tensor([42.0, -5.0])
        scaled = scale_rfu_torch(x, log_scale=False, max_rfu=None)
        assert torch.allclose(scaled, x.float())

    def test_negative_values_clamped(self):
        x = torch.tensor([-100.0, -1.0, 0.0])
        scaled = scale_rfu_torch(x, log_scale=True, max_rfu=RFU_MAX_VALUE)
        assert scaled.min() >= 0.0

    def test_output_dtype_float32(self):
        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        scaled = scale_rfu_torch(x)
        assert scaled.dtype == torch.float32

    def test_preserves_shape(self):
        x = torch.randn(2, 5, 4096).abs()
        scaled = scale_rfu_torch(x)
        assert scaled.shape == x.shape


# ---------------------------------------------------------------------------
# inverse_scale_rfu_torch
# ---------------------------------------------------------------------------


class TestInverseScaleRfuTorch:
    def test_roundtrip_log_clamp(self):
        """scale → inverse should recover original (within tolerance)."""
        x = torch.tensor([0.0, 500.0, 10000.0, 32768.0])
        scaled = scale_rfu_torch(x, log_scale=True, max_rfu=RFU_MAX_VALUE)
        recovered = inverse_scale_rfu_torch(scaled, log_scale=True, max_rfu=RFU_MAX_VALUE)
        assert torch.allclose(recovered, x, atol=1.0)

    def test_roundtrip_log_only(self):
        x = torch.tensor([0.0, 50.0, 5000.0])
        scaled = scale_rfu_torch(x, log_scale=True, max_rfu=None)
        recovered = inverse_scale_rfu_torch(scaled, log_scale=True, max_rfu=None)
        assert torch.allclose(recovered, x, atol=0.01)

    def test_roundtrip_linear(self):
        x = torch.tensor([0.0, 1000.0, 32768.0])
        scaled = scale_rfu_torch(x, log_scale=False, max_rfu=RFU_MAX_VALUE)
        recovered = inverse_scale_rfu_torch(scaled, log_scale=False, max_rfu=RFU_MAX_VALUE)
        assert torch.allclose(recovered, x, atol=1.0)

    def test_output_non_negative(self):
        x = torch.tensor([-0.5, 0.0, 0.5, 1.0])
        recovered = inverse_scale_rfu_torch(x, log_scale=True, max_rfu=RFU_MAX_VALUE)
        assert recovered.min() >= 0.0

    def test_trailing_singleton_preserved(self):
        x = torch.randn(2, 5, 100, 1).abs()
        scaled = scale_rfu_torch(x)
        recovered = inverse_scale_rfu_torch(scaled)
        assert recovered.shape == x.shape


# ---------------------------------------------------------------------------
# scale_rfu_numpy
# ---------------------------------------------------------------------------


class TestScaleRfuNumpy:
    def test_log_clamp_output_range(self):
        x = np.array([0.0, 100.0, 32768.0, 50000.0])
        scaled = scale_rfu_numpy(x, log_scale=True, max_rfu=RFU_MAX_VALUE)
        assert scaled.min() >= 0.0
        assert scaled.max() <= 1.0

    def test_matches_torch(self):
        """Numpy and torch scaling should produce same results."""
        x_np = np.array([0.0, 100.0, 1000.0, 10000.0, 32768.0])
        x_torch = torch.tensor(x_np)

        scaled_np = scale_rfu_numpy(x_np, log_scale=True, max_rfu=RFU_MAX_VALUE)
        scaled_torch = scale_rfu_torch(x_torch, log_scale=True, max_rfu=RFU_MAX_VALUE)

        np.testing.assert_allclose(
            scaled_np,
            scaled_torch.numpy(),
            atol=1e-6,
        )

    def test_log_only(self):
        x = np.array([0.0, 100.0])
        scaled = scale_rfu_numpy(x, log_scale=True, max_rfu=None)
        expected = np.log1p(np.clip(x, 0.0, None))
        np.testing.assert_allclose(scaled, expected)

    def test_linear(self):
        x = np.array([0.0, 16384.0, 32768.0])
        scaled = scale_rfu_numpy(x, log_scale=False, max_rfu=RFU_MAX_VALUE)
        np.testing.assert_allclose(scaled, [0.0, 0.5, 1.0])

    def test_identity(self):
        x = np.array([42.0, -5.0])
        scaled = scale_rfu_numpy(x, log_scale=False, max_rfu=None)
        np.testing.assert_allclose(scaled, x.astype(np.float64))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_rfu_max_value():
    assert RFU_MAX_VALUE == 32768
    assert RFU_MAX_VALUE == 2**15
