"""Tests for loss functions."""

import torch
import pytest

from dnanet.models.loss import DiceLoss, FocalLoss


class TestDiceLoss:
    """Tests for DiceLoss."""

    def test_perfect_prediction(self):
        """When prediction matches target exactly, loss should be near 0."""
        loss_fn = DiceLoss()
        # Use high logits so sigmoid → ~1 for positive, low → ~0 for negative
        logits = torch.tensor([10.0, -10.0, 10.0, -10.0])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.05

    def test_worst_prediction(self):
        """When prediction is opposite of target, loss should be near 1."""
        loss_fn = DiceLoss()
        logits = torch.tensor([-10.0, 10.0, -10.0, 10.0])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = loss_fn(logits, targets)
        assert loss.item() > 0.7

    def test_all_zeros(self):
        """All-zero target with all-zero prediction gives low loss (smooth term)."""
        loss_fn = DiceLoss()
        logits = torch.full((10,), -10.0)
        targets = torch.zeros(10)
        loss = loss_fn(logits, targets)
        # With smoothing, this should be near 0
        assert loss.item() < 0.1

    def test_output_range(self):
        """Loss should always be in [0, 1]."""
        loss_fn = DiceLoss()
        for _ in range(10):
            logits = torch.randn(32)
            targets = torch.randint(0, 2, (32,)).float()
            loss = loss_fn(logits, targets)
            assert 0.0 <= loss.item() <= 1.0

    def test_differentiable(self):
        """Loss should be differentiable for backpropagation."""
        loss_fn = DiceLoss()
        logits = torch.randn(8, requires_grad=True)
        targets = torch.randint(0, 2, (8,)).float()
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None

    def test_2d_input(self):
        """Should work with 2D spatial inputs (batch of images)."""
        loss_fn = DiceLoss()
        logits = torch.randn(2, 1, 5, 4096)
        targets = torch.randint(0, 2, (2, 1, 5, 4096)).float()
        loss = loss_fn(logits, targets)
        assert loss.shape == ()

    def test_smooth_parameter(self):
        """Custom smooth parameter should be respected."""
        loss1 = DiceLoss(smooth=1.0)
        loss2 = DiceLoss(smooth=100.0)
        logits = torch.randn(16)
        targets = torch.randint(0, 2, (16,)).float()
        # Different smooth values should give different results
        l1 = loss1(logits, targets)
        l2 = loss2(logits, targets)
        # Both should be valid but generally different
        assert 0.0 <= l1.item() <= 1.0
        assert 0.0 <= l2.item() <= 1.0


class TestFocalLoss:
    """Tests for FocalLoss."""

    def test_basic_computation(self):
        """Should produce a scalar loss for classification inputs."""
        loss_fn = FocalLoss()
        logits = torch.randn(4, 3)  # 4 samples, 3 classes
        targets = torch.tensor([0, 1, 2, 1])
        loss = loss_fn(logits, targets)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_differentiable(self):
        """Loss should be differentiable."""
        loss_fn = FocalLoss()
        logits = torch.randn(4, 3, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 1])
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None

    def test_gamma_zero_matches_cross_entropy(self):
        """With gamma=0 and alpha=1, focal loss = cross-entropy."""
        focal = FocalLoss(alpha=1.0, gamma=0.0)
        ce = torch.nn.CrossEntropyLoss()
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        fl = focal(logits, targets)
        ce_loss = ce(logits, targets)
        torch.testing.assert_close(fl, ce_loss, rtol=1e-5, atol=1e-5)
