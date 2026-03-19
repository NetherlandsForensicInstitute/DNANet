"""Tests for the ClassificationModule Lightning wrapper."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dnanet.models.peak_classifier import PeakClassificationModel
from dnanet.modules.classification import ClassificationModule


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    return PeakClassificationModel(
        num_classes=3, width=120, embedding_dim=0, hidden_channels=[16],
    )


@pytest.fixture
def module(model):
    return ClassificationModule(
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        num_classes=3,
        learning_rate=1e-3,
    )


@pytest.fixture
def batch_no_marker():
    """(peak_data, targets) — no marker embedding."""
    peak_data = torch.randn(8, 1, 120)
    targets = torch.randint(0, 3, (8,))
    return peak_data, targets


@pytest.fixture
def batch_with_marker():
    """(peak_data, marker_idx, targets) — with marker embedding."""
    peak_data = torch.randn(8, 1, 120)
    marker_idx = torch.randint(0, 28, (8,))
    targets = torch.randint(0, 3, (8,))
    return peak_data, marker_idx, targets


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestClassificationModule:
    def test_training_step_no_marker(self, module, batch_no_marker):
        loss = module.training_step(batch_no_marker, batch_idx=0)
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_training_step_with_marker(self, batch_with_marker):
        model = PeakClassificationModel(
            num_classes=3, width=120, embedding_dim=8, hidden_channels=[16],
        )
        mod = ClassificationModule(
            model=model, loss_fn=nn.CrossEntropyLoss(), num_classes=3,
        )
        loss = mod.training_step(batch_with_marker, batch_idx=0)
        assert loss.dim() == 0

    def test_validation_step(self, module, batch_no_marker):
        # Should not error
        module.validation_step(batch_no_marker, batch_idx=0)

    def test_forward(self, module, batch_no_marker):
        logits = module(batch_no_marker[0])
        assert logits.shape == (8, 3)

    def test_configure_optimizers_no_scheduler(self, module):
        """gamma=1.0 should not include a scheduler."""
        config = module.configure_optimizers()
        assert "optimizer" in config
        assert "lr_scheduler" not in config

    def test_configure_optimizers_with_scheduler(self, model):
        mod = ClassificationModule(
            model=model, loss_fn=nn.CrossEntropyLoss(),
            num_classes=3, scheduler_gamma=0.95,
        )
        config = mod.configure_optimizers()
        assert "lr_scheduler" in config

    def test_predict_step_no_marker(self, module, batch_no_marker):
        probs = module.predict_step(batch_no_marker, batch_idx=0)
        assert probs.shape == (8, 3)
        # Should be valid probabilities
        assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-5)

    def test_predict_step_with_marker(self, batch_with_marker):
        model = PeakClassificationModel(
            num_classes=3, width=120, embedding_dim=8, hidden_channels=[16],
        )
        mod = ClassificationModule(
            model=model, loss_fn=nn.CrossEntropyLoss(), num_classes=3,
        )
        probs = mod.predict_step(batch_with_marker, batch_idx=0)
        assert probs.shape == (8, 3)

    def test_metrics_update_and_reset(self, module, batch_no_marker):
        module.training_step(batch_no_marker, batch_idx=0)
        computed = module.train_metrics.compute()
        assert "train/accuracy" in computed
        module.train_metrics.reset()

    def test_hparams_saved(self, module):
        assert module.hparams.learning_rate == 1e-3
        assert module.hparams.num_classes == 3
