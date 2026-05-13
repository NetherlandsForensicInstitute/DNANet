"""Tests for the ReconstructionModule Lightning wrapper."""

from __future__ import annotations

import torch
import pytest
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from dnanet.models.autoencoder import Conv1dAutoencoder
from dnanet.modules.reconstruction import ReconstructionModule


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def autoencoder():
    return Conv1dAutoencoder(
        in_channels=1,
        hidden_channels=8,
        depth=2,
        input_length=512,
        kernel_size=5,
        compression=4,
    )


@pytest.fixture
def module(autoencoder, reconstruction_metrics_cfg):
    optimizer = AdamW(autoencoder.parameters(), lr=1e-3)
    return ReconstructionModule(
        model=autoencoder,
        optimizer=optimizer,
        metrics=reconstruction_metrics_cfg,
        learning_rate=1e-3,
    )


@pytest.fixture
def batch_self_supervised():
    """Single-element batch: target = input."""
    return (torch.randn(4, 1, 512),)


@pytest.fixture
def batch_with_target():
    """(input, target) batch."""
    x = torch.randn(4, 1, 512)
    return x, x.clone()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReconstructionModule:
    def test_training_step_self_supervised(self, module, batch_self_supervised):
        loss = module.training_step(batch_self_supervised, batch_idx=0)
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_training_step_with_target(self, module, batch_with_target):
        loss = module.training_step(batch_with_target, batch_idx=0)
        assert loss.dim() == 0

    def test_training_step_plain_tensor(self, module):
        """Should handle a plain tensor (not tuple) as input."""
        x = torch.randn(4, 1, 512)
        loss = module.training_step(x, batch_idx=0)
        assert loss.dim() == 0

    def test_validation_step(self, module, batch_self_supervised):
        module.validation_step(batch_self_supervised, batch_idx=0)

    def test_forward(self, module):
        x = torch.randn(2, 1, 512)
        out = module(x)
        assert out.shape == x.shape

    def test_default_loss_is_mse(self, autoencoder, reconstruction_metrics_cfg):
        mod = ReconstructionModule(
            model=autoencoder,
            optimizer=AdamW(autoencoder.parameters(), lr=1e-3),
            metrics=reconstruction_metrics_cfg,
        )
        assert isinstance(mod.loss_fn, nn.MSELoss)

    def test_custom_loss(self, autoencoder, reconstruction_metrics_cfg):
        mod = ReconstructionModule(
            model=autoencoder,
            loss_fn=nn.L1Loss(),
            optimizer=AdamW(autoencoder.parameters(), lr=1e-3),
            metrics=reconstruction_metrics_cfg,
        )
        x = torch.randn(2, 1, 512)
        loss = mod.training_step((x,), batch_idx=0)
        assert loss.dim() == 0

    def test_configure_optimizers_no_scheduler(self, module):
        config = module.configure_optimizers()
        assert 'optimizer' in config
        assert config['optimizer'] is module.optimizer
        assert 'lr_scheduler' not in config

    def test_configure_optimizers_with_scheduler(self, autoencoder, reconstruction_metrics_cfg):
        optimizer = AdamW(autoencoder.parameters(), lr=1e-3)
        scheduler = ExponentialLR(optimizer, gamma=0.95)
        mod = ReconstructionModule(
            model=autoencoder,
            optimizer=optimizer,
            metrics=reconstruction_metrics_cfg,
            scheduler=scheduler,
        )
        config = mod.configure_optimizers()
        assert 'lr_scheduler' in config
        assert config['lr_scheduler']['scheduler'] is scheduler

    def test_predict_step(self, module):
        x = torch.randn(2, 1, 512)
        out = module.predict_step((x,), batch_idx=0)
        assert out.shape == x.shape

    def test_metrics_update_and_reset(self, module, batch_self_supervised):
        module.training_step(batch_self_supervised, batch_idx=0)
        computed = module.train_metrics.compute()
        assert 'train/mse' in computed
        module.train_metrics.reset()

    @pytest.mark.skip(reason='FourierAutoencoder has no trainable parameters')
    def test_handles_trailing_singleton(self, reconstruction_metrics_cfg):
        """Should squeeze trailing dim from 4D output/target via FourierAutoencoder."""
        from dnanet.models.autoencoder import FourierAutoencoder

        ae = FourierAutoencoder(in_channels=1, signal_length=512, latent_coeffs=64)
        mod = ReconstructionModule(
            model=ae,
            optimizer=AdamW(ae.parameters(), lr=1e-3),
            metrics=reconstruction_metrics_cfg,
        )
        x = torch.randn(2, 1, 512)
        loss = mod.training_step((x,), batch_idx=0)
        assert loss.dim() == 0

    def test_hparams_saved(self, module):
        assert module.hparams.learning_rate == 1e-3
        assert not hasattr(module.hparams, 'scheduler_gamma')
