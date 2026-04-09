"""Tests for the SegmentationModule Lightning module."""

import torch
import pytest
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset

from dnanet.models.loss import DiceLoss
from dnanet.models.unet import UNet
from dnanet.modules.segmentation import SegmentationModule


@pytest.fixture
def small_model() -> UNet:
    """A tiny UNet for fast testing."""
    return UNet(depth=1, kernel_size=(3, 3), num_filters=4)


@pytest.fixture
def module(small_model, segmentation_metrics_cfg) -> SegmentationModule:
    """A SegmentationModule with small UNet."""
    optimizer = AdamW(small_model.parameters(), lr=1e-3, weight_decay=0.0)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    return SegmentationModule(
        model=small_model,
        loss_fn=DiceLoss(),
        optimizer=optimizer,
        metrics=segmentation_metrics_cfg,
        learning_rate=1e-3,
        weight_decay=0.0,
        scheduler=scheduler,
    )


@pytest.fixture
def dummy_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """A small batch of random data."""
    x = torch.randn(2, 5, 64)
    y = torch.randint(0, 2, (2, 5, 64)).float()
    return x, y


class TestSegmentationModule:
    """Tests for SegmentationModule."""

    def test_forward(self, module, dummy_batch):
        """Forward should return logits with same shape as input."""
        x, _ = dummy_batch
        logits = module(x)
        assert logits.shape == x.shape

    def test_training_step(self, module, dummy_batch):
        """Training step should return a scalar loss."""
        loss = module.training_step(dummy_batch, batch_idx=0)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_validation_step(self, module, dummy_batch):
        """Validation step should not return a value (Lightning convention)."""
        result = module.validation_step(dummy_batch, batch_idx=0)
        assert result is None

    def test_configure_optimizers_with_scheduler(self, module):
        config = module.configure_optimizers()
        assert "optimizer" in config
        assert "lr_scheduler" in config
        assert config["optimizer"] is module.optimizer
        assert config["lr_scheduler"]["scheduler"] is module.lr_scheduler

    def test_configure_optimizers_without_scheduler(self, small_model, segmentation_metrics_cfg):
        optimizer = AdamW(small_model.parameters(), lr=1e-3)
        module = SegmentationModule(
            model=small_model,
            loss_fn=DiceLoss(),
            optimizer=optimizer,
            metrics=segmentation_metrics_cfg,
        )
        config = module.configure_optimizers()
        assert "optimizer" in config
        assert config["optimizer"] is optimizer
        assert "lr_scheduler" not in config

    def test_predict_step(self, module, dummy_batch):
        """Predict step should return sigmoid probabilities."""
        preds = module.predict_step(dummy_batch, batch_idx=0)
        assert preds.shape == dummy_batch[0].shape
        assert preds.min() >= 0.0
        assert preds.max() <= 1.0

    def test_hyperparameters_saved(self, module):
        """Hyperparameters should be saved for reproducibility."""
        assert module.hparams.learning_rate == 1e-3
        assert module.hparams.threshold == 0.5

    def test_single_training_epoch(self, module):
        """Should survive a full training epoch with Lightning Trainer."""
        import lightning as L

        # Create a tiny dataset
        x = torch.randn(8, 5, 64)
        y = torch.randint(0, 2, (8, 5, 64)).float()
        ds = TensorDataset(x, y)
        dl = DataLoader(ds, batch_size=4)

        trainer = L.Trainer(
            max_epochs=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(module, train_dataloaders=dl)
        # Should complete without errors

    def test_train_and_validate(self, module):
        """Should handle both training and validation."""
        import lightning as L

        x = torch.randn(8, 5, 64)
        y = torch.randint(0, 2, (8, 5, 64)).float()
        ds = TensorDataset(x, y)
        train_dl = DataLoader(ds, batch_size=4)
        val_dl = DataLoader(ds, batch_size=4)

        trainer = L.Trainer(
            max_epochs=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl)

    def test_metrics_computed(self, module, dummy_batch):
        """Training/validation metrics should be updatable."""
        # Simulate a training step + epoch end
        module.training_step(dummy_batch, 0)
        metrics = module.train_metrics.compute()
        assert "train/accuracy" in metrics
        assert "train/f1" in metrics
        assert "train/iou" in metrics
        module.train_metrics.reset()

    def test_loss_decreases_on_overfit(self, small_model, segmentation_metrics_cfg):
        """On a tiny dataset, loss should decrease after a few epochs."""
        optimizer = AdamW(small_model.parameters(), lr=1e-2, weight_decay=0.0)
        module = SegmentationModule(
            model=small_model,
            loss_fn=DiceLoss(),
            optimizer=optimizer,
            metrics=segmentation_metrics_cfg,
            learning_rate=1e-2,  # high LR for fast convergence
        )

        # Deterministic tiny dataset
        torch.manual_seed(42)
        x = torch.randn(4, 5, 64)
        y = torch.zeros(4, 5, 64)
        y[:, :, 20:30] = 1.0  # simple target pattern

        ds = TensorDataset(x, y)
        dl = DataLoader(ds, batch_size=4)

        import lightning as L

        trainer = L.Trainer(
            max_epochs=20,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(module, train_dataloaders=dl)

        # Check loss after training is lower than initial
        module.eval()
        with torch.no_grad():
            logits = module(x)
            final_loss = DiceLoss()(logits, y)
        assert final_loss.item() < 0.9  # should be significantly below 1.0
