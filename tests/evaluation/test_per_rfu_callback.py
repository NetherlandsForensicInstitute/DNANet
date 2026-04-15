"""Tests for the RFU outcome Lightning callback."""

import csv

import numpy as np
import torch
import pytest

from dnanet.evaluation.callbacks import PerRFUOutcomeCallback


class FakeTrainer:
    def __init__(self, default_root_dir) -> None:
        self.default_root_dir = str(default_root_dir)
        self.is_global_zero = True


class FakeModule:
    pass


def test_per_rfu_callback_writes_outcome_npz(tmp_path):
    callback = PerRFUOutcomeCallback(threshold=0.5, filename="per_rfu_outcomes.npz")
    trainer = FakeTrainer(tmp_path)
    module = FakeModule()

    batch = (
        torch.zeros((1, 1, 4), dtype=torch.float32),
        torch.tensor([[[1, 1, 0, 0]]], dtype=torch.float32),
        [{
            "path": "sample.hid",
            "signal_image": np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32),
        }],
    )
    outputs = {"preds": torch.tensor([[[0.6, 0.4, 0.7, 0.2]]], dtype=torch.float32)}

    callback.on_test_epoch_start(trainer, module)
    callback.on_test_batch_end(trainer, module, outputs=outputs, batch=batch, batch_idx=0)
    callback.on_test_epoch_end(trainer, module)

    with np.load(tmp_path / "per_rfu_outcomes.npz") as data:
        assert data["tp_rfus"].tolist() == pytest.approx([10.0])
        assert data["fp_rfus"].tolist() == pytest.approx([30.0])
        assert data["fn_rfus"].tolist() == pytest.approx([20.0])


def test_per_rfu_callback_writes_outcome_csv_when_configured(tmp_path):
    callback = PerRFUOutcomeCallback(threshold=0.5, filename="per_rfu_outcomes.csv")
    trainer = FakeTrainer(tmp_path)
    module = FakeModule()

    batch = (
        torch.zeros((1, 1, 4), dtype=torch.float32),
        torch.tensor([[[1, 1, 0, 0]]], dtype=torch.float32),
        [{
            "path": "sample.hid",
            "signal_image": np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32),
        }],
    )
    outputs = {"preds": torch.tensor([[[0.6, 0.4, 0.7, 0.2]]], dtype=torch.float32)}

    callback.on_test_epoch_start(trainer, module)
    callback.on_test_batch_end(trainer, module, outputs=outputs, batch=batch, batch_idx=0)
    callback.on_test_epoch_end(trainer, module)

    with (tmp_path / "per_rfu_outcomes.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {"outcome": "tp", "rfu": "10"},
        {"outcome": "fp", "rfu": "30"},
        {"outcome": "fn", "rfu": "20"},
    ]


def test_per_rfu_callback_accepts_trailing_singleton_channel(tmp_path):
    callback = PerRFUOutcomeCallback(threshold=0.5)
    trainer = FakeTrainer(tmp_path)
    module = FakeModule()

    batch = (
        torch.zeros((1, 1, 3, 1), dtype=torch.float32),
        torch.tensor([[[[1], [0], [1]]]], dtype=torch.float32),
        [{
            "signal_image": np.array([[[100.0], [200.0], [300.0]]], dtype=np.float32),
        }],
    )
    outputs = {"preds": torch.tensor([[[[0.9], [0.8], [0.1]]]], dtype=torch.float32)}

    callback.on_test_epoch_start(trainer, module)
    callback.on_test_batch_end(trainer, module, outputs=outputs, batch=batch, batch_idx=0)
    outcomes = callback.metric.compute_outcomes()

    assert outcomes["tp_rfus"].tolist() == pytest.approx([100.0])
    assert outcomes["fp_rfus"].tolist() == pytest.approx([200.0])
    assert outcomes["fn_rfus"].tolist() == pytest.approx([300.0])


def test_per_rfu_callback_requires_signal_image_metadata():
    callback = PerRFUOutcomeCallback()

    with pytest.raises(ValueError, match="signal_image"):
        callback.on_test_batch_end(
            FakeTrainer("."),
            FakeModule(),
            outputs={"preds": torch.zeros((1, 1, 4), dtype=torch.float32)},
            batch=(
                torch.zeros((1, 1, 4), dtype=torch.float32),
                torch.zeros((1, 1, 4), dtype=torch.float32),
                [{"path": "sample.hid"}],
            ),
            batch_idx=0,
        )
