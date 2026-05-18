"""Tests for profile plotting evaluation callback."""

from __future__ import annotations

import numpy as np
import torch
import pytest
from matplotlib import pyplot as plt

import dnanet.evaluation.callbacks.profile_plot as profile_plot_module
from dnanet.evaluation.callbacks import ProfilePlotCallback


class FakeTrainer:
    def __init__(self, default_root_dir, *, is_global_zero: bool = True) -> None:
        self.default_root_dir = str(default_root_dir)
        self.is_global_zero = is_global_zero


class FakeModule:
    pass


@pytest.fixture(autouse=True)
def close_figures():
    plt.close("all")
    yield
    plt.close("all")


def _batch(size: int = 3) -> tuple[tuple[torch.Tensor, torch.Tensor, list[dict]], dict]:
    targets = torch.zeros((size, 3, 1, 4), dtype=torch.float32)
    targets[:, 1, 0, 1:3] = 1
    preds = torch.zeros((size, 3, 1, 4), dtype=torch.float32)
    preds[:, 2, 0, 2:4] = 0.9
    metadata = [
        {
            "path": f"sample {index}.hid",
            "signal_image": np.full((1, 4, 1), index + 1, dtype=np.float32),
        }
        for index in range(size)
    ]

    batch = torch.zeros((size, 1, 4, 1), dtype=torch.float32), targets, metadata
    outputs = {"preds": preds}
    return batch, outputs


def test_profile_plot_callback_writes_limited_pngs_to_evaluation_output_dir(tmp_path):
    callback = ProfilePlotCallback(num_profiles=2)
    trainer = FakeTrainer(tmp_path / "evaluation")
    batch, outputs = _batch(size=3)

    callback.on_test_epoch_start(trainer, FakeModule())
    callback.on_test_batch_end(
        trainer,
        FakeModule(),
        outputs=outputs,
        batch=batch,
        batch_idx=0,
    )
    callback.on_test_batch_end(
        trainer,
        FakeModule(),
        outputs=outputs,
        batch=batch,
        batch_idx=1,
    )

    plot_paths = sorted((tmp_path / "evaluation" / "plots").glob("*.png"))
    assert [path.name for path in plot_paths] == [
        "profile_0000_sample_0.png",
        "profile_0001_sample_1.png",
    ]
    assert plt.get_fignums() == []


def test_profile_plot_callback_respects_annotation_and_prediction_toggles(
    tmp_path,
    monkeypatch,
):
    calls = []

    def fake_plot_profile(signal, *, annotation=None, prediction=None, **kwargs):
        del kwargs
        calls.append({
            "signal": signal,
            "annotation": annotation,
            "prediction": prediction,
        })
        return plt.figure()

    monkeypatch.setattr(profile_plot_module, "plot_profile", fake_plot_profile)
    callback = ProfilePlotCallback(
        include_annotations=False,
        include_predictions=False,
        num_profiles=1,
    )
    batch, _outputs = _batch(size=1)

    callback.on_test_epoch_start(FakeTrainer(tmp_path), FakeModule())
    callback.on_test_batch_end(
        FakeTrainer(tmp_path),
        FakeModule(),
        outputs=None,
        batch=batch,
        batch_idx=0,
    )

    assert len(calls) == 1
    assert calls[0]["signal"].shape == (1, 4)
    assert calls[0]["annotation"] is None
    assert calls[0]["prediction"] is None
    assert plt.get_fignums() == []


def test_profile_plot_callback_passes_annotations_and_predictions(
    tmp_path,
    monkeypatch,
):
    calls = []

    def fake_plot_profile(signal, *, annotation=None, prediction=None, **kwargs):
        del kwargs
        calls.append({
            "signal": signal,
            "annotation": annotation,
            "prediction": prediction,
        })
        return plt.figure()

    monkeypatch.setattr(profile_plot_module, "plot_profile", fake_plot_profile)
    callback = ProfilePlotCallback(
        include_annotations=True,
        include_predictions=True,
        num_profiles=1,
    )
    batch, outputs = _batch(size=1)

    callback.on_test_epoch_start(FakeTrainer(tmp_path), FakeModule())
    callback.on_test_batch_end(
        FakeTrainer(tmp_path),
        FakeModule(),
        outputs=outputs,
        batch=batch,
        batch_idx=0,
    )

    assert len(calls) == 1
    assert calls[0]["annotation"].shape == (1, 4)
    assert calls[0]["prediction"].shape == (1, 4)
    np.testing.assert_array_equal(calls[0]["annotation"], np.array([[0, 1, 1, 0]]))
    np.testing.assert_array_equal(calls[0]["prediction"], np.array([[0, 0, 2, 2]]))


def test_profile_plot_callback_requires_predictions_when_enabled(tmp_path):
    callback = ProfilePlotCallback(
        include_predictions=True,
        num_profiles=1,
    )
    batch, _outputs = _batch(size=1)

    with pytest.raises(ValueError, match="include_predictions=True"):
        callback.on_test_batch_end(
            FakeTrainer(tmp_path),
            FakeModule(),
            outputs=None,
            batch=batch,
            batch_idx=0,
        )


def test_profile_plot_callback_thresholds_binary_probability_predictions(tmp_path, monkeypatch):
    calls = []

    def fake_plot_profile(signal, *, annotation=None, prediction=None, **kwargs):
        del signal, annotation, kwargs
        calls.append(prediction)
        return plt.figure()

    monkeypatch.setattr(profile_plot_module, "plot_profile", fake_plot_profile)
    callback = ProfilePlotCallback(include_predictions=True, num_profiles=1)

    targets = torch.zeros((1, 1, 4, 1), dtype=torch.float32)
    preds = torch.tensor([[[0.2], [0.6], [0.49], [0.51]]], dtype=torch.float32).unsqueeze(0)
    metadata = [{"path": "sample 0.hid", "signal_image": np.ones((1, 4, 1), dtype=np.float32)}]
    batch = torch.zeros((1, 1, 4, 1), dtype=torch.float32), targets, metadata
    outputs = {"preds": preds}

    callback.on_test_epoch_start(FakeTrainer(tmp_path), FakeModule())
    callback.on_test_batch_end(
        FakeTrainer(tmp_path),
        FakeModule(),
        outputs=outputs,
        batch=batch,
        batch_idx=0,
    )

    np.testing.assert_array_equal(calls[0], np.array([[0, 1, 0, 1]]))
