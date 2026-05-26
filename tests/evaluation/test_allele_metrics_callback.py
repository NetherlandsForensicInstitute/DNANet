"""Tests for allele metric evaluation callback."""

from __future__ import annotations

import numpy as np
import torch
import pytest

from dnanet.core.panel import Panel
from dnanet.core.allele import Allele
from dnanet.core.marker import Marker
from dnanet.core.annotation import AlleleAnnotation
from dnanet.evaluation.callbacks import AlleleMetricsCallback


def _marker(name: str, dye: int, alleles: list[str]) -> Marker:
    return Marker(
        name=name,
        dye_row=dye,
        alleles=frozenset(Allele(name=allele_name) for allele_name in alleles),
    )


class FakeAlleleCaller:
    def __init__(self, markers):
        self.markers = markers
        self.calls = []

    def call_alleles(self, prediction_image, signal_image, scaler, panel):
        self.calls.append({
            "prediction_shape": prediction_image.shape,
            "signal_shape": signal_image.shape,
            "scaler": scaler,
            "panel": panel,
        })
        return self.markers


class FakeModule:
    def __init__(self):
        self.logged = {}

    def log(self, name, value, **kwargs):
        del kwargs
        self.logged[name] = float(value)


def _batch(gt_marker: Marker) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    signal = np.zeros((1, 8, 1), dtype=np.float32)
    return (
        torch.zeros((1, 1, 8, 1), dtype=torch.float32),
        torch.zeros((1, 1, 8, 1), dtype=torch.float32),
        [{
            "allele_annotation": AlleleAnnotation([gt_marker]),
            "panel": Panel(markers=[gt_marker]),
            "path": "sample.hid",
            "scaler": np.arange(8),
            "signal_image": signal,
        }],
    )


def _peaknet_batch(gt_marker: Marker) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, list[dict]]:
    signal = np.zeros((1, 8), dtype=np.float32)
    inputs = (
        torch.zeros((1, 1, 8), dtype=torch.float32),
        torch.zeros((1, 1, 4), dtype=torch.float32),
        torch.zeros((1,), dtype=torch.long),
        torch.zeros((1, 2), dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
    )
    targets = torch.zeros((1, 1, 8), dtype=torch.long)
    metadata = [{
        "allele_annotation": AlleleAnnotation([gt_marker]),
        "panel": Panel(markers=[gt_marker]),
        "path": "sample.hid",
        "scaler": np.arange(8),
        "signal_image": signal,
    }]
    return inputs, targets, metadata


def test_allele_metrics_callback_logs_test_metrics():
    gt_marker = _marker("D5S818", 0, ["13", "15"])
    pred_marker = _marker("D5S818", 0, ["13", "14"])
    callback = AlleleMetricsCallback(allele_caller=FakeAlleleCaller([pred_marker]))
    module = FakeModule()

    callback.on_test_epoch_start(None, module)
    callback.on_test_batch_end(
        None,
        module,
        outputs={"preds": torch.zeros((1, 1, 8, 1), dtype=torch.float32)},
        batch=_batch(gt_marker),
        batch_idx=0,
    )
    callback.on_test_epoch_end(None, module)

    assert module.logged["test/allele_precision"] == pytest.approx(0.5)
    assert module.logged["test/allele_recall"] == pytest.approx(0.5)
    assert module.logged["test/allele_f1"] == pytest.approx(0.5)

    allele_call = callback.allele_caller.calls[0]
    assert allele_call["prediction_shape"] == (1, 8)
    assert allele_call["signal_shape"] == (1, 8)


def test_allele_metrics_callback_accepts_peaknet_metadata_batch():
    gt_marker = _marker("D5S818", 0, ["13"])
    pred_marker = _marker("D5S818", 0, ["13"])
    callback = AlleleMetricsCallback(allele_caller=FakeAlleleCaller([pred_marker]))
    module = FakeModule()

    callback.on_test_epoch_start(None, module)
    callback.on_test_batch_end(
        None,
        module,
        outputs={"preds": torch.zeros((1, 1, 8), dtype=torch.float32)},
        batch=_peaknet_batch(gt_marker),
        batch_idx=0,
    )
    callback.on_test_epoch_end(None, module)

    assert module.logged["test/allele_precision"] == pytest.approx(1.0)
    assert module.logged["test/allele_recall"] == pytest.approx(1.0)
    assert module.logged["test/allele_f1"] == pytest.approx(1.0)


def test_allele_metrics_callback_requires_metadata_batch():
    callback = AlleleMetricsCallback(allele_caller=FakeAlleleCaller([]))

    with pytest.raises(ValueError, match="metadata transformer"):
        callback.on_test_batch_end(
            None,
            FakeModule(),
            outputs={"preds": torch.zeros((1, 1, 8), dtype=torch.float32)},
            batch=(torch.zeros((1, 1, 8)), torch.zeros((1, 1, 8))),
            batch_idx=0,
        )
