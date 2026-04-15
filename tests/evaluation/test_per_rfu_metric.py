"""Tests for RFU outcome collection and later F1 binning."""

import numpy as np
import torch
import pytest

from dnanet.evaluation.metrics.per_RFU import (
    PerRFUOutcomeMetric,
    compute_binned_f1,
    load_rfu_outcome_npz,
    write_rfu_outcome_npz,
    compute_binned_f1_from_npz,
)


def test_per_rfu_metric_collects_only_tp_fp_fn_rfus():
    metric = PerRFUOutcomeMetric(threshold=0.5)

    preds = torch.tensor([[0.6, 0.4, 0.7, 0.2]])
    targets = torch.tensor([[1, 1, 0, 0]])
    rfu_values = torch.tensor([[10.0, 20.0, 30.0, 40.0]])

    metric.update(preds, targets, rfu_values)
    outcomes = metric.compute_outcomes()

    assert outcomes["tp_rfus"].tolist() == pytest.approx([10.0])
    assert outcomes["fp_rfus"].tolist() == pytest.approx([30.0])
    assert outcomes["fn_rfus"].tolist() == pytest.approx([20.0])


def test_per_rfu_metric_uses_configured_threshold():
    metric = PerRFUOutcomeMetric(threshold=0.75)

    preds = torch.tensor([[0.7, 0.8]])
    targets = torch.tensor([[1, 1]])
    rfu_values = torch.tensor([[100.0, 200.0]])

    metric.update(preds, targets, rfu_values)
    outcomes = metric.compute_outcomes()

    assert outcomes["tp_rfus"].tolist() == pytest.approx([200.0])
    assert outcomes["fn_rfus"].tolist() == pytest.approx([100.0])


def test_rfu_outcome_npz_round_trip(tmp_path):
    path = tmp_path / "per_rfu_outcomes.npz"
    outcomes = {
        "tp_rfus": torch.tensor([10.0, 11.0]),
        "fp_rfus": torch.tensor([20.0]),
        "fn_rfus": torch.tensor([30.0]),
    }

    write_rfu_outcome_npz(path, outcomes)

    with np.load(path) as data:
        assert data["tp_rfus"].tolist() == pytest.approx([10.0, 11.0])
        assert data["fp_rfus"].tolist() == pytest.approx([20.0])
        assert data["fn_rfus"].tolist() == pytest.approx([30.0])

    loaded = load_rfu_outcome_npz(path)
    assert loaded["tp_rfus"].tolist() == pytest.approx([10.0, 11.0])
    assert loaded["fp_rfus"].tolist() == pytest.approx([20.0])
    assert loaded["fn_rfus"].tolist() == pytest.approx([30.0])


def test_compute_binned_f1_from_saved_outcomes(tmp_path):
    path = tmp_path / "per_rfu_outcomes.npz"
    write_rfu_outcome_npz(
        path,
        {
            "tp_rfus": [10.0, 110.0],
            "fp_rfus": [20.0],
            "fn_rfus": [120.0],
        },
    )

    rows = compute_binned_f1_from_npz(path, [0.0, 100.0, 200.0])

    assert rows == compute_binned_f1(load_rfu_outcome_npz(path), [0.0, 100.0, 200.0])
    assert rows[0] == {
        "bin_left": 0.0,
        "bin_right": 100.0,
        "tp": 1,
        "fp": 1,
        "fn": 0,
        "support": 1,
        "precision": pytest.approx(0.5),
        "recall": pytest.approx(1.0),
        "f1": pytest.approx(2 / 3),
    }
    assert rows[1] == {
        "bin_left": 100.0,
        "bin_right": 200.0,
        "tp": 1,
        "fp": 0,
        "fn": 1,
        "support": 2,
        "precision": pytest.approx(1.0),
        "recall": pytest.approx(0.5),
        "f1": pytest.approx(2 / 3),
    }
