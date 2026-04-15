"""RFU outcome collection for later F1-by-RFU analysis."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


RFU_OUTCOMES: tuple[str, ...] = ("tp", "fp", "fn")


class PerRFUOutcomeMetric(Metric):
    """Collect RFU values for true positives, false positives, and false negatives.

    The metric intentionally does not bin RFU values during evaluation. It stores
    only the RFU values needed to compute precision, recall, and F1 later with
    arbitrary bin edges.
    """

    full_state_update = False
    is_differentiable = False
    higher_is_better = None

    def __init__(self, threshold: float = 0.5, **kwargs) -> None:
        """Initialize the accumulator with a prediction threshold."""
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("tp_rfus", default=[], dist_reduce_fx="cat")
        self.add_state("fp_rfus", default=[], dist_reduce_fx="cat")
        self.add_state("fn_rfus", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, targets: Tensor, rfu_values: Tensor) -> None:
        """Collect RFU values for TP, FP, and FN scanpoints."""
        preds = _to_tensor(preds, device=self.device)
        targets = _to_tensor(targets, device=self.device)
        rfu_values = _to_tensor(rfu_values, device=self.device)

        if preds.shape != targets.shape or preds.shape != rfu_values.shape:
            raise ValueError(
                "preds, targets, and rfu_values must have identical shapes, got "
                f"{tuple(preds.shape)}, {tuple(targets.shape)}, and "
                f"{tuple(rfu_values.shape)}."
            )

        pred_mask = preds >= self.threshold
        target_mask = targets.bool()
        valid_rfu_mask = torch.isfinite(rfu_values)

        rfu_values = rfu_values.float()
        self.tp_rfus.append(rfu_values[target_mask & pred_mask & valid_rfu_mask].detach())
        self.fp_rfus.append(rfu_values[(~target_mask) & pred_mask & valid_rfu_mask].detach())
        self.fn_rfus.append(rfu_values[target_mask & (~pred_mask) & valid_rfu_mask].detach())

    def compute(self) -> dict[str, Tensor]:
        """Return accumulated RFU values grouped by outcome."""
        return self.compute_outcomes()

    def compute_outcomes(self) -> dict[str, Tensor]:
        """Return accumulated RFU values as stable one-dimensional tensors."""
        return {
            "tp_rfus": _cat_tensors(self.tp_rfus, device=self.device),
            "fp_rfus": _cat_tensors(self.fp_rfus, device=self.device),
            "fn_rfus": _cat_tensors(self.fn_rfus, device=self.device),
        }


def write_rfu_outcome_file(
    path: str | Path,
    outcomes: Mapping[str, Tensor | np.ndarray | Sequence[float] | Sequence[Tensor]],
) -> Path:
    """Write RFU outcomes based on the filename extension."""
    output_path = Path(path)
    suffix = output_path.suffix.lower()
    if suffix == ".npz":
        return write_rfu_outcome_npz(output_path, outcomes)
    if suffix == ".csv":
        return write_rfu_outcome_csv(output_path, outcomes)
    raise ValueError(
        "RFU outcome filename must end with '.npz' or '.csv', got "
        f"{output_path.name!r}."
    )


def write_rfu_outcome_npz(
    path: str | Path,
    outcomes: Mapping[str, Tensor | np.ndarray | Sequence[float] | Sequence[Tensor]],
) -> Path:
    """Write RFU outcomes as a compressed NPZ archive."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{
                f"{outcome}_rfus": _outcome_array(outcomes, outcome)
                for outcome in RFU_OUTCOMES
            },
        )

    return output_path


def write_rfu_outcome_csv(
    path: str | Path,
    outcomes: Mapping[str, Tensor | np.ndarray | Sequence[float] | Sequence[Tensor]],
) -> Path:
    """Write RFU outcomes as compact ``outcome,rfu`` CSV rows."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("outcome", "rfu"))
        for outcome in RFU_OUTCOMES:
            for rfu in _outcome_array(outcomes, outcome):
                writer.writerow((outcome, f"{rfu:g}"))

    return output_path


def load_rfu_outcome_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load an RFU outcome NPZ written by :func:`write_rfu_outcome_npz`."""
    with np.load(Path(path)) as data:
        missing = [f"{outcome}_rfus" for outcome in RFU_OUTCOMES if f"{outcome}_rfus" not in data]
        if missing:
            raise ValueError(f"Missing RFU outcome array(s): {', '.join(missing)}.")

        return {
            f"{outcome}_rfus": np.asarray(data[f"{outcome}_rfus"], dtype=float).reshape(-1)
            for outcome in RFU_OUTCOMES
        }


def load_rfu_outcome_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Load an RFU outcome CSV written by :func:`write_rfu_outcome_csv`."""
    values: dict[str, list[float]] = {f"{outcome}_rfus": [] for outcome in RFU_OUTCOMES}

    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            outcome = row["outcome"]
            if outcome not in RFU_OUTCOMES:
                raise ValueError(f"Unknown RFU outcome {outcome!r}.")
            values[f"{outcome}_rfus"].append(float(row["rfu"]))

    return {
        key: np.asarray(outcome_values, dtype=float)
        for key, outcome_values in values.items()
    }


def compute_binned_f1(
    outcomes: Mapping[str, Tensor | np.ndarray | Sequence[float] | Sequence[Tensor]],
    bin_edges: Sequence[float],
) -> list[dict[str, float | int]]:
    """Compute precision, recall, and F1 for arbitrary RFU bins."""
    edges = np.asarray(bin_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("bin_edges must be a one-dimensional sequence with at least two values.")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("bin_edges must be strictly increasing.")

    tp = np.histogram(_outcome_array(outcomes, "tp"), bins=edges)[0]
    fp = np.histogram(_outcome_array(outcomes, "fp"), bins=edges)[0]
    fn = np.histogram(_outcome_array(outcomes, "fn"), bins=edges)[0]

    precision = _safe_divide_np(tp, tp + fp)
    recall = _safe_divide_np(tp, tp + fn)
    f1 = _safe_divide_np(2 * precision * recall, precision + recall)

    return [
        {
            "bin_left": float(edges[idx]),
            "bin_right": float(edges[idx + 1]),
            "tp": int(tp[idx]),
            "fp": int(fp[idx]),
            "fn": int(fn[idx]),
            "support": int(tp[idx] + fn[idx]),
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
        }
        for idx in range(len(edges) - 1)
    ]


def compute_binned_f1_from_npz(
    path: str | Path,
    bin_edges: Sequence[float],
) -> list[dict[str, float | int]]:
    """Load an RFU outcome NPZ and compute binned F1 rows."""
    return compute_binned_f1(load_rfu_outcome_npz(path), bin_edges)


def compute_binned_f1_from_csv(
    path: str | Path,
    bin_edges: Sequence[float],
) -> list[dict[str, float | int]]:
    """Load an RFU outcome CSV and compute binned F1 rows."""
    return compute_binned_f1(load_rfu_outcome_csv(path), bin_edges)


def _to_tensor(value: Tensor, *, device: torch.device) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device)
    return torch.as_tensor(value, device=device)


def _cat_tensors(values: Sequence[Tensor], *, device: torch.device) -> Tensor:
    if not values:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.cat([value.reshape(-1).float().to(device) for value in values])


def _outcome_array(
    outcomes: Mapping[str, Tensor | np.ndarray | Sequence[float] | Sequence[Tensor]],
    outcome: str,
) -> np.ndarray:
    key = f"{outcome}_rfus"
    values = outcomes.get(key, outcomes.get(outcome))
    if values is None:
        return np.empty(0, dtype=float)
    if isinstance(values, Tensor):
        return values.detach().cpu().numpy().astype(float, copy=False).reshape(-1)
    if isinstance(values, np.ndarray):
        return values.astype(float, copy=False).reshape(-1)
    if len(values) == 0:
        return np.empty(0, dtype=float)
    if all(isinstance(value, Tensor) for value in values):
        return _cat_tensors(values, device=torch.device("cpu")).numpy()
    return np.asarray(values, dtype=float).reshape(-1)


def _safe_divide_np(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros_like(numerator, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result
