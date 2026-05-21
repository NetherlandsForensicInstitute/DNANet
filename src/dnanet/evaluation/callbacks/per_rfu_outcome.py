"""Per-RFU outcome evaluation callback.

The per-RFU outcome is available during test runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path

import numpy as np
from loguru import logger
from lightning import Callback

from dnanet.evaluation.metrics.per_RFU import PerRFUOutcomeMetric, write_rfu_outcome_file
from dnanet.evaluation.callbacks.allele_metrics import AlleleMetricsCallback


if TYPE_CHECKING:
    from collections.abc import Mapping

    import lightning as L
    from torch import Tensor


class PerRFUOutcomeCallback(Callback):
    """Collect TP/FP/FN RFU values during Lightning test runs."""

    def __init__(
        self,
        threshold: float = 0.5,
        filename: str = "per_rfu_outcomes.npz",
        metric: PerRFUOutcomeMetric | None = None,
    ) -> None:
        """Initialize RFU outcome callback."""
        self.metric = metric or PerRFUOutcomeMetric(threshold=threshold)
        self.filename = filename

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Reset RFU outcome state before test epoch starts."""
        del trainer, pl_module
        logger.warning("Per-RFU outcome logging may create large output files.")
        self.metric.reset()

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Mapping[str, Tensor] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Update RFU outcome state from batch predictions, targets, and metadata."""
        del trainer, pl_module, batch_idx, dataloader_idx

        if outputs is None or "preds" not in outputs:
            raise ValueError(
                "PerRFUOutcomeCallback requires test_step to return a mapping "
                "with a 'preds' tensor."
            )

        metadata = AlleleMetricsCallback._metadata_from_batch(batch)
        targets = self._targets_from_batch(batch)
        preds = outputs["preds"].detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        if len(preds) != len(metadata) or len(targets) != len(metadata):
            raise ValueError(
                "Number of predictions, targets, and metadata entries must match, got "
                f"{len(preds)}, {len(targets)}, and {len(metadata)}."
            )

        for pred, target, sample_metadata in zip(preds, targets, metadata, strict=True):
            signal_image = sample_metadata.get("signal_image")
            if signal_image is None:
                raise ValueError("Missing signal_image in sample metadata.")

            self.metric.update(
                preds=self._as_2d_array(pred),
                targets=self._as_2d_array(target),
                rfu_values=self._as_2d_array(signal_image),
            )

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Write RFU outcome NPZ at testing end."""
        del pl_module
        outcomes = self.metric.compute()

        if getattr(trainer, "is_global_zero", True):
            write_rfu_outcome_file(self._output_path(trainer), outcomes)

        self.metric.reset()

    @staticmethod
    def _targets_from_batch(batch: Any) -> Tensor:
        if not isinstance(batch, (tuple, list)) or len(batch) != 3:
            raise ValueError(
                "PerRFUOutcomeCallback requires batches from a metadata transformer "
                "with shape (inputs, targets, metadata)."
            )

        targets = batch[1]
        if not hasattr(targets, "detach"):
            raise TypeError("Expected batch targets to be a tensor.")
        return targets

    @staticmethod
    def _as_2d_array(array: Any) -> np.ndarray:
        result = np.asarray(array)
        if result.ndim == 3 and result.shape[-1] == 1:
            result = result[..., 0]
        return result

    def _output_path(self, trainer: L.Trainer) -> Path:
        path = Path(self.filename)
        if path.is_absolute():
            return path

        root_dir = Path(getattr(trainer, "default_root_dir", ".") or ".")
        return root_dir / path
