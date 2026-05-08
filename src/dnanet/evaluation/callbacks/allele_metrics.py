"""Allele metric evaluation callback."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from collections.abc import Mapping, Sequence

import numpy as np
from lightning import Callback

from dnanet.core.annotation import AlleleAnnotation
from dnanet.evaluation.metrics.allele import AlleleRecall, AlleleF1Score, AllelePrecision


if TYPE_CHECKING:
    import lightning as L
    from torch import Tensor

    from dnanet.evaluation.allele_caller import AlleleCaller


class AlleleMetricsCallback(Callback):
    """Compute allele-level metrics during Lightning test runs."""

    def __init__(
        self,
        allele_caller: AlleleCaller,
        precision: AllelePrecision | None = None,
        recall: AlleleRecall | None = None,
        f1: AlleleF1Score | None = None,
        skip_missing_annotations: bool = True,
    ) -> None:
        """Initialize allele caller, metrics, and missing-annotation behavior."""
        self.allele_caller = allele_caller
        self.precision = precision or AllelePrecision()
        self.recall = recall or AlleleRecall()
        self.f1 = f1 or AlleleF1Score()
        self.skip_missing_annotations = skip_missing_annotations
        self._has_updates = False

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Reset allele metrics before a test epoch starts."""
        del trainer, pl_module
        self._reset_metrics()
        self._has_updates = False

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Mapping[str, Tensor] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Update allele metrics from test batch predictions and metadata."""
        del trainer, pl_module, batch_idx, dataloader_idx

        if outputs is None or "preds" not in outputs:
            raise ValueError(
                "AlleleMetricsCallback requires test_step to return a mapping "
                "with a 'preds' tensor."
            )

        metadata = self._metadata_from_batch(batch)
        preds = outputs["preds"].detach().cpu().numpy()
        if len(preds) != len(metadata):
            raise ValueError(
                "Number of predictions and metadata entries must match, got "
                f"{len(preds)} and {len(metadata)}."
            )

        for pred, sample_metadata in zip(preds, metadata, strict=True):
            allele_annotation = sample_metadata.get("allele_annotation")
            if allele_annotation is None:
                if self.skip_missing_annotations:
                    continue
                raise ValueError("Missing allele_annotation in sample metadata.")
            if not isinstance(allele_annotation, AlleleAnnotation):
                raise TypeError(
                    "Expected metadata['allele_annotation'] to be an AlleleAnnotation, "
                    f"got {type(allele_annotation).__name__}."
                )

            panel = sample_metadata.get("panel")
            if panel is None:
                raise ValueError("Missing panel in segmentation metadata.")

            pred_markers = self.allele_caller.call_alleles(
                prediction_image=self._as_2d_array(pred),
                signal_image=self._as_2d_array(sample_metadata["signal_image"]),
                scaler=np.asarray(sample_metadata["scaler"]),
                panel=panel,
            )
            ground_truth_markers = allele_annotation.data

            self.precision.update([ground_truth_markers], [pred_markers])
            self.recall.update([ground_truth_markers], [pred_markers])
            self.f1.update([ground_truth_markers], [pred_markers])
            self._has_updates = True

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log allele metrics at test epoch end."""
        del trainer
        if not self._has_updates:
            self.precision.update([[]], [[]])
            self.recall.update([[]], [[]])
            self.f1.update([[]], [[]])

        pl_module.log("test/allele_precision", self.precision.compute(), logger=True, sync_dist=True)
        pl_module.log("test/allele_recall", self.recall.compute(), logger=True, sync_dist=True)
        pl_module.log("test/allele_f1", self.f1.compute(), logger=True, sync_dist=True)
        self._reset_metrics()

    @staticmethod
    def _metadata_from_batch(batch: Any) -> Sequence[Mapping[str, Any]]:
        if not isinstance(batch, (tuple, list)) or len(batch) != 3:
            raise ValueError(
                "AlleleMetricsCallback requires batches from a metadata transformer "
                "with shape (inputs, targets, metadata)."
            )

        metadata = batch[2]
        if not isinstance(metadata, Sequence):
            raise TypeError("Expected batch metadata to be a sequence of mappings.")
        if not all(isinstance(sample_metadata, Mapping) for sample_metadata in metadata):
            raise TypeError("Expected every metadata entry to be a mapping.")
        return metadata

    @staticmethod
    def _as_2d_array(array: Any) -> np.ndarray:
        result = np.asarray(array)
        if result.ndim == 3 and result.shape[-1] == 1:
            result = result[..., 0]
        return result

    def _reset_metrics(self) -> None:
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
