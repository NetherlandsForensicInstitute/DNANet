"""Profile plot evaluation callback.

The profile plot is available during test runs."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any
from pathlib import Path

import numpy as np
from loguru import logger
from lightning import Callback
from matplotlib import pyplot as plt

from dnanet.evaluation.visualization import coerce_class_map, plot_profile
from dnanet.evaluation.callbacks.allele_metrics import AlleleMetricsCallback


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import lightning as L
    from torch import Tensor


class ProfilePlotCallback(Callback):
    """Save a limited number of evaluation profile plots."""

    def __init__(
        self,
        include_annotations: bool = True,
        include_predictions: bool = True,
        num_profiles: int = 5,
    ) -> None:
        """Initialize profile plotting options."""
        if num_profiles < 0:
            raise ValueError("num_profiles must be non-negative.")

        self.include_annotations = include_annotations
        self.include_predictions = include_predictions
        self.num_profiles = num_profiles
        self._saved_profiles = 0

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Reset saved-profile counter before test epoch starts."""
        del trainer, pl_module
        self._saved_profiles = 0

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Mapping[str, Tensor] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Save profile plots from current test batch."""
        del pl_module, batch_idx, dataloader_idx

        if self._saved_profiles >= self.num_profiles:
            return
        if not getattr(trainer, "is_global_zero", True):
            return

        metadata = AlleleMetricsCallback._metadata_from_batch(batch)
        targets = None
        if self.include_annotations:
            targets = self._targets_from_batch(batch).detach().cpu().numpy()

        preds = None
        if self.include_predictions:
            if outputs is None or "preds" not in outputs:
                raise ValueError(
                    "ProfilePlotCallback requires test_step to return a mapping "
                    "with a 'preds' tensor when include_predictions=True."
                )
            preds = outputs["preds"].detach().cpu().numpy()

        self._validate_lengths(metadata, targets, preds)

        output_dir = self._output_dir(trainer)
        output_dir.mkdir(parents=True, exist_ok=True)
        remaining = self.num_profiles - self._saved_profiles

        for index, sample_metadata in enumerate(metadata[:remaining]):
            signal = sample_metadata.get("signal_image")
            if signal is None:
                raise ValueError("Missing signal_image in sample metadata.")
            signal_for_plot = self._as_2d_array(signal)

            annotation = None
            if self.include_annotations and targets is not None:
                annotation = self._annotation_for_plot(
                    sample_metadata,
                    targets[index],
                    signal_shape=signal_for_plot.shape,
                )

            prediction = None
            if self.include_predictions and preds is not None:
                prediction = self._prediction_for_plot(
                    preds[index],
                    signal_shape=signal_for_plot.shape,
                )

            figure = plot_profile(
                signal_for_plot,
                annotation=annotation,
                prediction=prediction,
                title=self._title_for_plot(sample_metadata),
                figsize=(20, 10),
            )

            figure.savefig(
                self._profile_path(output_dir, sample_metadata),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(figure)
            self._saved_profiles += 1

            if self._saved_profiles >= self.num_profiles:
                break

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log where profile plots were saved."""
        del pl_module
        if getattr(trainer, "is_global_zero", True) and self._saved_profiles:
            logger.info(
                "Saved {} profile plot(s) to {}",
                self._saved_profiles,
                self._output_dir(trainer),
            )

    @staticmethod
    def _targets_from_batch(batch: Any) -> Tensor:
        if not isinstance(batch, (tuple, list)) or len(batch) != 3:
            raise ValueError(
                "ProfilePlotCallback requires batches from a metadata transformer "
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

    @classmethod
    def _annotation_for_plot(
        cls,
        sample_metadata: Mapping[str, Any],
        target: np.ndarray,
        *,
        signal_shape: tuple[int, int],
    ) -> np.ndarray:
        scanpoint_annotation = sample_metadata.get("scanpoint_annotation")
        if scanpoint_annotation is None:
            return coerce_class_map(target, signal_shape=signal_shape, source="annotation")

        data = getattr(scanpoint_annotation, "data", scanpoint_annotation)
        return coerce_class_map(data, signal_shape=signal_shape, source="annotation")

    @staticmethod
    def _prediction_for_plot(
        prediction: np.ndarray,
        *,
        signal_shape: tuple[int, int],
    ) -> np.ndarray:
        return coerce_class_map(prediction, signal_shape=signal_shape, source="prediction")

    @staticmethod
    def _validate_lengths(
        metadata: Sequence[Mapping[str, Any]],
        targets: np.ndarray | None,
        preds: np.ndarray | None,
    ) -> None:
        expected = len(metadata)
        lengths = {"metadata": expected}
        if targets is not None:
            lengths["targets"] = len(targets)
        if preds is not None:
            lengths["predictions"] = len(preds)

        if any(length != expected for length in lengths.values()):
            lengths_text = ", ".join(f"{name}={length}" for name, length in lengths.items())
            raise ValueError(
                "Number of profile plot inputs must match batch metadata; "
                f"got {lengths_text}."
            )

    def _output_dir(self, trainer: L.Trainer) -> Path:
        root_dir = Path(getattr(trainer, "default_root_dir", ".") or ".")
        return root_dir / "plots"

    def _profile_path(self, output_dir: Path, sample_metadata: Mapping[str, Any]) -> Path:
        sample_name = self._safe_sample_name(sample_metadata)
        return output_dir / f"profile_{self._saved_profiles:04d}_{sample_name}.png"

    @staticmethod
    def _title_for_plot(sample_metadata: Mapping[str, Any]) -> str | None:
        sample_path = sample_metadata.get("path")
        return str(sample_path) if sample_path is not None else None

    @staticmethod
    def _safe_sample_name(sample_metadata: Mapping[str, Any]) -> str:
        sample_path = sample_metadata.get("path")
        sample_name = Path(str(sample_path)).stem if sample_path is not None else "sample"
        sample_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_name).strip("._")
        return sample_name or "sample"
