"""Confusion matrix Lightning callback.

The confusion matrix is available during validation and test runs.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING
from pathlib import Path

import matplotlib


matplotlib.use('Agg')

import numpy as np
import seaborn as sns
from loguru import logger
from lightning import Callback
from matplotlib import pyplot as plt

from dnanet.core import LabelCategory


if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Mapping, Sequence

    import lightning as L
    from torch import Tensor


class ConfusionMatrixCallback(Callback):
    """Collect and save multiclass confusion matrices for validation/test runs."""

    def __init__(
        self,
        num_classes: int,
        class_names: Sequence[str] | None = None,
        output_dir: str = 'confusion_matrix',
        stages: Sequence[str] = ('val', 'test'),
        ignore_index: int | None = None,
        normalize: bool = True,
        threshold: float = 0.5,
        annot: bool = True,
        cmap: str = 'Blues',
        filename_prefix: str | None = None,
    ) -> None:
        """Initialize confusion matrix collection and output settings."""
        if num_classes < 2:
            raise ValueError('num_classes must be at least 2.')
        if class_names is not None and len(class_names) != num_classes:
            raise ValueError(
                'class_names length must match num_classes, got '
                f'{len(class_names)} and {num_classes}.'
            )

        unsupported = set(stages) - {'val', 'test'}
        if unsupported:
            raise ValueError(f'Unsupported confusion matrix stage(s): {sorted(unsupported)}.')

        self.num_classes = num_classes
        self.class_names = (
            list(class_names) if class_names is not None else self._default_class_names(num_classes)
        )
        self.output_dir = output_dir
        self.stages = frozenset(stages)
        self.ignore_index = ignore_index
        self.normalize = normalize
        self.threshold = threshold
        self.annot = annot
        self.cmap = cmap
        self.filename_prefix = filename_prefix
        self._matrices: dict[str, np.ndarray] = {}

    @staticmethod
    def _default_class_names(num_classes: int) -> list[str]:
        class_names = []
        for index in range(num_classes):
            try:
                class_names.append(LabelCategory.from_index(index).display_name)
            except IndexError:
                class_names.append(str(index))
        return class_names

    def on_validation_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Reset validation confusion matrix state."""
        del pl_module
        if self._should_collect('val', trainer):
            self._reset_stage('val')

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Mapping[str, Tensor] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Update validation confusion matrix state."""
        del pl_module, batch, batch_idx, dataloader_idx
        if self._should_collect('val', trainer):
            self._update_from_outputs('val', outputs)

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Save the latest validation confusion matrix."""
        del pl_module
        if self._should_collect('val', trainer):
            self._write_stage_outputs('val', trainer)

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Reset test confusion matrix state."""
        del pl_module
        if self._should_collect('test', trainer):
            self._reset_stage('test')

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Mapping[str, Tensor] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Update test confusion matrix state."""
        del pl_module, batch, batch_idx, dataloader_idx
        if self._should_collect('test', trainer):
            self._update_from_outputs('test', outputs)

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Save the test confusion matrix."""
        del pl_module
        if self._should_collect('test', trainer):
            self._write_stage_outputs('test', trainer)

    def _should_collect(self, stage: str, trainer: L.Trainer) -> bool:
        if stage not in self.stages:
            return False
        return not (stage == 'val' and getattr(trainer, 'sanity_checking', False))

    def _reset_stage(self, stage: str) -> None:
        self._matrices[stage] = np.zeros(
            (self.num_classes, self.num_classes),
            dtype=np.int64,
        )

    def _update_from_outputs(
        self,
        stage: str,
        outputs: Mapping[str, Tensor] | None,
    ) -> None:
        if outputs is None or 'metric_preds' not in outputs or 'targets' not in outputs:
            raise ValueError(
                "ConfusionMatrixCallback requires step outputs with 'metric_preds' "
                "and 'targets' tensors."
            )

        if stage not in self._matrices:
            self._reset_stage(stage)

        preds = self._to_numpy(outputs['metric_preds'])
        targets = self._to_numpy(outputs['targets'])
        pred_indices = self._as_class_indices(preds, name='metric_preds')
        target_indices = self._as_class_indices(targets, name='targets')

        if len(pred_indices) != len(target_indices):
            raise ValueError(
                'Confusion matrix predictions and targets must have the same number '
                f'of elements, got {len(pred_indices)} and {len(target_indices)}.'
            )

        mask = self._valid_label_mask(target_indices, pred_indices)
        target_indices = target_indices[mask]
        pred_indices = pred_indices[mask]

        encoded = self.num_classes * target_indices + pred_indices
        counts = np.bincount(
            encoded,
            minlength=self.num_classes * self.num_classes,
        ).reshape(self.num_classes, self.num_classes)
        self._matrices[stage] += counts

    @staticmethod
    def _to_numpy(value: Tensor | Any) -> np.ndarray:
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def _as_class_indices(self, values: np.ndarray, *, name: str) -> np.ndarray:
        if values.size == 0:
            return values.astype(np.int64).reshape(-1)

        class_values = values
        if np.issubdtype(class_values.dtype, np.floating):
            if class_values.ndim > 1 and class_values.shape[1] == self.num_classes:
                class_values = class_values.argmax(axis=1)
            elif class_values.ndim > 1 and class_values.shape[-1] == self.num_classes:
                class_values = class_values.argmax(axis=-1)
            elif self.num_classes == 2:
                class_values = class_values >= self.threshold
            elif np.all(np.isclose(class_values, np.rint(class_values))):
                class_values = np.rint(class_values)
            else:
                raise ValueError(
                    f'{name} contains floating values without a class dimension. '
                    'Return class indices from the Lightning step or configure a '
                    'binary confusion matrix.'
                )

        return class_values.astype(np.int64).reshape(-1)

    def _valid_label_mask(self, targets: np.ndarray, preds: np.ndarray) -> np.ndarray:
        mask = np.ones(targets.shape, dtype=bool)
        if self.ignore_index is not None:
            mask &= targets != self.ignore_index

        check_targets = targets[mask]
        check_preds = preds[mask]
        invalid_targets = check_targets[(check_targets < 0) | (check_targets >= self.num_classes)]
        invalid_preds = check_preds[(check_preds < 0) | (check_preds >= self.num_classes)]
        if invalid_targets.size or invalid_preds.size:
            raise ValueError(
                'Confusion matrix labels must be in [0, num_classes). '
                f'Invalid targets={np.unique(invalid_targets).tolist()}, '
                f'invalid predictions={np.unique(invalid_preds).tolist()}.'
            )

        return mask

    def _write_stage_outputs(self, stage: str, trainer: L.Trainer) -> None:
        if stage not in self._matrices:
            return
        if not getattr(trainer, 'is_global_zero', True):
            return

        output_dir = self._resolved_output_dir(trainer)
        output_dir.mkdir(parents=True, exist_ok=True)

        counts = self._matrices[stage]
        normalized = self._normalize_counts(counts)
        self._write_csv(
            output_dir / self._filename(stage, 'confusion_matrix_counts.csv'),
            counts,
        )
        self._write_csv(
            output_dir / self._filename(stage, 'confusion_matrix_normalized.csv'),
            normalized,
        )
        self._write_plot(
            output_dir / self._filename(stage, 'confusion_matrix.png'),
            normalized if self.normalize else counts,
            normalized=self.normalize,
            stage=stage,
        )
        logger.info('Saved {} confusion matrix outputs to {}', stage, output_dir)

    @staticmethod
    def _normalize_counts(counts: np.ndarray) -> np.ndarray:
        row_sums = counts.sum(axis=1, keepdims=True)
        return np.divide(
            counts,
            row_sums,
            out=np.zeros_like(counts, dtype=np.float64),
            where=row_sums != 0,
        )

    def _write_csv(self, path: Path, matrix: np.ndarray) -> None:
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['true_label', *self.class_names])
            for class_name, row in zip(self.class_names, matrix, strict=True):
                writer.writerow([class_name, *[self._format_value(value) for value in row]])

    @staticmethod
    def _format_value(value: np.number) -> str:
        if np.issubdtype(np.asarray(value).dtype, np.integer):
            return str(int(value))
        return f'{float(value):.10g}'

    def _write_plot(
        self,
        path: Path,
        matrix: np.ndarray,
        *,
        normalized: bool,
        stage: str,
    ) -> None:
        fig_width = max(6.0, 0.6 * self.num_classes)
        fig_height = max(5.0, 0.5 * self.num_classes)
        figure, axes = plt.subplots(figsize=(fig_width, fig_height))
        sns.heatmap(
            matrix,
            annot=self.annot,
            fmt='.2f' if normalized else 'd',
            cmap=self.cmap,
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar=True,
            ax=axes,
        )
        axes.set_xlabel('Predicted label')
        axes.set_ylabel('True label')
        axes.set_title(f'{stage.capitalize()} confusion matrix')
        figure.tight_layout()
        figure.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(figure)

    def _resolved_output_dir(self, trainer: L.Trainer) -> Path:
        path = Path(self.output_dir)
        if path.is_absolute():
            return path

        root_dir = Path(getattr(trainer, 'default_root_dir', '.') or '.')
        return root_dir / path

    def _filename(self, stage: str, name: str) -> str:
        prefix = stage if self.filename_prefix is None else self.filename_prefix
        if not prefix:
            return name
        return f'{prefix}_{name}'
