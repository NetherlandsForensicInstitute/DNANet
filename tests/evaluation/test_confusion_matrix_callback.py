"""Tests for the confusion matrix Lightning callback."""

from __future__ import annotations

import csv

import torch
import pytest
from matplotlib import pyplot as plt

from dnanet.evaluation.callbacks import ConfusionMatrixCallback


class FakeTrainer:
    def __init__(
        self,
        default_root_dir,
        *,
        is_global_zero: bool = True,
        sanity_checking: bool = False,
    ) -> None:
        self.default_root_dir = str(default_root_dir)
        self.is_global_zero = is_global_zero
        self.sanity_checking = sanity_checking


class FakeModule:
    pass


@pytest.fixture(autouse=True)
def close_figures():
    plt.close('all')
    yield
    plt.close('all')


def _read_csv(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.reader(handle))


def test_confusion_matrix_callback_writes_counts_normalized_csv_and_png(tmp_path):
    callback = ConfusionMatrixCallback(
        num_classes=3,
        class_names=['noise', 'allele', 'stutter'],
    )
    trainer = FakeTrainer(tmp_path)
    module = FakeModule()
    outputs = {
        'metric_preds': torch.tensor([0, 2, 1, 2]),
        'targets': torch.tensor([0, 1, 1, 2]),
    }

    callback.on_test_epoch_start(trainer, module)
    callback.on_test_batch_end(trainer, module, outputs=outputs, batch=None, batch_idx=0)
    callback.on_test_epoch_end(trainer, module)

    output_dir = tmp_path / 'confusion_matrix'
    assert _read_csv(output_dir / 'test_confusion_matrix_counts.csv') == [
        ['true_label', 'noise', 'allele', 'stutter'],
        ['noise', '1', '0', '0'],
        ['allele', '0', '1', '1'],
        ['stutter', '0', '0', '1'],
    ]
    assert _read_csv(output_dir / 'test_confusion_matrix_normalized.csv') == [
        ['true_label', 'noise', 'allele', 'stutter'],
        ['noise', '1', '0', '0'],
        ['allele', '0', '0.5', '0.5'],
        ['stutter', '0', '0', '1'],
    ]
    assert (output_dir / 'test_confusion_matrix.png').exists()
    assert plt.get_fignums() == []


def test_confusion_matrix_callback_ignores_configured_target_index(tmp_path):
    callback = ConfusionMatrixCallback(num_classes=3, ignore_index=-100)
    trainer = FakeTrainer(tmp_path)
    outputs = {
        'metric_preds': torch.tensor([0, 2, 2]),
        'targets': torch.tensor([0, -100, 1]),
    }

    callback.on_test_epoch_start(trainer, FakeModule())
    callback.on_test_batch_end(trainer, FakeModule(), outputs=outputs, batch=None, batch_idx=0)
    callback.on_test_epoch_end(trainer, FakeModule())

    assert _read_csv(tmp_path / 'confusion_matrix' / 'test_confusion_matrix_counts.csv') == [
        ['true_label', 'Unlabeled', 'Allele', 'Stutter'],
        ['Unlabeled', '1', '0', '0'],
        ['Allele', '0', '0', '1'],
        ['Stutter', '0', '0', '0'],
    ]


def test_confusion_matrix_callback_argmaxes_class_dimension(tmp_path):
    callback = ConfusionMatrixCallback(num_classes=3, filename_prefix='')
    trainer = FakeTrainer(tmp_path)
    outputs = {
        'metric_preds': torch.tensor(
            [
                [
                    [0.8, 0.1, 0.1],
                    [0.1, 0.7, 0.1],
                    [0.1, 0.2, 0.8],
                ]
            ]
        ),
        'targets': torch.tensor([[0, 1, 2]]),
    }

    callback.on_test_epoch_start(trainer, FakeModule())
    callback.on_test_batch_end(trainer, FakeModule(), outputs=outputs, batch=None, batch_idx=0)
    callback.on_test_epoch_end(trainer, FakeModule())

    assert _read_csv(tmp_path / 'confusion_matrix' / 'confusion_matrix_counts.csv') == [
        ['true_label', 'Unlabeled', 'Allele', 'Stutter'],
        ['Unlabeled', '1', '0', '0'],
        ['Allele', '0', '1', '0'],
        ['Stutter', '0', '0', '1'],
    ]


def test_confusion_matrix_callback_requires_step_outputs():
    callback = ConfusionMatrixCallback(num_classes=3)

    with pytest.raises(ValueError, match='metric_preds'):
        callback.on_test_batch_end(
            FakeTrainer('.'),
            FakeModule(),
            outputs=None,
            batch=None,
            batch_idx=0,
        )


def test_confusion_matrix_callback_skips_unconfigured_stage(tmp_path):
    callback = ConfusionMatrixCallback(num_classes=3, stages=('val',))
    trainer = FakeTrainer(tmp_path)

    callback.on_test_epoch_start(trainer, FakeModule())
    callback.on_test_batch_end(
        trainer,
        FakeModule(),
        outputs=None,
        batch=None,
        batch_idx=0,
    )
    callback.on_test_epoch_end(trainer, FakeModule())

    assert not (tmp_path / 'confusion_matrix').exists()
