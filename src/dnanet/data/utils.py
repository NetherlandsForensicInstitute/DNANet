"""Generic data utilities.

Functions here are small, stateless helpers that are dataset-agnostic and
kit-agnostic. Dataset-specific helpers (R&D filename parsing, ProvedIt
naming conventions, etc.) live in their respective ``DatasetStrategy``
implementations under ``dnanet.data.strategies.datasets``.

If this file grows beyond ~100 lines, it's time to split it.
"""

from __future__ import annotations

from typing import Tuple
from pathlib import Path

import numpy as np
import torch
import coolname
from sklearn.utils.class_weight import compute_class_weight


def generate_random_name() -> str:
    """Generate a random human-readable experiment name (e.g. 'BrilliantFalcon')."""
    return "".join(word.capitalize() for word in coolname.generate())


def find_files_by_suffix(root: str | Path, suffix: str) -> list[Path]:
    """Recursively find all files with a given suffix."""
    return list(Path(root).rglob(f"*{suffix}"))


def get_class_weights(dataset, num_classes: int = None) -> torch.Tensor:
    labels = []
    for i in range(len(dataset)):
        sample = dataset[i]
        label = sample[1] if isinstance(sample, (tuple, list)) else sample['label']
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        labels.extend(label.flatten())

    labels = np.array(labels)

    if num_classes is None:
        num_classes = int(labels.max()) + 1

    present_classes = np.unique(labels)

    weights = compute_class_weight(
        'balanced',
        classes=present_classes,
        y=labels
    )

    # Map back to full class array, missing classes get weight 0
    class_weights = np.zeros(num_classes)
    for cls, w in zip(present_classes, weights):
        class_weights[int(cls)] = w

    return torch.tensor(class_weights, dtype=torch.float32)