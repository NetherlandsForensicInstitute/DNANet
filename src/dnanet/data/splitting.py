"""Data splitting utilities for train/test/validation splits and k-fold CV."""

from __future__ import annotations

import itertools
import random
from typing import Any


def split_data_in_k_folds(
    data: list[Any], n_folds: int, seed: float | None = None
) -> list[list[Any]]:
    """Split a list into ``n_folds`` approximately equal-sized folds.

    The data is shuffled before splitting. If the total count is not evenly
    divisible by ``n_folds``, remainder items are distributed one-per-fold.

    Args:
        data: Items to split.
        n_folds: Number of folds.
        seed: Optional random seed for reproducibility.

    Returns:
        A list of ``n_folds`` lists, each containing a subset of the data.
    """
    random.seed(seed)
    shuffled = random.sample(data, len(data))

    remainder_count = len(shuffled) % n_folds
    if remainder_count > 0:
        main, remainder = shuffled[:-remainder_count], shuffled[-remainder_count:]
    else:
        main, remainder = shuffled, []

    folds: list[list[Any]] = [[] for _ in range(n_folds)]
    fold_size = len(shuffled) // n_folds
    indices = list(range(0, fold_size * n_folds + 1, fold_size))

    fold_cycle = itertools.cycle(range(n_folds))
    for i in range(len(indices) - 1):
        fold_idx = next(fold_cycle)
        folds[fold_idx].extend(main[indices[i] : indices[i + 1]])

    # Distribute remainder items one per fold
    for item in remainder:
        fold_idx = next(fold_cycle)
        folds[fold_idx].append(item)

    return folds
