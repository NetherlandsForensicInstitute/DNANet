import itertools
import random
from typing import Any, List, Optional

import numpy as np


def split_data_in_k_folds(data: List[Any], n_folds: int, seed: Optional[float] = None) \
        -> List[List[Any]]:
    """
    Split a list of instances into 'n_folds' separate folds.
    """
    random.seed(seed)

    folds = [[] for _ in range(n_folds)]
    fold_ind = 0
    # shuffle such that original data remains unshuffled
    shuffled_data = random.sample(data, len(data))
    # nr. of remaining instances that we are left with if the total number of images is not
    # divisible by the number of folds. we will divide those evenly over all folds.
    nr_remaining = len(shuffled_data) % n_folds
    if nr_remaining > 0:
        # cut off the remainder part from all instances
        data, remainder = shuffled_data[:-nr_remaining], shuffled_data[-nr_remaining:]
    else:
        data, remainder = shuffled_data, []

    fold_size = len(shuffled_data) // n_folds  # nr of instances to at least put in each fold
    split_inds = np.arange(0, fold_size + len(data), fold_size)  # create split indices
    # create infinite fold indexer that goes from 0 to n_folds and back to 0
    fold_indexer = itertools.cycle(range(n_folds))
    for i in range(len(split_inds) - 1):
        fold_ind = next(fold_indexer)
        images = data[split_inds[i]:split_inds[i + 1]]
        folds[fold_ind].extend(images)

    # put the remainder part in folds, instance by instance (so the number of instances per fold
    # is more balanced)
    for r in remainder:
        fold_ind = next(fold_indexer)
        folds[fold_ind].append(r)
    return folds
