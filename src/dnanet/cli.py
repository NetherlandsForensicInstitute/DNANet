"""Hydra-based CLI entry point for DNANet.

Design pattern: **Command**
    The CLI dispatches to task functions based on the ``task`` config key.
    Each task (train, evaluate, cross_validate) is a self-contained function
    that receives the full Hydra config. This is the *Command pattern*: the
    config object encapsulates everything needed to execute a task, and the
    dispatcher simply routes to the right handler.

    This replaces the original four separate scripts (``train.py``,
    ``evaluate.py``, ``cross_validate.py``, ``grid_search.py``) with a
    single unified entry point.

Usage:
    dnanet task=train data=dnanet_rd model=unet
    dnanet task=evaluate checkpoint=/path/to/model.ckpt
    dnanet task=cross_validate training.n_folds=5
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

import hydra
import torch

from dnanet import logging as dnanet_logging


if TYPE_CHECKING:
    from omegaconf import DictConfig


WORKSPACE_FOLDER = Path(__file__).parents[2]

@hydra.main(version_base=None, config_path=str(WORKSPACE_FOLDER / "conf"), config_name="config")
def main(cfg: DictConfig) -> None:
    """DNANet CLI — train, evaluate, or cross-validate DNA profile models."""
    # Configure logging first (before any other import triggers log messages)
    dnanet_logging.configure(verbosity=cfg.get("verbosity", "INFO"))

    task = cfg.get("task", "train")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    if task == "train":
        from dnanet.tasks.train import run
    elif task == "evaluate":
        from dnanet.tasks.evaluate import run
    elif task == "cross_validate":
        from dnanet.tasks.cross_validate import run
    else:
        raise ValueError(
            f"Unknown task: '{task}'. Choose from: train, evaluate, cross_validate"
        )

    run(cfg)


if __name__ == "__main__":
    main()
