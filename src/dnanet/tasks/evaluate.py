"""Evaluation task — load a trained model and compute metrics.

Design pattern: **Facade**
    Single entry point for model evaluation. Loads a checkpoint, runs
    predictions on the dataset, computes configured metrics, and saves
    results.

Usage::

    dnanet task=evaluate checkpoint=/path/to/best.ckpt
    dnanet task=evaluate checkpoint=/path/to/best.ckpt evaluation=segmentation
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, List, Mapping
from pathlib import Path

import numpy as np
import lightning as L
from loguru import logger
from omegaconf import OmegaConf, DictConfig, ListConfig
from hydra.utils import get_class, instantiate

from dnanet.tasks.train import _build_logger


if TYPE_CHECKING:
    from torch.utils.data import Dataset

    from dnanet.modules import BaseTaskModule



def _as_2d_array(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[-1] == 1:
        return array[..., 0]
    return array


def _save_results(
    results: List[Mapping[str, float]],
    output_dir: str,
    filename: str = "metrics.json",
) -> Path:
    """Save metric results to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path / filename
    metrics_path.write_text(json.dumps(results, indent=2))
    return metrics_path


def _build_callbacks(cfg: DictConfig) -> list[L.Callback]:
    """Build test-stage callbacks from evaluation config."""
    callbacks_cfg = cfg.evaluate.get("callbacks")
    if not callbacks_cfg:
        return []

    if isinstance(callbacks_cfg, DictConfig):
        callback_specs = callbacks_cfg.values()
    elif isinstance(callbacks_cfg, ListConfig):
        callback_specs = callbacks_cfg
    else:
        raise TypeError(
            "evaluate.callbacks must be a mapping or list of Hydra callback configs."
        )

    return [instantiate(callback_cfg, _convert_="partial") for callback_cfg in callback_specs]


def run(
    cfg: DictConfig,
    dataset: Dataset | None = None,
) -> dict[str, float]:
    """Run evaluation with a pre-loaded dataset.

    This is the primary programmatic entry point. It loads a checkpoint,
    runs predictions on the dataset, computes metrics, and saves results.

    Args:
        cfg: Composed Hydra config. Must include ``checkpoint`` path.
        dataset: The dataset to use, when none is specified, the dataset
            is loaded from the config.

    Returns:
        Dictionary of metric name -> value.
    """
    from dnanet.data.datamodule import DNANetDataModule

    L.seed_everything(cfg.seed, workers=True)

    # -- Validate config ---------------------------------------------------
    checkpoint_path = cfg.get("checkpoint")
    if not checkpoint_path:
        raise ValueError(
            "Evaluation requires a checkpoint path. "
            "Set it via: dnanet task=evaluate checkpoint=/path/to/model.ckpt"
        )
    logger.info("Loading checkpoint: {}", checkpoint_path)

    checkpoint_dir = Path(checkpoint_path).parent.parent  # go up from /checkpoints/
    checkpoint_config_path = checkpoint_dir / "config.yaml"

    if not checkpoint_config_path.exists():
        raise ValueError(
            f"Config file not found in checkpoint directory: {checkpoint_config_path}"
        )

    # Load the original training config from checkpoint
    checkpoint_cfg = OmegaConf.load(checkpoint_config_path)
    logger.info("Loaded checkpoint config from {}", checkpoint_config_path)

    # -- Build model and loss from checkpoint config -----------------------
    network = instantiate(checkpoint_cfg.model.architecture)
    loss_fn = instantiate(checkpoint_cfg.model.loss)


    eval_metrics = instantiate(cfg.evaluate.metrics, _convert_="partial")

    # -- Lightning module --------------------------------------------------
    # noinspection PyTypeChecker
    module_class: type[BaseTaskModule] = get_class(cfg.evaluate.lightning_module)


    model = module_class.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        metrics=eval_metrics,
        model=network,
        optimizer=None,
        loss_fn=loss_fn
    )
    model.eval()
    logger.info("Model loaded: {}", type(network).__name__)

    # -- Data --------------------------------------------------------------
    if not dataset:
        data_cfg = cfg.get('data')
        dataset = instantiate(data_cfg.dataset)

    # load datamodule with the same seed and split from the checkpoint config
    datamodule = instantiate(
        checkpoint_cfg.train.data_module,
        dataset=dataset,
        batch_size=cfg.evaluate.get("batch_size", 1),
        num_workers=cfg.evaluate.get("num_workers", 0),
    )
    datamodule.setup("test")

    # -- Predict -----------------------------------------------------------
    trainer = L.Trainer(
        default_root_dir=cfg.output_dir,
        callbacks=_build_callbacks(cfg),
        enable_progress_bar=True,
        logger=_build_logger(cfg),
        devices=1,
    )

    if trainer.logger is not None:
        trainer.logger.log_hyperparams(cfg)

    logger.info("Running predictions...")
    logger.warning("Evaluating on entire dataset (no split applied).")
    results = trainer.test(model, dataloaders=datamodule.train_dataloader()) # FIXME: use datamodule test dataloader


    # -- Save results ------------------------------------------------------
    if results:
        metrics_path = _save_results(results, cfg.output_dir)
        logger.info("Results saved to {}", metrics_path)

    # Save config
    config_path = Path(cfg.output_dir) / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(OmegaConf.to_yaml(cfg))

    # Optionally save predictions
    if cfg.evaluate.get("save_predictions", False):
        # TODO do not do a second loop of predictions to save them to disk
        pred_arrays = trainer.predict(model, dataloaders=datamodule.train_dataloader(), return_predictions=True) # FIXME: use datamodule test dataloader
        pred_dir = Path(cfg.output_dir) / cfg.evaluate.get("predictions_dir", "predictions")
        pred_dir.mkdir(parents=True, exist_ok=True)
        for i, pred in enumerate(pred_arrays):
            np.save(pred_dir / f"prediction_{i:04d}.npy", pred)
        logger.info("Predictions` saved to {}", pred_dir)

    return results
