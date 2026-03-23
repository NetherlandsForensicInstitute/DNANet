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
from pathlib import Path

import lightning as L
import numpy as np
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from dnanet.evaluation.metrics import (
    allele_f1_score,
    allele_precision,
    allele_recall,
    average_binary_iou,
    pixel_f1_score,
    pixel_precision,
    pixel_recall,
)


# Metric name -> callable mapping
_METRIC_REGISTRY: dict[str, callable] = {
    "pixel_precision": pixel_precision,
    "pixel_recall": pixel_recall,
    "pixel_f1_score": pixel_f1_score,
    "average_binary_iou": average_binary_iou,
}

# Allele metrics require a different interface (marker sequences)
_ALLELE_METRIC_REGISTRY: dict[str, callable] = {
    "allele_precision": allele_precision,
    "allele_recall": allele_recall,
    "allele_f1_score": allele_f1_score,
}


def _compute_pixel_metrics(
    ground_truths: list[np.ndarray],
    predictions: list[np.ndarray],
    metric_names: list[str],
) -> dict[str, float]:
    """Compute pixel-level metrics."""
    results = {}
    for name in metric_names:
        if name in _METRIC_REGISTRY:
            value = _METRIC_REGISTRY[name](ground_truths, predictions)
            results[name] = float(value)
            logger.info("  {}: {:.4f}", name, value)
    return results


def _save_results(
    results: dict[str, float],
    output_dir: str,
    filename: str = "metrics.json",
) -> Path:
    """Save metric results to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path / filename
    metrics_path.write_text(json.dumps(results, indent=2))
    return metrics_path


def run(cfg: DictConfig) -> dict[str, float]:
    """Run model evaluation.

    Args:
        cfg: Composed Hydra config. Must include ``checkpoint`` path.

    Returns:
        Dictionary of metric name -> value.
    """
    L.seed_everything(cfg.seed, workers=True)

    # -- Validate config ---------------------------------------------------
    checkpoint_path = cfg.get("checkpoint")
    if not checkpoint_path:
        raise ValueError(
            "Evaluation requires a checkpoint path. "
            "Set it via: dnanet task=evaluate checkpoint=/path/to/model.ckpt"
        )
    logger.info("Loading checkpoint: {}", checkpoint_path)

    # -- Build model and load checkpoint -----------------------------------
    model_cfg = cfg.model
    network = instantiate(model_cfg.architecture)
    loss_fn = instantiate(model_cfg.loss)

    # Determine module type from training config
    training_type = cfg.training.get("type", "segmentation")

    from dnanet.tasks.train import _resolve_module_class, _build_module
    ModuleClass = _resolve_module_class(training_type)

    # Load from checkpoint
    module = ModuleClass.load_from_checkpoint(
        checkpoint_path,
        model=network,
        loss_fn=loss_fn,
    )
    module.eval()
    logger.info("Model loaded: {}", type(network).__name__)

    # -- Data --------------------------------------------------------------
    data_cfg = cfg.get("data")
    if data_cfg and data_cfg.get("root"):
        from dnanet.data.loading import load_dataset
        dataset = load_dataset(data_cfg)
        return run_with_data(cfg, dataset)

    logger.warning(
        "No data config found. Use run_with_data(cfg, dataset) or "
        "pass a data config group (e.g. data=dnanet_rd)."
    )
    return {}


def run_with_data(
    cfg: DictConfig,
    dataset: "InMemoryDataset",
) -> dict[str, float]:
    """Run evaluation with a pre-loaded dataset.

    This is the primary programmatic entry point. It loads a checkpoint,
    runs predictions on the dataset, computes metrics, and saves results.

    Args:
        cfg: Composed Hydra config. Must include ``checkpoint`` path.
        dataset: A loaded :class:`~dnanet.data.dataset.InMemoryDataset`.

    Returns:
        Dictionary of metric name -> value.
    """
    from dnanet.data.datamodule import DNANetDataModule
    from dnanet.data.dataset import InMemoryDataset

    L.seed_everything(cfg.seed, workers=True)

    # -- Validate config ---------------------------------------------------
    checkpoint_path = cfg.get("checkpoint")
    if not checkpoint_path:
        raise ValueError(
            "Evaluation requires a checkpoint path. "
            "Set it via: dnanet task=evaluate checkpoint=/path/to/model.ckpt"
        )
    logger.info("Loading checkpoint: {}", checkpoint_path)

    # -- Build model and load checkpoint -----------------------------------
    model_cfg = cfg.model
    network = instantiate(model_cfg.architecture)
    loss_fn = instantiate(model_cfg.loss)

    training_type = cfg.training.get("type", "segmentation")
    from dnanet.tasks.train import _resolve_module_class
    ModuleClass = _resolve_module_class(training_type)

    module = ModuleClass.load_from_checkpoint(
        checkpoint_path,
        model=network,
        loss_fn=loss_fn,
    )
    module.eval()
    logger.info("Model loaded: {}", type(network).__name__)

    # -- Data --------------------------------------------------------------
    datamodule = DNANetDataModule(
        dataset=dataset,
        batch_size=cfg.training.batch_size,
        val_fraction=cfg.training.get("val_fraction", 0.2),
        num_workers=cfg.training.get("num_workers", 0),
        seed=cfg.seed,
    )
    datamodule.setup("test")

    # -- Predict -----------------------------------------------------------
    trainer = L.Trainer(
        default_root_dir=cfg.output_dir,
        enable_progress_bar=True,
        logger=False,
    )

    logger.info("Running predictions...")
    predictions_list = trainer.predict(module, datamodule.val_dataloader())

    # Collect predictions and ground truths as numpy arrays
    pred_arrays: list[np.ndarray] = []
    gt_arrays: list[np.ndarray] = []

    for batch_preds in predictions_list:
        pred_arrays.extend(
            batch_preds[i].cpu().numpy()
            for i in range(batch_preds.shape[0])
        )

    for i in range(len(datamodule._val_dataset)):
        _, y = datamodule._val_dataset[i]
        gt_arrays.append(y.numpy())

    # -- Compute metrics ---------------------------------------------------
    eval_cfg = cfg.get("evaluation", {})
    metric_names = list(eval_cfg.get("metrics", []))
    results: dict[str, float] = {}

    if metric_names:
        logger.info("Computing metrics...")
        results = _compute_pixel_metrics(gt_arrays, pred_arrays, metric_names)
    else:
        logger.warning("No metrics configured in evaluation config.")

    # -- Save results ------------------------------------------------------
    if results:
        metrics_path = _save_results(results, cfg.output_dir)
        logger.info("Results saved to {}", metrics_path)

    # Save config
    config_path = Path(cfg.output_dir) / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(OmegaConf.to_yaml(cfg))

    # Optionally save predictions
    if eval_cfg.get("save_predictions", False):
        pred_dir = Path(cfg.output_dir) / eval_cfg.get("predictions_dir", "predictions")
        pred_dir.mkdir(parents=True, exist_ok=True)
        for i, pred in enumerate(pred_arrays):
            np.save(pred_dir / f"prediction_{i:04d}.npy", pred)
        logger.info("Predictions saved to {}", pred_dir)

    return results
