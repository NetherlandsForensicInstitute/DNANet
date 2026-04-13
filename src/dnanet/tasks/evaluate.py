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
from typing import Any
from pathlib import Path

import numpy as np
import lightning as L
from loguru import logger
from omegaconf import OmegaConf, DictConfig
from hydra.utils import get_class, instantiate
from torch.utils.data import Subset, Dataset

from dnanet.modules import BaseTaskModule


def _raw_eval_items(dataset: Dataset) -> list[Any]:
    """Return untransformed validation items when the dataset exposes them."""
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        if hasattr(base_dataset, "data"):
            return [base_dataset.data[i] for i in dataset.indices]
        return [base_dataset[i] for i in dataset.indices]

    if hasattr(dataset, "data"):
        return list(dataset.data)

    return [dataset[i] for i in range(len(dataset))]


def _as_2d_array(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[-1] == 1:
        return array[..., 0]
    return array


def _compute_allele_metrics(
    raw_items: list[Any],
    predictions: list[np.ndarray],
    metric_cfg: dict[str, Any] | DictConfig | None,
    allele_caller_cfg: dict[str, Any] | DictConfig | None,
) -> dict[str, float]:
    """Compute configured allele metrics when raw caller inputs are available."""
    if allele_caller_cfg is None:
        logger.warning(
            "Allele metrics are configured, but evaluation.allele_caller is missing.",
        )
        return {}

    allele_caller = instantiate(allele_caller_cfg)
    ground_truth_markers = []
    predicted_markers = []
    skipped_items = 0

    for raw_item, prediction in zip(raw_items, predictions, strict=True):
        meta = getattr(raw_item, "meta", None)
        called_alleles = meta.get("called_alleles") if hasattr(meta, "get") else None
        panel = getattr(raw_item, "adjusted_panel", None)
        signal_image = getattr(raw_item, "data", None)

        if called_alleles is None or panel is None or signal_image is None:
            skipped_items += 1
            continue

        predicted_markers.append(
            allele_caller.call_alleles(
                prediction_image=_as_2d_array(prediction),
                signal_image=_as_2d_array(signal_image),
                scaler=raw_item.scaler,
                panel=panel,
            )
        )
        ground_truth_markers.append(tuple(called_alleles))

    if skipped_items:
        logger.warning(
            "Skipped allele metrics for {} validation samples without raw allele metadata.",
            skipped_items,
        )

    if not ground_truth_markers:
        logger.warning("No allele-call annotations were available for evaluation.")
        return {}

    return _compute_metrics(metric_cfg, ground_truth_markers, predicted_markers)


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

    # -- Build model and load checkpoint -----------------------------------
    model_cfg = cfg.model
    network = instantiate(model_cfg.architecture)
    loss_fn = instantiate(model_cfg.loss)

    eval_metrics = instantiate(cfg.evaluation.metrics, _convert_="partial")

    # -- Lightning module --------------------------------------------------
    # noinspection PyTypeChecker
    module_class: type[BaseTaskModule] = get_class(cfg.evaluation.lightning_module)


    module = module_class.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        metrics=eval_metrics,
        model=network,
        optimizer=None,
        loss_fn=loss_fn
    )
    module.eval()
    logger.info("Model loaded: {}", type(network).__name__)

    # -- Data --------------------------------------------------------------
    if not dataset:
        data_cfg = cfg.get('data')
        dataset = instantiate(data_cfg.dataset)

    datamodule = DNANetDataModule(
        dataset=dataset,
        batch_size=cfg.evaluation.get("batch_size", 1),
        val_fraction= None, # always use the entire dataset for evaluation
        num_workers=cfg.evaluation.get("num_workers", 0),
        seed=cfg.seed,
    )
    datamodule.setup("test")

    # -- Predict -----------------------------------------------------------
    predictor = L.Trainer(
        default_root_dir=cfg.output_dir,
        enable_progress_bar=True,
        logger=False,
    )

    logger.info("Running predictions...")
    predictions_list = predictor.predict(module, datamodule.train_dataloader())

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
    ## TODO compute pixel and allele scores
    eval_cfg = cfg.get("evaluation", {})
    pixel_metric_cfg = eval_cfg.get("pixel_metrics")
    allele_metric_cfg = eval_cfg.get("allele_metrics")
    results: dict[str, float] = {}

    has_pixel_metrics = pixel_metric_cfg is not None and len(pixel_metric_cfg) > 0
    has_allele_metrics = allele_metric_cfg is not None and len(allele_metric_cfg) > 0

    if has_pixel_metrics:
        logger.info("Computing pixel metrics...")
        results.update(_compute_pixel_metrics(gt_arrays, pred_arrays, pixel_metric_cfg))

    if has_allele_metrics:
        logger.info("Computing allele metrics...")
        raw_items = _raw_eval_items(datamodule._val_dataset)
        results.update(
            _compute_allele_metrics(
                raw_items,
                pred_arrays,
                allele_metric_cfg,
                eval_cfg.get("allele_caller"),
            )
        )

    if not has_pixel_metrics and not has_allele_metrics:
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
