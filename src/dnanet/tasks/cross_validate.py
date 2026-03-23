"""Cross-validation task — k-fold training and evaluation.

Design pattern: **Facade**
    Orchestrates k-fold cross-validation by repeatedly running train + evaluate
    cycles and aggregating metrics across folds.

Usage::

    dnanet task=cross_validate training.n_folds=5 model=unet
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import lightning as L
import numpy as np
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from dnanet.data.splitting import split_data_in_k_folds
from dnanet.tasks.train import _build_callbacks, _build_logger, _build_module


def run(cfg: DictConfig) -> dict[str, dict[str, float]]:
    """Run k-fold cross-validation.

    Args:
        cfg: Composed Hydra config. Must include ``training.n_folds``.

    Returns:
        Dictionary with ``"per_fold"`` and ``"aggregate"`` keys containing
        metric results.

    Note:
        Dataset loading from config is not yet wired. Use
        :func:`run_with_data` with a pre-loaded dataset instead.
    """
    L.seed_everything(cfg.seed, workers=True)

    data_cfg = cfg.get("data")
    if data_cfg and data_cfg.get("root"):
        from dnanet.data.loading import load_dataset
        dataset = load_dataset(data_cfg)
        return run_with_data(cfg, dataset)

    logger.warning(
        "No data config found. Use run_with_data(cfg, dataset) or "
        "pass a data config group (e.g. data=dnanet_rd)."
    )
    return {"per_fold": [], "aggregate": {}}


def run_with_data(
    cfg: DictConfig,
    dataset: "InMemoryDataset",
) -> dict[str, dict[str, float]]:
    """Run k-fold cross-validation with a pre-loaded dataset.

    This is the primary programmatic entry point. It splits the dataset
    into k folds, trains a fresh model on each fold, evaluates on the
    held-out fold, and aggregates metrics.

    Args:
        cfg: Composed Hydra config. Must include ``training.n_folds``.
        dataset: A loaded :class:`~dnanet.data.dataset.InMemoryDataset`.

    Returns:
        Dictionary with ``"per_fold"`` and ``"aggregate"`` keys containing
        metric results.
    """
    from dnanet.data.datamodule import HIDTorchDataset
    from dnanet.data.dataset import InMemoryDataset
    from torch.utils.data import DataLoader

    L.seed_everything(cfg.seed, workers=True)

    n_folds = cfg.training.get("n_folds", 5)
    logger.info("Starting {}-fold cross-validation", n_folds)

    # -- Split into folds --------------------------------------------------
    all_items = list(dataset)
    folds = split_data_in_k_folds(all_items, n_folds, seed=cfg.seed)
    logger.info("Dataset: {} samples, split into {} folds: sizes = {}",
                len(all_items), n_folds, [len(f) for f in folds])

    # -- Per-fold train + evaluate -----------------------------------------
    all_fold_results: list[dict[str, float]] = []
    agg_metrics: dict[str, list[float]] = defaultdict(list)

    eval_cfg = cfg.get("evaluation", {})
    metric_names = list(eval_cfg.get("metrics", []))

    for fold_idx in range(n_folds):
        logger.info("=" * 60)
        logger.info("Fold {}/{}", fold_idx + 1, n_folds)
        logger.info("=" * 60)

        # Create train/test split for this fold
        test_items = folds[fold_idx]
        train_items = [
            item for i, fold in enumerate(folds) if i != fold_idx for item in fold
        ]

        logger.info("Train: {} samples, Test: {} samples",
                     len(train_items), len(test_items))

        # Build fresh model for each fold
        model_cfg = cfg.model
        network = instantiate(model_cfg.architecture)
        loss_fn = instantiate(model_cfg.loss)
        module = _build_module(cfg, network, loss_fn)

        # Optional validation split from training fold
        val_fraction = cfg.training.get("val_fraction", 0.0)
        if val_fraction > 0:
            split_idx = int(len(train_items) * (1 - val_fraction))
            val_items = train_items[split_idx:]
            train_items = train_items[:split_idx]
            val_loader = DataLoader(
                HIDTorchDataset(val_items),
                batch_size=cfg.training.batch_size,
                shuffle=False,
                pin_memory=True,
            )
        else:
            val_loader = None

        train_loader = DataLoader(
            HIDTorchDataset(train_items),
            batch_size=cfg.training.batch_size,
            shuffle=True,
            pin_memory=True,
        )

        # Fold output directory
        fold_dir = Path(cfg.output_dir) / f"fold_{fold_idx + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Train
        trainer = L.Trainer(
            max_epochs=cfg.training.max_epochs,
            callbacks=_build_callbacks(cfg),
            logger=_build_logger(cfg),
            default_root_dir=str(fold_dir),
            deterministic=True,
            enable_progress_bar=True,
        )

        try:
            trainer.fit(
                module,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            )
        except KeyboardInterrupt:
            logger.warning("Training interrupted at fold {}", fold_idx + 1)
            break

        # Evaluate on test fold
        test_dataset = HIDTorchDataset(test_items)
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            pin_memory=True,
        )

        predictions = trainer.predict(module, test_loader)
        pred_arrays = [
            predictions[b][i].cpu().numpy()
            for b in range(len(predictions))
            for i in range(predictions[b].shape[0])
        ]
        gt_arrays = [
            test_dataset[i][1].numpy()
            for i in range(len(test_dataset))
        ]

        # Compute metrics
        fold_results: dict[str, float] = {}
        if metric_names:
            from dnanet.tasks.evaluate import _compute_pixel_metrics
            fold_results = _compute_pixel_metrics(gt_arrays, pred_arrays, metric_names)

        all_fold_results.append(fold_results)
        for name, value in fold_results.items():
            agg_metrics[name].append(value)

        # Save fold results
        fold_metrics_path = fold_dir / "metrics.json"
        fold_metrics_path.write_text(json.dumps(fold_results, indent=2))
        logger.info("Fold {} results saved to {}", fold_idx + 1, fold_metrics_path)

    # -- Aggregate results -------------------------------------------------
    aggregate: dict[str, float] = {}
    for name, values in agg_metrics.items():
        mean = float(np.mean(values))
        std = float(np.std(values))
        aggregate[f"{name}_mean"] = mean
        aggregate[f"{name}_std"] = std
        logger.info("  {}: {:.4f} ± {:.4f}", name, mean, std)

    # Save aggregate results
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agg_path = output_dir / "aggregate_metrics.json"
    agg_path.write_text(json.dumps(aggregate, indent=2))
    logger.info("Aggregate results saved to {}", agg_path)

    # Save config
    config_path = output_dir / "config.yaml"
    config_path.write_text(OmegaConf.to_yaml(cfg))

    return {
        "per_fold": all_fold_results,
        "aggregate": aggregate,
    }
