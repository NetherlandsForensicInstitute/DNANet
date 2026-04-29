"""Cross-validation task — k-fold training and evaluation.

Usage::

    dnanet task=cross_validate splitting=cross_validate data=<config> model=<config>
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
from typing import TYPE_CHECKING, cast

import numpy as np
import lightning as L
from loguru import logger
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from torch.utils.data import DataLoader, default_collate

from dnanet.tasks.train import _build_logger, _build_callbacks
from dnanet.data.splitting import dataset_splitter, KFoldSplitResult


if TYPE_CHECKING:
    from torch.utils.data import Dataset


def run(cfg: DictConfig) -> dict[str, list | dict]:
    """Run k-fold cross-validation.

    Args:
        cfg: Composed Hydra config. Must include ``splitting.k_folds``.

    Returns:
        Dictionary with ``"per_fold"`` and ``"aggregate"`` keys.
    """
    L.seed_everything(cfg.seed, workers=True)

    data_cfg = cfg.get("data")
    if data_cfg is None:
        raise ValueError(
            "Cross-validation requires a dataset. "
            "Set it via: dnanet task=cross_validate data=your_dataset"
        )

    dataset = instantiate(data_cfg.dataset)
    return run_with_data(cfg, dataset)


def run_with_data(
    cfg: DictConfig,
    dataset: Dataset,
) -> dict[str, list | dict]:
    """Run k-fold cross-validation with a pre-loaded dataset.

    Args:
        cfg: Composed Hydra config. Must include ``splitting.k_folds``.
        dataset: A loaded dataset.

    Returns:
        Dictionary with ``"per_fold"`` and ``"aggregate"`` keys.
    """
    L.seed_everything(cfg.seed, workers=True)

    if not cfg.splitting.get("k_folds"):
        raise ValueError(
            "task=cross_validate requires k-fold splitting. "
            "Run with: dnanet task=cross_validate splitting=cross_validate ..."
        )

    split_kwargs: dict = OmegaConf.to_container(cfg.splitting, resolve=True)  # type: ignore[assignment]
    folds, _test_set = cast(KFoldSplitResult, dataset_splitter(dataset, **split_kwargs))

    _transform = getattr(dataset, "transform", None)
    collate_fn = _transform.collate_fn if _transform is not None else default_collate

    k_folds = len(folds)
    logger.info("Starting {}-fold cross-validation", k_folds)

    all_fold_metrics: list[dict[str, float]] = []
    agg_metrics: dict[str, list[float]] = defaultdict(list)

    for fold_idx, (train_set, val_set) in enumerate(folds):
        logger.info("=" * 60)
        logger.info("Fold {}/{}", fold_idx + 1, k_folds)
        logger.info("Train: {} samples, Val: {} samples", len(train_set), len(val_set))

        fold_dir = Path(cfg.output_dir) / f"fold_{fold_idx + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        network = instantiate(cfg.model.architecture)
        optimizer = instantiate(cfg.train.optimizer, params=network.parameters())
        scheduler = (
            instantiate(cfg.train.scheduler, optimizer=optimizer)
            if cfg.train.get("scheduler")
            else None
        )
        module = instantiate(
            cfg.train.lightning_module,
            model=network,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            _convert_="partial",
        )

        train_loader = DataLoader(
            train_set,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=cfg.train.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        trainer = L.Trainer(
            max_epochs=cfg.train.max_epochs,
            callbacks=_build_callbacks(cfg),
            logger=_build_logger(cfg),
            default_root_dir=str(fold_dir),
            deterministic=False,
            enable_progress_bar=True,
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
        )

        try:
            trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        except KeyboardInterrupt:
            logger.warning("Training interrupted at fold {}", fold_idx + 1)
            break

        fold_metrics = {
            k: float(v.item() if hasattr(v, "item") else v)
            for k, v in trainer.callback_metrics.items()
        }
        all_fold_metrics.append(fold_metrics)
        for name, value in fold_metrics.items():
            agg_metrics[name].append(value)

        fold_dir.joinpath("metrics.json").write_text(json.dumps(fold_metrics, indent=2))
        logger.info("Fold {} complete: {}", fold_idx + 1, fold_metrics)

    aggregate: dict[str, float] = {}
    for name, values in agg_metrics.items():
        mean = float(np.mean(values))
        std = float(np.std(values))
        aggregate[f"{name}_mean"] = mean
        aggregate[f"{name}_std"] = std
        logger.info("  {}: {:.4f} ± {:.4f}", name, mean, std)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2))
    output_dir.joinpath("config.yaml").write_text(OmegaConf.to_yaml(cfg))
    logger.info("Cross-validation complete. Results saved to {}", output_dir)

    return {"per_fold": all_fold_metrics, "aggregate": aggregate}
