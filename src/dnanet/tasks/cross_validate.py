"""Cross-validation task — k-fold training and evaluation.

Usage::

    dnanet task=cross_validate splitting=cross_validate data=<config> model=<config>
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict

import numpy as np
import mlflow
import lightning as L
from loguru import logger
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from flatten_dict import flatten
from torch.utils.data import DataLoader, default_collate

from dnanet.tasks.train import (
    _build_logger,
    _build_callbacks,
    _validate_dataset_for_callbacks,
    _configure_module_for_callbacks,
)
from dnanet.data.splitting import KFoldSplitResult, dataset_splitter


if TYPE_CHECKING:
    from torch.utils.data import Dataset


@contextmanager
def _null_context():
    yield (None, None)


def _is_mlflow_logger(cfg: DictConfig) -> bool:
    log_cfg = cfg.get("logging")
    return bool(log_cfg and log_cfg.get("logger_type") == "mlflow")


@contextmanager
def _mlflow_parent_run(cfg: DictConfig, k_folds: int):
    """Context manager: start parent MLflow run, yield mlflow_cfg, end on exit."""
    mlflow_cfg = cfg.logging.mlflow
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "dnanet"))
    with mlflow.start_run(run_name=f"cross_validate_{k_folds}fold") as parent_run:
        mlflow.log_params(flatten(cfg, reducer=lambda x, y: f'{x}/{y}' if x else f'{y}'))
        mlflow.set_tags({
            "model": cfg.get("model", {}).get("name", "unknown"),
            "data": cfg.data.get("name", "unknown"),
            "task": "cross_validate",
            "k_folds": str(k_folds),
        })
        yield mlflow_cfg, parent_run


def _build_fold_logger(
    cfg: DictConfig, mlflow_cfg: DictConfig | None, fold_idx: int
) -> L.pytorch.loggers.Logger | None:
    if mlflow_cfg is None:
        return _build_logger(cfg)
    nested_run = mlflow.start_run(
        nested=True, run_name=f"fold_{fold_idx + 1}"
    )
    mlflow.set_tag("fold", fold_idx + 1)
    return L.pytorch.loggers.MLFlowLogger(
        experiment_name=mlflow_cfg.get("experiment_name", "dnanet"),
        tracking_uri=mlflow_cfg.get("tracking_uri", "mlruns"),
        log_model=mlflow_cfg.get("log_model", False),
        run_id=nested_run.info.run_id,
    )


def run(
    cfg: DictConfig,
    dataset: Dataset | None = None,
) -> dict[str, list | dict]:
    """Run k-fold cross-validation.

    Args:
        cfg: Composed Hydra config. Must include ``splitting.k_folds``.

    Returns:
        Dictionary with ``"per_fold"`` and ``"aggregate"`` keys.
    """
    L.seed_everything(cfg.seed, workers=True)
    
    # Validate config
    if not dataset:
        data_cfg = cfg.get("data")
        if data_cfg is None:
            raise ValueError(
                "Cross-validation requires a dataset. "
                "Set it via: dnanet task=cross_validate data=your_dataset"
            )
        dataset = instantiate(data_cfg.dataset)
    _validate_dataset_for_callbacks(cfg, dataset)

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

    use_mlflow = _is_mlflow_logger(cfg)
    ctx = _mlflow_parent_run(cfg, k_folds) if use_mlflow else None

    with (ctx if ctx is not None else _null_context()) as mlflow_info:
        mlflow_cfg = mlflow_info[0] if use_mlflow else None

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
            _configure_module_for_callbacks(cfg, module)

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

            fold_logger = _build_fold_logger(cfg, mlflow_cfg, fold_idx)
            trainer = L.Trainer(
                max_epochs=cfg.train.max_epochs,
                callbacks=_build_callbacks(cfg),
                logger=fold_logger,
                default_root_dir=str(fold_dir),
                deterministic=False,
                enable_progress_bar=True,
                log_every_n_steps=1,
                check_val_every_n_epoch=1,
            )
            if trainer.logger is not None:
                trainer.logger.log_hyperparams(cfg)

            interrupted = False
            try:
                trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
            except KeyboardInterrupt:
                logger.warning("Training interrupted at fold {}", fold_idx + 1)
                interrupted = True
            finally:
                if use_mlflow:
                    mlflow.end_run()  # ends nested fold run

            fold_metrics = {
                k: float(v.item() if hasattr(v, "item") else v)
                for k, v in trainer.callback_metrics.items()
            }
            all_fold_metrics.append(fold_metrics)
            for name, value in fold_metrics.items():
                agg_metrics[name].append(value)

            fold_dir.joinpath("metrics.json").write_text(json.dumps(fold_metrics, indent=2))
            logger.info("Fold {} complete: {}", fold_idx + 1, fold_metrics)

            if interrupted:
                break

        aggregate: dict[str, float] = {}
        for name, values in agg_metrics.items():
            mean = float(np.mean(values))
            std = float(np.std(values))
            aggregate[f"{name}_mean"] = mean
            aggregate[f"{name}_std"] = std
            logger.info("  {}: {:.4f} ± {:.4f}", name, mean, std)

        if use_mlflow:
            mlflow.log_metrics(aggregate)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2))
    output_dir.joinpath("config.yaml").write_text(OmegaConf.to_yaml(cfg))
    logger.info("Cross-validation complete. Results saved to {}", output_dir)

    return {"per_fold": all_fold_metrics, "aggregate": aggregate}
