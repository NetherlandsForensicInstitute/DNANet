"""Training task — runs a single training run with Lightning Trainer.

Design pattern: **Facade**
    This module is the single entry point for training. It reads the
    composed Hydra config and wires together:
    - Model architecture (from ``cfg.model``)
    - Loss function (from ``cfg.model.loss``)
    - Lightning module (SegmentationModule)
    - DataModule (DNANetDataModule)
    - Callbacks (early stopping, checkpointing)
    - Logger (MLflow or TensorBoard)
    - Trainer

Usage::

    dnanet task=train model=unet training.max_epochs=20

The Hydra config is composed from ``conf/config.yaml`` defaults.
"""

from __future__ import annotations

import lightning as L
from hydra.utils import instantiate
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from loguru import logger
from omegaconf import DictConfig

from dnanet.modules.segmentation import SegmentationModule


def _build_callbacks(cfg: DictConfig) -> list[L.Callback]:
    """Build Lightning callbacks from training config."""
    callbacks: list[L.Callback] = []

    # Early stopping
    es_cfg = cfg.training.get("early_stopping")
    if es_cfg:
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg.monitor,
                patience=es_cfg.patience,
                min_delta=es_cfg.get("min_delta", 0.0),
                mode=es_cfg.get("mode", "min"),
                verbose=True,
            )
        )

    # Model checkpointing
    ckpt_cfg = cfg.training.get("checkpoint")
    if ckpt_cfg:
        callbacks.append(
            ModelCheckpoint(
                monitor=ckpt_cfg.monitor,
                save_top_k=ckpt_cfg.get("save_top_k", 1),
                mode=ckpt_cfg.get("mode", "min"),
                dirpath=f"{cfg.output_dir}/checkpoints",
                filename="best-{epoch:02d}-{val/loss:.4f}",
            )
        )

    return callbacks


def _build_logger(cfg: DictConfig) -> L.pytorch.loggers.Logger | None:
    """Build a Lightning logger from the logging config group."""
    log_cfg = cfg.get("logging")
    if log_cfg is None:
        return None

    # MLflow
    if "tracking_uri" in log_cfg:
        return L.pytorch.loggers.MLFlowLogger(
            experiment_name=log_cfg.get("experiment_name", "dnanet"),
            tracking_uri=log_cfg.tracking_uri,
            log_model=log_cfg.get("log_model", False),
        )

    # CSV Logger or others that can be instantiated
    elif log_cfg.get("logger", {}).get("_target_", False):
        return instantiate(log_cfg.logger)
    
    return None



def run(cfg: DictConfig) -> None:
    """Run model training.

    Args:
        cfg: Composed Hydra config with sections: model, training, data,
            logging, seed, output_dir.
    """
    L.seed_everything(cfg.seed, workers=True)

    # -- Model architecture + loss -----------------------------------------
    model_cfg = cfg.model
    network = instantiate(model_cfg.architecture)
    loss_fn = instantiate(model_cfg.loss)

    logger.info("Model: {}", type(network).__name__)
    logger.info(
        "Parameters: {:,}",
        sum(p.numel() for p in network.parameters()),
    )

    # -- Lightning module --------------------------------------------------
    module = SegmentationModule(
        model=network,
        loss_fn=loss_fn,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.get("weight_decay", 0.0),
        scheduler_gamma=cfg.training.get("scheduler", {}).get("gamma", 1.0),
    )

    # -- Data --------------------------------------------------------------
    # DataModule creation is deferred to the CLI layer (cli.py) which has
    # access to the dataset. For now, we expect it to be passed in or
    # instantiated from config.
    # TODO: Wire up DataModule instantiation from cfg.data in Phase 5.
    logger.info("Training config: {} epochs, lr={}, batch_size={}",
                cfg.training.max_epochs,
                cfg.training.learning_rate,
                cfg.training.batch_size)

    # -- Trainer -----------------------------------------------------------
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=_build_callbacks(cfg),
        logger=_build_logger(cfg),
        default_root_dir=cfg.output_dir,
        deterministic=True,
        enable_progress_bar=True,
    )

    logger.info("Trainer ready — launch with trainer.fit(module, datamodule=dm)")

    # Full integration (datamodule wiring) will be completed in Phase 5.
    # For now, the module + trainer are returned for programmatic use.
    return trainer, module
