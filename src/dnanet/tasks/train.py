"""Training task — runs a single training run with Lightning Trainer.

Design pattern: **Facade**
    This module is the single entry point for training. It reads the
    composed Hydra config and wires together:
    - Model architecture (from ``cfg.model``)
    - Loss function (from ``cfg.model.loss``)
    - Lightning module (auto-selected by task type)
    - DataModule (auto-selected by task type)
    - Callbacks (early stopping, checkpointing)
    - Logger (MLflow, CSV, or TensorBoard)
    - Trainer

Usage::

    dnanet task=train model=unet training=segmentation
    dnanet task=train model=autoencoder training=reconstruction
    dnanet task=train model=peak_classifier data=peaks_rd training=classification
    dnanet task=train model=peaknet training=peaknet
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import lightning as L
from loguru import logger
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


if TYPE_CHECKING:
    from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Module registry: maps training type to Lightning module class
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from torch.utils.data import Dataset




def _build_callbacks(cfg: DictConfig) -> list[L.Callback]:
    """Build Lightning callbacks from training config."""
    # callbacks: list[L.Callback] = [EpochConsoleLogger()]
    callbacks: list[L.Callback] = []

    # Early stopping
    es_cfg = cfg.train.get('early_stopping')
    if es_cfg:
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg.monitor,
                patience=es_cfg.patience,
                min_delta=es_cfg.get('min_delta', 0.0),
                mode=es_cfg.get('mode', 'min'),
                verbose=True,
            )
        )

    # Model checkpointing
    ckpt_cfg = cfg.train.get('checkpoint')
    if ckpt_cfg:
        callbacks.append(
            ModelCheckpoint(
                monitor=ckpt_cfg.monitor,
                save_top_k=ckpt_cfg.get('save_top_k', 1),
                mode=ckpt_cfg.get('mode', 'min'),
                dirpath=f'{cfg.output_dir}/checkpoints',
                filename='epoch-{epoch:03d}-val_loss-{val/loss:.4f}',
                auto_insert_metric_name=False,
            )
        )

    return callbacks


def _build_logger(cfg: DictConfig) -> L.pytorch.loggers.Logger | None:
    """Build a Lightning logger from the logging config group."""
    log_cfg = cfg.get('logging')
    if log_cfg is None:
        return None

    logger_type = log_cfg.get('logger_type', '')

    if logger_type == 'mlflow' and 'mlflow' in log_cfg:
        mlflow_cfg = log_cfg.mlflow
        return L.pytorch.loggers.MLFlowLogger(
            experiment_name=mlflow_cfg.get('experiment_name', 'dnanet'),
            tracking_uri=mlflow_cfg.get('tracking_uri', 'mlruns'),
            log_model=mlflow_cfg.get('log_model', False),
        )
    elif logger_type == 'tensorboard':
        return L.pytorch.loggers.TensorBoardLogger(
            save_dir=cfg.output_dir,
            name='tensorboard',
        )
    elif logger_type == 'csv':
        return L.pytorch.loggers.CSVLogger(
            save_dir=cfg.output_dir,
            name='csv_logs',
        )

    # Hydra _target_ instantiation
    target_cfg = log_cfg.get('logger')
    if target_cfg and target_cfg.get('_target_'):
        return instantiate(target_cfg)

    return None


def _load_pretrained_weights(network, cfg: DictConfig) -> None:
    """Load pre-trained sub-model weights if checkpoint paths are given.

    Supports loading pre-trained autoencoder and/or peak classifier weights
    into a CombinedClassifier before training the combiner.
    """
    import torch

    ae_ckpt = cfg.model.get('autoencoder_checkpoint')
    pc_ckpt = cfg.model.get('peak_classifier_checkpoint')

    if ae_ckpt and hasattr(network, 'autoencoder'):
        logger.info('Loading pre-trained autoencoder from: {}', ae_ckpt)
        state = torch.load(ae_ckpt, map_location='cpu', weights_only=True)
        # Handle Lightning checkpoint wrapping
        if 'state_dict' in state:
            prefix = 'model.'
            ae_state = {
                k.removeprefix(prefix): v
                for k, v in state['state_dict'].items()
                if k.startswith(prefix)
            }
            network.autoencoder.load_state_dict(ae_state, strict=False)
        else:
            network.autoencoder.load_state_dict(state, strict=False)

    if pc_ckpt and hasattr(network, 'peak_classifier'):
        logger.info('Loading pre-trained peak classifier from: {}', pc_ckpt)
        state = torch.load(pc_ckpt, map_location='cpu', weights_only=True)
        if 'state_dict' in state:
            prefix = 'model.'
            pc_state = {
                k.removeprefix(prefix): v
                for k, v in state['state_dict'].items()
                if k.startswith(prefix)
            }
            network.peak_classifier.load_state_dict(pc_state, strict=False)
        else:
            network.peak_classifier.load_state_dict(state, strict=False)


def _save_config(cfg: DictConfig, output_dir: str) -> None:
    """Save the resolved Hydra config to the output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config_path = output_path / 'config.yaml'
    config_path.write_text(OmegaConf.to_yaml(cfg))
    logger.info('Config saved to {}', config_path)


def run(
    cfg: DictConfig,
    dataset: Dataset | None = None,
) -> tuple[L.Trainer, L.LightningModule]:
    """Run model training.

    Args:
        cfg: Composed Hydra config with sections: model, training, data,
            logging, seed, output_dir.
        dataset: The dataset to use, when none is specified, the dataset
            is loaded from the config.

    Returns:
        Tuple of (trainer, module) for programmatic access.
    """
    L.seed_everything(cfg.seed, workers=True)

    # -- Model architecture + loss -----------------------------------------
    model_cfg = cfg.model
    network = instantiate(model_cfg.architecture)

    # Load pre-trained sub-model weights (e.g. autoencoder, peak classifier)
    _load_pretrained_weights(network, cfg)

    logger.info('Model: {}', type(network).__name__)
    logger.info(
        'Parameters: {:,}',
        sum(p.numel() for p in network.parameters()),
    )

    # -- Optimizer and scheduler -------------------------------------------
    optimizer = instantiate(cfg.train.optimizer, params=network.parameters())

    if cfg.train.get('scheduler'):
        scheduler = instantiate(cfg.train.scheduler, optimizer=optimizer)
    else:
        scheduler = None


    # -- Lightning module --------------------------------------------------
    module = instantiate(
        cfg.train.lightning_module,
        model=network,
        optimizer=optimizer,
        scheduler=scheduler,
        _convert_='partial' # convert OmegaDict to standard dict since this is not a supported type for instantiating
    )


    # -- Resume from checkpoint if provided --------------------------------
    ckpt_path = cfg.get('checkpoint')
    if ckpt_path:
        logger.info('Will resume training from checkpoint: {}', ckpt_path)


    # -- Data --------------------------------------------------------------
    if not dataset:
        if (data_cfg := cfg.get('data')) is None:
            raise ValueError(
                'Training requires a dataset. Set it via: dnanet task=train data=your_dataset'
            )

        dataset = instantiate(data_cfg.dataset)

    datamodule = instantiate(cfg.train.data_module, dataset=dataset)
    datamodule.setup('fit')

    logger.info(
        'Training config: {} epochs, lr={}, batch_size={}',
        cfg.train.max_epochs,
        cfg.train.learning_rate,
        cfg.train.batch_size,
    )

    # -- Trainer -----------------------------------------------------------
    trainer = L.Trainer(
        max_epochs=cfg.train.max_epochs,
        callbacks=_build_callbacks(cfg),
        logger=_build_logger(cfg),
        default_root_dir=cfg.output_dir,
        deterministic=False,
        enable_progress_bar=True,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
    )

    if datamodule is not None:
        ckpt_path = cfg.get('checkpoint')
        logger.info('Starting training...')
        trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
        logger.info('Training complete!')
        _save_config(cfg, cfg.output_dir)
    else:
        logger.info(
            'No data config found. Use run_with_data() or '
            'pass a data config group (e.g. data=dnanet_rd).'
        )

    return trainer, module
