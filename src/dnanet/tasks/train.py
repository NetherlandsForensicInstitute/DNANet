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

from calendar import c
from typing import TYPE_CHECKING
from pathlib import Path

import torch
import lightning as L
from loguru import logger
from omegaconf import OmegaConf, DictConfig, ListConfig
from hydra.utils import instantiate
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from dnanet.data.transformer import AlleleMetadataTransformer
from dnanet.data.utils import get_class_weights

if TYPE_CHECKING:
    from torch.utils.data import Dataset


_ALLELE_METRICS_CALLBACK_TARGET = 'dnanet.evaluation.callbacks.AlleleMetricsCallback'
_ALLELE_METRICS_TRAIN_TYPES = frozenset({'segmentation', 'segmentation_mc', 'peaknet'})

def _build_callbacks(cfg: DictConfig) -> list[L.Callback]:
    """Build Lightning callbacks from training config."""
    callbacks: list[L.Callback] = []
    train_cfg = cfg.get('train')
    if train_cfg is None:
        return callbacks

    # Early stopping
    es_cfg = train_cfg.get('early_stopping')
    if es_cfg:
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg.monitor,
                patience=es_cfg.patience,
                min_delta=es_cfg.get('min_delta', 0.0),
                mode=es_cfg.get('mode', 'min'),
                verbose=False,
                check_finite=True,
            )
        )

    # Model checkpointing
    ckpt_cfg = train_cfg.get('checkpoint')
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

    callbacks.extend(
        _instantiate_configured_callbacks(
            train_cfg.get('callbacks'),
            config_path='train.callbacks',
        )
    )
    return callbacks


def _instantiate_configured_callbacks(
    callbacks_cfg: DictConfig | ListConfig | None,
    *,
    config_path: str,
) -> list[L.Callback]:
    """Instantiate callbacks from a Hydra mapping/list config."""
    if not callbacks_cfg:
        return []

    if isinstance(callbacks_cfg, DictConfig):
        callback_specs = callbacks_cfg.values()
    elif isinstance(callbacks_cfg, ListConfig):
        callback_specs = callbacks_cfg
    else:
        raise TypeError(f'{config_path} must be a mapping or list of Hydra callback configs.')

    return [instantiate(callback_cfg, _convert_='partial') for callback_cfg in callback_specs]


def _configured_callback_targets(
    callbacks_cfg: DictConfig | ListConfig | None,
    *,
    config_path: str,
) -> list[str]:
    """Return configured callback targets without instantiating them."""
    if not callbacks_cfg:
        return []

    if isinstance(callbacks_cfg, DictConfig):
        callback_specs = callbacks_cfg.values()
    elif isinstance(callbacks_cfg, ListConfig):
        callback_specs = callbacks_cfg
    else:
        raise TypeError(f'{config_path} must be a mapping or list of Hydra callback configs.')

    targets: list[str] = []
    for callback_cfg in callback_specs:
        target = callback_cfg.get('_target_')
        if target:
            targets.append(str(target))
    return targets


def _uses_validation_allele_metrics(cfg: DictConfig) -> bool:
    """Return whether training callbacks require validation allele metadata/predictions."""
    train_cfg = cfg.get('train')
    if train_cfg is None:
        return False

    targets = _configured_callback_targets(
        train_cfg.get('callbacks'),
        config_path='train.callbacks',
    )
    return _ALLELE_METRICS_CALLBACK_TARGET in targets


def _validate_dataset_for_callbacks(cfg: DictConfig, dataset: Dataset) -> None:
    """Validate that configured callbacks are compatible with the dataset transform."""
    if not _uses_validation_allele_metrics(cfg):
        return

    train_cfg = cfg.get('train')
    train_type = str(train_cfg.get('type', '')) if train_cfg is not None else ''
    if train_type not in _ALLELE_METRICS_TRAIN_TYPES:
        raise ValueError(
            'AlleleMetricsCallback validation support is only available for '
            f'{sorted(_ALLELE_METRICS_TRAIN_TYPES)} training types, got {train_type!r}.'
        )

    transform = getattr(dataset, 'transform', None)
    if transform is None:
        raise ValueError(
            'AlleleMetricsCallback validation support requires a dataset transform. '
            'Set model.data_transform_active with an AlleleMetadataTransformer'
        )
    if isinstance(transform, AlleleMetadataTransformer):
        return

    raise ValueError(
        'AlleleMetricsCallback validation support requires metadata batches from '
        'AlleleMetadataTransformer. Set model.data_transform_active'
    )


def _configure_module_for_callbacks(cfg: DictConfig, module: L.LightningModule) -> None:
    """Enable module callback outputs needed by configured training callbacks."""
    if _uses_validation_allele_metrics(cfg):
        module.enable_validation_callback_preds = True


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
            tags={
                'model': cfg.get('model', {}).get('name', 'Unknown'),
                'data': cfg.data.get('name', 'Unknown'),
                'task': cfg.get('train', {}).get('type', 'Unknown'),
            },
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


def _load_state_dict(checkpoint: str):
    """Load a PyTorch state dict from a checkpoint file.

    Handles both plain PyTorch checkpoints and PyTorch Lightning checkpoints. For the latter, strip ``model.`` prefix
    from parameters and discard parameters not belonging to the model.
    """
    import torch

    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    # Handle Lightning checkpoint wrapping
    if 'state_dict' in state:
        prefix = 'model.'
        state_dict = {
            k.removeprefix(prefix): v for k, v in state['state_dict'].items() if k.startswith(prefix)
        }
        return state_dict

    return state


def _load_pretrained_weights(network, cfg: DictConfig) -> None:
    """Load pre-trained sub-model weights if checkpoint paths are given.

    Supports loading pre-trained autoencoder and/or peak classifier weights
    into a CombinedClassifier before training the combiner.
    """
    ae_ckpt = cfg.model.get('autoencoder_checkpoint')
    if ae_ckpt and hasattr(network, 'autoencoder'):
        logger.info('Loading pre-trained autoencoder from: {}', ae_ckpt)
        ae_state = _load_state_dict(ae_ckpt)
        network.autoencoder.load_state_dict(ae_state, strict=False)

    pc_ckpt = cfg.model.get('peak_classifier_checkpoint')
    if pc_ckpt and hasattr(network, 'peak_classifier'):
        logger.info('Loading pre-trained peak classifier from: {}', pc_ckpt)
        pc_state = _load_state_dict(pc_ckpt)
        network.peak_classifier.load_state_dict(pc_state, strict=False)


def _save_config(cfg: DictConfig, output_dir: str) -> None:
    """Save the resolved Hydra config to the output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config_path = output_path / 'config.yaml'
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True))
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

    # -- Data --------------------------------------------------------------
    if not dataset:
        if (data_cfg := cfg.get('data')) is None:
            raise ValueError(
                'Training requires a dataset. Set it via: dnanet task=train data=your_dataset'
            )

        dataset = instantiate(data_cfg.dataset)
    _validate_dataset_for_callbacks(cfg, dataset)

    datamodule = instantiate(cfg.train.data_module, dataset=dataset, **cfg.splitting)
    datamodule.setup('fit')
    
    
    if "num_classes" in cfg.model and cfg.get('class_balance', False):
        class_weights = get_class_weights(
            datamodule._train_dataset, 
            num_classes=cfg.model.num_classes
        )
        new_loss_fn = {
            "_target_": cfg.model.loss._target_,
            "weight": {
                "_target_": "torch.Tensor",
                "data": class_weights.tolist()
            }
        }
        cfg.train.lightning_module.loss_fn = new_loss_fn
        cfg.model.loss = new_loss_fn
        logger.info(f"Calculated and applied class weights to Loss Function")
    
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
        lr_scheduler=scheduler,
        _convert_='partial',  # convert OmegaDict to standard dict since this is not a supported type for instantiating
    )
    _configure_module_for_callbacks(cfg, module)

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

    if trainer.logger is not None:
        trainer.logger.log_hyperparams(cfg)

    # Resume training from checkpoint path if provided
    if ckpt_path := cfg.get('checkpoint'):
        logger.info('Will resume training from checkpoint: {}', ckpt_path)

    _save_config(cfg, cfg.output_dir)
    logger.info('Starting training...')
    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
    logger.info('Training complete!')
    logger.info('Saved to {}', cfg.output_dir)

    return trainer, module
