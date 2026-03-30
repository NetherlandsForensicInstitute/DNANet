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

import lightning as L
from loguru import logger
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from dnanet.data.dataset import InMemoryDataset


# ---------------------------------------------------------------------------
# Module registry: maps training type to Lightning module class
# ---------------------------------------------------------------------------

_MODULE_REGISTRY: dict[str, str] = {
    'segmentation': 'dnanet.modules.segmentation.SegmentationModule',
    'classification': 'dnanet.modules.classification.ClassificationModule',
    'reconstruction': 'dnanet.modules.reconstruction.ReconstructionModule',
    'peaknet': 'dnanet.modules.peaknet.PeakNetModule',
}


def _resolve_module_class(training_type: str):
    """Import and return the Lightning module class for the given training type."""
    if training_type == 'segmentation':
        from dnanet.modules.segmentation import SegmentationModule

        return SegmentationModule
    elif training_type == 'classification':
        from dnanet.modules.classification import ClassificationModule

        return ClassificationModule
    elif training_type == 'reconstruction':
        from dnanet.modules.reconstruction import ReconstructionModule

        return ReconstructionModule
    elif training_type == 'peaknet':
        from dnanet.modules.peaknet import PeakNetModule

        return PeakNetModule
    else:
        raise ValueError(
            f'Unknown training type: {training_type!r}. Choose from: {list(_MODULE_REGISTRY.keys())}'
        )


def _build_callbacks(cfg: DictConfig) -> list[L.Callback]:
    """Build Lightning callbacks from training config."""
    callbacks: list[L.Callback] = []

    # Early stopping
    es_cfg = cfg.training.get('early_stopping')
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
    ckpt_cfg = cfg.training.get('checkpoint')
    if ckpt_cfg:
        callbacks.append(
            ModelCheckpoint(
                monitor=ckpt_cfg.monitor,
                save_top_k=ckpt_cfg.get('save_top_k', 1),
                mode=ckpt_cfg.get('mode', 'min'),
                dirpath=f'{cfg.output_dir}/checkpoints',
                filename='best-{epoch:02d}-{val/loss:.4f}',
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


def _build_module(
    cfg: DictConfig,
    network,
    loss_fn,
) -> L.LightningModule:
    """Build the appropriate Lightning module for the training type."""
    training_type = cfg.training.get('type', 'segmentation')
    ModuleClass = _resolve_module_class(training_type)

    # Common kwargs
    kwargs = {
        'model': network,
        'loss_fn': loss_fn,
        'learning_rate': cfg.training.learning_rate,
        'weight_decay': cfg.training.get('weight_decay', 0.0),
        'scheduler_gamma': cfg.training.get('scheduler', {}).get('gamma', 1.0),
    }

    # Module-specific kwargs
    if training_type == 'classification':
        kwargs['num_classes'] = cfg.model.architecture.get('num_classes', 2)
    elif training_type == 'peaknet':
        kwargs['num_classes'] = cfg.model.architecture.get('num_classes', 2)
    elif training_type == 'segmentation':
        kwargs['threshold'] = cfg.training.get('threshold', 0.5)

    return ModuleClass(**kwargs)


def _build_datamodule(
    cfg: DictConfig,
    dataset: InMemoryDataset,
) -> L.LightningDataModule:
    """Build the appropriate DataModule for the training type.

    Automatically selects the right DataModule based on the training type:
    - ``segmentation`` → :class:`DNANetDataModule` (full images)
    - ``classification`` → :class:`PeakDataModule` (peak windows)
    - ``peaknet`` → :class:`PeakNetDataModule` (full images + peaks)
    - ``reconstruction`` → :class:`ReconstructionDataModule` (preprocessed)
    """
    training_type = cfg.training.get('type', 'segmentation')

    common_kwargs = dict(
        batch_size=cfg.training.batch_size,
        val_fraction=cfg.training.get('val_fraction', 0.2),
        num_workers=cfg.training.get('num_workers', 0),
        seed=cfg.seed,
    )

    if training_type == 'peaknet':
        from dnanet.data.datamodule import PeakNetDataModule

        return PeakNetDataModule(
            dataset=dataset,
            peak_threshold=cfg.model.get('peak_threshold', 25),
            window_size=cfg.model.get('window_size', 120),
            include_max_pool_dyes=cfg.model.architecture.get(
                'peak_classifier',
                {},
            ).get('include_max_pool_dyes', False),
            **common_kwargs,
        )

    elif training_type == 'classification':
        # Check if we have a pre-extracted PeakWindowDataset
        from dnanet.data.peak_dataset import PeakWindowDataset

        if isinstance(dataset, PeakWindowDataset):
            from dnanet.data.datamodule import PeakDataModule

            return PeakDataModule(dataset=dataset, **common_kwargs)

        # Otherwise create PeakWindowDataset on the fly from HID profiles
        peak_cfg = cfg.data.get('peak_extraction', {})
        peak_dataset = PeakWindowDataset(
            base_dataset=dataset,
            threshold=peak_cfg.get('threshold', 40),
            window_size=peak_cfg.get('window_size', 120),
            labels=list(peak_cfg.get('labels', ['noise', 'allele'])),
            include_max_pool_dyes=peak_cfg.get('include_max_pool_dyes', False),
            preprocess=peak_cfg.get('preprocess', True),
            smooth_keep_factor=peak_cfg.get('smooth_keep_factor', 0.4),
            log_scale=peak_cfg.get('log_scale', True),
            max_rfu_value=peak_cfg.get('max_rfu_value', 32768),
            use_ground_truth=peak_cfg.get('use_ground_truth', True),
        )
        from dnanet.data.datamodule import PeakDataModule

        return PeakDataModule(dataset=peak_dataset, **common_kwargs)

    elif training_type == 'reconstruction':
        from dnanet.data.datamodule import ReconstructionDataModule

        preprocessing = cfg.data.get('preprocessing', {})
        return ReconstructionDataModule(
            dataset=dataset,
            log_scale=True,
            max_rfu=preprocessing.get('max_rfu_value', None),
            n_dyes=cfg.model.architecture.get('in_channels', 5),
            **common_kwargs,
        )

    else:
        # Default: segmentation
        from dnanet.data.datamodule import DNANetDataModule

        return DNANetDataModule(dataset=dataset, **common_kwargs)


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


def run(cfg: DictConfig) -> tuple[L.Trainer, L.LightningModule]:
    """Run model training.

    Args:
        cfg: Composed Hydra config with sections: model, training, data,
            logging, seed, output_dir.

    Returns:
        Tuple of (trainer, module) for programmatic access.
    """
    L.seed_everything(cfg.seed, workers=True)

    # -- Model architecture + loss -----------------------------------------
    model_cfg = cfg.model
    network = instantiate(model_cfg.architecture)
    loss_fn = instantiate(model_cfg.loss)

    # Load pre-trained sub-model weights (e.g. autoencoder, peak classifier)
    _load_pretrained_weights(network, cfg)

    logger.info('Model: {}', type(network).__name__)
    logger.info(
        'Parameters: {:,}',
        sum(p.numel() for p in network.parameters()),
    )

    # -- Lightning module --------------------------------------------------
    module = _build_module(cfg, network, loss_fn)

    # -- Resume from checkpoint if provided --------------------------------
    ckpt_path = cfg.get('checkpoint')
    if ckpt_path:
        logger.info('Will resume training from checkpoint: {}', ckpt_path)

    # -- Data --------------------------------------------------------------
    datamodule = None
    data_cfg = cfg.get('data')
    if data_cfg and data_cfg.get('root'):
        from dnanet.data.loading import load_dataset

        dataset = load_dataset(data_cfg)
        datamodule = _build_datamodule(cfg, dataset)

    logger.info(
        'Training config: {} epochs, lr={}, batch_size={}',
        cfg.training.max_epochs,
        cfg.training.learning_rate,
        cfg.training.batch_size,
    )

    # -- Trainer -----------------------------------------------------------
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=_build_callbacks(cfg),
        logger=_build_logger(cfg),
        default_root_dir=cfg.output_dir,
        deterministic=True,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    if datamodule is not None:
        ckpt_path = cfg.get('checkpoint')
        trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
        logger.info('Training complete!')
        _save_config(cfg, cfg.output_dir)
    else:
        logger.info(
            'No data config found. Use run_with_data() or '
            'pass a data config group (e.g. data=dnanet_rd).'
        )

    return trainer, module


def run_with_data(
    cfg: DictConfig,
    dataset: InMemoryDataset,
) -> tuple[L.Trainer, L.LightningModule]:
    """Run training with a pre-loaded dataset.

    This is the primary programmatic entry point. It builds the full
    training pipeline and actually trains the model.

    Args:
        cfg: Composed Hydra config.
        dataset: A loaded :class:`~dnanet.data.dataset.InMemoryDataset`.

    Returns:
        Tuple of (trainer, module) after training completes.
    """
    L.seed_everything(cfg.seed, workers=True)

    # -- Model architecture + loss -----------------------------------------
    model_cfg = cfg.model
    network = instantiate(model_cfg.architecture)
    loss_fn = instantiate(model_cfg.loss)

    _load_pretrained_weights(network, cfg)

    logger.info('Model: {}', type(network).__name__)
    logger.info(
        'Parameters: {:,}',
        sum(p.numel() for p in network.parameters()),
    )

    # -- Lightning module --------------------------------------------------
    module = _build_module(cfg, network, loss_fn)

    # -- Data --------------------------------------------------------------
    datamodule = _build_datamodule(cfg, dataset)

    # -- Trainer -----------------------------------------------------------
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=_build_callbacks(cfg),
        logger=_build_logger(cfg),
        default_root_dir=cfg.output_dir,
        deterministic=True,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    ckpt_path = cfg.get('checkpoint')
    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
    logger.info('Training complete!')
    _save_config(cfg, cfg.output_dir)

    return trainer, module
