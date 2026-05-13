from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightning as L
import numpy as np
import torch
from hydra.utils import get_class, instantiate
from loguru import logger
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset

from dnanet.data.image import HIDImage
from dnanet.data.transformer import AlleleMetadataTransformer, TransformDataCallable
from dnanet.evaluation.visualization import coerce_class_map, plot_profile
from dnanet.tasks.train import _build_logger

if TYPE_CHECKING:
    from dnanet.modules.base import BaseTaskModule


class _DebugPredictionDataset(Dataset):
    """Wrap a raw HID dataset with the transform expected by a checkpoint."""

    def __init__(
        self,
        dataset: Dataset,
        transformer: TransformDataCallable[HIDImage],
    ) -> None:
        self.dataset = dataset
        self.metadata_transformer = AlleleMetadataTransformer(transformer)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        image = self._get_image(index)
        inputs, targets, metadata = self.metadata_transformer(image)
        metadata = dict(metadata)
        metadata["scanpoint_annotation"] = (
            image.annotation.data if image.annotation is not None else None
        )
        return inputs, targets, metadata

    def collate_fn(
        self,
        batch: list[tuple[Any, Any, dict[str, Any]]],
    ) -> tuple[Any, Any, list[dict[str, Any]]]:
        return self.metadata_transformer.collate_fn(batch)

    def _get_image(self, index: int) -> HIDImage:
        if hasattr(self.dataset, "get_image"):
            image = self.dataset.get_image(index)
        else:
            image = self.dataset[index]

        if not isinstance(image, HIDImage):
            raise TypeError(
                "The debug task expects a HIDImage dataset. "
                f"Received {type(image)!r} at index {index}."
            )
        return image


def _load_checkpoint_config(checkpoint_path: str | Path) -> DictConfig:
    checkpoint_dir = Path(checkpoint_path).parent.parent
    checkpoint_config_path = checkpoint_dir / "config.yaml"
    if not checkpoint_config_path.exists():
        raise ValueError(
            f"Config file not found in checkpoint directory: {checkpoint_config_path}"
        )
    logger.info("Loaded checkpoint config from {}", checkpoint_config_path)
    return OmegaConf.load(checkpoint_config_path)


def _clear_transform_fields(node: Any) -> Any:
    if isinstance(node, dict):
        cleaned: dict[str, Any] = {}
        for key, value in node.items():
            cleaned[key] = None if key == "transform" else _clear_transform_fields(value)
        return cleaned

    if isinstance(node, list):
        return [_clear_transform_fields(value) for value in node]

    return node


def _instantiate_raw_dataset(cfg: DictConfig, checkpoint_cfg: DictConfig) -> Dataset:
    dataset_cfg = OmegaConf.select(cfg, "data.dataset")
    if dataset_cfg is None:
        dataset_cfg = OmegaConf.select(checkpoint_cfg, "data.dataset")
    if dataset_cfg is None:
        raise ValueError("Debug requires a data.dataset configuration.")

    raw_dataset_cfg = OmegaConf.create(
        _clear_transform_fields(
            OmegaConf.to_container(dataset_cfg, resolve=False),
        )
    )
    return instantiate(raw_dataset_cfg)


def _instantiate_checkpoint_transform(checkpoint_cfg: DictConfig) -> TransformDataCallable[HIDImage]:
    transform_cfg = OmegaConf.select(checkpoint_cfg, "data.dataset.transform")
    if transform_cfg is None:
        transform_cfg = OmegaConf.select(checkpoint_cfg, "train.data_transform")
    if transform_cfg is None:
        raise ValueError(
            "Checkpoint config does not define a data transform for debug prediction."
        )

    transformer = instantiate(transform_cfg)
    if not isinstance(transformer, TransformDataCallable):
        raise TypeError(
            "Checkpoint transform must be a TransformDataCallable, "
            f"received {type(transformer)!r}."
        )
    return transformer


def _load_checkpoint_module(
    checkpoint_cfg: DictConfig,
    checkpoint_path: str | Path,
) -> BaseTaskModule:
    network = instantiate(checkpoint_cfg.model.architecture)
    loss_fn = instantiate(checkpoint_cfg.model.loss)
    module_class: type[BaseTaskModule] = get_class(
        checkpoint_cfg.train.lightning_module._target_
    )
    module = module_class.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        model=network,
        optimizer=None,
        loss_fn=loss_fn,
    )
    module.eval()
    logger.info("Model loaded: {}", type(network).__name__)
    return module


def _build_predict_dataloader(
    dataset: _DebugPredictionDataset,
    *,
    num_examples: int,
    num_workers: int,
) -> DataLoader:
    subset = Subset(dataset, range(num_examples))
    return DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def _as_2d_array(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[-1] == 1:
        return array[..., 0]
    if array.ndim == 1:
        return array[np.newaxis, :]
    return array


def _prediction_for_plot(
    prediction: torch.Tensor | np.ndarray,
    signal_shape: tuple[int, int],
) -> np.ndarray:
    if isinstance(prediction, torch.Tensor):
        pred_array = prediction.detach().cpu().numpy()
    else:
        pred_array = np.asarray(prediction)

    return coerce_class_map(pred_array, signal_shape=signal_shape, source="prediction")


def run(
    cfg: DictConfig,
    dataset: Dataset | None = None,
) -> None:
    """Load a checkpoint, run a few predictions, and plot them."""
    L.seed_everything(cfg.seed, workers=True)

    checkpoint_path = cfg.get("checkpoint")
    if not checkpoint_path:
        raise ValueError(
            "Debug requires a checkpoint path. "
            "Set it via: dnanet task=debug checkpoint=/path/to/model.ckpt"
        )

    checkpoint_cfg = _load_checkpoint_config(checkpoint_path)
    module = _load_checkpoint_module(checkpoint_cfg, checkpoint_path)

    if dataset is None:
        dataset = _instantiate_raw_dataset(cfg, checkpoint_cfg)

    transform = _instantiate_checkpoint_transform(checkpoint_cfg)
    predict_dataset = _DebugPredictionDataset(dataset, transform)

    num_examples = min(int(cfg.get("debug_num_examples", 5)), len(predict_dataset))
    if num_examples <= 0:
        raise ValueError("Debug requires at least one sample to plot.")

    num_workers = int(
        cfg.get(
            "debug_num_workers",
            OmegaConf.select(cfg, "train.num_workers")
            or OmegaConf.select(checkpoint_cfg, "train.num_workers")
            or 0,
        )
    )
    dataloader = _build_predict_dataloader(
        predict_dataset,
        num_examples=num_examples,
        num_workers=num_workers,
    )

    trainer = L.Trainer(
        default_root_dir=cfg.output_dir,
        enable_progress_bar=True,
        logger=_build_logger(cfg),
        devices=1,
    )

    logger.info(
        "Running debug predictions for {} sample(s) from {}",
        num_examples,
        type(dataset).__name__,
    )
    predictions = trainer.predict(
        module,
        dataloaders=dataloader,
        return_predictions=True,
    )

    for index in range(num_examples):
        _inputs, _targets, metadata = predict_dataset[index]
        signal = _as_2d_array(np.asarray(metadata["signal_image"]))
        annotation = metadata["scanpoint_annotation"]
        if annotation is not None:
            annotation = coerce_class_map(
                np.asarray(annotation),
                signal_shape=signal.shape,
                source="annotation",
            )

        prediction = _prediction_for_plot(predictions[index], signal.shape)
        figure = plot_profile(
            signal,
            annotation=annotation,
            prediction=prediction,
            title=str(metadata["path"]),
            figsize=(20, 10),
        )
        plt.show()
        plt.close(figure)
