"""Dataset loading from Hydra config — bridges config to HIDDataset.

This module provides ``load_dataset(cfg.data)`` which reads the data config
group and returns a fully loaded ``InMemoryDataset`` ready for training.

Usage::

    from dnanet.data.loading import load_dataset
    dataset = load_dataset(cfg.data)
    trainer, module = run_with_data(cfg, dataset)
"""

from __future__ import annotations

from loguru import logger
from omegaconf import DictConfig

from dnanet.data.dataset import InMemoryDataset
from dnanet.data.hid_dataset import HIDDataset
from dnanet.data.strategies.registry import StrategyRegistry


def load_dataset(data_cfg: DictConfig) -> InMemoryDataset:
    """Load a dataset from a Hydra data config group.

    Configures the ``StrategyRegistry`` and constructs an ``HIDDataset``
    from the config parameters.

    Args:
        data_cfg: The ``cfg.data`` section of the composed Hydra config.

    Returns:
        A loaded ``InMemoryDataset`` (specifically ``HIDDataset``).

    Example config (``conf/data/dnanet_rd.yaml``)::

        kit: PPF6C
        dataset_strategy: NFI_RND
        root: data/2p_5p_Dataset_NFI/Raw data .HID files
        annotations_path: data/2p_5p_Dataset_NFI/txt_annotations_2024
        hid_to_annotations_path: data/2p_5p_Dataset_NFI/2p_5p_hid_to_annotation.csv
        best_ladder_paths_csv: data/2p_5p_Dataset_NFI/best_ladder_paths_DTH.csv
        ladder_alleles_csv: data/2p_5p_Dataset_NFI/ladder_alleles.csv
        analysis_threshold_type: DTH
        data_loading_strategy: superior
        adjustment_of_annotations: complete
    """
    # Configure strategies
    kit = data_cfg.get("kit", "PPF6C")
    dataset_strategy = data_cfg.get("dataset_strategy", "NFI_RND")

    logger.info("Configuring kit={}, dataset_strategy={}", kit, dataset_strategy)
    StrategyRegistry.configure_kit(kit)
    StrategyRegistry.configure_dataset(dataset_strategy)

    # Build HIDDataset
    dataset = HIDDataset(
        root=data_cfg.root,
        analysis_threshold_type=data_cfg.get("analysis_threshold_type", "DTH"),
        adjustment_of_annotations=data_cfg.get("adjustment_of_annotations"),
        limit=data_cfg.get("limit"),
        skip_if_invalid_ladder=data_cfg.get("skip_if_invalid_ladder", False),
        include_size_standard=data_cfg.get("include_size_standard", False),
        data_loading_strategy=data_cfg.get("data_loading_strategy", "superior"),
    )

    logger.info("Dataset loaded: {} images", len(dataset))
    return dataset
