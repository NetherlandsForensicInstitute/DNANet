from __future__ import annotations
from typing import TYPE_CHECKING

from DNAnet.data.strategies.kit_strategies.scaling_strategy import ScalingStrategy, ScalingStrategyOptions, \
    get_scaling_strategy

if TYPE_CHECKING:
    from DNAnet.data.strategies.dataset_strategies.Abstract_DatasetStrategy import (
        DatasetStrategy,
    )


class StrategyRegistry:
    """
    Central registry for user-configured strategies.
    Only supports one active kit strategy and one active dataset strategy at a time, as these are typically
    configured at the start of a run and then used globally.
    """
    _scaling_strategy: ScalingStrategy | None = None
    _dataset_strategy: DatasetStrategy | None = None

    # --- configure ---

    @classmethod
    def configure_kit(cls, strategy: ScalingStrategy | str, **kwargs):
        if isinstance(strategy, ScalingStrategy):
            cls._scaling_strategy = strategy
        elif isinstance(strategy, str):
            if strategy not in ScalingStrategyOptions.__members__:
                raise ValueError(
                    f"Unknown scaling strategy: '{strategy}'. Valid: {list(ScalingStrategyOptions.__members__)}"
                )
            cls._scaling_strategy = get_scaling_strategy(strategy, **kwargs)
        else:
            raise TypeError(f"Expected ScalingStrategy or str, got {type(strategy)}")

    @classmethod
    def configure_dataset(cls, strategy: DatasetStrategy):
        cls._dataset_strategy = strategy

    # --- get ---

    @classmethod
    def get_scaling_strategy(cls) -> ScalingStrategy:
        if cls._scaling_strategy is None:
            raise RuntimeError(
                "No scaling strategy configured. Call configure_kit() first."
            )
        return cls._scaling_strategy

    @classmethod
    def get_dataset(cls) -> DatasetStrategy:
        if cls._dataset_strategy is None:
            raise RuntimeError(
                "No dataset strategy configured. Call configure_dataset() first."
            )
        return cls._dataset_strategy
