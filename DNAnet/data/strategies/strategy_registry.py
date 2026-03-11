from __future__ import annotations
from typing import Annotated, Union, TYPE_CHECKING
from DNAnet.data.strategies.kit_strategies.kit import Kit, KitStandards, get_standard_kit

if TYPE_CHECKING:
    from DNAnet.data.strategies.dataset_strategies.Abstract_DatasetStrategy import DatasetStrategy
    

class StrategyRegistry:
    _kit_strategy: Kit | None = None
    _dataset_strategy: DatasetStrategy | None = None

    # --- configure ---

    @classmethod
    def configure_kit(cls, strategy: Kit | str):
        if isinstance(strategy, Kit):
            cls._kit_strategy = strategy
        elif isinstance(strategy, str):
            if strategy not in KitStandards.__members__:
                raise ValueError(f"Unknown kit standard: '{strategy}'. Valid: {list(KitStandards.__members__)}")
            cls._kit_strategy = get_standard_kit(strategy)
        else:
            raise TypeError(f"Expected Kit or str, got {type(strategy)}")

    @classmethod
    def configure_dataset(cls, strategy: DatasetStrategy):
        cls._dataset_strategy = strategy

    # --- get ---

    @classmethod
    def get_kit(cls) -> Kit:
        if cls._kit_strategy is None:
            raise RuntimeError("No kit strategy configured. Call configure_kit() first.")
        return cls._kit_strategy

    @classmethod
    def get_dataset(cls) -> DatasetStrategy:
        if cls._dataset_strategy is None:
            raise RuntimeError("No dataset strategy configured. Call configure_dataset() first.")
        return cls._dataset_strategy