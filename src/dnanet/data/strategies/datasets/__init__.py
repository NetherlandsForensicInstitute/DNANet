"""Concrete dataset strategy implementations."""

from dnanet.data.strategies.datasets.nfi_rnd import NFIRnDStrategy
from dnanet.data.strategies.datasets.provedit import ProvedItStrategy

DATASET_STRATEGIES: dict[str, type] = {
    "NFI_RND": NFIRnDStrategy,
    "PROVEDIT": ProvedItStrategy,
}


def get_dataset_strategy(name: str) -> type:
    """Look up a dataset strategy class by name.

    Raises:
        ValueError: If the name is not recognized.
    """
    cls = DATASET_STRATEGIES.get(name.upper())
    if cls is None:
        raise ValueError(
            f"Unknown dataset strategy '{name}'. "
            f"Available: {list(DATASET_STRATEGIES.keys())}"
        )
    return cls


__all__ = [
    "NFIRnDStrategy",
    "ProvedItStrategy",
    "DATASET_STRATEGIES",
    "get_dataset_strategy",
]
