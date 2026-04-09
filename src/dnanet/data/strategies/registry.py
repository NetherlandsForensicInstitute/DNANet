"""Strategy registry — central access point for active strategies.

Design pattern: **Service Locator**
    The registry holds the currently active scaling and dataset strategies.
    It's configured once at startup (typically in ``cli.py`` or the task
    runner) and then queried by components that need strategy-specific
    behavior (e.g. ``HIDImage``, ``Panel``).

    Why not dependency injection everywhere? Some deep call chains (e.g.
    inside ``Panel.fill_allele_bins``) make pure DI impractical without
    threading a context object through every function. The registry is a
    pragmatic compromise: explicit configuration, global read access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from dnanet.data.strategies.scaling import ScalingStrategy, get_scaling_strategy


if TYPE_CHECKING:
    from dnanet.data.strategies.datasets.dataset import DatasetStrategy


class StrategyRegistry:
    """Holds the active kit (scaling) and dataset strategies.

    Class-level state — configured once, read many times.
    """

    _scaling_strategy: ScalingStrategy | None = None
    _dataset_strategy: type[DatasetStrategy] | None = None

    # -- Configuration ---------------------------------------------------- #

    @classmethod
    def configure_kit(cls, strategy: ScalingStrategy | str, **kwargs) -> None:
        """Set the active scaling strategy.

        Args:
            strategy: Either a ``ScalingStrategy`` instance or a name
                (e.g. ``"PPF6C"``) to look up via ``get_scaling_strategy``.
        """
        if isinstance(strategy, ScalingStrategy):
            cls._scaling_strategy = strategy
        elif isinstance(strategy, str):
            cls._scaling_strategy = get_scaling_strategy(strategy, **kwargs)
        else:
            raise TypeError(f"Expected ScalingStrategy or str, got {type(strategy)}")
        logger.debug(f"Set scaling strategy: {strategy}")

    @classmethod
    def configure_dataset(cls, strategy: type[DatasetStrategy] | str) -> None:
        """Set the active dataset strategy.

        Args:
            strategy: Either a ``DatasetStrategy`` subclass or a name
                (e.g. ``"NFI_RND"``, ``"PROVEDIT"``) to look up.
        """
        if isinstance(strategy, str):
            from dnanet.data.strategies.datasets import get_dataset_strategy
            cls._dataset_strategy = get_dataset_strategy(strategy)
        else:
            cls._dataset_strategy = strategy

    # -- Access ----------------------------------------------------------- #

    @classmethod
    def get_scaling_strategy(cls) -> ScalingStrategy:
        """Return the active scaling strategy.

        Raises:
            RuntimeError: If no strategy has been configured.
        """
        if cls._scaling_strategy is None:
            raise RuntimeError(
                "No scaling strategy configured. Call StrategyRegistry.configure_kit() first."
            )
        return cls._scaling_strategy

    @classmethod
    def get_dataset_strategy(cls) -> type[DatasetStrategy]:
        """Return the active dataset strategy class.

        Raises:
            RuntimeError: If no strategy has been configured.
        """
        if cls._dataset_strategy is None:
            raise RuntimeError(
                "No dataset strategy configured. "
                "Call StrategyRegistry.configure_dataset() first."
            )
        return cls._dataset_strategy

    @classmethod
    def reset(cls) -> None:
        """Clear all registered strategies (useful in tests)."""
        cls._scaling_strategy = None
        cls._dataset_strategy = None
