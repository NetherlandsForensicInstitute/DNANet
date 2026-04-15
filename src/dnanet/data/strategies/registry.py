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

    _dataset_strategy: type[DatasetStrategy] | None = None



    # -- Access ----------------------------------------------------------- #



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
