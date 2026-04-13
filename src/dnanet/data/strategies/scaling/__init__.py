"""Scaling strategy interfaces and concrete implementations."""

from dnanet.data.strategies.scaling.kit import STRKit
from dnanet.data.strategies.scaling.scaling import (
    ScalingStrategy,
    SizeStandardParseResult,
)
from dnanet.data.strategies.scaling.globalfiler import GlobalFilerStrategy
from dnanet.data.strategies.scaling.powerplex_y23 import PowerplexY23
from dnanet.data.strategies.scaling.size_standard import (
    WEN_ILS,
    GENESCAN_600_LIZ,
    SYNTHETIC_GENESCAN_600_LIZ,
    SizeStandard,
)
from dnanet.data.strategies.scaling.powerplex_fusion_6c import (
    PowerPlexFusion6CStrategy,
)


SCALING_STRATEGIES: dict[str, type[ScalingStrategy]] = {
    'PPF6C': PowerPlexFusion6CStrategy,
    'GLOBALFILER': GlobalFilerStrategy,
    'Y23': PowerplexY23,
}

def get_scaling_strategy(name: str, **kwargs) -> ScalingStrategy:
    """Instantiate a scaling strategy by name.

    Args:
        name: Strategy name (e.g. "PPF6C", "GLOBALFILER").
        **kwargs: Additional arguments passed to the strategy constructor.

    Raises:
        ValueError: If the name is not recognized.
    """
    cls = SCALING_STRATEGIES.get(name.upper())
    if cls is None:
        raise ValueError(
            f"Unknown scaling strategy '{name}'. Available: {list(SCALING_STRATEGIES.keys())}"
        )
    return cls(**kwargs)

__all__ = [
    "SCALING_STRATEGIES",
    "STRKit",
    "SizeStandard",
    "WEN_ILS",
    "GENESCAN_600_LIZ",
    "SYNTHETIC_GENESCAN_600_LIZ",
    "ScalingStrategy",
    "SizeStandardParseResult",
    "get_scaling_strategy",
    "GlobalFilerStrategy",
    "PowerPlexFusion6CStrategy",
    "PowerplexY23",
]
