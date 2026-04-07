"""Dataset and scaling strategy exports.

Design pattern: **Strategy**
    Different forensic DNA kits (PowerPlex Fusion 6C, GlobalFiler) and datasets
    (NFI R&D, ProvedIt) require different processing logic for:
    - Parsing the internal size standard (different peak patterns, validation)
    - Rescaling the electropherogram to base-pair coordinates
    - Categorizing files (sample vs. ladder vs. control)
    - Parsing annotations (different file formats per dataset)

    The Strategy pattern encapsulates each variant behind a common interface:
    - ``ScalingStrategy`` (ABC) — kit-specific size standard parsing & rescaling
    - ``DatasetStrategy`` (ABC) — dataset-specific file handling & annotations

    At runtime, the user selects a strategy via Hydra config, and it's injected
    into the data pipeline. No ``if/elif`` chains needed.

Design pattern: **Registry**
    ``StrategyRegistry`` is a simple service locator that holds the active
    strategies. It's configured once at startup and provides global access.
    This replaces scattered global state and makes dependencies explicit.
"""

from dnanet.data.strategies.datasets import (
    DATASET_STRATEGIES,
    DatasetStrategy,
    FileCategory,
    NFIRnDStrategy,
    ProvedItStrategy,
    get_dataset_strategy,
)
from dnanet.data.strategies.registry import StrategyRegistry
from dnanet.data.strategies.scaling import (
    GENESCAN_600_LIZ,
    SCALING_STRATEGIES,
    GlobalFilerStrategy,
    PowerPlexFusion6CStrategy,
    PowerplexY23,
    ScalingStrategy,
    STRKit,
    SYNTHETIC_GENESCAN_600_LIZ,
    SizeStandard,
    SizeStandardParseResult,
    WEN_ILS,
    get_scaling_strategy,
)

__all__ = [
    "DATASET_STRATEGIES",
    "DatasetStrategy",
    "FileCategory",
    "NFIRnDStrategy",
    "ProvedItStrategy",
    "get_dataset_strategy",
    "SCALING_STRATEGIES",
    "STRKit",
    "SizeStandard",
    "WEN_ILS",
    "GENESCAN_600_LIZ",
    "SYNTHETIC_GENESCAN_600_LIZ",
    "GlobalFilerStrategy",
    "PowerPlexFusion6CStrategy",
    "PowerplexY23",
    "ScalingStrategy",
    "SizeStandardParseResult",
    "get_scaling_strategy",
    "StrategyRegistry",
]
