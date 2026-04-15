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

    At runtime, the user selects strategies via Hydra config. Kit scaling is
    injected into the data pipeline directly. Dataset strategy is also stored
    in ``StrategyRegistry`` for shared dataset label and split helpers.

Design pattern: **Registry**
    ``StrategyRegistry`` is a simple service locator for the active dataset
    strategy only. Scaling strategies are not stored globally.
"""

from dnanet.data.strategies.scaling import (
    WEN_ILS,
    GENESCAN_600_LIZ,
    SYNTHETIC_GENESCAN_600_LIZ,
    STRKit,
    PowerplexY23,
    SizeStandard,
    ScalingStrategy,
    GlobalFilerStrategy,
    SizeStandardParseResult,
    PowerPlexFusion6CStrategy,
)
from dnanet.data.strategies.datasets import (
    FileCategory,
    NFIRnDStrategy,
    DatasetStrategy,
    ProvedItStrategy,
)


__all__ = [
    "DatasetStrategy",
    "FileCategory",
    "NFIRnDStrategy",
    "ProvedItStrategy",
    "STRKit",
    "SizeStandard",
    "WEN_ILS",
    "GENESCAN_600_LIZ",
    "SYNTHETIC_GENESCAN_600_LIZ",
    "GlobalFilerStrategy",
    "PowerPlexFusion6CStrategy",
    "PowerplexY23",
    "ScalingStrategy",
    "SizeStandardParseResult"
]
