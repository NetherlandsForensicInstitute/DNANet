"""Kit and dataset strategies for handling different forensic DNA kits and datasets.

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

from dnanet.data.strategies.scaling import ScalingStrategy, SizeStandardParseResult
from dnanet.data.strategies.registry import StrategyRegistry

__all__ = [
    "ScalingStrategy",
    "SizeStandardParseResult",
    "StrategyRegistry",
]
