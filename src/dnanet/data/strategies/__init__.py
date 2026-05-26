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

    At runtime, the user selects strategies via Hydra config. Both kit scaling
    and dataset strategies are injected into the data pipeline directly.
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
    'DatasetStrategy',
    'FileCategory',
    'NFIRnDStrategy',
    'ProvedItStrategy',
    'STRKit',
    'SizeStandard',
    'WEN_ILS',
    'GENESCAN_600_LIZ',
    'SYNTHETIC_GENESCAN_600_LIZ',
    'GlobalFilerStrategy',
    'PowerPlexFusion6CStrategy',
    'PowerplexY23',
    'ScalingStrategy',
    'SizeStandardParseResult',
]
