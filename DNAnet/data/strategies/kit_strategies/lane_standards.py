from enum import Enum
import numpy as np
from typing import Union
from numpy.typing import NDArray


BASE_PAIR_START, BASE_PAIR_END = 60, 480
RESCALE_SIZE = 4096
VAL_THRESHOLD = 5.0

# Enum for internal size standards
class InternalSizeStandard(Enum):
    GENESCAN_600_LIZ = "GENESCAN_600_LIZ"
    WEN_ILS = "WEN_ILS"
    SYNTHETIC_GENESCAN_600_LIZ = "SYNTHETIC_GENESCAN_600_LIZ"
    # Add more standards as needed


# Size standard base pair values for different kits
# For GENESCAN_600_LIZ, I have omitted the first 2 values because they are drowned out by primer flare.
GENESCAN_600_LIZ_BPS: NDArray[np.int_] = np.array(
    [20, 40, 60, 80, 100, 114, 120, 140, 160, 180, 200, 
     214, 220, 240, 250, 260, 280, 300, 314, 320, 340, 360, 
     380, 400, 414, 420, 440, 460, 480, 500, 514, 520, 540, 
     560, 580, 600], dtype=int)[2:]

WEN_ILS_BPS: NDArray[np.int_] = np.array([65, 80, 100, 120, 140, 160, 180,
                                          200, 225, 250, 275, 300, 325,
                                          350, 375, 400, 425, 450, 475], dtype=int)

# For synthetic data, we use the same values as GENESCAN_600_LIZ, except we exclude the last 7 values
# Why? because last 7 values are non-existent in syntetic data
SYNTHETIC_GENESCAN_600_LIZ_BPS: NDArray[np.int_] = GENESCAN_600_LIZ_BPS[:-7]


# Mapping from enum or string to BPS array
SIZE_STANDARD_BPS_MAP = {
    InternalSizeStandard.GENESCAN_600_LIZ: GENESCAN_600_LIZ_BPS,
    InternalSizeStandard.WEN_ILS: WEN_ILS_BPS,
    InternalSizeStandard.SYNTHETIC_GENESCAN_600_LIZ: SYNTHETIC_GENESCAN_600_LIZ_BPS,
    # Add more mappings as needed
}



def get_size_standard_bps(standard: Union[InternalSizeStandard, str]) -> np.ndarray:
    """
    Get the BPS array for a given size standard enum or string.
    """
    if isinstance(standard, str):
        try:
            standard = InternalSizeStandard[standard.upper()]
        except KeyError:
            raise ValueError(
                f"Unknown size standard '{standard}'. "
                f"Valid options: {list_available_size_standards()}"
            )
    return SIZE_STANDARD_BPS_MAP[standard]

def list_available_size_standards() -> list[str]:
    """Return a list of valid size standard strings."""
    return [e.name for e in InternalSizeStandard]