from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


# Enum for internal size standards
class InternalSizeStandardOptions(Enum):
    GENESCAN_600_LIZ = "GENESCAN_600_LIZ"
    WEN_ILS = "WEN_ILS"
    SYNTHETIC_GENESCAN_600_LIZ = "SYNTHETIC_GENESCAN_600_LIZ"


@dataclass(frozen=True)
class InternalSizeStandard:
    name: str
    expected_bps: np.ndarray



def get_internal_size_standard(internal_standard: InternalSizeStandardOptions) -> InternalSizeStandard:
    if internal_standard == InternalSizeStandardOptions.GENESCAN_600_LIZ:
        return InternalSizeStandard(InternalSizeStandardOptions.GENESCAN_600_LIZ.name, GENESCAN_600_LIZ_BPS)
    elif internal_standard == InternalSizeStandardOptions.WEN_ILS:
        return InternalSizeStandard(InternalSizeStandardOptions.WEN_ILS.name, WEN_ILS_BPS)
    elif internal_standard == InternalSizeStandardOptions.SYNTHETIC_GENESCAN_600_LIZ:
        return InternalSizeStandard(InternalSizeStandardOptions.SYNTHETIC_GENESCAN_600_LIZ.name, SYNTHETIC_GENESCAN_600_LIZ_BPS)

    raise ValueError(f"Internal standard name is not found: {internal_standard.name}")


# Size standard base pair values for different kits
# For GENESCAN_600_LIZ, I have omitted the first 2 values because they are drowned out by primer flare.
GENESCAN_600_LIZ_BPS: NDArray[np.int_] = np.array(
    [
        20,
        40,
        60,
        80,
        100,
        114,
        120,
        140,
        160,
        180,
        200,
        214,
        220,
        240,
        250,
        260,
        280,
        300,
        314,
        320,
        340,
        360,
        380,
        400,
        414,
        420,
        440,
        460,
        480,
        500,
        514,
        520,
        540,
        560,
        580,
        600,
    ],
    dtype=int,
)[2:]

WEN_ILS_BPS: NDArray[np.int_] = np.array(
    [
        65,
        80,
        100,
        120,
        140,
        160,
        180,
        200,
        225,
        250,
        275,
        300,
        325,
        350,
        375,
        400,
        425,
        450,
        475,
    ],
    dtype=int,
)

# For synthetic data, we use the same values as GENESCAN_600_LIZ, except we exclude the last 7 values
# Why? because last 7 values are non-existent in syntetic data
SYNTHETIC_GENESCAN_600_LIZ_BPS: NDArray[np.int_] = GENESCAN_600_LIZ_BPS[:-7]

