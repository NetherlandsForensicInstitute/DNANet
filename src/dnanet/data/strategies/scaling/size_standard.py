"""Internal size standard definitions for forensic DNA kits.

A size standard is a set of DNA fragments with known sizes (in base pairs)
that are co-electrophoresed with the sample. By detecting these fragments'
peaks, the software can build a base-pair-to-scan-point calibration curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SizeStandard:
    """Expected base-pair positions for a size standard.

    Attributes:
        expected_bps: 1D array of expected fragment sizes in base pairs.
    """

    expected_bps: np.ndarray


# GeneScan 600 LIZ — first 2 values omitted (drowned by primer flare)
GENESCAN_600_LIZ = SizeStandard(
    expected_bps=np.array(
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
)

# WEN ILS — used by PowerPlex Fusion 6C
WEN_ILS = SizeStandard(
    expected_bps=np.array(
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
)

# Synthetic variant of GeneScan 600 LIZ (last 7 values excluded)
SYNTHETIC_GENESCAN_600_LIZ = SizeStandard(expected_bps=GENESCAN_600_LIZ.expected_bps[:-7])
