"""Shared test fixtures for DNANet."""

import numpy as np
import pytest

from dnanet.core import Allele, Annotation, Marker, Panel, Prediction


@pytest.fixture
def sample_alleles() -> tuple[Allele, ...]:
    """A handful of alleles for testing."""
    return (
        Allele(name="12", base_pair=120.0, left_bin=0.4, right_bin=0.4, height=1500.0),
        Allele(name="15", base_pair=132.0, left_bin=0.5, right_bin=0.5, height=2200.0),
        Allele(name="13.2", base_pair=125.6, left_bin=0.4, right_bin=0.4, height=800.0),
    )


@pytest.fixture
def sample_marker(sample_alleles) -> Marker:
    """A single autosomal marker."""
    return Marker(name="D3S1358", dye_row=0, alleles=sample_alleles)


@pytest.fixture
def amel_marker() -> Marker:
    """The amelogenin sex marker."""
    return Marker(
        name="AMEL",
        dye_row=1,
        alleles=(
            Allele(name="X", base_pair=107.0, left_bin=0.5, right_bin=0.5, height=3000.0),
        ),
    )


@pytest.fixture
def sample_panel(sample_marker, amel_marker) -> Panel:
    """A minimal panel with two markers."""
    return Panel(markers=[sample_marker, amel_marker])


@pytest.fixture
def segmentation_mask() -> np.ndarray:
    """A small fake segmentation mask."""
    return np.zeros((5, 100), dtype=np.float32)
