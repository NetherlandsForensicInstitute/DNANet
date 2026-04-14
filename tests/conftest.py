"""Shared test fixtures for DNANet."""

from pathlib import Path

import numpy as np
import pytest

from dnanet.core.panel import Panel
from dnanet.core.allele import Allele
from dnanet.core.marker import Marker
from dnanet.data.strategies.registry import StrategyRegistry


# ---------------------------------------------------------------------------
# Resource paths
# ---------------------------------------------------------------------------

TESTS_DIR = Path(__file__).parent
RESOURCES_DIR = TESTS_DIR / "resources"
PANEL_PATH = RESOURCES_DIR / "panel.xml"
LADDER_ALLELES_CSV = RESOURCES_DIR / "ladder_alleles.csv"

RD_DIR = RESOURCES_DIR / "profiles" / "RD"
PROVEDIT_DIR = RESOURCES_DIR / "PROVEDIt"


# ---------------------------------------------------------------------------
# Core domain fixtures (synthetic / in-memory)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_alleles() -> frozenset[Allele]:
    """A handful of alleles for testing."""
    return frozenset([
        Allele(name="12", base_pair=120.0, left_bin=0.4, right_bin=0.4, height=1500.0),
        Allele(name="15", base_pair=132.0, left_bin=0.5, right_bin=0.5, height=2200.0),
        Allele(name="13.2", base_pair=125.6, left_bin=0.4, right_bin=0.4, height=800.0),
        ]
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
        alleles=frozenset(
            [Allele(name="X", base_pair=107.0, left_bin=0.5, right_bin=0.5, height=3000.0)]
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


@pytest.fixture
def segmentation_metrics_cfg() -> dict[str, object]:
    return {
        "accuracy": {
            "_target_": "torchmetrics.classification.BinaryAccuracy",
            "threshold": 0.5,
        },
        "precision": {
            "_target_": "torchmetrics.classification.BinaryPrecision",
            "threshold": 0.5,
        },
        "recall": {
            "_target_": "torchmetrics.classification.BinaryRecall",
            "threshold": 0.5,
        },
        "f1": {
            "_target_": "torchmetrics.classification.BinaryF1Score",
            "threshold": 0.5,
        },
        "iou": {
            "_target_": "torchmetrics.classification.BinaryJaccardIndex",
            "threshold": 0.5,
        },
    }


@pytest.fixture
def classification_metrics_cfg() -> dict[str, object]:
    return {
        "accuracy": {
            "_target_": "torchmetrics.classification.MulticlassAccuracy",
            "num_classes": 3,
            "average": "micro",
        },
        "precision": {
            "_target_": "torchmetrics.classification.MulticlassPrecision",
            "num_classes": 3,
            "average": "macro",
        },
        "recall": {
            "_target_": "torchmetrics.classification.MulticlassRecall",
            "num_classes": 3,
            "average": "macro",
        },
        "f1": {
            "_target_": "torchmetrics.classification.MulticlassF1Score",
            "num_classes": 3,
            "average": "macro",
        },
    }


@pytest.fixture
def reconstruction_metrics_cfg() -> dict[str, object]:
    return {
        "mse": {
            "_target_": "torchmetrics.regression.MeanSquaredError",
        },
    }


@pytest.fixture
def peaknet_metrics_cfg() -> dict[str, object]:
    return {
        "accuracy": {
            "_target_": "torchmetrics.classification.MulticlassAccuracy",
            "num_classes": 3,
            "average": "micro",
        },
        "f1": {
            "_target_": "torchmetrics.classification.MulticlassF1Score",
            "num_classes": 3,
            "average": "macro",
        },
    }


@pytest.fixture
def evaluation_pixel_metrics_cfg() -> dict[str, object]:
    return {
        "pixel_precision": {
            "_target_": "dnanet.evaluation.metrics.pixel.pixel_precision",
            "_partial_": True,
            "threshold": 0.5,
        },
        "pixel_recall": {
            "_target_": "dnanet.evaluation.metrics.pixel.pixel_recall",
            "_partial_": True,
            "threshold": 0.5,
        },
        "pixel_f1_score": {
            "_target_": "dnanet.evaluation.metrics.pixel.pixel_f1_score",
            "_partial_": True,
            "threshold": 0.5,
        },
        "average_binary_iou": {
            "_target_": "dnanet.evaluation.metrics.pixel.average_binary_iou",
            "_partial_": True,
            "threshold": 0.5,
        },
    }


# ---------------------------------------------------------------------------
# Strategy fixtures (configure global kit strategy for integration tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def ppf6c_kit():
    """Configure the PPF6C scaling strategy globally."""
    StrategyRegistry.configure_kit("PPF6C")
    yield
    StrategyRegistry.reset()


@pytest.fixture
def globalfiler_kit():
    """Configure the GlobalFiler scaling strategy globally."""
    StrategyRegistry.configure_kit("GLOBALFILER")
    yield
    StrategyRegistry.reset()


# ---------------------------------------------------------------------------
# Dataset fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def nfi_rnd_kit():
    """Configure PPF6C kit + NFI_RND dataset strategy."""
    StrategyRegistry.configure_kit("PPF6C")
    StrategyRegistry.configure_dataset("NFI_RND")
    yield
    StrategyRegistry.reset()


@pytest.fixture
def globalfiler_provedit():
    """Configure GlobalFiler kit + PROVEDIT dataset strategy."""
    StrategyRegistry.configure_kit("GLOBALFILER")
    StrategyRegistry.configure_dataset("PROVEDIT")
    yield
    StrategyRegistry.reset()


# ---------------------------------------------------------------------------
# Real-data fixtures (from test resources)
# ---------------------------------------------------------------------------

@pytest.fixture
def ppf6c_panel() -> Panel:
    """The PPF6C panel loaded from the real XML file."""
    return Panel.from_xml(PANEL_PATH)
