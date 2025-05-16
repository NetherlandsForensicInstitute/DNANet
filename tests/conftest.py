import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, Union

import numpy as np
import pytest

from DNAnet.data.data_models import Annotation, Panel
from DNAnet.data.data_models.hid_dataset import HIDDataset
from DNAnet.data.data_models.hid_image import HIDImage, Ladder
from DNAnet.data.parsing import parse_called_alleles


def pytest_configure():
    """Global variables for tests can be configured here and accessed in any test
    using `pytest.<variable>`.
    """
    tests_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    pytest.RESOURCES_DIR = tests_dir / 'resources'
    pytest.PANEL_PATH = pytest.RESOURCES_DIR / "panel.xml"


@pytest.fixture
def hid_dataset_rd():
    return HIDDataset(
        root=pytest.RESOURCES_DIR / "profiles" / "RD",
        panel=pytest.PANEL_PATH,
        annotations_path=pytest.RESOURCES_DIR / "profiles" / "RD",
        hid_to_annotations_path=(pytest.RESOURCES_DIR / "profiles" /
                                 "RD" / "rd_hid_annotations_mapping.csv"),
        analysis_threshold_type="DTH",
        best_ladder_paths_csv=(pytest.RESOURCES_DIR / "profiles" /
                               "RD" / "test_best_ladder_paths.csv")
    )


@pytest.fixture
def hid_image():
    annotation_path = os.path.join(pytest.RESOURCES_DIR, "profiles", "RD", "1A2_A01_01_annotation.npy")
    return HIDImage(
        path=os.path.join(pytest.RESOURCES_DIR, "profiles", "RD", "1A2_A01_01.hid"),
        annotation=Annotation(image=np.load(annotation_path))
    )


@pytest.fixture
def ladder():
    ladder_path = os.path.join(pytest.RESOURCES_DIR, "profiles", "RD", "Ladder_G03_21.hid")
    panel = Panel(pytest.PANEL_PATH)
    ladder = Ladder(ladder_path, panel)
    return ladder


@pytest.fixture
def hid_image_with_ladder(ladder):
    annotation_path = os.path.join(pytest.RESOURCES_DIR,  "profiles", "RD",
                                   "1A2_A01_01_annotation.npy")
    image = HIDImage(
        path=os.path.join(pytest.RESOURCES_DIR, "profiles", "RD", "1A2_A01_01.hid"),
        annotation=Annotation(image=np.load(annotation_path)),
        panel=ladder._panel
    )
    image.meta["called_alleles"] = parse_called_alleles(
        os.path.join(pytest.RESOURCES_DIR, "profiles", "RD", "Dataset 1 DTH_AlleleReport.txt"),
        ladder.panel,
        "1_11148_1A2"
    )
    return image


@pytest.fixture
def called_alleles():
    return parse_called_alleles(
        os.path.join(pytest.RESOURCES_DIR, "profiles", "RD", "Dataset 1 DTH_AlleleReport.txt"),
        ladder.panel,
        "1_11148_1A2"
    )


class SupportsBool(Protocol):
    def __bool__(self) -> bool:
        pass


@dataclass
class SkipCondition(Mapping[str, Any]):
    condition: SupportsBool
    description: str

    @property
    def reason(self) -> str:
        return f"{self.description} ({self.condition})"

    def __bool__(self):
        return bool(self.condition)

    def __getitem__(self, k: str) -> Union[str, Any]:
        return getattr(self, k)

    def __len__(self) -> int:
        return 2

    def __iter__(self) -> Iterator[str]:
        yield "condition"
        yield "reason"


SKIP_MODELS = SkipCondition(
    condition=int(os.environ.get("TEST_SKIP_MODELS", 0)),
    description="TEST_SKIP_MODELS environment variable set to non-zero value",
)

SKIP_DATASETS = SkipCondition(
    condition=int(os.environ.get("TEST_SKIP_DATASETS", 0)),
    description="TEST_SKIP_DATASETS environment variable set to non-zero value",
)
