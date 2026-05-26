"""Integration tests for annotation parsing with real AlleleReport files."""

import numpy as np
import pytest

from dnanet.core.panel import Panel
from dnanet.data.hid_dataset import HIDDataset
from dnanet.data.image import HIDImage
from dnanet.data.ladders.ladder import Ladder
from dnanet.data.ladders.ladder_allele_catalog import LadderAlleleCatalog
from dnanet.data.strategies import NFIRnDStrategy, PowerPlexFusion6CStrategy
from tests.conftest import PANEL_PATH, RD_DIR


class TestParseCalledAlleles:
    """Test annotation parsing against the real AlleleReport TXT files.

    These tests verify backward compatibility with the original DNANet
    annotation parser output.
    """

    @pytest.fixture
    def nfi_rnd_kit(self):
        """Return PPF6C scaling."""
        yield PowerPlexFusion6CStrategy()


    @pytest.fixture
    def panel(self) -> Panel:
        return Panel.from_xml(PANEL_PATH)

    @pytest.fixture
    def allele_report(self) -> str:
        return str(RD_DIR / "Dataset 1 DTH_AlleleReport.txt")

    def test_parse_sample_1a2(self, nfi_rnd_kit, allele_report, nfi_rnd_dataset):
        """Parse sample '1_11148_1A2' — the primary test sample."""
        annotations = nfi_rnd_dataset.parse_annotations(
            allele_report,
            nfi_rnd_kit,
        )
        markers = annotations["1_11148_1A2"]
        assert markers is not None
        assert len(markers.data) == 27  # matches original test

    def test_amel_alleles(self, nfi_rnd_kit, allele_report, nfi_rnd_dataset):
        """AMEL should have X and Y alleles with correct heights."""
        dataset_strat = nfi_rnd_dataset
        markers = dataset_strat.parse_annotations(allele_report, nfi_rnd_kit)["1_11148_1A2"]
        amel = markers.data[0]
        assert amel.name == "AMEL"
        assert amel.dye_row == 0
        assert len(amel.alleles) == 2

        assert {a.name for a in amel.alleles} == {"X", "Y"}

    def test_d3s1358_alleles(self, nfi_rnd_kit, allele_report, nfi_rnd_dataset):
        """D3S1358 should have alleles 14, 15, 17."""
        dataset_strat = nfi_rnd_dataset
        markers = dataset_strat.parse_annotations(allele_report, nfi_rnd_kit)["1_11148_1A2"]
        d3 = next(m for m in markers.data if m.name == "D3S1358")
        allele_names = {a.name for a in d3.alleles}
        assert allele_names == {"14", "15", "17"}

    def test_d3s1358_heights(self, nfi_rnd_kit, allele_report, monkeypatch, nfi_rnd_dataset):
        """D3S1358 allele heights should match reference."""
        dataset_strat = nfi_rnd_dataset
        monkeypatch.setattr(type(dataset_strat), "READ_ANNOTATION_HEIGHTS", True)
        markers = dataset_strat.parse_annotations(allele_report, nfi_rnd_kit)["1_11148_1A2"]
        d3 = next(m for m in markers.data if m.name == "D3S1358")
        height_map = {a.name: a.height for a in d3.alleles}
        assert height_map["14"] == 3950.0
        assert height_map["15"] == 8780.0
        assert height_map["17"] == 6486.0

    def test_unknown_sample_raises_error(self, nfi_rnd_kit, allele_report, nfi_rnd_dataset):
        """Requesting a non-existent sample should return None."""
        dataset_strat = nfi_rnd_dataset
        with pytest.raises(KeyError):
            dataset_strat.parse_annotations(allele_report, nfi_rnd_kit)["NONEXISTENT_SAMPLE"]

    def test_d1s1656_microvariant(self, nfi_rnd_kit, allele_report, nfi_rnd_dataset):
        """D1S1656 should parse micro-variants like 15.3 and 18.3."""
        dataset_strat = nfi_rnd_dataset
        markers = dataset_strat.parse_annotations(allele_report, nfi_rnd_kit)["1_11148_1A2"]
        d1 = next(m for m in markers.data if m.name == "D1S1656")
        allele_names = {a.name for a in d1.alleles}
        assert "15.3" in allele_names
        assert "18.3" in allele_names

    def test_second_sample(self, nfi_rnd_kit, nfi_rnd_dataset):
        """Parse the second sample '3_10196_1A2'."""
        report = str(RD_DIR / "Dataset 1 DTH_AlleleReport.txt")
        dataset_strat = nfi_rnd_dataset
        markers = dataset_strat.parse_annotations(report, nfi_rnd_kit)["3_10196_1A2"]
        assert markers is not None
        assert len(markers.data) > 0

    def test_markers_have_correct_dye_rows(self, nfi_rnd_kit, panel, allele_report, nfi_rnd_dataset):
        """Dye rows from annotation parsing should match the panel."""
        dataset_strat = nfi_rnd_dataset
        markers = dataset_strat.parse_annotations(allele_report, nfi_rnd_kit)["1_11148_1A2"]
        for marker in markers.data:
            expected_dye = panel.get_dye_row(marker.name)
            assert marker.dye_row == expected_dye, (
                f"{marker.name}: expected dye {expected_dye}, got {marker.dye_row}"
            )

    def test_ladder_adjusted_scanpoint_annotation_matches_reference_mask(self, nfi_rnd_kit):
        """Adjusted-panel annotations should stay aligned with the reference mask."""
        image = HIDImage(
            path=RD_DIR / '1A2_A01_01.hid',
            scaling_strategy=nfi_rnd_kit,
            load_in_memory=True,
        )
        assert image.data is not None

        allele_report = RD_DIR / 'Dataset 1 DTH_AlleleReport.txt'
        dth_strategy = NFIRnDStrategy(annotation_type='DTH')
        dth_annotations = dth_strategy.parse_annotations(allele_report, nfi_rnd_kit)
        sample_annotation = dth_annotations['1_11148_1A2']

        default_scanpoint = HIDDataset._translate_allele_to_scanpoint_annotation(
            allele_annotation=sample_annotation,
            adjusted_panel=nfi_rnd_kit.panel,
            scaler=image.scaler,
            include_size_standard=False,
            scaling_strategy=nfi_rnd_kit,
        ).data

        adjusted_panel = Ladder.create_adjusted_panel(
            ladder_path=RD_DIR / 'Ladder_G03_21.hid',
            catalog=LadderAlleleCatalog.from_panel(nfi_rnd_kit.panel),
            data_loading_strategy='superior',
            scaling_strategy=nfi_rnd_kit,
            dataset_strategy=dth_strategy,
        )
        assert adjusted_panel is not None

        adjusted_scanpoint = HIDDataset._translate_allele_to_scanpoint_annotation(
            allele_annotation=sample_annotation,
            adjusted_panel=adjusted_panel,
            scaler=image.scaler,
            include_size_standard=False,
            scaling_strategy=nfi_rnd_kit,
        ).data

        reference = np.load(RD_DIR / '1A2_A01_01_annotation.npy')
        if reference.ndim == 3:
            reference = reference[:, :, 0]

        default_diff = int(np.abs(default_scanpoint.astype(int) - reference.astype(int)).sum())
        adjusted_diff = int(np.abs(adjusted_scanpoint.astype(int) - reference.astype(int)).sum())

        assert adjusted_diff < default_diff
        assert adjusted_diff <= 100
