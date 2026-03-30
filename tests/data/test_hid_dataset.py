"""Tests for HIDDataset — file discovery, annotation mapping, and loading."""

import csv
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import RD_DIR, LADDER_ALLELES_CSV
from dnanet.data.hid_dataset import HIDDataset
from dnanet.data.strategies.registry import StrategyRegistry


# ---------------------------------------------------------------------------
# Ladder path loading (static method)
# ---------------------------------------------------------------------------

class TestLadderPathLoading:
    def test_loads_real_csv(self, nfi_rnd_kit):
        mapping = HIDDataset._load_ladder_paths(RD_DIR / "best_ladder_paths.csv")
        assert len(mapping) > 0

    def test_empty_rows_skipped(self, nfi_rnd_kit, tmp_path):
        csv_path = tmp_path / "ladders.csv"
        csv_path.write_text("image_path,ladder_path\n,\nsample1,ladder1\n")
        mapping = HIDDataset._load_ladder_paths(csv_path)
        assert len(mapping) == 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestHIDDatasetValidation:
    def test_invalid_adjustment_raises(self, nfi_rnd_kit):
        with pytest.raises(ValueError, match="adjustment_of_annotations"):
            HIDDataset(root=RD_DIR, adjustment_of_annotations="bad")

    def test_empty_root_raises(self, nfi_rnd_kit, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No valid HID images"):
            HIDDataset(root=empty_dir)


# ---------------------------------------------------------------------------
# Integration: loading from test resources
# ---------------------------------------------------------------------------

class TestHIDDatasetIntegration:
    def test_load_from_rd_dir(self, nfi_rnd_kit):
        """Load from test resources — should find 2 sample HID files."""
        ds = HIDDataset(
            root=RD_DIR,
            hid_to_annotations_path=RD_DIR / "2p_5p_hid_to_annotation.csv",
            best_ladder_paths_csv=RD_DIR / "best_ladder_paths.csv",
            ladder_alleles_csv=LADDER_ALLELES_CSV,
            analysis_threshold_type="DTH",
        )
        assert len(ds) >= 1  # at least one sample should load

    def test_images_have_correct_shape(self, nfi_rnd_kit):
        ds = HIDDataset(
            root=RD_DIR,
            hid_to_annotations_path=RD_DIR / "2p_5p_hid_to_annotation.csv",
            best_ladder_paths_csv=RD_DIR / "best_ladder_paths.csv",
            ladder_alleles_csv=LADDER_ALLELES_CSV,
        )
        for img in ds:
            assert img.data is not None
            assert img.data.shape == (5, 4096, 1)

    def test_limit_parameter(self, nfi_rnd_kit):
        ds = HIDDataset(
            root=RD_DIR,
            hid_to_annotations_path=RD_DIR / "2p_5p_hid_to_annotation.csv",
            best_ladder_paths_csv=RD_DIR / "best_ladder_paths.csv",
            ladder_alleles_csv=LADDER_ALLELES_CSV,
            limit=1,
        )
        assert len(ds) == 1

    def test_repr(self, nfi_rnd_kit):
        ds = HIDDataset(
            root=RD_DIR,
            hid_to_annotations_path=RD_DIR / "2p_5p_hid_to_annotation.csv",
            best_ladder_paths_csv=RD_DIR / "best_ladder_paths.csv",
            ladder_alleles_csv=LADDER_ALLELES_CSV,
        )
        r = repr(ds)
        assert "HIDDataset" in r
        assert "RD" in r

    def test_split_produces_valid_subsets(self, nfi_rnd_kit):
        ds = HIDDataset(
            root=RD_DIR,
            hid_to_annotations_path=RD_DIR / "2p_5p_hid_to_annotation.csv",
            best_ladder_paths_csv=RD_DIR / "best_ladder_paths.csv",
            ladder_alleles_csv=LADDER_ALLELES_CSV,
        )
        if len(ds) >= 2:
            train, val = ds.split(0.5, seed=42)
            assert len(train) + len(val) == len(ds)

    def test_annotations_populated(self, nfi_rnd_kit):
        """Images with annotation mapping should have non-None annotations."""
        ds = HIDDataset(
            root=RD_DIR,
            hid_to_annotations_path=RD_DIR / "2p_5p_hid_to_annotation.csv",
            best_ladder_paths_csv=RD_DIR / "best_ladder_paths.csv",
            ladder_alleles_csv=LADDER_ALLELES_CSV,
        )
        annotated = [img for img in ds if img.annotation is not None]
        assert len(annotated) > 0, "Expected at least one image with annotation"

    def test_adjustment_complete(self, nfi_rnd_kit):
        """Loading with annotation adjustment should not crash."""
        ds = HIDDataset(
            root=RD_DIR,
            hid_to_annotations_path=RD_DIR / "2p_5p_hid_to_annotation.csv",
            best_ladder_paths_csv=RD_DIR / "best_ladder_paths.csv",
            ladder_alleles_csv=LADDER_ALLELES_CSV,
            adjustment_of_annotations="complete",
        )
        assert len(ds) >= 1
