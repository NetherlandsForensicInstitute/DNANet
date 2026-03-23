"""Tests for annotation CSV parsing and array conversion."""

import csv

import numpy as np
import pytest

from dnanet.tools.annotation_stats.parser import (
    CATEGORY_NAMES,
    DYE_CHANNEL_NAMES,
    ScanPointAnnotation,
    annotations_to_csv_rows,
    parse_csv_file,
    parse_full_annotations,
    read_and_group_annotations,
)


# ---------------------------------------------------------------------------
# parse_csv_file
# ---------------------------------------------------------------------------

class TestParseCsvFile:
    def test_parses_valid_csv(self, tmp_path):
        path = tmp_path / "test.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user", "date", "profile", "dye", "x0", "x1",
                             "peak_idx", "category", "version"])
            writer.writerow(["Alice", "2024-01-01", "sample1", "blue",
                             "10", "20", "500", "Allele", "1.0"])
        result = parse_csv_file(path)
        assert len(result) == 1
        assert result[0] == ("Alice", "sample1", "blue", 10, 20, 500, "Allele")

    def test_skips_empty_category(self, tmp_path):
        path = tmp_path / "test.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user", "date", "profile", "dye", "x0", "x1",
                             "peak_idx", "category", "version"])
            writer.writerow(["Alice", "2024-01-01", "sample1", "blue",
                             "10", "20", "", "", "1.0"])
        result = parse_csv_file(path)
        assert len(result) == 0

    def test_skips_size_standard_dye(self, tmp_path):
        path = tmp_path / "test.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user", "date", "profile", "dye", "x0", "x1",
                             "peak_idx", "category", "version"])
            writer.writerow(["Alice", "2024-01-01", "sample1", "orange",
                             "10", "20", "500", "Allele", "1.0"])
        result = parse_csv_file(path)
        assert len(result) == 0  # orange = dye index 5, filtered

    def test_empty_rfu_defaults_to_negative(self, tmp_path):
        path = tmp_path / "test.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user", "date", "profile", "dye", "x0", "x1",
                             "peak_idx", "category", "version"])
            writer.writerow(["Alice", "2024-01-01", "sample1", "blue",
                             "10", "20", "", "Allele", "1.0"])
        result = parse_csv_file(path)
        assert result[0][5] == -1

    def test_nonexistent_file_returns_empty(self, tmp_path):
        result = parse_csv_file(tmp_path / "nonexistent.csv")
        assert result == []


# ---------------------------------------------------------------------------
# read_and_group_annotations
# ---------------------------------------------------------------------------

class TestReadAndGroupAnnotations:
    @pytest.fixture
    def annotation_folder(self, tmp_path):
        path = tmp_path / "annotations.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user", "date", "profile", "dye", "x0", "x1",
                             "peak_idx", "category", "version"])
            writer.writerow(["Alice", "", "sample1", "blue", "10", "20", "500", "Allele", "1.0"])
            writer.writerow(["Alice", "", "sample1", "green", "30", "40", "300", "Stutter", "1.0"])
            writer.writerow(["Bob", "", "sample1", "blue", "50", "60", "400", "Allele", "1.0"])
            # 1-pixel annotation (should be filtered)
            writer.writerow(["Alice", "", "sample1", "blue", "70", "70", "100", "Allele", "1.0"])
        return tmp_path

    def test_groups_by_sample(self, annotation_folder):
        result = read_and_group_annotations(annotation_folder, per_annotator=False)
        assert "sample1" in result
        assert len(result["sample1"]) == 3  # 1-pixel one filtered out

    def test_groups_per_annotator(self, annotation_folder):
        result = read_and_group_annotations(annotation_folder, per_annotator=True)
        assert ("Alice", "sample1") in result
        assert ("Bob", "sample1") in result
        assert len(result[("Alice", "sample1")]) == 2  # 1-pixel filtered

    def test_returns_scan_point_annotations(self, annotation_folder):
        result = read_and_group_annotations(annotation_folder)
        ann = result["sample1"][0]
        assert isinstance(ann, ScanPointAnnotation)
        assert ann.dye_index == 0  # blue
        assert ann.start == 10
        assert ann.end == 20
        assert ann.label == "Allele"


# ---------------------------------------------------------------------------
# parse_full_annotations
# ---------------------------------------------------------------------------

class TestParseFullAnnotations:
    def test_produces_arrays(self, tmp_path):
        path = tmp_path / "ann.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user", "date", "profile", "dye", "x0", "x1",
                             "peak_idx", "category", "version"])
            writer.writerow(["Alice", "", "sample1", "blue", "10", "20", "500", "Allele", "1.0"])
        result = parse_full_annotations(tmp_path)
        assert "sample1" in result
        arr = result["sample1"]
        assert arr.shape == (5, 4096)
        # "Allele" should be category index 1
        allele_idx = CATEGORY_NAMES.index("Allele")
        assert arr[0, 10:20].sum() == allele_idx * 10
        assert arr[1:, :].sum() == 0  # Other dyes empty


# ---------------------------------------------------------------------------
# annotations_to_csv_rows
# ---------------------------------------------------------------------------

class TestAnnotationsToCsvRows:
    def test_basic_conversion(self):
        arr = np.zeros((5, 100), dtype=np.int8)
        allele_idx = CATEGORY_NAMES.index("Allele")
        arr[0, 10:20] = allele_idx
        rows = annotations_to_csv_rows("sample1", arr)
        assert len(rows) == 1
        assert rows[0][2] == "sample1"  # profile name
        assert rows[0][3] == "blue"     # dye
        assert rows[0][4] == "10"       # x0
        assert rows[0][5] == "20"       # x1

    def test_multiple_regions(self):
        arr = np.zeros((5, 100), dtype=np.int8)
        allele_idx = CATEGORY_NAMES.index("Allele")
        stutter_idx = CATEGORY_NAMES.index("Stutter")
        arr[0, 10:20] = allele_idx
        arr[0, 50:60] = stutter_idx
        rows = annotations_to_csv_rows("sample1", arr)
        assert len(rows) == 2

    def test_per_annotator_tuple(self):
        arr = np.zeros((5, 100), dtype=np.int8)
        allele_idx = CATEGORY_NAMES.index("Allele")
        arr[0, 10:20] = allele_idx
        rows = annotations_to_csv_rows(("Alice", "sample1"), arr, per_annotator=True)
        assert rows[0][0] == "Alice"

    def test_with_rfu_array(self):
        arr = np.zeros((5, 100), dtype=np.int8)
        allele_idx = CATEGORY_NAMES.index("Allele")
        arr[0, 10:20] = allele_idx
        rfu = np.zeros((5, 100), dtype=np.float32)
        rfu[0, 15] = 500.0
        rows = annotations_to_csv_rows("sample1", arr, rfu_array=rfu)
        assert rows[0][6] == "500"  # max_rfu
