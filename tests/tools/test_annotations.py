"""Tests for AnnotationStore — CSV-based annotation persistence."""

import csv

import pytest

from dnanet.tools.labeltool.annotations import (
    HEADER,
    LABELTOOL_VERSION,
    AnnotationStore,
    _build_label_lookup,
)


# ---------------------------------------------------------------------------
# _build_label_lookup
# ---------------------------------------------------------------------------

class TestBuildLabelLookup:
    def test_returns_dict(self):
        lookup = _build_label_lookup()
        assert isinstance(lookup, dict)

    def test_contains_allele(self):
        lookup = _build_label_lookup()
        assert "Allele" in lookup
        color, alpha = lookup["Allele"]
        assert color == "green"
        assert alpha == 0.4

    def test_contains_empty_string(self):
        lookup = _build_label_lookup()
        assert "" in lookup
        assert lookup[""][0] == "gray"

    def test_contains_none(self):
        lookup = _build_label_lookup()
        assert None in lookup


# ---------------------------------------------------------------------------
# AnnotationStore.ensure_file
# ---------------------------------------------------------------------------

class TestAnnotationStoreEnsureFile:
    def test_creates_file_with_header(self, tmp_path):
        path = tmp_path / "test.csv"
        store = AnnotationStore(path)
        store.ensure_file()
        assert path.exists()

        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == HEADER

    def test_does_not_overwrite_existing(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text("existing content")
        store = AnnotationStore(path)
        store.ensure_file()
        assert path.read_text() == "existing content"


# ---------------------------------------------------------------------------
# AnnotationStore.read_all
# ---------------------------------------------------------------------------

class TestAnnotationStoreReadAll:
    def test_reads_csv_file(self, tmp_path):
        path = tmp_path / "annotations.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            writer.writerow(["Alice", "2024-01-01", "sample1", "blue",
                             "10", "20", "-1", "Allele", "1.0"])
        store = AnnotationStore(path)
        rows = store.read_all()
        assert len(rows) == 1
        assert rows[0]["user"] == "Alice"
        assert rows[0]["profile"] == "sample1"

    def test_reads_folder(self, tmp_path):
        for i in range(3):
            path = tmp_path / f"file{i}.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(HEADER)
                writer.writerow(["Bob", "2024-01-01", f"sample{i}", "blue",
                                 "10", "20", "-1", "Stutter", "1.0"])
        store = AnnotationStore(tmp_path)
        rows = store.read_all()
        assert len(rows) == 3

    def test_empty_on_nonexistent(self, tmp_path):
        store = AnnotationStore(tmp_path / "nonexistent.csv")
        rows = store.read_all()
        assert rows == []


# ---------------------------------------------------------------------------
# AnnotationStore.load_spans_by_profile
# ---------------------------------------------------------------------------

class TestAnnotationStoreLoadSpans:
    @pytest.fixture
    def store_with_data(self, tmp_path):
        path = tmp_path / "annotations.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            writer.writerow(["Alice", "2024-01-01", "sample1", "blue",
                             "10", "20", "5", "Allele", "1.0"])
            writer.writerow(["Bob", "2024-01-01", "sample1", "green",
                             "30", "40", "-1", "Stutter", "1.0"])
            writer.writerow(["Alice", "2024-01-01", "sample2", "red",
                             "50", "60", "-1", "PullUp", "1.0"])
        return AnnotationStore(path)

    def test_groups_by_profile(self, store_with_data):
        spans = store_with_data.load_spans_by_profile()
        assert "sample1" in spans
        assert "sample2" in spans
        assert len(spans["sample1"]) == 2
        assert len(spans["sample2"]) == 1

    def test_filters_by_user(self, store_with_data):
        spans = store_with_data.load_spans_by_profile(user="Alice")
        assert len(spans["sample1"]) == 1
        assert spans["sample1"][0]["annotator"] == "Alice"

    def test_span_has_color(self, store_with_data):
        spans = store_with_data.load_spans_by_profile()
        allele_span = spans["sample1"][0]
        assert allele_span["color"] == "green"
        assert allele_span["alpha"] == 0.4


# ---------------------------------------------------------------------------
# AnnotationStore.save_spans
# ---------------------------------------------------------------------------

class TestAnnotationStoreSaveSpans:
    def test_save_creates_file(self, tmp_path):
        path = tmp_path / "out.csv"
        store = AnnotationStore(path)

        spans = [
            {"ax": None, "dye": "blue", "x0": 10.5, "x1": 20.3,
             "peak_idx": 15, "category": "Allele"},
        ]
        store.save_spans(
            "sample1", spans, user="Alice",
            dye_names=["blue", "green", "yellow", "red", "purple", "orange"],
        )

        assert path.exists()
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["profile"] == "sample1"
        assert rows[0]["user"] == "Alice"
        assert rows[0]["x0"] == "10"
        assert rows[0]["x1"] == "20"

    def test_save_replaces_same_user_profile(self, tmp_path):
        path = tmp_path / "out.csv"
        store = AnnotationStore(path)
        dye_names = ["blue", "green", "yellow", "red", "purple", "orange"]

        # First save
        spans1 = [{"dye": "blue", "x0": 10, "x1": 20, "peak_idx": -1, "category": "Allele"}]
        store.save_spans("sample1", spans1, user="Alice", dye_names=dye_names)

        # Second save (same user+profile)
        spans2 = [{"dye": "red", "x0": 50, "x1": 60, "peak_idx": -1, "category": "Stutter"}]
        store.save_spans("sample1", spans2, user="Alice", dye_names=dye_names)

        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1  # Replaced, not appended
        assert rows[0]["dye"] == "red"

    def test_save_preserves_other_users(self, tmp_path):
        path = tmp_path / "out.csv"
        store = AnnotationStore(path)
        dye_names = ["blue", "green", "yellow", "red", "purple", "orange"]

        # Alice saves
        store.save_spans("sample1",
                         [{"dye": "blue", "x0": 10, "x1": 20, "peak_idx": -1, "category": "Allele"}],
                         user="Alice", dye_names=dye_names)

        # Bob saves same profile
        store.save_spans("sample1",
                         [{"dye": "green", "x0": 30, "x1": 40, "peak_idx": -1, "category": "Stutter"}],
                         user="Bob", dye_names=dye_names)

        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2  # Both preserved
