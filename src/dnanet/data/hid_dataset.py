"""HIDDataset — loads HID files from disk into an InMemoryDataset.

Design pattern: **Facade**
    ``HIDDataset`` is the single entry point for loading forensic DNA profiles
    from a directory of HID files. It orchestrates:
    - File discovery and filtering (via ``DatasetStrategy``)
    - Annotation mapping (CSV lookup)
    - Ladder loading and panel adjustment
    - HIDImage construction with lazy data loading
    - Filtering of invalid images

Design pattern: **Template Method** (inherited)
    ``split()`` and ``split_k_fold()`` are inherited from ``InMemoryDataset``
    and can be overridden for replica-aware splitting.

Usage::

    from dnanet.data.strategies.registry import StrategyRegistry
    StrategyRegistry.configure_kit("PPF6C")
    StrategyRegistry.configure_dataset("NFI_RND")

    dataset = HIDDataset(
        root="data/2p_5p_Dataset_NFI/Raw data .HID files",
        annotations_path="data/2p_5p_Dataset_NFI/txt_annotations_2024",
        hid_to_annotations_path="data/2p_5p_Dataset_NFI/2p_5p_hid_to_annotation.csv",
        best_ladder_paths_csv="data/2p_5p_Dataset_NFI/best_ladder_paths_DTH.csv",
        ladder_alleles_csv="data/2p_5p_Dataset_NFI/ladder_alleles.csv",
    )
"""

from __future__ import annotations

import csv
import random
from typing import Any, Generator
from pathlib import Path

import numpy as np
from loguru import logger

from dnanet.core.panel import Panel
from dnanet.core.types import PathLike
from dnanet.data.image import HIDImage
from dnanet.data.ladder import Ladder, LadderAlleleCatalog
from dnanet.data.dataset import SimpleDataset, InMemoryDataset
from dnanet.data.parsing.hid import get_peak_data
from dnanet.data.strategies.registry import StrategyRegistry


class HIDDataset(InMemoryDataset):
    """Load HID files from a directory into an in-memory dataset.

    This is the primary dataset class for forensic DNA profiles. It:
    1. Walks a root directory for ``.hid`` files
    2. Filters to sample files (excluding ladders, controls)
    3. Matches each sample to its annotation and best ladder
    4. Builds an adjusted panel from the ladder
    5. Creates ``HIDImage`` instances and filters out invalid ones

    Requires ``StrategyRegistry`` to be configured with a kit (scaling)
    strategy before construction.

    Args:
        root: Root directory containing HID files (searched recursively).
        annotations_path: Directory containing annotation TXT files.
        hid_to_annotations_path: CSV mapping HID filenames to annotation names.
        best_ladder_paths_csv: CSV mapping sample stems to best ladder paths.
        ladder_alleles_csv: CSV with expected ladder alleles per dye.
        analysis_threshold_type: ``"DTH"`` (high) or ``"DTL"`` (low).
        adjustment_of_annotations: ``"top"`` or ``"complete"`` peak adjustment,
            or ``None`` to skip.
        limit: Maximum number of images to load.
        shuffle: Randomize iteration order.
        skip_if_invalid_ladder: Skip images whose ladder fails to parse.
        include_size_standard: Include the 6th dye channel in the data.
        data_loading_strategy: ``"raw"``, ``"analyzed"``, or ``"superior"``.
    """

    def __init__(
        self,
        root: PathLike,
        annotations_path: PathLike | None = None,
        hid_to_annotations_path: PathLike | None = None,
        best_ladder_paths_csv: PathLike | None = None,
        ladder_alleles_csv: PathLike | None = None,
        analysis_threshold_type: str = "DTH",
        adjustment_of_annotations: str | None = None,
        limit: int | None = None,
        shuffle: bool = False,
        skip_if_invalid_ladder: bool = False,
        include_size_standard: bool = False,
        data_loading_strategy: str = "superior",
    ) -> None:
        super().__init__(shuffle)

        self.root = Path(root)
        self.annotations_path = Path(annotations_path) if annotations_path else None
        self.analysis_threshold_type = analysis_threshold_type
        self.adjustment_of_annotations = adjustment_of_annotations
        self.skip_if_invalid_ladder = skip_if_invalid_ladder
        self.include_size_standard = include_size_standard
        self.data_loading_strategy = data_loading_strategy

        if adjustment_of_annotations and adjustment_of_annotations not in ("top", "complete"):
            raise ValueError(
                f"adjustment_of_annotations must be 'top' or 'complete', "
                f"got {adjustment_of_annotations!r}"
            )

        # Get strategies from registry
        self._scaling = StrategyRegistry.get_scaling_strategy()
        self._dataset_strategy = StrategyRegistry.get_dataset_strategy()
        self._default_panel = self._scaling.panel

        # Load annotation mapping: hid_stem -> (annotation_name, annotation_file_path)
        self._annotation_map: dict[str, tuple[str, Path]] = {}
        if hid_to_annotations_path and self.annotations_path:
            self._annotation_map = self._create_annotation_mapping(
                Path(hid_to_annotations_path),
                self.annotations_path,
                analysis_threshold_type,
            )
            logger.info("Loaded {} annotation mappings", len(self._annotation_map))

        # Load best ladder paths: sample_stem -> ladder_path
        self._ladder_paths: dict[str, Path] = {}
        if best_ladder_paths_csv:
            self._ladder_paths = self._load_ladder_paths(Path(best_ladder_paths_csv))
            logger.info("Loaded {} ladder path mappings", len(self._ladder_paths))

        # Load ladder allele catalog
        self._ladder_catalog: LadderAlleleCatalog | None = None
        if ladder_alleles_csv:
            self._ladder_catalog = LadderAlleleCatalog.from_csv(Path(ladder_alleles_csv))

        # Collect files, apply limit, load images
        file_entries = list(self._collect_files())
        logger.info("Found {} sample files to process", len(file_entries))

        if limit and shuffle:
            file_entries = random.sample(file_entries, min(limit, len(file_entries)))
        elif limit:
            file_entries = file_entries[:limit]

        self._data = list(self._load_images(file_entries))

        if len(self._data) == 0:
            raise ValueError(
                f"No valid HID images found in {self.root}. "
                f"Check paths and StrategyRegistry configuration."
            )

        logger.info("Loaded {} valid HID images", len(self._data))

        # Optional annotation adjustment
        if adjustment_of_annotations:
            logger.info("Adjusting annotations (method={})", adjustment_of_annotations)
            self._data = [
                img.adjust_annotations(adjustment_of_annotations)
                for img in self._data
            ]

    # -- Annotation mapping ------------------------------------------------ #

    @staticmethod
    def _create_annotation_mapping(
        hid_to_annotations_path: Path,
        annotations_dir: Path,
        threshold_type: str,
    ) -> dict[str, tuple[str, Path]]:
        """Map HID stems to (annotation_name, annotation_file_path).

        The CSV has columns: ``Raw hid name``, ``DTH name 2024``, ``DTL name 2024``.
        The annotation TXT file is derived from the first digit of the HID stem:
        e.g. ``1A2`` → ``Dataset 1 DTH_AlleleReport.txt``.
        """
        if threshold_type not in ("DTH", "DTL"):
            raise ValueError(
                f"analysis_threshold_type must be 'DTH' or 'DTL', got {threshold_type!r}"
            )

        mapping: dict[str, tuple[str, Path]] = {}
        with open(hid_to_annotations_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_name = row.get("Raw hid name", "").strip()
                if not raw_name:
                    continue
                hid_stem = Path(raw_name).stem
                ann_name = row.get(f"{threshold_type} name 2024", "").strip()
                if not ann_name:
                    continue

                # Derive annotation file: first char of stem is the dataset number
                dataset_num = hid_stem[0]
                txt_name = f"Dataset {dataset_num} {threshold_type}_AlleleReport.txt"
                mapping[hid_stem] = (ann_name, annotations_dir / txt_name)

        return mapping

    # -- Ladder path loading ----------------------------------------------- #

    @staticmethod
    def _load_ladder_paths(csv_path: Path) -> dict[str, Path]:
        """Load sample → ladder path mapping from CSV.

        The CSV has columns: ``image_path``, ``ladder_path``.
        """
        mapping: dict[str, Path] = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stem = row["image_path"].strip()
                ladder = row["ladder_path"].strip()
                if stem and ladder:
                    mapping[stem] = Path(ladder)
        return mapping

    # -- File collection --------------------------------------------------- #

    def _collect_files(self) -> Generator[dict[str, Any], None, None]:
        """Walk root directory and yield metadata for each sample file."""
        strategy = self._dataset_strategy
        no_annotation_count = 0
        non_sample_count = 0

        for hid_path in sorted(self.root.rglob("*.hid")):
            # Categorize using the dataset strategy
            category = strategy.categorize_file(hid_path.name)
            if category != "sample":
                non_sample_count += 1
                continue

            stem = hid_path.stem
            annotation = self._annotation_map.get(stem)

            if not annotation and self._annotation_map:
                # We have an annotation map but this sample isn't in it
                no_annotation_count += 1
                continue

            # Look up ladder path from CSV, or fall back to strategy
            ladder_path = self._ladder_paths.get(stem)
            if ladder_path is None and not self._ladder_paths:
                # No CSV mapping — use strategy to find a nearby ladder
                ladder_path = strategy.find_ladder_for_sample(hid_path)

            yield {
                "path": hid_path,
                "ladder_path": ladder_path,
                "annotation": annotation,  # (name, file_path) or None
            }

        if non_sample_count:
            logger.info("Skipped {} non-sample files", non_sample_count)
        if no_annotation_count:
            logger.info("Skipped {} files without annotation mapping", no_annotation_count)

    # -- Ladder building --------------------------------------------------- #

    def _build_ladder(
        self,
        ladder_path: Path,
        ladder_cache: dict[str, Panel | None],
    ) -> Panel | None:
        """Build an adjusted panel from a ladder HID file.

        Results are cached by ladder path to avoid redundant parsing.
        """
        cache_key = str(ladder_path)
        if cache_key in ladder_cache:
            return ladder_cache[cache_key]

        # Resolve the ladder path relative to root if needed
        resolved = self._resolve_ladder_path(ladder_path)
        if resolved is None or not resolved.exists():
            logger.warning("Ladder file not found: {}", ladder_path)
            ladder_cache[cache_key] = None
            return None

        # Parse the raw HID data from the ladder file
        raw = get_peak_data(resolved, "raw")
        if raw is None:
            logger.warning("Could not parse ladder HID: {}", resolved.name)
            ladder_cache[cache_key] = None
            return None

        # Parse size standard from the last channel
        ss_lane = np.array(raw[-1])
        try:
            ss_result = self._scaling.parse_size_standard(ss_lane)
        except ValueError as e:
            logger.warning("Ladder size standard invalid for {}: {}", resolved.name, e)
            ladder_cache[cache_key] = None
            return None

        if ss_result is None:
            logger.warning("Ladder size standard parsing returned None for {}", resolved.name)
            ladder_cache[cache_key] = None
            return None

        # Rescale the ladder data
        data = raw[:, ss_result.rescaled_indices][..., np.newaxis]
        scaler = ss_result.scaler

        if self._ladder_catalog is None:
            logger.warning("No ladder allele catalog; cannot build adjusted panel")
            ladder_cache[cache_key] = None
            return None

        # Build Ladder object with adjusted panel
        ladder = Ladder.from_hid_data(
            path=resolved,
            data=data,
            scaler=scaler,
            catalog=self._ladder_catalog,
            default_panel=self._default_panel,
            num_dyes=self._scaling.kit.num_dyes,
        )

        adjusted = ladder.adjusted_panel if ladder else None
        ladder_cache[cache_key] = adjusted

        if adjusted:
            logger.debug("Built adjusted panel from ladder {}", resolved.name)
        else:
            logger.warning("Ladder failed for {}", resolved.name)

        return adjusted

    def _resolve_ladder_path(self, ladder_path: Path) -> Path | None:
        """Resolve a ladder path that may be relative to the project root."""
        if ladder_path.is_absolute() and ladder_path.exists():
            return ladder_path

        # Try relative to CWD first
        if ladder_path.exists():
            return ladder_path

        # The CSV may contain paths like "resources/data/2p_5p_Dataset_NFI/..."
        # Try stripping known prefixes and resolving relative to root's parent
        path_str = str(ladder_path)
        # Try to find the "Raw data .HID files" part and resolve from root
        for prefix in ("resources/data/2p_5p_Dataset_NFI/", "data/2p_5p_Dataset_NFI/"):
            if prefix in path_str:
                relative = path_str.split(prefix, 1)[1]
                candidate = self.root.parent / relative
                if candidate.exists():
                    return candidate

        # Try relative to root's parent (the dataset folder)
        for part in ("Raw data .HID files",):
            if part in path_str:
                idx = path_str.index(part)
                relative = path_str[idx:]
                candidate = self.root.parent / relative
                if candidate.exists():
                    return candidate

        # Last resort: just the filename, search in root
        matches = list(self.root.rglob(ladder_path.name))
        if matches:
            return matches[0]

        return None

    # -- Image loading ----------------------------------------------------- #

    def _load_images(
        self, file_entries: list[dict[str, Any]]
    ) -> Generator[HIDImage, None, None]:
        """Create HIDImage instances, loading data and filtering invalid ones."""
        ladder_cache: dict[str, Panel | None] = {}
        loaded = 0
        skipped_data = 0
        skipped_alleles = 0
        skipped_ladder = 0

        for entry in file_entries:
            path: Path = entry["path"]
            ladder_path: Path | None = entry["ladder_path"]
            annotation = entry["annotation"]  # (name, file) or None

            # Build adjusted panel from ladder
            panel = self._default_panel
            if ladder_path:
                adjusted = self._build_ladder(ladder_path, ladder_cache)
                if adjusted:
                    panel = adjusted
                elif self.skip_if_invalid_ladder:
                    skipped_ladder += 1
                    continue

            # Prepare annotation info
            ann_name, ann_file = annotation if annotation else (None, None)

            # Extract NOC from filename
            noc = self._dataset_strategy.get_contributors(path.name)

            # Create HIDImage
            image = HIDImage(
                path=path,
                panel=panel,
                annotations_file=ann_file,
                include_size_standard=self.include_size_standard,
                data_loading_strategy=self.data_loading_strategy,
                meta={
                    "annotations_name": ann_name,
                    "ladder_path": str(ladder_path) if ladder_path else None,
                    "noc": noc,
                },
            )

            # Trigger lazy load and validate
            if image.data is None:
                skipped_data += 1
                logger.debug("Skipping {}: no data", path.name)
                continue

            if self._annotation_map and image.annotation is None:
                skipped_alleles += 1
                logger.debug("Skipping {}: no annotation/called alleles", path.name)
                continue

            loaded += 1
            if loaded % 50 == 0:
                logger.info("Loaded {}/{} images...", loaded, len(file_entries))

            yield image

        if skipped_data:
            logger.warning("Skipped {} images with missing data", skipped_data)
        if skipped_alleles:
            logger.warning("Skipped {} images with missing annotations", skipped_alleles)
        if skipped_ladder:
            logger.warning("Skipped {} images with invalid ladders", skipped_ladder)

    # -- Dunder ----------------------------------------------------------- #

    def __repr__(self) -> str:
        return f"HIDDataset(root={self.root.name}, n={len(self._data)})"
