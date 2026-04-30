"""HIDDataset — lazy, cache-first dataset of forensic DNA profiles.

Design pattern: **Facade + Lazy Loading**
    ``HIDDataset`` is the single entry point for loading HID files. At
    construction time it only builds (or validates) an on-disk memmap cache
    and loads a lightweight index into memory. No pixel/signal data is read
    at init. ``__getitem__`` memmaps the single requested row per call.

Design pattern: **Template Method** (inherited)
    ``DatasetStrategy.split`` operates on ``__len__`` and lightweight path
    access via ``HIDDataset.images``, which returns stub ``HIDImage`` objects
    with just the path populated.

Usage::

    dataset = HIDDataset(
        root="data/2p_5p_Dataset_NFI/Raw data .HID files",
        scaling_strategy=PowerPlexFusion6CStrategy(),
        dataset_strategy=NFIRnDStrategy(),
        cache_dir="data/cache/dnanet_rd",
    )
"""

from __future__ import annotations

import os
import json
import random
from typing import TYPE_CHECKING, Any, List, Tuple, Optional, Generator
from pathlib import Path

import numpy as np
from tqdm import tqdm
from loguru import logger
from torch.utils.data import Dataset

from dnanet.data.cache import (
    IndexEntry,
    MemmapCacheReader,
    MemmapCacheWriter,
    compute_key,
    is_complete,
    cache_key_dir,
    validate_fingerprint,
)
from dnanet.data.image import HIDImage
from dnanet.data.dataset import TransformableDataset
from dnanet.core.annotation import AlleleAnnotation, ScanpointAnnotation
from dnanet.data.ladders.ladder import Ladder
from dnanet.data.cache.fingerprint import build_config_payload
from dnanet.data.preprocessing.peaks import find_peak_boundary, find_peak_idx_near_or_in_range
from dnanet.data.ladders.ladder_allele_catalog import LadderAlleleCatalog


if TYPE_CHECKING:
    from dnanet.core.panel import Panel
    from dnanet.core.types import PathLike
    from dnanet.core.marker import Marker
    from dnanet.data.transformer import TransformDataCallable
    from dnanet.data.strategies.scaling import ScalingStrategy
    from dnanet.data.strategies.datasets import DatasetStrategy


class HIDDataset(Dataset, TransformableDataset):
    """Lazy, cache-backed dataset of HID profiles.

    On construction:
      1. A cache directory keyed by config hash is located (or created).
      2. If the cache is complete and its stored fingerprint still matches the
         current source files, only the index parquet is loaded.
      3. Otherwise, source HIDs are parsed once, fully pre-processed
         (rescaling + scanpoint annotation + optional annotation adjustment),
         and streamed into the memmap cache.

    At read time (``__getitem__``) three memmaps are opened lazily (once per
    worker process) and a single row is copied out to build a fresh
    ``HIDImage``.
    """

    def __init__(
        self,
        root: PathLike,
        scaling_strategy: ScalingStrategy,
        dataset_strategy: DatasetStrategy,
        cache_dir: PathLike,
        adjustment_of_annotations: str | None = None,
        limit: int | None = None,
        skip_if_invalid_ladder: bool = False,
        skip_if_no_annotation: bool = True,
        include_size_standard: bool = False,
        data_loading_strategy: str = 'superior',
        transform: TransformDataCallable | None = None,
        # When True, the cache is fully realized into RAM after build/load.
        load_in_memory: bool = False,
        # When True, HIDs without annotations are still cached (for eval/labeltool).
        allow_missing_annotations: bool = False,
        use_cache: bool = True,
    ) -> None:
        super().__init__()

        self.root = Path(root)
        self.adjustment_of_annotations = adjustment_of_annotations
        self.skip_if_invalid_ladder = skip_if_invalid_ladder
        self.skip_if_no_annotation = skip_if_no_annotation
        self.include_size_standard = include_size_standard
        self.data_loading_strategy = data_loading_strategy
        self._transform = transform
        self.load_in_memory = load_in_memory
        self.allow_missing_annotations = allow_missing_annotations
        self._scaling = scaling_strategy
        self._dataset_strategy = dataset_strategy
        self._default_panel = self._scaling.panel
        self._use_cache = use_cache

        if adjustment_of_annotations and adjustment_of_annotations not in ('top', 'complete'):
            raise ValueError(
                f"adjustment_of_annotations must be 'top' or 'complete', "
                f'got {adjustment_of_annotations!r}'
            )

        # ----- Cache discovery / build -------------------------------------
        key = compute_key(
            root=self.root,
            scaling_strategy=self._scaling,
            dataset_strategy=self._dataset_strategy,
            data_loading_strategy=self.data_loading_strategy,
            include_size_standard=self.include_size_standard,
            adjustment_of_annotations=self.adjustment_of_annotations,
            skip_if_invalid_ladder=self.skip_if_invalid_ladder,
            allow_missing_annotations=self.allow_missing_annotations,
        )
        
        self._cache_dir = cache_key_dir(
            Path(cache_dir) if self._use_cache else Path('/tmp/var/dnanet-cache/'),
            key
        )

        config_payload = build_config_payload(
            root=self.root,
            scaling_strategy=self._scaling,
            dataset_strategy=self._dataset_strategy,
            data_loading_strategy=self.data_loading_strategy,
            include_size_standard=self.include_size_standard,
            adjustment_of_annotations=self.adjustment_of_annotations,
            skip_if_invalid_ladder=self.skip_if_invalid_ladder,
            allow_missing_annotations=self.allow_missing_annotations,
        )

        self._resolve_cache(config_payload)

        self._reader = MemmapCacheReader(self._cache_dir)
        self._index: List[IndexEntry] = self._reader.load_index()
        self._paths: List[Path] = [Path(e.path) for e in self._index]

        # ----- Optional downsample (indexes only; cache stays intact) ------
        if limit is not None and limit < len(self._index):
            sampled = random.sample(range(len(self._index)), limit)
            sampled.sort()
            self._index = [self._index[i] for i in sampled]
            self._paths = [self._paths[i] for i in sampled]
            self._row_remap: list[int] = sampled
            logger.info('Limiting to {} files (random sample)', len(self._index))
        else:
            self._row_remap = list(range(len(self._index)))

        if len(self._index) == 0:
            raise ValueError(
                f'No valid HID images found in {self.root}. Check paths and strategy configuration.'
            )

        logger.info(
            f'Transforming all samples with {self.transform.__class__}'
            if self.transform is not None
            else 'No transform applied to samples'
        )
        if self.load_in_memory:
            self._load_cache_into_ram()

        logger.info(
            'HIDDataset ready: {} indexed samples (cache {})', len(self._index), self._cache_dir
        )

    # -- In-memory materialization ---------------------------------------- #

    _RAM_BUDGET_FRACTION = 0.5  # refuse if cache exceeds this fraction of total RAM.

    def _load_cache_into_ram(self) -> None:
        """Copy the three memmaps into RAM-resident arrays, with a hard RAM guard."""
        estimated = self._reader.memmap_bytes()
        total_ram = self._total_ram_bytes()
        budget = int(total_ram * self._RAM_BUDGET_FRACTION) if total_ram else 0

        if budget and estimated > budget:
            raise RuntimeError(
                f'load_in_memory=True refused: cache would use '
                f'{estimated / 1e9:.2f} GB which exceeds '
                f'{self._RAM_BUDGET_FRACTION:.0%} of total RAM '
                f'({total_ram / 1e9:.2f} GB). Set load_in_memory=False '
                f'and stream from the memmap instead.'
            )

        logger.info('Materializing cache into RAM: ~{:.2f} GB', estimated / 1e9)
        self._reader.materialize()

    @staticmethod
    def _total_ram_bytes() -> int:
        """Best-effort total physical RAM in bytes; 0 if not determinable."""
        try:
            return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        except (ValueError, AttributeError, OSError):
            return 0

    # -- Cache resolution -------------------------------------------------- #

    def _resolve_cache(self, config_payload: dict[str, Any]) -> None:
        """Ensure the cache for our key exists and matches current sources.

        Returns the final list of source file entries (may be empty on a warm
        cache hit). The list is only computed when needed for either
        fingerprint validation on a hit, or a full/partial build on a miss.
        """
        # Collect sources. This is used for (a) fingerprint validation and
        # (b) driving a fresh build; it's a cheap walk (stat-only).
        file_entries = list(self._dataset_strategy.collect_dataset_files(self.root, self._scaling))

        if is_complete(self._cache_dir) and self._use_cache:
            source_paths = [e[0] for e in file_entries]
            if validate_fingerprint(self._cache_dir, config_payload, source_paths):
                logger.info('Cache hit: {}', self._cache_dir)
                return
            logger.warning('Cache fingerprint stale at {}; rebuilding', self._cache_dir)

        logger.info(
            'Building cache at {} from {} source files',
            self._cache_dir,
            len(file_entries),
        )
        self._build_cache(file_entries, config_payload)
        return

    def _build_cache(
        self,
        file_entries: list,
        config_payload: dict[str, Any],
    ) -> None:
        source_paths = [e[0] for e in file_entries]
        with MemmapCacheWriter(self._cache_dir) as writer:
            resume = writer.resume_paths()
            if resume:
                remaining = [e for e in file_entries if str(e[0]) not in resume]
                logger.info(
                    'Resuming build: {} already cached, {} remaining',
                    len(file_entries) - len(remaining),
                    len(remaining),
                )
            else:
                remaining = file_entries

            for image in self._load_images(remaining):
                if image.data is None or image.scaler is None:
                    logger.debug('Skipping {} during cache build (no data/scaler)', image.path)
                    continue
                if image.annotation is None and not self.allow_missing_annotations:
                    logger.debug('Skipping {} during cache build (no annotation)', image.path)
                    continue
                writer.write(image)
            writer.finalize(config_payload, source_paths)

    # -- Source → fully-preprocessed HIDImage ------------------------------ #

    def _load_images(
        self,
        file_entries: list[Tuple[Path, Any, Path | None]],
    ) -> Generator[HIDImage, None, None]:
        """Parse source HIDs and yield fully pre-processed HIDImages.

        All heavy preprocessing (profile parsing, size-standard rescaling,
        scaler extraction, allele→scanpoint translation, annotation
        adjustment) happens here so the resulting image is cache-ready.
        """
        skipped_data = 0
        skipped_alleles = 0
        skipped_ladder = 0

        for entry in tqdm(file_entries, desc='Building cache', total=len(file_entries)):
            path: Path = entry[0]
            annotation = entry[1]
            ladder_path: Path | None = entry[2]

            allele_annotation: AlleleAnnotation | None = (
                annotation if isinstance(annotation, AlleleAnnotation) else None
            )

            current_panel: Panel = self._default_panel
            if ladder_path is not None:
                adjusted = Ladder.create_adjusted_panel(
                    ladder_path=ladder_path,
                    catalog=LadderAlleleCatalog.from_panel(self._default_panel),
                    data_loading_strategy=self.data_loading_strategy,
                    scaling_strategy=self._scaling,
                    dataset_strategy=self._dataset_strategy,
                )
                if adjusted:
                    current_panel = adjusted
                elif self.skip_if_invalid_ladder:
                    skipped_ladder += 1
                    continue

            image = HIDImage(
                path=path,
                scaling_strategy=self._scaling,
                adjusted_panel=current_panel,
                include_size_standard=self.include_size_standard,
                data_loading_strategy=self.data_loading_strategy,
                allele_annotation=allele_annotation,
                load_in_memory=True,  # force in-memory just long enough to write it
            )

            if image.data is None:
                skipped_data += 1
                logger.debug('Skipping {}: no data', path.name)
                continue

            if isinstance(annotation, AlleleAnnotation):
                scanpoint_annotation = self._translate_allele_to_scanpoint_annotation(
                    allele_annotation=annotation,
                    adjusted_panel=current_panel,
                    scaler=image.scaler,
                    include_size_standard=self.include_size_standard,
                    scaling_strategy=self._scaling,
                )
            else:
                scanpoint_annotation = annotation

            if self.adjustment_of_annotations and scanpoint_annotation is not None:
                scanpoint_annotation = self._adjust_annotations(
                    [image],
                    [scanpoint_annotation],
                    adjustment_type=self.adjustment_of_annotations,
                )[0]

            image.annotation = scanpoint_annotation

            if image.annotation is None and not self.allow_missing_annotations:
                skipped_alleles += 1
                logger.debug('{}: no annotation/called alleles', path.name)
                continue

            yield image

        if skipped_data:
            logger.warning('Skipped {} images with missing data', skipped_data)
        if skipped_alleles:
            logger.warning('Skipped {} images with missing annotations', skipped_alleles)
        if skipped_ladder:
            logger.warning('Skipped {} images with invalid ladders', skipped_ladder)

    @staticmethod
    def _translate_allele_to_scanpoint_annotation(
        allele_annotation: AlleleAnnotation,
        adjusted_panel: Panel,
        scaler: np.ndarray,
        include_size_standard: bool,
        scaling_strategy: ScalingStrategy,
    ) -> ScanpointAnnotation:
        """Translate allele-level annotation to a scanpoint binary mask."""
        kit_num_dyes = scaling_strategy.kit.num_dyes

        num_dyes = kit_num_dyes if include_size_standard else kit_num_dyes - 1
        scanpoint_annotation = np.zeros(
            (num_dyes, scaling_strategy.scanpoint_resolution),
            dtype=np.int8,
        )
        for locus in allele_annotation.data:
            for allele in locus.alleles:
                _, left_bin, right_bin = adjusted_panel.get_allele_basepair_and_bins(
                    locus.name, allele.name
                )
                left_scanpoint = np.argmin(np.abs(scaler - left_bin))
                right_scanpoint = np.argmin(np.abs(scaler - right_bin))
                scanpoint_annotation[locus.dye_row, left_scanpoint:right_scanpoint] = 1

        return ScanpointAnnotation(data=scanpoint_annotation)

    @staticmethod
    def _adjust_annotations(
        profiles: List[HIDImage],
        annotations: List[Optional[ScanpointAnnotation]],
        adjustment_type: str = 'top',
        threshold: int = 0,
    ) -> List[Optional[ScanpointAnnotation]]:
        """Adjust annotations to mark peak tops ('top') or full peak extents ('complete')."""
        assert len(profiles) == len(annotations)

        for profile, annotation in zip(profiles, annotations, strict=True):
            if annotation is None:
                continue
            if profile.data is None:
                continue

            for dye_idx, dye_data in enumerate(profile.data):
                _annotations = np.where(annotation.data[dye_idx] == 1)[0]
                if _annotations.size == 0:
                    continue
                annotation_groups = np.split(
                    _annotations, np.where(np.diff(_annotations) != 1)[0] + 1
                )
                for ann_group in annotation_groups:
                    annotation.data[dye_idx, ann_group] = 0.0
                    peak_idx = find_peak_idx_near_or_in_range(dye_data, ann_group, threshold)

                    if peak_idx.size == 0:
                        logger.warning(
                            f'No peak found above {threshold}rfu. '
                            f'Original annotation is removed '
                            'and no adjustment is applied '
                            f'(dye {dye_idx}, bin {ann_group}, '
                            f'rfus {dye_data[ann_group].flatten()}).'
                        )
                    else:
                        if adjustment_type == 'complete':
                            start, end = find_peak_boundary(dye_data, int(peak_idx), threshold)
                            annotation.data[dye_idx, np.arange(start, end + 1)] = 1.0
                        elif adjustment_type == 'top':
                            annotation.data[dye_idx, peak_idx] = 1.0
                        else:
                            raise ValueError(
                                'Unknown adjustment type found: '
                                f'{adjustment_type}. Please provide'
                                ' either `top` or `complete`.'
                            )
        return annotations

    # -- Properties -------------------------------------------------------- #

    @property
    def transform(self) -> TransformDataCallable | None:
        return self._transform

    @property
    def images(self) -> List[HIDImage]:
        """Lightweight stub images exposing just ``.path`` for split logic.

        Full ``HIDImage`` objects are only materialized by ``__getitem__``.
        """
        return [self._stub_image(i) for i in range(len(self._index))]

    @property
    def dataset_strategy(self) -> DatasetStrategy:
        return self._dataset_strategy

    # -- Dunder ------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        return f'HIDDataset(root={self.root.name}, n={len(self._index)})'

    def __getitem__(self, index: int) -> Any:
        image = self.get_image(index)

        if self._transform:
            return self._transform(image)
        return image

    # -- Internal helpers -------------------------------------------------- #

    def _stub_image(self, idx: int) -> HIDImage:
        """Empty HIDImage with only ``.path`` set, for split-logic consumers."""
        entry = self._index[idx]
        return HIDImage(
            path=entry.path,
            scaling_strategy=self._scaling,
            include_size_standard=self.include_size_standard,
            load_in_memory=False,
        )

    def get_stub_image(self, index: int) -> HIDImage:
        """Return a lightweight image object with only split metadata populated."""
        return self._stub_image(index)

    def get_image(self, index: int) -> HIDImage:
        """Materialize a single cached HID image without applying transforms."""
        entry = self._index[index]
        row = self._row_remap[index]
        data, annotation_arr, scaler = self._reader.get_row(row)
        return self._materialize(entry, data, annotation_arr, scaler)

    def _materialize(
        self,
        entry: IndexEntry,
        data: np.ndarray,
        annotation_arr: np.ndarray,
        scaler: np.ndarray,
    ) -> HIDImage:
        """Build a full HIDImage from cache arrays + sidecar JSON metadata."""
        from dnanet.core.panel import Panel
        from dnanet.core.marker import Marker

        allele_json = self._reader.allele_json(entry.allele_key)
        allele_annotation: AlleleAnnotation | None = None
        if allele_json:
            markers: list[Marker] = [Marker.from_dict(d) for d in json.loads(allele_json)]
            allele_annotation = AlleleAnnotation(data=markers)

        panel_json = self._reader.panel_json(entry.panel_key)
        adjusted_panel: Panel | None = None
        if panel_json:
            adjusted_panel = Panel(
                markers=tuple(Marker.from_dict(d) for d in json.loads(panel_json))
            )

        meta = json.loads(entry.meta_json) if entry.meta_json else {}

        image = HIDImage(
            path=entry.path,
            scaling_strategy=self._scaling,
            adjusted_panel=adjusted_panel,
            include_size_standard=self.include_size_standard,
            allele_annotation=allele_annotation,
            load_in_memory=False,
            meta=meta,
        )
        image._data = data
        image._scaler = scaler
        image.annotation = ScanpointAnnotation(data=annotation_arr) if entry.has_annotation else None
        return image
