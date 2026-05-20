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
from typing import TYPE_CHECKING, Any, List, Tuple, Generator
from pathlib import Path
from collections import defaultdict

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
from dnanet.core.constants import LabelCategory
from dnanet.core.annotation import SpanAnnotation, AlleleAnnotation, ScanpointAnnotation
from dnanet.data.ladders.ladder import Ladder
from dnanet.data.cache.fingerprint import build_config_payload
from dnanet.data.preprocessing.peaks import (
    find_peak_boundary,
    find_valley_idx_in_range,
    find_peak_idx_near_or_in_range,
    find_absolute_peak_idx_in_range,
)
from dnanet.data.ladders.ladder_allele_catalog import LadderAlleleCatalog


if TYPE_CHECKING:
    from dnanet.core.panel import Panel
    from dnanet.core.types import PathLike
    from dnanet.core.marker import Marker
    from dnanet.data.transformer import TransformDataCallable
    from dnanet.data.strategies.scaling import ScalingStrategy
    from dnanet.data.strategies.datasets import DatasetStrategy


# ---------------------------------------------------------------------------
# Per-class span-adjustment dispatch
# ---------------------------------------------------------------------------
# Maps each LabelCategory to the function used to locate the representative
# scanpoint within an annotated region.  ``None`` means the class has no
# adjustment defined yet; its span is kept as-is with a logged warning.
#
# Signature of every entry: (signal: np.ndarray, index_range: np.ndarray,
#                             threshold: float) -> np.ndarray
_CLASS_ADJUST_FN: dict[LabelCategory, object] = {
    LabelCategory.UNLABELED: None,  # background — never adjusted
    LabelCategory.ALLELE: find_peak_idx_near_or_in_range,
    LabelCategory.STUTTER: find_peak_idx_near_or_in_range,
    LabelCategory.PULL_UP: find_peak_idx_near_or_in_range,
    LabelCategory.BLEED_THROUGH: find_absolute_peak_idx_in_range,  # ABS for peak/valley indifference
    LabelCategory.SPIKE: find_peak_idx_near_or_in_range,
    LabelCategory.DYE_BLOB: find_peak_idx_near_or_in_range,
    LabelCategory.ARTEFACT: find_peak_idx_near_or_in_range,
    LabelCategory.UNCLEAR: find_peak_idx_near_or_in_range,
    LabelCategory.SHOULDER: find_peak_idx_near_or_in_range,
    LabelCategory.FOREIGN_DNA: find_peak_idx_near_or_in_range,
    LabelCategory.OVERLOADING_ARTEFACT: find_peak_idx_near_or_in_range,
}

# Classes for which the 'complete' adjustment type is meaningful (i.e. the
# region can be expanded to a full peak boundary via find_peak_boundary).
# For all other classes the 'complete' mode falls back to 'top'.
_CLASS_SUPPORTS_COMPLETE: frozenset[LabelCategory] = frozenset(
    {
        LabelCategory.ALLELE,
        LabelCategory.STUTTER,
        LabelCategory.SHOULDER,
        LabelCategory.FOREIGN_DNA,
    }
)


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
        cache_dir: PathLike | None = None,
        adjustment_of_annotations: str | None = None,
        limit: int | None = None,
        skip_if_invalid_ladder: bool = False,
        include_size_standard: bool = False,
        data_loading_strategy: str = 'superior',
        transform: TransformDataCallable | None = None,
        # When True, the cache is fully realized into RAM after build/load.
        load_in_memory: bool = False,
        # When True, HIDs without annotations are still cached (for eval/labeltool).
        allow_missing_annotations: bool = False,
        # When False, skip fingerprint validation and accept cache as-is.
        cache_validate: bool = True,
    ) -> None:
        super().__init__()

        self.root = Path(root)
        self.adjustment_of_annotations = adjustment_of_annotations
        self.skip_if_invalid_ladder = skip_if_invalid_ladder
        self.include_size_standard = include_size_standard
        self.data_loading_strategy = data_loading_strategy
        self._transform = transform
        self.load_in_memory = load_in_memory
        self.allow_missing_annotations = allow_missing_annotations
        self._scaling = scaling_strategy
        self._dataset_strategy = dataset_strategy
        self._default_panel = self._scaling.panel

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

        self._cache_validate = cache_validate
        self._use_cache = False if cache_dir is None else True
        self._cache_dir = cache_key_dir(
            Path(cache_dir) if cache_dir is not None else Path('/tmp/var/dnanet-cache/'), key
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
        logger.info('Looking for dataset cache...')
        file_entries = list(self._dataset_strategy.collect_dataset_files(self.root, self._scaling, allow_missing_annotations=self.allow_missing_annotations))

        if is_complete(self._cache_dir) and self._use_cache:
            source_paths = [e[0] for e in file_entries]
            if self._cache_validate and validate_fingerprint(
                self._cache_dir, config_payload, source_paths, root=self.root
            ):
                logger.info('Cache hit: {}', self._cache_dir)
                return
            if not self._cache_validate:
                logger.info('Cache validation disabled, using {} as-is', self._cache_dir)
                return
            logger.warning('Cache fingerprint stale at {}; rebuilding', self._cache_dir)

        logger.info(
            'Building {}cache at {} from {} source files',
            f'{"temp-" if not self._use_cache else ""}',
            self._cache_dir,
            len(file_entries),
        )
        self._build_cache(file_entries, config_payload, root=self.root)
        return

    def _build_cache(
        self,
        file_entries: list,
        config_payload: dict[str, Any],
        root: PathLike | None = None,
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
                writer.write(image)
            writer.finalize(config_payload, source_paths, root=root)

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

        pbar_desc = f'Building {"temp-" if not self._use_cache else ""}cache'
        for entry in tqdm(file_entries, desc=pbar_desc, total=len(file_entries)):
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
                # Adjustment is interleaved per-allele during translation to
                # prevent overlapping bins from merging into one contiguous
                # block before peak-finding runs.
                scanpoint_annotation = self._translate_allele_to_scanpoint_annotation(
                    allele_annotation=annotation,
                    adjusted_panel=current_panel,
                    scaler=image.scaler,
                    scaling_strategy=self._scaling,
                    profile_data=image.data if self.adjustment_of_annotations else None,
                    adjustment_type=self.adjustment_of_annotations,
                )
            elif isinstance(annotation, SpanAnnotation):
                # Adjustment runs per class-layer on the 3-D tensor before
                # argmax-flattening so that finer-grained class labels (e.g.
                # shoulder) are not erased by coarser ones during collapse.
                if self.adjustment_of_annotations:
                    adjusted_2d = self._adjust_and_flatten_span_annotation(
                        profile=image,
                        span_annotation=annotation,
                        adjustment_type=self.adjustment_of_annotations,
                    )
                    scanpoint_annotation = ScanpointAnnotation(adjusted_2d)
                else:
                    scanpoint_annotation = self._dataset_strategy._span_to_scanpoint_annotation(
                        annotation.data, path.stem
                    )
            elif isinstance(annotation, ScanpointAnnotation) and self.adjustment_of_annotations:
                # Is there a scenario where we're already loading a ScanpointAnnotation
                # and want to adjust it? If so, that'd be done here.
                logger.warning(
                    'Adjust annotations is provided but direct ScanpointAnnotations are not adjusted'
                )
                scanpoint_annotation = annotation
            elif annotation is not None:
                logger.info(f'Encountered unknown annotation type: {type(annotation)}')
                scanpoint_annotation = annotation
            else:
                scanpoint_annotation = annotation

            if scanpoint_annotation and not self.include_size_standard:
                scanpoint_annotation = ScanpointAnnotation(scanpoint_annotation.data[:-1])

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
        scaling_strategy: ScalingStrategy,
        profile_data: np.ndarray | None = None,
        adjustment_type: str | None = None,
        threshold: int = 0,
    ) -> ScanpointAnnotation:
        """Translate allele-level annotation to a scanpoint binary mask.

        When *profile_data* and *adjustment_type* are both supplied, adjustment
        is interleaved per allele **before** any scanpoints are written to the
        output array.  This avoids the bin-overlap bug where two overlapping
        allele bins merge into one contiguous block of 1s prior to adjustment,
        making the subsequent peak-finding treat them as a single allele.

        Without those arguments the method falls back to marking the full bin
        range as 1.  Either way, a two-pass approach is used: scanpoint ranges
        are collected first (per dye row), then written with a forced 0-gap
        inserted at the boundary between any adjacent or overlapping bins so
        that isolated islands of 1s are always preserved.
        """
        scanpoint_annotation = np.zeros(
            (scaling_strategy.kit.num_dyes, scaling_strategy.scanpoint_resolution),
            dtype=np.int8,
        )

        do_adjust = adjustment_type is not None and profile_data is not None

        # Pass 1: resolve scanpoint ranges for every allele, grouped by dye.
        intervals_by_dye: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for locus in allele_annotation.data:
            for allele in locus.alleles:
                _, left_bin, right_bin = adjusted_panel.get_allele_basepair_and_bins(
                    locus.name, allele.name
                )
                left_sp = int(np.argmin(np.abs(scaler - left_bin)))
                right_sp = int(np.argmin(np.abs(scaler - right_bin)))
                intervals_by_dye[locus.dye_row].append((left_sp, right_sp))

        # Pass 2: write each allele, then enforce 0-gaps at adjacent/overlapping boundaries.
        for dye_row, intervals in intervals_by_dye.items():
            sorted_ivs = sorted(intervals)

            if do_adjust:
                assert profile_data is not None  # guaranteed by do_adjust
                dye_signal = profile_data[dye_row]
                for left_sp, right_sp in sorted_ivs:
                    ann_range = np.arange(left_sp, right_sp)
                    if ann_range.size == 0:
                        continue
                    peak_idx = find_peak_idx_near_or_in_range(dye_signal, ann_range, threshold)
                    if peak_idx.size == 0:
                        logger.warning(
                            'No peak found above {}rfu. Original annotation removed '
                            '(dye {}, bin {}:{}, rfus {}).',
                            threshold,
                            dye_row,
                            left_sp,
                            right_sp,
                            dye_signal[ann_range].flatten(),
                        )
                    elif adjustment_type == 'top':
                        scanpoint_annotation[dye_row, peak_idx] = 1
                    elif adjustment_type == 'complete':
                        start, end = find_peak_boundary(dye_signal, int(peak_idx), threshold)
                        scanpoint_annotation[dye_row, start : end + 1] = 1
                    else:
                        raise ValueError(
                            f'Unknown adjustment_type {adjustment_type!r}. Use "top" or "complete".'
                        )
            else:
                for left_sp, right_sp in sorted_ivs:
                    scanpoint_annotation[dye_row, left_sp:right_sp] = 1

                # Insert 0-gaps so adjacent/overlapping bins stay as separate islands.
                for i in range(len(sorted_ivs) - 1):
                    right_i = sorted_ivs[i][1]
                    left_next = sorted_ivs[i + 1][0]
                    if right_i >= left_next and left_next > 0:
                        scanpoint_annotation[dye_row, left_next - 1] = 0

        return ScanpointAnnotation(data=scanpoint_annotation)

    def _adjust_and_flatten_span_annotation(
        self,
        profile: HIDImage,
        span_annotation: SpanAnnotation,
        adjustment_type: str,
        threshold: int = 0,
    ) -> np.ndarray:
        """Adjust a 3-D span annotation per class, then flatten to a 2-D label array.

        For each dye channel and each non-background class, contiguous annotated
        regions are found and reduced to a single peak scanpoint (``'top'``) or
        a full peak extent (``'complete'``), exactly as in
        :meth:`_adjust_annotations_binary` but operating on each class layer
        independently.  Reducing to point/boundary annotations before the argmax
        collapse eliminates the overlap-loss bug where a finer-grained class
        (e.g. ``shoulder``) that falls entirely inside a coarser one
        (e.g. ``allele``) is erased because argmax picks the lowest class index
        for positions shared by both spans.

        Args:
            profile: The HID profile whose signal is used for peak detection.
            span_annotation: The ``(num_dyes, scanpoints, num_classes)`` tensor.
            adjustment_type: ``'top'`` or ``'complete'``.
            threshold: Minimum RFU for a scanpoint to be considered a peak.

        Returns:
            ``(num_dyes, scanpoints)`` int8 array of class indices ready for
            wrapping in :class:`ScanpointAnnotation`.
        """
        if profile.data is None:
            return span_annotation.data.argmax(axis=-1).astype(np.int8)

        span_data = span_annotation.data.copy()
        num_classes = span_data.shape[-1]

        for dye_idx, dye_signal in enumerate(profile.data):
            for class_idx in range(1, num_classes):  # skip background (class 0)
                category = LabelCategory.from_index(class_idx)
                find_fn = _CLASS_ADJUST_FN.get(category)

                class_layer = span_data[dye_idx, :, class_idx]
                regions = np.where(class_layer == 1)[0]
                if regions.size == 0:
                    continue

                if find_fn is None:
                    logger.warning(
                        'No span adjustment defined for class {} ({}). '
                        'Keeping original span annotation.',
                        class_idx,
                        category.name,
                    )
                    continue

                groups = np.split(regions, np.where(np.diff(regions) != 1)[0] + 1)
                for group in groups:
                    if len(group) == 1:
                        # We assume that this span-annotation of 1px is not something an annotator would (conciously) do, hence we skip
                        logger.trace(f"Skipping 1px wide annotation 'group': {(dye_idx, group, class_idx)}")
                        continue
                    
                    span_data[dye_idx, group, class_idx] = 0
                    rep_idx = find_fn(dye_signal, group, threshold)  # type: ignore[operator]
                    if rep_idx.size == 0:
                        logger.warning(
                            'No representative point found above {}rfu. '
                            'Annotation removed (dye {}, class {}, bin {}:{}).',
                            threshold,
                            dye_idx,
                            category.name,
                            group[0],
                            group[-1],
                        )
                        continue

                    use_complete = (
                        adjustment_type == 'complete' and category in _CLASS_SUPPORTS_COMPLETE
                    )
                    if use_complete:
                        start, end = find_peak_boundary(dye_signal, int(rep_idx[0]), threshold)
                        span_data[dye_idx, start : end + 1, class_idx] = 1
                    else:
                        span_data[dye_idx, rep_idx, class_idx] = 1

        # Flatten: highest (most specific) class index wins when two peaks
        # happen to land on the same scanpoint after adjustment.
        weights = span_data * np.arange(num_classes, dtype=np.int8)
        return weights.max(axis=-1).astype(np.int8)

    # -- Properties -------------------------------------------------------- #

    @property
    def transform(self) -> TransformDataCallable | None:
        """Optional transform applied to each sample in ``__getitem__``."""
        return self._transform

    @property
    def images(self) -> List[HIDImage]:
        """Lightweight stub images exposing just ``.path`` for split logic.

        Full ``HIDImage`` objects are only materialized by ``__getitem__``.
        """
        return [self._stub_image(i) for i in range(len(self._index))]

    @property
    def dataset_strategy(self) -> DatasetStrategy:
        """Dataset strategy used for file discovery and annotation parsing."""
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
