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

    from dnanet.data.strategies import NFIRnDStrategy, PowerPlexFusion6CStrategy

    dataset = HIDDataset(
        root="data/2p_5p_Dataset_NFI/Raw data .HID files",
        scaling_strategy=PowerPlexFusion6CStrategy(),
        dataset_strategy=NFIRnDStrategy(),
    )
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Tuple, Optional, Generator

import numpy as np
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm

from dnanet.core.annotation import Annotation, AlleleAnnotation, ScanpointAnnotation
from dnanet.data.dataset import TransformableDataset
from dnanet.data.image import HIDImage
from dnanet.data.ladders.ladder import Ladder
from dnanet.data.ladders.ladder_allele_catalog import LadderAlleleCatalog
from dnanet.data.preprocessing.peaks import find_peak_boundary, find_peak_idx_near_or_in_range
from dnanet.data.strategies.datasets import DatasetStrategy
from dnanet.data.strategies.scaling import ScalingStrategy

if TYPE_CHECKING:
    from dnanet.core.panel import Panel
    from dnanet.core.types import PathLike
    from dnanet.data.transformer import TransformDataCallable


class HIDDataset(Dataset, TransformableDataset):
    """Load HID files from a directory into an in-memory dataset.

    This is the primary dataset class for forensic DNA profiles. It:
    1. Walks a root directory for ``.hid`` files
    2. Filters to sample files (excluding ladders, controls)
    3. Matches each sample to its annotation and best ladder
    4. Builds an adjusted panel from the ladder
    5. Creates ``HIDImage`` instances and filters out invalid ones

    Requires a kit scaling strategy and dataset strategy to be passed during
    construction. The dataset strategy provides dataset-specific file
    collection, annotations, labels, and split helpers.

    Args:
        root: Root directory containing HID files (searched recursively).
        ladder_alleles_csv: CSV with expected ladder alleles per dye.
        analysis_threshold_type: ``"DTH"`` (high) or ``"DTL"`` (low).
        adjustment_of_annotations: ``"top"`` or ``"complete"`` peak adjustment,
            or ``None`` to skip.
        limit: Maximum number of images to load.
        skip_if_invalid_ladder: Skip images whose ladder fails to parse.
        include_size_standard: Include the 6th dye channel in the data.
        data_loading_strategy: ``"raw"``, ``"analyzed"``, or ``"superior"``.
    """

    def __init__(
        self,
        root: PathLike,
        scaling_strategy: ScalingStrategy,
        dataset_strategy: DatasetStrategy,
        analysis_threshold_type: str = 'DTH',
        adjustment_of_annotations: str | None = None,
        limit: int | None = None,
        skip_if_invalid_ladder: bool = False,
        include_size_standard: bool = False,
        data_loading_strategy: str = "superior",
        transform: TransformDataCallable | None = None,
        load_in_memory: bool = False,
    ) -> None:
        super().__init__()

        self.root = Path(root)
        self.analysis_threshold_type = analysis_threshold_type
        self.adjustment_of_annotations = adjustment_of_annotations
        self.skip_if_invalid_ladder = skip_if_invalid_ladder
        self.include_size_standard = include_size_standard
        self.data_loading_strategy = data_loading_strategy
        self._transform = transform
        self.load_in_memory = load_in_memory
        self._scaling = scaling_strategy
        self._dataset_strategy = dataset_strategy
        self._default_panel = self._scaling.panel

        if adjustment_of_annotations and adjustment_of_annotations not in ('top', 'complete'):
            raise ValueError(
                f"adjustment_of_annotations must be 'top' or 'complete', "
                f'got {adjustment_of_annotations!r}'
            )


        # Collect files, apply limit, load images
        file_entries = list(self._dataset_strategy.collect_dataset_files(self.root, self._scaling))
        logger.info('Found {} sample files to process', len(file_entries))

        self._data: List[HIDImage] = list(self._load_images(file_entries))

        if limit:
            self._data = random.sample(self._data, min(limit, len(self._data)))
            logger.info('Limiting to {} files (random sample)', len(self._data))

        if len(self._data) == 0:
            raise ValueError(
                f'No valid HID images found in {self.root}. '
                f'Check paths and strategy configuration.'
            )

        logger.info(
            f'Transforming all samples with {self.transform.__class__}'
            if self.transform is not None
            else 'No transform applied to samples'
        )

        logger.info('Loaded {} valid HID images', len(self._data))

    # -- Image loading ----------------------------------------------------- #

    def _load_images(
        self, file_entries: List[Tuple[Path, Annotation | None, Path | None]]
    ) -> Generator[HIDImage, None, None]:
        """Create HIDImage instances, loading data and filtering invalid ones."""
        ladder_cache: dict[str, Panel | None] = {}

        skipped_data = 0
        skipped_alleles = 0
        skipped_ladder = 0

        for entry in tqdm(file_entries, desc='Loading images', total=len(file_entries), ):
            path: Path = entry[0]
            ladder_path: Path | None = entry[2]
            annotation = entry[1]  # (name, file) or None

            if isinstance(annotation, AlleleAnnotation):
                allele_annotation = annotation
            else:
                allele_annotation = None

            # Build adjusted panel from ladder
            _current_panel = self._default_panel
            if ladder_path:
                adjusted = Ladder.create_adjusted_panel(
                    ladder_path=ladder_path,
                    catalog=LadderAlleleCatalog.from_panel(self._default_panel),
                    data_loading_strategy=self.data_loading_strategy,
                    scaling_strategy=self._scaling,
                    dataset_strategy=self.dataset_strategy
                )
                if adjusted:
                    _current_panel = adjusted
                elif self.skip_if_invalid_ladder:
                    skipped_ladder += 1
                    continue

            # Create HIDImage
            image = HIDImage(
                path=path,
                scaling_strategy=self._scaling,
                dataset_strategy=self._dataset_strategy,
                adjusted_panel=_current_panel,
                include_size_standard=self.include_size_standard,
                data_loading_strategy=self.data_loading_strategy,
                allele_annotation=allele_annotation,
                load_in_memory=self.load_in_memory,
            )

            # Trigger lazy load and validate
            # TODO: should we always force data to be loaded from disk?
            if image.data is None:
                skipped_data += 1
                logger.debug('Skipping {}: no data', path.name)
                continue

            if isinstance(annotation, AlleleAnnotation):
                scanpoint_annotation = self._translate_allele_to_scanpoint_annotation(
                    allele_annotation=annotation,
                    adjusted_panel=_current_panel,
                    scaler=image.scaler,
                    include_size_standard=self.include_size_standard,
                    scaling_strategy=self._scaling,
                )
            else:
                scanpoint_annotation = annotation

            if self.adjustment_of_annotations:
                scanpoint_annotation = self._adjust_annotations(
                    [image], [scanpoint_annotation], adjustment_type=self.adjustment_of_annotations
                )[0]

            image.annotation = scanpoint_annotation

            if image.annotation is None:
                skipped_alleles += 1
                logger.debug('{}: no annotation/called alleles', path.name)

            yield image

        if skipped_data:
            logger.warning('Skipped {} images with missing data', skipped_data)
        if skipped_alleles:
            logger.warning('Skipped {} images with missing annotations', skipped_alleles)
        if skipped_ladder:
            logger.warning('Skipped {} images with invalid ladders', skipped_ladder)

    @staticmethod
    def _translate_allele_to_scanpoint_annotation(
        allele_annotation: AlleleAnnotation, adjusted_panel: Panel, scaler: np.ndarray, include_size_standard: bool, scaling_strategy: ScalingStrategy
    ) -> ScanpointAnnotation:
        """Translates allele annotation to scanpoint annotation.

        Achieves this by finding the closest scanpoint indices for the left and right bins
        of each allele using the provided scaler and adjusted panel. The entire allelic bin is annotated as 1.

        All non-annotated scanpoints are labeled as 0, while annotated scanpoints are labeled as 1.
        The resulting scanpoint annotation is a binary matrix of shape (num_dyes, num_scanpoints)

            :param allele_annotation: AlleleAnnotation object containing the allele annotations to be translated.
            :param adjusted_panel: Panel object containing the adjusted panel information to be used for translation.
            :param scaler: numpy array containing the scaler values to be used for finding the closest scanpoint indices.
            :return: ScanpointAnnotation object containing the translated scanpoint annotation.
        """
        kit_num_dyes = scaling_strategy.kit.num_dyes

        num_dyes = kit_num_dyes if include_size_standard else kit_num_dyes - 1
        scanpoint_annotation = np.zeros(
            (
                num_dyes,
                scaling_strategy.scanpoint_resolution,
            ),
            dtype=np.int8,
        )
        for locus in allele_annotation.data:
            for allele in locus.alleles:
                # for each allele, find the left and right bin of the allele using the panel that has been adjusted by the corresponding ladder.
                _, left_bin, right_bin = adjusted_panel.get_allele_basepair_and_bins(
                    locus.name, allele.name
                )

                # use the scaler to find the closest scanpoint indices for the left and right bins of the allele
                left_scanpoint = np.argmin(np.abs(scaler - left_bin))
                right_scanpoint = np.argmin(np.abs(scaler - right_bin))

                scanpoint_annotation[locus.dye_row, left_scanpoint:right_scanpoint] = 1

        return ScanpointAnnotation(data=scanpoint_annotation)

    @staticmethod
    def _adjust_annotations(
        profiles: List[HIDImage],
        annotations: List[Optional[ScanpointAnnotation]],
        adjustment_type: str = 'top',
        threshold: int = 40,
    ) -> List[Optional[ScanpointAnnotation]]:
        """Adjust the annotation of the image.

        If `adjustment_type` is 'top', (by default) we label the top of the peak, instead of the
        entire bin. If the type is 'complete', we find the entire peak and label this.
        Note that the original image annotations are overwritten in place.
        """
        # TODO: This function is now called for single images, remove the loop?

        assert len(profiles) == len(annotations)

        for profile, annotation in zip(profiles, annotations, strict=True):
            if annotation is None:
                continue
            if profile.data is None:
                continue

            for dye_idx, dye_data in enumerate(profile.data):
                # find indices of groups of positive annotations
                _annotations = np.where(annotation.data[dye_idx] == 1)[0]
                if _annotations.size == 0:  # no annotation present in this dye
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
                            # find the boundary of the peak and annotate the range
                            start, end = find_peak_boundary(dye_data, int(peak_idx), threshold)
                            annotation.data[dye_idx, np.arange(start, end + 1)] = 1.0
                        elif adjustment_type == 'top':
                            # label only the top of the peak
                            annotation.data[dye_idx, peak_idx] = 1.0
                        else:
                            raise ValueError(
                                'Unknown adjustment type found: '
                                f'{adjustment_type}. Please provide'
                                ' either `top` or `complete`.'
                            )
        return annotations

    @property
    def transform(self) -> TransformDataCallable | None:
        return self._transform

    @property
    def images(self) -> List[HIDImage]:
        return self._data
    @property
    def data(self) -> List[HIDImage]:
        """Protected property list of HIDImages in the dataset."""
        return self._data

    @property
    def dataset_strategy(self) -> DatasetStrategy:
        """Protected property for the dataset strategy."""
        return self._dataset_strategy

    # -- Dunder ----------------------------------------------------------- #

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self._data)

    def __repr__(self) -> str:
        """String representation of the dataset, containing the root path and length."""
        return f'HIDDataset(root={self.root.name}, n={len(self._data)})'

    def __getitem__(self, index: int) -> Any:
        item = self._data[index]

        if self._transform:
            item = self._transform(item)

        return item
