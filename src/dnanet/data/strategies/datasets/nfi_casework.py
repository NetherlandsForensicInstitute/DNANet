"""NFI Zaaksdata strategy.

Handles the NFI Zaaksdata

Warning:
    This strategy is developed specifically for our in-house casework.
    Altough file-collection and caching logic might be of use for your own implementation,
    using this strategy for other/your own data does not make sense.

    Please see the documentation about developing your own strategy, or use the two other strategies for the open-source data.
"""

import hashlib
import itertools
import json
import os
import pickle
from pathlib import Path
from typing import Callable, Dict, Tuple, Mapping, Sequence, Generator

import numpy as np
from loguru import logger
from sklearn.model_selection import (
    KFold,
    train_test_split,
)
from torch.utils.data import Subset

from dnanet.core.annotation import Annotation, AlleleAnnotation, ScanpointAnnotation
from dnanet.core.constants import LabelCategory
from dnanet.core.types import PathLike
from dnanet.data.strategies.datasets.dataset import FileCategory, DatasetStrategy
from dnanet.data.strategies.datasets.nfi_rnd import NFIRnDStrategy
from dnanet.data.strategies.scaling.scaling import ScalingStrategy


class NFICaseStrategy(DatasetStrategy):
    _ROBOT_NAMES = ('3500XL_A', '3500XL_B', '3500XL_C', '3500XL_D')
    _CACHE_DIR = Path('/tmp/.nfi_zaaksdata_cache/')

    def __init__(self,
                 annotation_type: str = 'ATLT',
                 robot_selection: Sequence[str] | None = None,
                 span_annotations_path: PathLike | None = None,
        ) -> None:
        """
        Initialize the NFI casework strategy

        Available annotation types:
        - AT: only include profiles with a "high" analytical threshold (allele annotation)
        - LT: only include profiles with a low threshold (allele annotation)
        - ATLT: include profiles with either a high or low threshold (allele annotation)
        - span: include profiles with a span annotation (scanpoint annotation)
        """
        super().__init__()
        self.annotation_type = annotation_type
        self._robot_selection = robot_selection
        self._span_annotations_path = span_annotations_path


    def collect_dataset_files(
        self, root_path: str | Path, scaling_strategy: ScalingStrategy, **kwargs
    ) -> Generator[Tuple[Path, Annotation | None, Path | None], None, None]:
        """Collect all .HID files in the casework folders.

        Finds all .HID files and their corresponding annotation and ladder.
        Since this involves a lot of files and subdirectories, the result is cached into the user's /tmp folder
        when running for the first time, allowing subsequent runs to load the files faster compared to doing a full walk again.

        Args:
            root_path: The path in which to find the data, must contain a 'hids', and 'annotations' folder.
            scaling_strategy: The Scaling strategy to use for the data, PPF6C for our casework.
            **kwargs: Extra parameters allowed by the abstract method, not used here.

        Yields:
            A tuple of HID path, annotation, and ladder path
        """
        cache_key = hashlib.md5(
            json.dumps(
                {
                    'root_path': str(root_path),
                    'scaling_strategy': scaling_strategy.__class__.__name__,
                    **self.cache_signature(),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        cache_file = self._CACHE_DIR / f'collect-{cache_key}'

        if cache_file.exists():
            logger.debug(f'Reading collect_dataset_files from cache: {cache_file}')
            with cache_file.open('rb') as f:
                yield from pickle.load(f)
            return

        results = list(self._collect_dataset_files_uncached(root_path, scaling_strategy))

        cache_file.parent.mkdir(exist_ok=True, parents=True)
        with cache_file.open('wb') as f:
            pickle.dump(results, f)

        yield from results

    def _collect_dataset_files_uncached(
        self,
        root_path: str | Path,
        scaling_strategy: ScalingStrategy,
        folder_cache: bool = True,
    ) -> Generator[Tuple[Path, Annotation | None, Path | None], None, None]:
        path = Path(root_path)

        # Collect all .HID files from the robot folders
        hids_folder = path / 'hids'
        robots = self.find_robot_files(
            hids_folder,
            self._robot_selection,
            cache=folder_cache,
        )
        resolve_annotation = self._build_annotation_resolver(path, scaling_strategy, self.annotation_type, self._span_annotations_path)

        for hid_file in robots:
            file_category = self.categorize_file(hid_file.name)
            if file_category != 'sample':
                logger.debug(f'Skipping {file_category} file: {hid_file.stem}')
                continue

            ladder = self.find_ladder_for_sample(hid_file)
            yield (hid_file, resolve_annotation(hid_file), ladder)

    @classmethod
    def _build_annotation_resolver(
        cls,
        path: Path,
        scaling_strategy: ScalingStrategy,
        annotation_type: str,
        span_annotations_path: PathLike | None = None,
    ) -> Callable[[Path], Annotation | None]:
        """Build a per-HID annotation resolver for the configured annotation type.

        The dataset collector iterates robot files once and delegates
        annotation lookup to the callable returned here. Each annotation type
        can therefore prepare its own lookup state up front while the core HID
        filtering and ladder collection logic remains centralized in
        :meth:`_collect_dataset_files_uncached`.

        Args:
            path: Dataset root containing the annotation directories.
            scaling_strategy: Scaling strategy required to parse annotations.

        Returns:
            A callable that takes a HID path and returns the corresponding
            parsed annotation, or ``None`` when no annotation is available.

        Raises:
            ValueError: If ``self.annotation_type`` is not supported.
        """
        if annotation_type == 'span':
            # parse span annotations
            if span_annotations_path is None:
                span_annotations_path = path / 'span_annotations'
            span_annotations_path = Path(span_annotations_path)
            hid_to_annotation = cls._parse_span_annotation(span_annotations_path, scaling_strategy)

            def resolve_annotation(hid_file: Path) -> ScanpointAnnotation | None:
                return hid_to_annotation.get(hid_file.stem)

            return resolve_annotation
        elif annotation_type == 'AT' or annotation_type == 'LT' or annotation_type == 'ATLT':
            # parse analyst annotations
            annotations_folder = path / 'annotations'
            annotation_mapping = cls.find_annotation_files(annotations_folder)

            # find what run_id corresponds to what annotation type (AT/LT/init) from runid_type.csv
            run_id_path = path / 'runid_type.csv'
            if not run_id_path.exists():
                raise FileNotFoundError(f'Could not find runid_type.csv in {path}')
            run_id_type_mapping = {
                row[0]: row[1] for row in np.genfromtxt(run_id_path, delimiter=',', dtype=str)
            }

            def resolve_annotation(hid_file: Path) -> AlleleAnnotation | None:
                run_id = hid_file.stem.split('_')[0]
                sample_name = hid_file.stem.rsplit('_', 1)[0]

                if cls._is_correct_annotation_type(run_id, annotation_type, run_id_type_mapping):
                    annotation = annotation_mapping.get(run_id)
                    if not annotation:
                        return None

                    allele_annotation_map = cls.parse_annotations(
                        annotation, scaling_strategy=scaling_strategy
                    )

                    # Usually there's only one sample <-> annotation per annotation file
                    if len(allele_annotation_map) == 1:
                        return next(iter(allele_annotation_map.values()))
                    elif len(allele_annotation_map) > 1:
                        return allele_annotation_map[sample_name]

                return None

            return resolve_annotation
        else:
            raise ValueError(f'Invalid annotation type: {annotation_type}')


    @staticmethod
    def _is_correct_annotation_type(run_id: str, annotation_type: str, annotation_type_mapping: dict[str, str]) -> bool:
        """
        Check if the run_id corresponds to the requested annotation type.

        When a run_id ends with an L, it is assumed to be a LT profile.
        When the run_id is found in the csv with an added L, it is also assumed to be a LT profile.
        When a run_id does not end in an L, we check the type from the csv.
        When the run_id is not found in the csv, we assume it is an AT profile.
        """
        if annotation_type == 'ATLT':
            # when ATLT is selected, include all annotations
            return True
        elif run_id.endswith('L'):
            # assume the run-id is an LT profile if it ends with an L
            return annotation_type == 'LT'
        elif run_id + 'L' in annotation_type_mapping:
            # sometimes the run-id is changed to end with an L, check also for this option.
            return annotation_type == 'LT'
        elif run_id in annotation_type_mapping:
            # if the run-id is found in the csv, check if it matches the requested annotation type
            return annotation_type_mapping[run_id] == annotation_type
        elif annotation_type == 'AT':
            # if the run-id is not found in the csv, assume it is AT
            return True
        # do not include LT profiles when AT is requested
        return False

    def cache_signature(self) -> dict:  # noqa: D102
        return {
            'class': self.__class__.__name__,
            'annotation_type': self.annotation_type,
            **(
                {'robot_selection': tuple(set(self._robot_selection))}
                if self._robot_selection
                else {}
            ),
        }

    @classmethod
    def categorize_file(cls, file_name: str) -> FileCategory:  # noqa: D102
        if not file_name.endswith('.hid'):
            logger.warning(f'Encountered non-HID file: {file_name}')
            return 'unknown'

        fname = file_name.lower()
        if 'ladder' in fname:
            return 'ladder'
        elif 'control' in fname or 'blanco' in fname:
            return 'control'
        return 'sample'

    @classmethod
    def find_robot_files(
        cls,
        robots_folder: Path,
        selected_robots: Sequence[str] | None = None,
        robot_limit: int | None = None,
        cache: bool = True,
    ) -> Generator[Path, None, None]:
        """Collect all files in the robot's casework folders.

        Args:
            robots_folder: The root in which the robot folders reside
            selected_robots: Allows for a subselection of robot folder names (e.g. `('3500XL_A',)`). Defaults to None.
            robot_limit: Limits the amount of files returned from a robot. Defaults to None.
            cache: Whether to use cache saved in /tmp/ to prevent walking the whole folder again between runs.

        Yields:
            File paths of .hid's found in the robot folders.
        """
        robot_names = cls._ROBOT_NAMES if not selected_robots else selected_robots
        for robot_name in robot_names:
            robot_folder = robots_folder / robot_name
            if not robot_folder.exists():
                continue
            logger.info(f'Retrieving files from {robot_name}')
            yield from itertools.islice(
                cls._scan_directory_structure(robot_folder, cache=cache), robot_limit
            )

    @classmethod
    def find_annotation_files(cls, annotations_folder: Path):  # noqa: D102
        # Collect annotation files
        annotations = annotations_folder.iterdir()
        return {
            annotation.stem.split('_')[0]: annotation
            for annotation in annotations
            if annotation.suffix in ('.txt', '.csv')
        }

    @classmethod
    def _scan_directory_structure(cls, path: PathLike, cache: bool = True):
        _cache_file = cls._CACHE_DIR / f'{Path(path).stem}-cache'
        if cache and _cache_file.exists():
            logger.debug(f'Reading folder contents from cache: {_cache_file}')
            with _cache_file.open('rb') as f:
                return pickle.load(f)

        files = list(cls._scan_directory_structure_uncached(path))

        if cache:
            _cache_file.parent.mkdir(exist_ok=True, parents=True)
            with _cache_file.open('wb') as f:
                pickle.dump(files, f)

        return files

    @classmethod
    def _scan_directory_structure_uncached(cls, path: PathLike) -> Generator[Path, None, None]:
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_file():
                    _path = Path(entry.path)
                    if _path.suffix == '.hid':
                        yield _path
                elif entry.is_dir():
                    yield from cls._scan_directory_structure_uncached(entry.path)

    @classmethod
    def _split(
        cls,
        dataset,
        fraction: float | None = None,
        seed: int | None = None,
        k_folds: int | None = None,
        test_fraction: float = 0.0,
        **kwargs,
    ):
        dataset_indices = np.arange(len(dataset.images))
        match (fraction, k_folds):
            # Case: only train/val split, no folds, no test fraction
            case (float(), None) if test_fraction == 0.0:
                train_idx, val_idx = train_test_split(
                    dataset_indices, train_size=fraction, random_state=seed
                )
                return Subset(dataset, train_idx), Subset(dataset, val_idx)
            # Case: train/val/test split, no folds
            case (float(), None):
                train_val_idx, test_idx = train_test_split(
                    dataset_indices, test_size=test_fraction, random_state=seed
                )
                train_idx, val_idx = train_test_split(
                    train_val_idx, train_size=fraction, random_state=seed
                )

                return (
                    Subset(dataset, train_idx),
                    Subset(dataset, val_idx),
                    Subset(dataset, test_idx),
                )

            # Case: K-folds without test_fraction
            case (None, int()) if test_fraction == 0.0:
                k_fold_indices = KFold(n_splits=k_folds, random_state=seed, shuffle=True).split(
                    dataset_indices
                )

                return [
                    (Subset(dataset, train), Subset(dataset, val)) for train, val in k_fold_indices
                ]

            # Case: K-Folds with test-fraction
            case (None, int()):
                to_be_folded, test_idx = train_test_split(dataset_indices, test_size=test_fraction)
                k_fold_indices = KFold(n_splits=k_folds, shuffle=True, random_state=seed).split(
                    to_be_folded
                )

                return [
                    (Subset(dataset, train), Subset(dataset, val)) for train, val in k_fold_indices
                ], Subset(dataset, test_idx)

            case _:
                raise ValueError(
                    f'Provide either a fraction in (0, 1) or 2 <= k_folds <= len(dataset), not both. Got {fraction=}, {k_folds=}'
                )

    @classmethod
    def find_ladder_for_sample(  # noqa: D102
        cls, sample_path: Path, ladder_mapping: Dict[str, Path] | None = None
    ) -> Path | None:
        _ladders = list(sample_path.parent.glob('*ladder*.hid', case_sensitive=False))
        match len(_ladders):
            case 0:
                _ladder = None
            case 1:
                _ladder = _ladders[0]
            case _:
                logger.trace(f'Multiple ladders found, taking first: {sample_path.stem} -> {tuple(map(lambda x: x.stem, _ladders))}')
                _ladder = _ladders[0]
        return _ladder

    def get_annotation_classes(self) -> list[str]:
        """Return the annotation class labels produced by this strategy."""
        if self.annotation_type == 'span':
            return LabelCategory.label_names()
        return ['noise', 'allele']

    @classmethod
    def get_number_of_contributors(cls, file_name: str) -> int | None:  # noqa: D102
        raise AttributeError("Casework profiles don't have a known NoC")

    @classmethod
    def get_sample_id(cls, file_name: str) -> str:  # noqa: D102
        return super().get_sample_id(file_name)

    @classmethod
    def parse_annotations(  # noqa: D102
        cls, annotation_source: str | Path, scaling_strategy: ScalingStrategy
    ) -> Mapping[str, AlleleAnnotation]:
        return NFIRnDStrategy.parse_annotations(
            annotation_source=annotation_source, scaling_strategy=scaling_strategy
        )
