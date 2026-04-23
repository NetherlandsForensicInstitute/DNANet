"""NFI Zaaksdata strategy.

Handles the NFI Zaaksdata

Warning:
    This strategy is developed specifically for our in-house casework.
    Altough file-collection and caching logic might be of use for your own implementation,
    using this strategy for other/your own data does not make sense.

    Please see the documentation about developing your own strategy, or use the two other strategies for the open-source data.
"""

import os
import json
import pickle
import hashlib
import itertools
from typing import Dict, Tuple, Mapping, Sequence, Generator
from pathlib import Path

from loguru import logger

from dnanet.core.types import PathLike
from dnanet.core.annotation import AlleleAnnotation, ScanpointAnnotation
from dnanet.data.strategies.scaling.scaling import ScalingStrategy
from dnanet.data.strategies.datasets.dataset import FileCategory, DatasetStrategy
from dnanet.data.strategies.datasets.nfi_rnd import NFIRnDStrategy


class NFICaseStrategy(DatasetStrategy):
    _ROBOT_NAMES = ('3500XL_A', '3500XL_B', '3500XL_C', '3500XL_D')
    _CACHE_DIR = Path('/tmp/.nfi_zaaksdata_cache/')

    def __init__(self, annotation_type: str, robot_selection: Sequence[str] | None = None) -> None:
        super().__init__()
        self.annotation_type = (annotation_type,)
        self._robot_selection = robot_selection

    def collect_dataset_files(
        self, root_path: str | Path, scaling_strategy: ScalingStrategy, **kwargs
    ) -> Generator[
        Tuple[Path, ScanpointAnnotation | AlleleAnnotation | None, Path | None], None, None
    ]:
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
    ) -> Generator[
        Tuple[Path, ScanpointAnnotation | AlleleAnnotation | None, Path | None], None, None
    ]:
        path = Path(root_path)

        # Collect all .HID files from the robot folders
        hids_folder = path / 'hids'
        robots = self.find_robot_files(hids_folder, self._robot_selection)

        # Collect all annotation .txt/.csv files and map from run_id -> annotation file
        annotations_folder = path / 'annotations'
        annotation_mapping = self.find_annotation_files(annotations_folder)

        file_statistics = {'Total files': 0, 'Missing Annotation': 0, 'Missing Ladder(s)': 0}

        for _file in robots:
            if self.categorize_file(_file.name) != 'sample':
                continue

            _run_id = _file.stem.split('_')[0]
            _annotation = annotation_mapping.get(_run_id)

            _ladders = list(_file.parent.glob('*ladder*.hid', case_sensitive=False))
            match len(_ladders):
                case 0:
                    _ladder = None
                case 1:
                    _ladder = _ladders[0]
                case _:
                    # logger.warning(f'Multiple ladders found, taking first: {_file.stem} -> {tuple(map(lambda x: x.stem, _ladders))}')
                    _ladder = _ladders[0]

            file_statistics['Total files'] += 1
            file_statistics['Missing Annotation'] += 1 if _annotation is None else 0
            file_statistics['Missing Ladder(s)'] += 1 if not _ladders else 0

            _allele_annotation = None
            if _annotation:
                _allele_annotation_map = NFIRnDStrategy.parse_annotations(
                    _annotation, scaling_strategy=scaling_strategy
                )
                if _allele_annotation_map:
                    _allele_annotation = list(_allele_annotation_map.values())[0]

            yield (_file, _allele_annotation, _ladder)
        logger.info(file_statistics)

    def cache_signature(self) -> dict:
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
    def categorize_file(cls, file_name: str) -> FileCategory:
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
    ) -> Generator[Path, None, None]:
        """Collect all files in the robot's casework folders.

        Args:
            robots_folder: The root in which the robot folders reside
            selected_robots: Allows for a subselection of robot folder names (e.g. `('3500XL_A',)`). Defaults to None.
            robot_limit: Limits the amount of files returned from a robot. Defaults to None.

        Yields:
            File paths of .hid's found in the robot folders.
        """
        robot_names = cls._ROBOT_NAMES if not selected_robots else selected_robots
        for robot_name in robot_names:
            robot_folder = robots_folder / robot_name
            if not robot_folder.exists():
                continue
            logger.info(f'Retrieving files from {robot_name}')
            yield from itertools.islice(cls._scan_directory_structure(robot_folder), robot_limit)

    @classmethod
    def find_annotation_files(cls, annotations_folder: Path):
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
    def _split(cls, dataset, **kwargs):
        raise NotImplementedError('Splitting not configured yet')

    @classmethod
    def find_ladder_for_sample(
        cls, sample_path: Path, ladder_mapping: Dict[str, Path] | None = None
    ) -> Path | None:
        return super().find_ladder_for_sample(sample_path, ladder_mapping)

    @staticmethod
    def get_annotation_classes() -> list[str]:
        pass

    @classmethod
    def get_number_of_contributors(cls, file_name: str) -> int | None:
        return super().get_number_of_contributors(file_name)

    @classmethod
    def get_sample_id(cls, file_name: str) -> str:
        return super().get_sample_id(file_name)

    @classmethod
    def parse_annotations(
        cls, annotation_source: str | Path, scaling_strategy: ScalingStrategy
    ) -> Mapping[str, ScanpointAnnotation | AlleleAnnotation]:
        return super().parse_annotations(annotation_source, scaling_strategy)
