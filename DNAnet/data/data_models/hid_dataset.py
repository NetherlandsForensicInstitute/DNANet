import csv
import logging
import os
import random
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set, Tuple, Union

import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from DNAnet.data.caching import _load_cached_hf_data, write_to_hf_cache
from DNAnet.data.data_models import Panel
from DNAnet.data.data_models.base import InMemoryDataset, SimpleDataset
from DNAnet.data.data_models.hid_image import HIDImage, Ladder
from DNAnet.data.data_models.structs import AlleleAnnotation, ScanpointAnnotation
from DNAnet.data.split import split_data_in_k_folds
from DNAnet.data.strategies.dataset_strategies import DatasetStrategy, NFI_RND_DatasetStrategy
from DNAnet.data.strategies.kit_strategies.scaling_strategy import ScalingStrategy
from DNAnet.data.strategies.sample_validation_strategy import SampleValidationStrategy
from DNAnet.data.strategies.strategy_registry import StrategyRegistry
from DNAnet.typing import PathLike
from DNAnet.utils import (
    get_noc_from_rd_file_name,
    get_prefix_from_filename,
    is_rd_hid_filename
)

LOGGER = logging.getLogger('dnanet')


class HIDDataset(InMemoryDataset):
    """
    A class that can load the HID images from the 2p-5p R&D dataset with their annotations.

    :param root: root folder containing the (subdirectories containing) HID files.
    :param annotations_path: location of the folder containing annotations.
    :param panel: path to read the panel from.
    :param hid_to_annotations_path: path of the file that maps hid files to annotations.
    :param best_ladder_paths_csv: path of the file that maps the path of a hid image to the file
        path of its best ladder (produced by scripts/select_ladder_for_images.py).
    :param limit: number of hid files to read.
    :param use_cache: whether to read data from the cache.
    :param cache_path: location of the cache to read files from (if use_cache is true), or to
    write the files to (if use_cache is false).
    :param adjustment_of_annotations: the adjustment to be applied to the annotations, either
    'complete' to label the entire peak, or 'top' to only label the top of the peak.
    :param shuffle: whether the dataset should be shuffled when iterating.
    :param skip_if_invalid_ladder: whether to skip images that have invalid ladders.
    :param analysis_threshold_type: the analysis threshold type to use for annotations (either
        `DTH` (high) or `DTL` (low).
    :param ground_truth_as_annotations: whether to load the ground truth donor alleles as
        annotations.
    :param include_size_standard: whether to include the size standard in the HIDImage.
    :param data_loading_strategy: the strategy to load the data, either "raw", "analyzed" or "superior".
        "raw" means loading the raw data, "analyzed" means loading the analyzed data,
        and "superior" means loading the raw data and applying baseline subtraction
    :param group_replicas_in_split: whether to put measurements from the same profile (replicas)
    in the same set when splitting, and balance the number of profiles per noc, if false all
    replicas will be mixed.
    """

    def __init__(self,
                 root: PathLike,
                 annotations_path: Optional[PathLike] = None,
                 panel: Optional[PathLike] = None,
                 hid_to_annotations_path: Optional[PathLike] = None,
                 best_ladder_paths_csv: Optional[PathLike] = None,
                 limit: Optional[int] = None,
                 use_cache: Optional[bool] = False,
                 cache_path: Optional[PathLike] = None,
                 adjustment_of_annotations: Optional[str] = None,
                 shuffle: Optional[bool] = False,
                 skip_if_invalid_ladder: Optional[bool] = False,
                 analysis_threshold_type: Optional[str] = 'DTL',
                 ground_truth_as_annotations: Optional[bool] = False,
                 include_size_standard: bool = False,
                 data_loading_strategy: str = "superior",
                 group_replicas_in_split: Optional[bool] = False,
                 dataset_strategy: Optional[DatasetStrategy] = None,
                 scaling_strategy: Optional[ScalingStrategy] = None,
                 sample_validation_strategy: Optional[SampleValidationStrategy] = None):
        super().__init__(shuffle)
        self.root = str(root)
        self.limit = limit
        self.cache_path = cache_path
        self.analysis_threshold_type = analysis_threshold_type
        self.skip_if_invalid_ladder = skip_if_invalid_ladder
        self.adjustment_of_annotations = adjustment_of_annotations
        self.ground_truth_as_annotations = ground_truth_as_annotations
        self.include_size_standard = include_size_standard
        self.data_loading_strategy = data_loading_strategy
        self.group_replicas_in_split = group_replicas_in_split
        self.dataset_strategy = dataset_strategy
        self.scaling_strategy = scaling_strategy
        self.sample_validation_strategy = sample_validation_strategy or (lambda image: True)
        self.annotations_path = annotations_path
        self.hid_to_annotations_path = hid_to_annotations_path
        self.best_ladder_paths_csv = best_ladder_paths_csv
        self.panel_path = panel

        # If cache path is given and use_cache is set to true, load cached data.
        if cache_path and use_cache:
            LOGGER.info(f"Loading data from arrow cache: {cache_path}")
            if ground_truth_as_annotations:
                if panel is None:
                    raise ValueError("Need panel when loading ground truth annotations")
                self._panel = Panel(panel_path=panel)
            self._data = _load_cached_hf_data(cache_path, limit, include_size_standard)
        # Otherwise, read data and cache (optional)
        else:
            LOGGER.info(f"Loading raw data from {self.root}")
            self._validate_dataset_args(annotations_path, panel)

            self._panel = Panel(panel_path=panel)
            # Map the hid file names to the annotation (optional for non-RD datasets)
            if annotations_path and hid_to_annotations_path:
                self.annotation_dict = self._create_annotation_mapping_rd(
                    analysis_threshold_type=analysis_threshold_type,
                    hid_to_annotations_path=hid_to_annotations_path,
                    annotations_path=annotations_path,
                )
            else:
                self.annotation_dict = {}
            # Map the image path to the path of the best fitting ladder
            self.best_ladder_paths = self.load_best_ladder_paths(best_ladder_paths_csv)

            # Create a list of .hid files, with their ladder paths and annotations
            self._files = self._collect_and_filter_file_paths()
            if self.limit:
                # Make sure to shuffle files before limiting, otherwise we would always take the
                # first `limit` when loading images.
                if self.shuffle:
                    self._files = random.Random().sample(self._files, self.limit)
                else:
                    self._files = self._files[:self.limit]
                LOGGER.info(f"Limiting dataset to {self.limit}")

            # Create all HIDImages and filter out invalid ones
            self._data = list(self.load_images_from_files())
            if len(self._data) == 0:
                raise ValueError("Zero hid images found when loading data!")
            LOGGER.info(f"Found {len(self._data)} HIDImages after filtering out images with "
                        f"missing data or missing called alleles.")

            if adjustment_of_annotations:
                LOGGER.info(f"Adjusting image annotations (type {adjustment_of_annotations}).")
                self._data = [im.adjust_annotations(adjustment_of_annotations) for im in self._data]

            if self.cache_path:
                LOGGER.info(f"Writing data to arrow cache: {self.cache_path}")
                write_to_hf_cache(self.cache_path, self._data, include_size_standard)

        # TODO improve loading of annotations and ground truth annotations
        if self.ground_truth_as_annotations:
            LOGGER.info("Loading ground truth annotations")
            for image in self._data:
                # we want to store the ground truth donor alleles in meta['called_alleles'],
                # therefore we copy the original 'called_alleles' into 'called_alleles_manual'
                if "called_alleles" in image._meta:
                    image._meta["called_alleles_manual"] = image._meta["called_alleles"]
                try:
                    if self.dataset_strategy:
                        image._meta["called_alleles"] = self.dataset_strategy.load_donor_alleles(
                            image.path.stem
                        )
                    else:
                        image._meta["called_alleles"] = NFI_RND_DatasetStrategy.load_donor_alleles(image.path.stem, self._panel)
                except ValueError as e:
                    LOGGER.warning("Could not load ground truth annotations for %s: %s", image.path, e)

        # Final validation pass to ensure cached or pre-filtered data respects the strategy.
        if self.sample_validation_strategy:
            before = len(self._data)
            self._data = [im for im in self._data if self.sample_validation_strategy(im)]
            if before != len(self._data):
                LOGGER.info("Filtered out %d images that failed sample validation", before - len(self._data))

    def _validate_dataset_args(self,
                               annotations_path: Optional[PathLike],
                               panel_path: Optional[PathLike]):
        """
        Check the presence and validity of dataset arguments.
        """
        if self.adjustment_of_annotations is not None and \
                self.adjustment_of_annotations not in ["top", "complete"]:
            raise ValueError(
                "Unknown adjustment type for annotations found: "
                f"{self.adjustment_of_annotations}. Provide `top` or `complete`."
            )

        if not panel_path:
            raise ValueError("Panel path missing.")

        if not annotations_path and not self.ground_truth_as_annotations:
            raise ValueError("Annotations path missing.")

    def _collect_and_filter_file_paths(self) -> List[Dict[str, Union[str, PathLike, List[str]]]]:
        """
        Scans the root directory and composes a list all files to consider for this dataset.
        Store each file's properties (full_path, ladders, annotations) in dictionaries.
        """
        # Retrieve the folder structure of the root path
        LOGGER.info("Walking directory to retrieve all hid files...")
        root_path_structure = self._scan_directory_structure(self.root)

        # Flatten the folder structure into tuples and file-lists
        flat_folder_structure = self._flatten_directory_structure(root_path_structure)

        # Clean up results, only keep folders with HID files in them
        flat_folder_structure = {
            os.path.join(self.root, *folder_path): [f for f in folder_content if
                                                    Path(f).suffix == ".hid"] for
            folder_path, folder_content in flat_folder_structure.items()
        }

        full_file_list = []
        # Keep track of the number of files we filter out because they have no annotation
        no_annotation_counter = 0
        non_sample_counter = 0
        # Loop over each folder and its corresponding files
        for folder, files in tqdm(flat_folder_structure.items(), "Processing folders"):
            if len(files) == 0:  # no hid files in this folder
                continue

            # Retrieve viable HID files, filter by RD naming only when no dataset strategy is provided
            hid_files = files if self.dataset_strategy else list(filter(is_rd_hid_filename, files))

            # Now for each file in the folder, save its corresponding ladder (if there is any)
            # and match the file with its corresponding annotation (if present).
            for file in hid_files:
                if self.dataset_strategy:
                    try:
                        if self.dataset_strategy.categorize_file(Path(file).name) != "sample":
                            non_sample_counter += 1
                            continue
                    except ValueError as e:
                        LOGGER.warning("Skipping image: categorization failed for %s (%s)", file, e)
                        non_sample_counter += 1
                        continue

                # get the annotation name and actual file containing annotations
                _annotation = self.annotation_dict.get(Path(file).stem)

                if not _annotation and not self.ground_truth_as_annotations:
                    # we only want to keep hid files that have an annotation
                    no_annotation_counter += 1
                    continue

                full_path = os.path.join(folder, file)
                full_file_list.append(
                    {
                        "full_path": full_path,
                        "ladder_path": self.best_ladder_paths.get(Path(full_path).stem),
                        "annotation": _annotation,
                    }
                )

        # Logging some statistics of our dataset
        LOGGER.info(f"Found {len(full_file_list)} .hid files")
        LOGGER.info(f"Removed {no_annotation_counter} .hid files without annotation")
        if self.dataset_strategy:
            LOGGER.info(f"Removed {non_sample_counter} non-sample files based on dataset strategy")

        return full_file_list

    @staticmethod
    def load_best_ladder_paths(best_ladders_csv_path: Optional[PathLike]) -> Dict[str, str]:
        """
        Loads a csv file containing for every HID image path the path of the best corresponding
        ladder. If not provided, an empty dictionary is returned, meaning that no ladders will
        be considered when loading the data.
        """
        if not best_ladders_csv_path:
            LOGGER.info("No csv path for best ladders is provided.")
            return dict()

        LOGGER.info(f"Loading best ladder paths from {best_ladders_csv_path}.")
        with open(best_ladders_csv_path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            next(reader)
            data = {k: v if v != '' else None for (k, v) in reader}
        return data

    @staticmethod
    def _create_annotation_mapping_rd(
            analysis_threshold_type: str,
            hid_to_annotations_path: PathLike,
            annotations_path: PathLike
    ) -> Dict[str, Tuple[str, PathLike]]:
        """
        Maps the .hid file name (e.g. 1A2_A01_01) to a tuple of the annotations name (e.g.
        1L_11148_1A2) and annotations file (e.g. <path to>/Dataset 1 DTL_AlleleReport.txt).
        """
        hid_to_annotations_mapping: Dict[str, Tuple[str, PathLike]] = {}
        # DTH/DTL is part of the .txt file name with the annotations we need, and indicates that
        # profiles where measured with either high or low detection threshold
        if analysis_threshold_type not in ["DTH", "DTL"]:
            raise ValueError("Provide analysis threshold type DTH or DTL when loading RD data, "
                             f"not {analysis_threshold_type}.")

        with open(hid_to_annotations_path, "r") as f:
            reader = csv.DictReader(f, delimiter=",")
            for row in reader:
                if row['Raw hid name'] != "":
                    hid_file = Path(row['Raw hid name']).stem
                    annotation_name = row[f"{analysis_threshold_type} name 2024"]
                    txt_name = f"Dataset {hid_file[0]} {analysis_threshold_type}_AlleleReport.txt"
                    hid_to_annotations_mapping[hid_file] = \
                        (annotation_name, annotations_path / Path(txt_name))

        return hid_to_annotations_mapping

    def _scan_directory_structure(self, path: PathLike) -> Dict[str, Union[dict, list]]:
        """
        Scans a directory path and returns a nested dictionary of folders and files.
        """
        structure = {}
        with os.scandir(path) as entries:
            files = []
            for entry in entries:
                if entry.is_file():
                    files.append(entry.name)
                elif entry.is_dir():
                    # Recursively scan subdirectories and add them as nested dictionaries
                    structure[entry.name] = self._scan_directory_structure(entry.path)
            if files:
                structure['files'] = files

        return structure

    def _flatten_directory_structure(
            self,
            structure: Dict[str, Union[dict, list]],
            parent_path: tuple = ()
    ) -> dict:
        """
        Recursively flatten a nested directory structure into a dictionary with tuple keys and
        files as values. Add a parent folder to each of the flattened keys.
        """
        flattened = {}
        for key, value in structure.items():
            if key == 'files':
                # If 'files' key, store files in the current path
                flattened[parent_path] = value
            else:
                # If it's a folder, recursively flatten it
                nested_path = parent_path + (key,)
                flattened.update(self._flatten_directory_structure(value, nested_path))

        return flattened

    def load_images_from_files(self) -> Generator[HIDImage, None, None]:
        """
        From the hid file path, ladder file path(s) and annotation information, create the actual
        HIDImages by parsing those file attributes. Some HIDImages might be filtered out due to
        missing either the called alleles or peak data.
        """
        files_progress_bar = tqdm(self._files, desc=f"Loading data from {self.root}")
        with logging_redirect_tqdm():
            for file_attributes in files_progress_bar:
                path, ladder_path, (annotation_name, annotation_file) = (
                    file_attributes["full_path"],
                    file_attributes["ladder_path"],
                    file_attributes["annotation"] if file_attributes["annotation"] else (None, None)
                )
                # create a ladder from the ladder_path if present
                ladder = Ladder(ladder_path, self._panel) if ladder_path else None
                if self.skip_if_invalid_ladder and ladder is None:
                    LOGGER.warning("Skipping image: Invalid/Missing ladder (%s)", path)
                    continue

                # take the ladder as panel if present, otherwise use the default panel
                panel = ladder._panel if (ladder and ladder._panel) else self._panel
                _image = HIDImage(
                    path=path,
                    annotations_file=annotation_file,
                    panel=panel,
                    include_size_standard=self.include_size_standard,
                    data_loading_strategy=self.data_loading_strategy,
                    use_ground_truth_as_annotations=self.ground_truth_as_annotations,
                    meta={
                        "annotations_name": annotation_name,
                        "ladder_path": ladder_path,
                        "noc": get_noc_from_rd_file_name(Path(path).stem)
                    }
                )

                # Some filtering based on needed properties for a valid HIDImage
                if _image.data is None:
                    LOGGER.warning("Skipping image: Missing data (%s)", path)
                    continue
                elif not self.ground_truth_as_annotations and _image.meta.get("called_alleles") is None:
                    LOGGER.warning("Skipping image: Missing called_alleles (%s)", path)
                    continue
                elif not self.sample_validation_strategy(_image):
                    LOGGER.warning("Skipping image: Failed sample validation (%s)", path)
                    continue

                yield _image

    def split(self, fraction: float, seed: Optional[float] = None) \
            -> Tuple['SimpleDataset', 'SimpleDataset']:
        if not 0 < fraction < 1:
            raise ValueError(f"Fraction should be between 0 and 1, got {fraction}.")

        if self.group_replicas_in_split:
            LOGGER.info("Splitting HIDDataset, taking into account splitting by replicas and "
                        "balancing number of donors")
            return self._split_by_replicas_and_noc(fraction, seed)
        else:
            return super(HIDDataset, self).split(fraction, seed)

    def _split_by_replicas_and_noc(self, fraction: float, seed: Optional[float] = None) \
            -> Tuple['SimpleDataset', 'SimpleDataset']:
        """
        Split the dataset by ensuring that replicas are put in the same set and the number of
        donors are balanced in the two sets.
        """
        random.seed(seed)
        # divide all hid images per prefix and number of donors
        hids_per_prefix_per_nr_donors = self._get_hids_per_prefix_per_noc()

        hids_1, hids_2 = [], []
        for nr_donors, hids_per_prefix in hids_per_prefix_per_nr_donors.items():
            # split all prefixes for a NoC-value into two sets, then find all images belonging
            # to those prefixes
            all_prefixes = list(hids_per_prefix.keys())
            split_idx = int(len(all_prefixes) * fraction)
            random.shuffle(all_prefixes)
            hids_1.extend(list(chain.from_iterable(
                [hids_per_prefix[prefix] for prefix in all_prefixes[:split_idx]])))
            hids_2.extend(list(chain.from_iterable(
                [hids_per_prefix[prefix] for prefix in all_prefixes[split_idx:]])))
        # shuffle again, since replicas (with the same prefix) are still grouped together
        random.shuffle(hids_1)
        random.shuffle(hids_2)
        return SimpleDataset(data=hids_1, shuffle=self.shuffle), \
               SimpleDataset(data=hids_2, shuffle=self.shuffle)

    def split_k_fold(self, n_folds: int, seed: Optional[float] = None) \
            -> List[Tuple[SimpleDataset, SimpleDataset]]:
        if self.group_replicas_in_split:
            LOGGER.info(f"Splitting HIDDataset in {n_folds}-fold, taking into account splitting by "
                        "replicas and balancing number of donors")
            return self._split_k_fold_by_replicas_and_noc(n_folds, seed)
        else:
            return super(HIDDataset, self).split_k_fold(n_folds, seed)

    def split_by_genotypes(self, genotypes: Set[str]) -> Tuple['SimpleDataset', 'SimpleDataset']:
        """
        Split images into two datasets based on contributor IDs. Images with contributors that
        are a subset of `genotypes` go to A; images disjoint from `genotypes` go to B; mixed or
        unparseable contributors are discarded.
        """
        if not self.dataset_strategy:
            raise ValueError("split_by_genotypes requires a dataset_strategy to extract contributors.")

        community_A_images: List[HIDImage] = []
        community_B_images: List[HIDImage] = []
        ambiguous_images: List[HIDImage] = []

        for img in self._data:
            try:
                contribs = set(self.dataset_strategy.get_contributors(img.path.name))
            except Exception as e:
                LOGGER.warning("Could not extract contributors for %s: %s", img.path, e)
                ambiguous_images.append(img)
                continue

            if contribs.issubset(genotypes):
                community_A_images.append(img)
            elif contribs.isdisjoint(genotypes):
                community_B_images.append(img)
            else:
                ambiguous_images.append(img)

        LOGGER.info("Split by genotypes: A=%d, B=%d, ambiguous=%d",
                    len(community_A_images), len(community_B_images), len(ambiguous_images))

        community_A_dataset = SimpleDataset(data=community_A_images, shuffle=self.shuffle)
        community_B_dataset = SimpleDataset(data=community_B_images, shuffle=self.shuffle)

        return community_A_dataset, community_B_dataset

    def _split_k_fold_by_replicas_and_noc(self, n_folds: int, seed: Optional[float] = None) \
            -> List[Tuple[SimpleDataset, SimpleDataset]]:
        """
        Create a k-fold split by ensuring that replicas are put in the same set and the number of
        donors are balanced in each fold.
        """
        random.seed(seed)
        # divide all hid images per prefix and number of donors
        hids_per_prefix_per_nr_donors = self._get_hids_per_prefix_per_noc()

        folds = [[] for _ in range(n_folds)]
        for hids_per_prefix in hids_per_prefix_per_nr_donors.values():
            all_prefixes = list(hids_per_prefix.keys())
            random.shuffle(all_prefixes)
            # first split all prefixes per noc in k-folds, then select all hid images belonging
            # to those prefixes
            splitted_prefixes = split_data_in_k_folds(all_prefixes, n_folds, seed)
            for i, prefixes in enumerate(splitted_prefixes):
                images = list(chain.from_iterable([hids_per_prefix[p] for p in prefixes]))
                folds[i].extend(images)

        train_test_folds = self._make_train_test_splits(folds)
        return train_test_folds

    def _get_hids_per_prefix_per_noc(self) -> Dict[str, Dict[str, List[HIDImage]]]:
        """
        Get per number of contributors and per prefix (i.e. `1A2`) the belonging HIDImages as
        a dictionary. The number of contributors is the second number in the prefix, e.g. '2' in
        `1A2`.
        """
        hids_per_prefix_per_nr_donors = defaultdict(lambda: defaultdict(list))
        for im in self._data:
            prefix = get_prefix_from_filename(im.path.stem)
            noc = prefix[-1]
            hids_per_prefix_per_nr_donors[noc][prefix].append(im)
        return hids_per_prefix_per_nr_donors

    def get_hid_image_by_name(self, hid_file_name: PathLike) -> Optional[HIDImage]:
        """
        Retrieve a HIDImage from the dataset by comparing the HID file names (that is
        1A2_A01_01 (with or without .hid file extension)). Return None if no images were found.
        """
        hid_file_name = str(Path(hid_file_name).stem)

        image = [im for im in self._data if im.path.stem == hid_file_name]
        if len(image) > 1:
            raise ValueError(f"Found more than one ({len(image)}) for file name {hid_file_name}")
        if len(image) == 0:
            return None
        return image[0]

    def serialize(self) -> Dict[str, Union[str, int, float, bool, None, dict]]:
        """Return a minimal serializable representation of this dataset."""
        return {
            "class": self.__class__.__name__,
            "root": self.root,
            "annotations_path": str(self.annotations_path) if self.annotations_path else None,
            "hid_to_annotations_path": str(self.hid_to_annotations_path) if self.hid_to_annotations_path else None,
            "best_ladder_paths_csv": str(self.best_ladder_paths_csv) if self.best_ladder_paths_csv else None,
            "panel_path": str(self.panel_path) if self.panel_path else None,
            "limit": self.limit,
            "use_cache": bool(self.cache_path) if self.cache_path else False,
            "cache_path": str(self.cache_path) if self.cache_path else None,
            "adjustment_of_annotations": self.adjustment_of_annotations,
            "shuffle": self.shuffle,
            "skip_if_invalid_ladder": self.skip_if_invalid_ladder,
            "analysis_threshold_type": self.analysis_threshold_type,
            "ground_truth_as_annotations": self.ground_truth_as_annotations,
            "group_replicas_in_split": self.group_replicas_in_split,
            "kit": {
                "name": self.kit.name,
                "size_standard": getattr(self.kit.size_standard, "name", str(self.kit.size_standard)),
                "panel_path": str(self.kit.panel_path) if self.kit and self.kit.panel_path else None,
            } if self.kit else None,
            "dataset_strategy": self.dataset_strategy.serialize() if self.dataset_strategy else None,
            "scaling_strategy": self.scaling_strategy.__class__.__name__ if self.scaling_strategy else None,
            "sample_validation_strategy": getattr(self.sample_validation_strategy, "__name__", str(self.sample_validation_strategy)),
            "num_samples": len(self._data),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.serialize()})"

    @staticmethod
    def _translate_allele_to_scanpoint_annotation(allele_annotation: AlleleAnnotation, adjusted_panel: Panel, scaler: np.ndarray) -> ScanpointAnnotation:
        """
        Translates allele annotation to scanpoint annotation by finding the closest scanpoint indices for the left and right bins
        of each allele using the provided scaler and adjusted panel. The entire allelic bin is annotated as 1.

        All non annotated scanpoints are labeled as 0, while annotated scanpoints are labeled as 1.
        The resulting scanpoint annotation is a binary matrix of shape (num_dyes, num_scanpoints)

            :param allele_annotation: AlleleAnnotation object containing the allele annotations to be translated.
            :param adjusted_panel: Panel object containing the adjusted panel information to be used for translation.
            :param scaler: numpy array containing the scaler values to be used for finding the closest scanpoint indices.
            :return: ScanpointAnnotation object containing the translated scanpoint annotation.
        """
        scanpoint_annotation = np.zeros((StrategyRegistry.get_scaling_strategy().kit.num_dyes,
                                         StrategyRegistry.get_scaling_strategy().scanpoint_resolution), dtype=np.int8)
        for locus in allele_annotation.annotation:
            for allele in locus.alleles:
                # for each allele, find the left and right bin of the allele using the panel that has been adjusted by the corresponding ladder.
                _, left_bin, right_bin = adjusted_panel.get_allele_basepair_and_bins(locus.name, allele.name)

                # use the scaler to find the closest scanpoint indices for the left and right bins of the allele
                left_scanpoint = np.argmin(np.abs(scaler - left_bin))
                right_scanpoint = np.argmin(np.abs(scaler - right_bin))

                scanpoint_annotation[locus.dye_row, left_scanpoint : right_scanpoint] = 1

        return ScanpointAnnotation(annotation=scanpoint_annotation)
