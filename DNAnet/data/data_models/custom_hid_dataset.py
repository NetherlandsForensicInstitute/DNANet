import logging
from pathlib import Path
from typing import List, Optional, Set, Tuple
from DNAnet.data.data_models.base import InMemoryDataset, SimpleDataset
from DNAnet.data.data_models.dna_models import Panel
from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.data.dataset_compatibility.dataset_strategy import DatasetStrategy, categorize_files
from DNAnet.data.kit_compatibility.kit import Kit
from DNAnet.data.dataset_compatibility.utils import find_files_by_suffix
from DNAnet.data.kit_compatibility.scaling_strategy import EPGScalingStrategy
from DNAnet.data.strategies.sample_validation_strategy import SampleValidationStrategy
from DNAnet.typing import PathLike


LOGGER = logging.getLogger('dnanet')


class CustomHIDDataset(InMemoryDataset):
    def __init__(self, 
                 root_path: PathLike,
                 panel_path: PathLike, 
                 shuffle: Optional[bool] = False,
                 limit: Optional[int] = None,
                 adjustment_of_annotations: Optional[str] = None,
                 ground_truth_as_annotations: bool = False,
                #  size_standard: str = InternalSizeStandard.WEN_ILS.value,
                #  file_categorization_strategy: FileCategorizationStrategy = lambda file_name: "sample",
                 sample_validation_strategy: SampleValidationStrategy = lambda image: True,
                 kit: Optional[Kit] = None,
                 dataset_strategy: Optional[DatasetStrategy] = None,
                 scaling_strategy: Optional[EPGScalingStrategy] = None,
                ):
        super().__init__(shuffle)

        self.root_path = root_path
        self.files = find_files_by_suffix(root_path, ".hid")

        self.panel_path = Path(panel_path)
        self.panel = Panel(panel_path)

        self.limit = limit
        self.adjustment_of_annotations = adjustment_of_annotations
        self.ground_truth_as_annotations = ground_truth_as_annotations

        # self.size_standard = size_standard
        # self.file_categorization_strategy = file_categorization_strategy
        self.kit = kit
        self.dataset_strategy = dataset_strategy
        self.scaling_strategy = scaling_strategy
        self.sample_validation_strategy = sample_validation_strategy

        self.categorized_files = categorize_files(self.files, self.dataset_strategy)
        self.sample_files = self.categorized_files["sample"]

        LOGGER.info(f"Found {len(self.sample_files)} files in {self.root_path}")
        unvalidated_images = [
            HIDImage(path=f, 
                     panel=self.panel, 
                     kit=self.kit, 
                     dataset_strategy=self.dataset_strategy,
                     scaling_strategy=self.scaling_strategy,
                     use_ground_truth_as_annotations=self.ground_truth_as_annotations
                     )
            for f in self.sample_files
        ]

        validated_images = [
            image for image in unvalidated_images
            if self.sample_validation_strategy(image)
        ]
        LOGGER.info(f"Number of valid images: {len(validated_images)}")
        
        # Move limiting logic here, after validation
        if limit is not None:
            validated_images = validated_images[:limit]
            LOGGER.info(f"Limiting dataset to {self.limit}")
        LOGGER.info(f"Number of files limited to: {len(validated_images)}")
        self._data = validated_images

        if self.adjustment_of_annotations:
            self._data = [
                im.adjust_annotations(self.adjustment_of_annotations)
                for im in self._data
            ]







    # Also I would like to add the `split_by_genotypes` method over, 
    # but we have to adjust it and check that it works (best by writing tests) 
    # because the `extract_contributor_ids` method does not exist in the `dataset_strategy`, 
    # rather `get_contributors``
    def split_by_genotypes(self, genotypes: Set[int]) -> Tuple['SimpleDataset', 'SimpleDataset']:
        """
        Splits a set of genotypes into two datasets: one with images whose contributors are a subset of genotypes,
        and one with images whose contributors are disjoint from genotypes. Ambiguous images are discarded.

        :param genotypes: A set of genotype IDs to split the dataset by. e.g.: {39, 40, 41, 42, 43}
        :return: A tuple of two CustomHIDDataset instances.
        """
        community_A_images: List[HIDImage] = []
        community_B_images: List[HIDImage] = []
        ambiguous_images: List[HIDImage] = []

        for img in self._data:
            contribs = set(self.file_categorization_strategy.extract_contributor_ids(img.path.name))
            if contribs.issubset(genotypes):
                community_A_images.append(img)
            elif contribs.isdisjoint(genotypes):
                community_B_images.append(img)
            else:
                ambiguous_images.append(img)  # contributors from both groups, discard or flag

        LOGGER.info(f"✅ Community A: {len(community_A_images)} images")
        LOGGER.info(f"✅ Community B: {len(community_B_images)} images")
        LOGGER.info(f"⚠️ Ambiguous: {len(ambiguous_images)} images discarded due to overlap")

        community_A_dataset = SimpleDataset(data=community_A_images, shuffle=self.shuffle)
        community_B_dataset = SimpleDataset(data=community_B_images, shuffle=self.shuffle)

        return community_A_dataset, community_B_dataset

    def __str__(self):
        return (
            "CustomHIDDataset:\n"
            f"  files: \"<{len(self.files)} files>\"\n"
            f"  panel: \"{str(self.panel)}\"\n"
            f"  shuffle: \"{str(self.shuffle)}\"\n"
            f"  limit: \"{str(self.limit)}\"\n"
            f"  size_standard: \"{str(self.size_standard)}\"\n"
            f"  file_categorization_strategy: \"{getattr(self.file_categorization_strategy, '__name__', str(self.file_categorization_strategy))}\"\n"
            f"  sample_validation_strategy: \"{getattr(self.sample_validation_strategy, '__name__', str(self.sample_validation_strategy))}\""
        )
