import os
from functools import lru_cache
from itertools import chain, islice
from pathlib import Path
from typing import Generator, Optional, Sequence, Tuple

import datasets
import numpy as np
from datasets import Array3D, Features, Value, load_from_disk
from datasets import Dataset as HFDataset
from tqdm import tqdm

from DNAnet.data.data_models import Annotation, Panel
from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.typing import PathLike
from DNAnet.utils import dict_to_marker_list, get_noc_from_rd_file_name, marker_list_to_dict


def _load_cached_hf_data(cache_path: PathLike, limit: Optional[int], include_size_standard: bool) -> Sequence[HIDImage]:
    """
    Load a sequence of cached HIDImages from `cache_path` and limit the number of images to return,
    if desired.
    """
    directories = os.listdir(cache_path)
    # the 'directories' variable is already the cache
    if any(file.endswith(".arrow") for file in directories):  # noqa
        img_count, images = read_from_hf_cache(cache_path, include_size_standard)
    else:
        # multiple cache are inside the directory
        image_generators = []
        img_count = 0
        for directory in directories:
            img_count_chunk, img_gen = read_from_hf_cache(
                str(os.path.join(cache_path, directory)),
                include_size_standard
            )
            image_generators.append(img_gen)
            img_count += img_count_chunk
        images = chain(*image_generators)
    if limit:
        img_count = limit
        images = islice(images, limit)
    return list(tqdm(images, desc="Loading cached data", total=img_count))

def _cached_arrow(cache_directory: str, include_size_standard: bool) -> HFDataset:
    """
    Load a Huggingface Dataset from an arrow cache. The cache is loaded
    with the format set to numpy to avoid unnecessary conversions when creating HIDImages from the cache.

        :param cache_directory: the path to the cache directory
        :param include_size_standard: whether to include the size standard in the HIDImage
        :return: a Huggingface Dataset with the cache data
    """
    ds = load_from_disk(cache_directory)
    ds.set_format(
        type="numpy",
        columns=["images", "annotations"],
        output_all_columns=True
    )
    num_of_dyes = ds.features["images"].shape[0]
    if num_of_dyes == 5 and include_size_standard:
        raise ValueError("include_size_standard is true however only 5 dyes were found in the cache. "
                         "Please load a cache with 6 dyes or turn off include_size_standard.")
    return ds


def read_from_hf_cache(
        cache_directory: PathLike,
        include_size_standard: bool,
        batch_size: int = 256,
) -> Tuple[int, Generator[HIDImage, None, None]]:
    """
    Read a cache from `cache_directory` and return a generator of HIDImages. The generator
    reads the data in batches of `batch_size` to avoid loading the entire cache into memory at once.

        :param cache_directory: the path to the cache directory
        :param batch_size: the number of images to read in each batch
        :param include_size_standard: whether to include the size standard in the HIDImage
        :return: a tuple containing the total number of images in the cache and a generator of HIDImages
    """
    ds = _cached_arrow(str(cache_directory), include_size_standard)
    def gen():
        for batch in ds.iter(batch_size=batch_size):
            imgs = batch["images"]
            anns = batch["annotations"]
            paths = batch["hid_paths"]
            scalers = batch["scalers"]
            panels = batch["panel_contents"]
            alleles = batch["called_alleles"]
            for img, ann, p, sc, panel_json, ca in zip(
                    imgs, anns, paths, scalers, panels, alleles
            ):
                item = {
                    "images": img,
                    "annotations": ann,
                    "hid_paths": p,
                    "scalers": sc,
                    "panel_contents": panel_json,
                    "called_alleles": ca
                }
                yield create_from_huggingface_dataset(item, include_size_standard)
    return len(ds), gen()


def create_from_huggingface_dataset(item: dict, include_size_standard: bool) -> HIDImage:
    """
    Takes item (dictionary) from Huggingface Dataset as
    input and creates a HIDImage

    :param item: dictionary containing all relevant attributes for an HIDImage
    :param include_size_standard: whether to include the size standard in the HIDImage
    :return: HIDImage
    """
    annotation = Annotation(image=np.asarray(item['annotations'], dtype='int8'))
    img = HIDImage(path=item['hid_paths'],
                   panel=Panel(panel_contents=dict_to_marker_list(item['panel_contents'], True)),
                   include_size_standard=include_size_standard,
                   annotation=annotation,
                   use_cache=True)
    img._data = np.asarray(item['images'], dtype='int16')
    img._scaler = np.array(item['scalers'])
    img._meta['called_alleles'] = dict_to_marker_list(item['called_alleles'], True)
    img._meta['noc'] = get_noc_from_rd_file_name(Path(item['hid_paths']).name)
    return img


def write_to_hf_cache(cache_path: PathLike, hid_images: Sequence[HIDImage], include_size_standard: bool):
    """
    Write a sequence of HIDImages to an arrow cache.
    """
    Path(cache_path).mkdir(exist_ok=True)

    num_of_dyes = 6 if include_size_standard else 5

    images, annotations, hid_paths, scalers, panel_contents, called_alleles = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for image in hid_images:
        images.append(image.data)
        annotations.append(image.annotation.image)
        hid_paths.append(str(image.path))
        scalers.append(image._scaler.tolist())
        panel_contents.append(marker_list_to_dict(image._panel._panel, as_json=True))
        called_alleles.append(
            marker_list_to_dict(image.meta["called_alleles"], as_json=True)
        )

    data_to_cache = HFDataset.from_dict(
        mapping={
            "images": images,
            "annotations": annotations,
            "hid_paths": hid_paths,
            "scalers": scalers,
            "panel_contents": panel_contents,
            "called_alleles": called_alleles,
        },
        features=Features(
            {
                "images": Array3D(shape=(num_of_dyes, 4096, 1), dtype="int16"),
                "annotations": Array3D(shape=(num_of_dyes, 4096, 1), dtype="int8"),
                "hid_paths": Value(dtype="string"),
                "scalers": datasets.Sequence(feature=Value(dtype="float64")),
                "panel_contents": Value(dtype="string"),
                "called_alleles": Value(dtype="string"),
            }
        ),
    )
    data_to_cache.save_to_disk(cache_path)
