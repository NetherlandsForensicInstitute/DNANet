"""HuggingFace-based caching for HIDImage datasets.

Caching avoids re-parsing thousands of HID binary files on every run.
The cache stores pre-processed data in Apache Arrow format (via the
HuggingFace ``datasets`` library) for fast memory-mapped loading.

Design pattern: **Adapter**
    These functions adapt between DNANet's ``HIDImage`` domain objects and
    HuggingFace's ``Dataset`` tabular format. ``write_to_cache`` serializes
    HIDImages into a flat table; ``load_from_cache`` reconstructs them.
"""

from __future__ import annotations

import json
import os
from itertools import chain, islice
from pathlib import Path
from typing import Generator, Sequence

import datasets
import numpy as np
from datasets import Array3D, Features, Value
from datasets import Dataset as HFDataset
from datasets import load_from_disk
from loguru import logger
from tqdm import tqdm

from dnanet.core.annotation import ScanpointAnnotation
from dnanet.core.types import PathLike
from dnanet.data.image import HIDImage


def load_from_cache(
    cache_path: PathLike,
    limit: int | None = None,
    include_size_standard: bool = False,
) -> list[HIDImage]:
    """Load HIDImages from an Arrow cache.

    Supports both single-directory caches and multi-shard caches
    (a directory containing multiple Arrow subdirectories).

    Args:
        cache_path: Path to the cache directory.
        limit: Maximum number of images to load.
        include_size_standard: Whether the cache includes 6 dye channels.

    Returns:
        List of reconstructed HIDImages.
    """
    entries = os.listdir(cache_path)

    if any(f.endswith(".arrow") for f in entries):
        count, gen = _read_shard(str(cache_path), include_size_standard)
    else:
        generators = []
        count = 0
        for subdir in entries:
            shard_count, shard_gen = _read_shard(
                os.path.join(cache_path, subdir), include_size_standard
            )
            generators.append(shard_gen)
            count += shard_count
        gen = chain(*generators)

    if limit:
        count = limit
        gen = islice(gen, limit)

    return list(tqdm(gen, desc="Loading cached data", total=count))


def write_to_cache(
    cache_path: PathLike,
    images: Sequence[HIDImage],
    include_size_standard: bool = False,
) -> None:
    """Serialize HIDImages to an Arrow cache.

    Args:
        cache_path: Directory to write the cache to.
        images: HIDImages to cache.
        include_size_standard: Whether data includes 6 dye channels.
    """
    cache_path = Path(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)

    data: dict[str, list] = {
        "images": [],
        "annotations": [],
        "hid_paths": [],
        "scalers": [],
        "panel_contents": [],
        "called_alleles": [],
    }

    # Determine dimensions from the first image
    if not images:
        logger.warning("No images to cache")
        return

    first = images[0]
    if first.data is None:
        raise ValueError("First image has no data; cannot determine cache dimensions")
    num_dyes = first.data.shape[0]
    signal_length = first.data.shape[1]

    for image in images:
        data["images"].append(image.data)
        data["annotations"].append(
            image.annotation.data if image.annotation else np.zeros_like(image.data, dtype=np.int8)
        )
        data["hid_paths"].append(str(image.path))
        data["scalers"].append(image._scaler.tolist())
        data["panel_contents"].append(
            json.dumps([m.to_dict() for m in image._panel.markers])
            if image._panel else "[]"
        )
        called = image.meta.get("called_alleles", [])
        data["called_alleles"].append(
            json.dumps([m.to_dict() for m in called]) if called else "[]"
        )

    ds = HFDataset.from_dict(
        data,
        features=Features({
            "images": Array3D(shape=(num_dyes, signal_length, 1), dtype="int16"),
            "annotations": Array3D(shape=(num_dyes, signal_length, 1), dtype="int8"),
            "hid_paths": Value(dtype="string"),
            "scalers": datasets.Sequence(feature=Value(dtype="float64")),
            "panel_contents": Value(dtype="string"),
            "called_alleles": Value(dtype="string"),
        }),
    )
    ds.save_to_disk(str(cache_path))
    logger.info("Cache written to {} ({} images)", cache_path, len(images))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _read_shard(
    directory: str, include_size_standard: bool, batch_size: int = 256
) -> tuple[int, Generator[HIDImage, None, None]]:
    """Read a single Arrow cache shard as a generator of HIDImages."""
    ds = load_from_disk(directory)
    ds.set_format(type="numpy", columns=["images", "annotations"], output_all_columns=True)

    def gen() -> Generator[HIDImage, None, None]:
        for batch in ds.iter(batch_size=batch_size):
            for img, ann, path, scaler, panel_json, ca_json in zip(
                batch["images"],
                batch["annotations"],
                batch["hid_paths"],
                batch["scalers"],
                batch["panel_contents"],
                batch["called_alleles"],
            ):
                yield _reconstruct_image(img, ann, path, scaler, panel_json, ca_json, include_size_standard)

    return len(ds), gen()


def _reconstruct_image(
    img_data, ann_data, path, scaler, panel_json, alleles_json, include_size_standard
) -> HIDImage:
    """Reconstruct an HIDImage from cached column data."""
    from dnanet.core.marker import Marker
    from dnanet.core.panel import Panel

    annotation = ScanpointAnnotation(data=np.asarray(ann_data, dtype="int8"))

    # Reconstruct panel from JSON
    panel_dicts = json.loads(panel_json) if isinstance(panel_json, str) else panel_json
    markers = tuple(Marker.from_dict(m) for m in panel_dicts)
    panel = Panel(markers=markers) if markers else None

    image = HIDImage(
        path=path,
        panel=panel,
        include_size_standard=include_size_standard,
        annotation=annotation,
        use_cache=True,
    )
    image._data = np.asarray(img_data, dtype="int16")
    image._scaler = np.array(scaler)

    # Reconstruct called alleles
    allele_dicts = json.loads(alleles_json) if isinstance(alleles_json, str) else alleles_json
    if allele_dicts:
        image._meta["called_alleles"] = [Marker.from_dict(m) for m in allele_dicts] # FIXME

    # NoC extraction is dataset-specific — done by the DatasetStrategy,
    # not hard-coded here. The caller can set image._meta["noc"] after loading.

    return image
