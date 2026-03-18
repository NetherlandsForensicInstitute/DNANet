import json
import logging
import pickle
from pathlib import Path
from typing import Optional, List, Iterator, Sequence, Dict, Any, Tuple

import numpy as np
from tqdm import tqdm

from DNAnet.data.data_models import Marker, Annotation
from DNAnet.data.data_models.extracted_peak import ExtractedPeak, SCAN_TO_BASE
from DNAnet.data.data_models.hid_dataset import HIDDataset
from DNAnet.data.preprocessing.peak_extraction import extract_peak_windows
from DNAnet.data.preprocessing.peak_utils import filter_peaks_AT_LT
from DNAnet.data.preprocessing.preprocess_item import RFU_MAX_VALUE, preprocess_peak
from DNAnet.typing import PathLike

LOGGER = logging.getLogger('dnanet')


class PeakWindowDataset(HIDDataset):
    """
    A dataset that extracts peak windows from HIDImages, with optional preprocessing and caching.
    Each item is an ExtractedPeak object containing the data around the peak center and its annotation.

     Caching:
      - If use_cache_peaks=True and cache_path_peaks is provided, peaks will be loaded from cache if it
      exists, otherwise they will be extracted and written to cache for future use.
      - The cache stores only the necessary fields to reconstruct ExtractedPeak objects, and loads them
      as memory-mapped numpy arrays for efficient access without loading everything into RAM at once.
    """
    def __init__(self, *args,
                 threshold: float,
                 window_size: int,
                 filter_peaks: bool,
                 preprocess: bool = True,
                 smooth_keep_factor: Optional[float] = 0.4,
                 log_scale: bool = True,
                 max_rfu_value: Optional[int] = RFU_MAX_VALUE,
                 use_cache_peaks: Optional[bool] = False,
                 cache_path_peaks: Optional[PathLike] = None,
                 ground_truth_as_annotations: bool = False,
                 labels: List[str] = ["allele", "noise"],
                 include_max_pool_dyes: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.window_size = window_size
        self.filter_peaks = filter_peaks
        self.preprocess = preprocess
        self.smooth_keep_factor = smooth_keep_factor
        self.log_scale = log_scale
        self.max_rfu_value = max_rfu_value
        self.use_cache_peaks = use_cache_peaks
        self.cache_path_peaks = cache_path_peaks
        self.ground_truth_as_annotations = ground_truth_as_annotations
        self.include_max_pool_dyes = include_max_pool_dyes
        self.label_to_idx = dict(zip(labels, range(len(labels))))
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()} # invert the mapping

        if cache_path_peaks is not None and use_cache_peaks is False:
            LOGGER.info("Cache does not exist yet, creating it now...")
            self.write_peak_cache(shard_size=1_000_000)
            self.use_cache_peaks = True

        self._data = list(tqdm(self.__iter__(), desc="Extracting peak windows", unit="peaks"))

        LOGGER.info(f"Initialized PeakWindowDataset with threshold={threshold}, window_size={window_size}")

    def __iter__(self) -> Iterator[ExtractedPeak]:
        if self._data and isinstance(self._data[0], ExtractedPeak):
            return super().__iter__()

        if self.use_cache_peaks:
            out = Path(self.cache_path_peaks)
            if self.cache_path_peaks is not None:
                if not out.exists() or not any(out.iterdir()):
                    self.write_peak_cache(shard_size=1_000_000)
                return self.cache_iterator()


        return self.iterate_HIDImages()

    def iterate_HIDImages(self) -> Iterator[ExtractedPeak]:
        for image in super().__iter__():

            peaks = extract_peak_windows(image, self.threshold, self.window_size,
                                         use_ground_truth_labels=self.ground_truth_as_annotations,
                                         include_raw_annotation=False,
                                         include_max_pool_dyes=self.include_max_pool_dyes)

            if self.filter_peaks:
                threshold_type = self.analysis_threshold_type if self.analysis_threshold_type else "AT"
                peaks = filter_peaks_AT_LT(peaks, threshold_type)
            for peak in peaks:
                if self.preprocess:
                    yield preprocess_peak(peak,
                                          smooth_keep_factor=self.smooth_keep_factor,
                                          log_scale=self.log_scale,
                                          max_rfu_value=self.max_rfu_value)
                else:
                    yield peak

    def cache_iterator(self,
                       ) -> Iterator[ExtractedPeak]:
        """
        Loads shards in natural (sorted) order, memory-maps arrays, and reconstructs
        ExtractedPeak objects
        """
        root = Path(self.cache_path_peaks)
        meta = json.loads((root / "meta.json").read_text())
        window_size: int = int(meta["window_size"])
        peak_count: int = int(meta["num_peaks"])
        # threshold_type = self.analysis_threshold_type if self.analysis_threshold_type else "AT"


        # load markers once
        with open(root / "markers.pkl", "rb") as f:
            markers: Sequence[Marker] = pickle.load(f)

        # A tiny stub to attach to ExtractedPeak.image so peak.image._data exists
        class _ImageStub:
            def __init__(self, data_1d: np.ndarray, markers_seq: Sequence[Marker]):
                # expose both ._data and .data for compatibility
                self._data = data_1d
                self.data = data_1d
                # mimic image._panel._panel if ever accessed
                self._panel = type("PanelStub", (), {"_panel": markers_seq})()

        # Iterate shards in natural (sorted) order; no shuffling
        with tqdm(desc=f"Loading cached data from (ExtractedPeaks)", unit="peaks", total=peak_count) as pbar:
            for shard_name in sorted(meta["shards"]):
                base = root / shard_name

                image_data = np.load(base / "image_data.npy", mmap_mode="r")                # (N, W) int16
                annotation = np.load(base / "annotation.npy", mmap_mode="r")                # (N, W) int16
                annotation_label = np.load(base / "annotation_label.npy", mmap_mode="r")    # (N,) int16
                dye_index = np.load(base / "dye_index.npy", mmap_mode="r")                  # (N,)  int16
                center_idx = np.load(base / "center_index.npy", mmap_mode="r")              # (N,)  int16
                peak_height = np.load(base / "peak_height.npy", mmap_mode="r")              # (N,)  int16
                window_start = np.load(base / "window_start.npy", mmap_mode="r")            # (N,)  int16

                rows = int(image_data.shape[0])
                pbar.set_description(f"Loading cached data from (ExtractedPeaks) {shard_name}")

                for i in range(rows):
                    # Build an ExtractedPeak *instance* without running its __init__
                    peak = object.__new__(ExtractedPeak)  # type: ignore

                    # Required attributes
                    peak.window_start = int(window_start[i])
                    peak.peak_height = int(peak_height[i])
                    peak.window_size = int(window_size)
                    peak.original_peak_center_index = int(center_idx[i])
                    peak.dye_index = int(dye_index[i])
                    peak.peak_basepair = SCAN_TO_BASE(peak.original_peak_center_index)


                    # Optional attributes from your class
                    peak.markers = markers
                    peak.is_normalized = False
                    peak.include_max_pool_dyes = False
                    peak.marker = peak._find_marker()

                    # Attach data and annotation directly
                    # Note: keep them as views to memmap rows to minimize RAM
                    data_row = image_data[i]
                    peak._data = data_row

                    # Provide a minimal .image that exposes _data (as requested)
                    peak.image = _ImageStub(data_row, markers)
                    peak._annotation = Annotation(image=annotation[i], labels=self.idx_to_label[annotation_label[i]])

                    pbar.update(1)


                    if self.preprocess:
                        yield preprocess_peak(peak,
                                              smooth_keep_factor=self.smooth_keep_factor,
                                              log_scale=self.log_scale,
                                              max_rfu_value=self.max_rfu_value)
                    else:
                        yield peak


    def write_peak_cache(
            self,
            shard_size: int = 100_000,
    ) -> None:
        """
        Writes a streaming cache of peaks to disk. Only stores:
          - image_data:       int16 array   (N, W)    <-- peak.image._data
          - annotation:       int16 array   (N, W)    <-- peak._annotation
          - dye_index:        int16 array   (N,)
          - center_index:     int16 array   (N,)      <-- peak.original_peak_center_index
          - peak_height:      int16 array   (N,)
          - window_start:     int16 array   (N,)
        Global metadata (single value for whole cache):
          - window_size:      int16

        Files:
          cache/
            meta.json
            markers.pkl                # Sequence[Marker]
            shard_000/
              image_data.npy
              annotation.npy
              annotation_label.npy
              dye_index.npy
              center_index.npy
              peak_height.npy
              window_start.npy
            shard_001/
              ...
        """

        LOGGER.info(f"Writing peak cache to {self.cache_path_peaks}...")

        out = Path(self.cache_path_peaks)
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f"Cache path {out} already exists and is not empty. Manually delete it or choose a different path.")
        out.mkdir(parents=True, exist_ok=True)

        # buffers
        image_buffer, annotation_buffer = [], []
        dye_buf, ctr_buf, hgt_buf, wst_buf, ann_label_buf = [], [], [], [], []
        peak_count = 0

        shard_idx = 0
        window_size_meta: Optional[int] = None
        markers_saved = False

        def flush():
            nonlocal shard_idx
            if not image_buffer:  # nothing to flush
                return
            base = out / f"shard_{shard_idx:03d}"
            base.mkdir(exist_ok=True)

            # convert lists -> numpy (int16 everywhere per requirement)
            image_data = np.asarray(image_buffer, dtype=np.int16)
            annotation = np.asarray(annotation_buffer, dtype=np.int16)
            annotation_label = np.asarray(ann_label_buf, dtype=np.int16)
            dye_index = np.asarray(dye_buf, dtype=np.int16)
            center_idx = np.asarray(ctr_buf, dtype=np.int16)
            peak_height = np.asarray(hgt_buf, dtype=np.int16)
            window_start = np.asarray(wst_buf, dtype=np.int16)

            # save arrays
            np.save(base / "image_data.npy", image_data)
            np.save(base / "annotation.npy", annotation)
            np.save(base / "annotation_label.npy", annotation_label)
            np.save(base / "dye_index.npy", dye_index)
            np.save(base / "center_index.npy", center_idx)
            np.save(base / "peak_height.npy", peak_height)
            np.save(base / "window_start.npy", window_start)

            # clear buffers
            image_buffer.clear(); annotation_buffer.clear()
            dye_buf.clear(); ctr_buf.clear(); hgt_buf.clear(); wst_buf.clear(); ann_label_buf.clear()
            LOGGER.debug(f"Wrote shard {base} with {len(image_data)} peaks.")
            shard_idx += 1

        # stream from your dataset
        for image in super().__iter__():
            peaks = extract_peak_windows(image, self.threshold, self.window_size, use_ground_truth_labels=False, include_raw_annotation=True)


            for peak in peaks:
                # capture global metadata, markers once
                if window_size_meta is None:
                    window_size_meta = int(peak.window_size)
                if not markers_saved:
                    with open(out / "markers.pkl", "wb") as f:
                        pickle.dump(peak.markers, f, protocol=pickle.HIGHEST_PROTOCOL)
                    markers_saved = True

                # append required fields
                image_buffer.append(np.asarray(peak.data, dtype=np.int16))
                annotation_buffer.append(np.asarray(peak.annotation.image, dtype=np.int16))
                ann_label_buf.append(np.int16(self.label_to_idx[peak.annotation.label]))
                dye_buf.append(np.int16(peak.dye_index))
                ctr_buf.append(np.int16(peak.original_peak_center_index))
                hgt_buf.append(np.int16(peak.peak_height))
                wst_buf.append(np.int16(peak.window_start))
                peak_count += 1

                if len(image_buffer) >= shard_size:
                    flush()

        flush()  # final

        # write meta
        meta: Dict[str, Any] = {
            "schema_version": 2,
            "window_size": int(window_size_meta or 0),
            "dtype": "int16",
            "num_peaks": peak_count,
            "shards": sorted([p.name for p in out.glob("shard_*")]),
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2))

        LOGGER.info(f"Wrote peak cache to {self.cache_path_peaks} with {peak_count} total peaks in {shard_idx} shards.")


    def split(self, fraction: float, seed: Optional[float] = None) -> Tuple['SimpleDataset', 'SimpleDataset']:
        if self.group_replicas_in_split:
            LOGGER.warning("Splitting with group_replicas_in_split=True is not implemented for PeakWindowDataset. "
                           "Continuing with a regular random split that may split replicas across datasets.")
            self.group_replicas_in_split = False
        return super().split(fraction, seed)
