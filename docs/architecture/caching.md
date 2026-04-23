# Dataset Caching

`HIDDataset` persists fully pre-processed profiles to a memmap-backed cache so
that `__init__` only touches a small on-disk index and `__getitem__` streams
single rows from disk on demand. No pickle, no dependency additions; only
NumPy memmaps + Parquet + JSON.

## Cache directory layout

Each config hashes to a key; the cache for that key lives under

```
<cache_dir>/<key>/
    data.bin           # (N, D, L)  int16    — fluorescence signal
    annotation.bin     # (N, D, L)  int8     — scanpoint mask
    scaler.bin         # (N, L)     float32  — base-pair scaler
    index.parquet      # per-row: path, has_annotation, panel_key, allele_key, meta_json
    panels.json        # {panel_key: adjusted_panel_json}  — deduped
    alleles.json       # {allele_key: allele_annotation_json}  — deduped
    shapes.json        # row counts, per-array shapes and dtypes
    fingerprint.json   # hash of source files + config; validated on load
    manifest.jsonl     # append-only writer log (removed at finalize)
    _COMPLETE          # sentinel, only touched after a successful finalize
```

All three `.bin` files are row-major, fixed-shape arrays. A single sample is
retrieved by slicing row index `i` out of each memmap — O(1) regardless of
cache size.

## Cache key vs. fingerprint

Two hashes drive invalidation:

| hash | input | stored as | purpose |
|---|---|---|---|
| **key** | config only (strategies, adjustment, flags, `cache_version`) | 16-char hex used as the cache directory name | routes a config to its cache dir |
| **fingerprint** | config + every source file's `(path, mtime_ns, size)` | full SHA256 in `fingerprint.json` | detects source changes on disk; mismatched → rebuild |

Changing any config parameter produces a different key and a fresh cache
directory. Touching a source HID re-stamps its mtime and invalidates the
fingerprint in the existing directory.

`CACHE_VERSION` (in `layout.py`) is bumped whenever the on-disk layout
changes; old caches automatically miss and rebuild.

## Write path (resumable)

`MemmapCacheWriter` is streaming and crash-safe:

1. `write(image)` appends one row to each `.bin`, flushes, then appends one
   JSON line to `manifest.jsonl`. Binary bytes hit disk before the manifest
   line commits, so a crash leaves — at worst — a trailing partial `.bin`
   write that is truncated on resume.
2. On reopen, `_recover_from_manifest()` reads the manifest, truncates each
   `.bin` to the committed row count, and rewrites the manifest dropping any
   partial trailing line.
3. Callers can call `writer.resume_paths()` to skip sources already written.
4. `finalize()` streams the manifest into `index.parquet`, writes the deduped
   sidecars, writes `shapes.json` and `fingerprint.json`, deletes the
   manifest, and touches `_COMPLETE`.

## Deduped sidecars (panels + allele annotations)

The panel JSON is large (~126 KB per row) and nearly identical across rows
(it only varies per ladder). Allele annotations are smaller (~13 KB) and
shared across replicas of the same mixture. Storing either inline per row
in the Parquet index pins them in RAM — at 87k casework rows, that's
**~12 GB** just for the Python strings returned by `load_index()`.

At finalize time the writer stream-interns both JSON strings into
`{key → json_string}` sidecars and the Parquet row keeps only small integer
keys. The reader loads each sidecar lazily (on first `panel_json(key)` /
`allele_json(key)` call) and caches it for the process lifetime. Forked
DataLoader workers drop the cache in `__getstate__` and reload per worker.

Measured on the R&D dataset (325 rows, 26 unique panels, 24 unique
annotations):

| metric | inline | deduped |
|---|---|---|
| `index.parquet` on disk | 745 KB | **11 KB** |
| `self._index` in RAM | ~46 MB | **~0.03 MB** |
| Projected for 87k casework rows | ~12 GB | ~3–5 MB |

## Reader: lazy, fork-safe memmaps

`MemmapCacheReader` holds no `numpy.memmap` handles until `get_row` is
called, and the memmaps it does open are keyed by `os.getpid()` so that
each forked DataLoader worker opens its own independent handles. The
`__getstate__` hook clears both memmaps and sidecar caches before
cross-process pickling.

```python
reader = MemmapCacheReader(cache_dir)
index  = reader.load_index()              # small: paths + int keys only
data, ann, scaler = reader.get_row(42)    # opens memmaps lazily, slices one row
panel_js  = reader.panel_json(entry.panel_key)
allele_js = reader.allele_json(entry.allele_key)
```

## In-memory realization (`load_in_memory=True`)

For small caches, `HIDDataset(load_in_memory=True)` copies each memmap into
a RAM-resident `ndarray` so `__getitem__` slices from memory rather than
disk. This is gated by a **hard RAM guard**: the dataset refuses to
materialize if the cache would exceed 50% of total physical RAM, surfacing
a descriptive `RuntimeError` instead of silently triggering OOM.

```python
HIDDataset(
    ...,
    load_in_memory=True,   # copy memmaps into RAM after cache load
)
# RuntimeError: load_in_memory=True refused: cache would use 42.3 GB,
#   exceeds 50% of total RAM (64.0 GB). Set load_in_memory=False.
```

## Nullable annotations (`allow_missing_annotations=True`)

Evaluation and labeltool workflows need to read HIDs that have no annotation
attached. When `allow_missing_annotations=True`, the cache build keeps rows
whose `annotation` is `None`: the writer stores a zero-filled placeholder in
`annotation.bin` (to keep the memmap uniform) and records `has_annotation:
False` on the Parquet row. `__getitem__` reads that flag and returns
`image.annotation = None` for those rows.

Default is `False` — training pipelines continue to drop unannotated rows.

## Cache inspection tool

Running

```
python -m dnanet.data.cache.inspect <cache_dir_or_root>
```

prints on-disk sizes, the Parquet schema, approximate in-RAM cost of each
string column, sidecar dedup ratios, and projected memmap-realization cost
if `load_in_memory=True`. Pass either a single key directory or a parent
`cache_dir` to walk all keys.

Typical output:

```
=== data/cache/dnanet_rd/1a1685307ca20e23 ===
  complete marker:  True
  rows (n):         325
  on-disk:
    data.bin                15.97 MB
    annotation.bin           7.99 MB
    scaler.bin               5.32 MB
    index.parquet            0.01 MB
    panels.json              3.74 MB
    alleles.json             0.35 MB
  index in-RAM (string columns, approx):
    path                     0.03 MB
    STRING TOTAL             0.03 MB
  panels.json:   26 unique (avg=128598 B, max=128643 B)
  alleles.json:  24 unique (avg=13097 B, max=16685 B)
  memmap realize: 0.03 GB if load_in_memory=True
```

## API surface

```python
from dnanet.data.cache import (
    IndexEntry,            # frozen dataclass: row, path, has_annotation, panel_key, allele_key, meta_json
    MemmapCacheReader,
    MemmapCacheWriter,
    compute_key,           # 16-char config hash → cache dir name
    compute_fingerprint,   # config + source stamps → SHA256 dict
    validate_fingerprint,
    is_complete,
    cache_key_dir,
)
```

The writer/reader are used transparently by `HIDDataset`; direct use is only
needed for tools that introspect or rebuild caches.
