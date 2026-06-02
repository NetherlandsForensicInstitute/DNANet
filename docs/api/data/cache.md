# Memmap Cache

The `dnanet.data.cache` package provides a lazy, memmap-backed disk cache for
`HIDDataset`. It enables large datasets to be loaded without keeping all pixel
data in RAM.

## Layout

Each cache key maps to a directory containing:

| File | Contents |
|------|----------|
| `data.bin` | `(N, D, L)` int16 — fluorescence signal |
| `annotation.bin` | `(N, D, L)` int8 — scanpoint mask |
| `scaler.bin` | `(N, L)` float32 — base-pair mapping |
| `index.parquet` | N rows of path + sidecar keys |
| `panels.json` | Deduped adjusted panels |
| `alleles.json` | Deduped allele annotations |
| `shapes.json` | Row counts, array shapes, dtypes |
| `fingerprint.json` | SHA256 of sources + config |
| `manifest.jsonl` | Append-only writer log |
| `_COMPLETE` | Sentinel for successful write |

## Reader

```python
from dnanet.data.cache.reader import IndexEntry, MemmapCacheReader
```

`MemmapCacheReader` opens memmaps lazily per-process (fork-safe for DataLoader
workers). Use `load_index()` to read the index without touching pixel data,
then `get_row(idx)` to retrieve `(data, annotation, scaler)` slices.

## Writer

```python
from dnanet.data.cache.writer import MemmapCacheWriter
```

`MemmapCacheWriter` appends rows to binary files and writes a JSONL manifest.
On `finalize()`, the manifest is converted to `index.parquet` with deduped
JSON sidecars.

**Usage:**

```python
with MemmapCacheWriter(cache_dir) as writer:
    for image in images:
        writer.write(image)
    writer.finalize(config_payload, source_paths)
```

## Fingerprinting

Two hashes are used:

- **key**: 16-char hex derived from config only. Drives the cache directory name.
- **fingerprint**: full SHA256 from config + source file contents. Stored
  inside the cache dir and validated on load.

Content-based validation (SHA256 of file contents) ensures cached datasets
survive copies to NFS shares, mount point changes, and file renames.
