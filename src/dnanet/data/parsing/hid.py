"""HID (ABIF) binary file parser.

This module parses HID/FSA files — the binary format used by Applied Biosystems
for storing forensic DNA electropherogram data. It's a domain-specific parser
that decodes the ABIF container format to extract raw fluorescence signals.

Design pattern: **Layered Parsing Pipeline**
    The parsing proceeds in three well-defined layers, each with a single
    responsibility:

    1. ``parse_hid_header`` — reads the file header (magic bytes, offsets)
    2. ``parse_hid_directory`` — reads the directory of data elements
    3. ``parse_hid_data`` — decodes each element's raw bytes into typed values

    The public API is ``parse_hid(path)`` which chains all three, and
    ``get_peak_data(path, strategy)`` which extracts the dye channels.

    This module is ported almost verbatim from the original DNANet — it's
    well-isolated, domain-specific binary parsing with no external coupling.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Union, Sequence
from collections import Counter

import numpy as np
from loguru import logger

from dnanet.data.preprocessing.baseline import baseline_superior


if TYPE_CHECKING:
    from dnanet.core.types import PathLike
    from dnanet.data.strategies.scaling.scaling import ScalingStrategy


ElementValue = Union[int, Sequence[int], str, bytes, np.ndarray]

# ABIF Header (34 bytes total):
# 4s (ABIF), H (version), 4s (dir_name), I (dir_tag_num), H (dir_type),
# H (dir_elem_size), I (dir_num_elem), I (dir_data_size), I (dir_data_offset), I (dir_data_handle)
_HEADER_STRUCT = struct.Struct('>4sH4sIHHIIII')

# ABIF Directory Entry (28 bytes total):
# 4s (name), I (tag_num), H (elem_type), H (elem_size), I (num_elem),
# I (data_size), I (data_offset), I (data_handle)
_ENTRY_STRUCT = struct.Struct('>4sIHHIII')


def _is_valid_type(elem_type: int) -> bool:
    return elem_type <= 19 or elem_type == 1024


def _decode_entry(mem: memoryview, entry_offset: int, fields: tuple) -> tuple[str, object]:
    """Decode one directory entry located at entry_offset into (key, value)."""
    name_bytes, tag_num, elem_type, _, n_elem, data_size, elem_data_offset = fields

    # Data <= 4 bytes is stored directly inside the data_offset field itself
    data_start = entry_offset + 20 if data_size <= 4 else elem_data_offset

    key = f'{name_bytes.decode("ascii", errors="replace")}_{tag_num}'
    return key, _decode_element(mem[data_start : data_start + data_size], elem_type, n_elem)


def parse_abif_bytes(raw: bytes) -> dict[str, object] | None:
    """Parse an ABIF binary payload (HID or FSA)."""
    if len(raw) < 128 or raw[:4] != b'ABIF':
        if raw[:7] == b'version':
            logger.error('Git LFS stub detected. Run `git lfs pull` to fetch real files.')
        else:
            logger.error('Not a valid ABIF file (missing ABIF magic bytes).')
        return None

    _, _, _, _, _, elem_size, num_elements, _, data_offset, _ = _HEADER_STRUCT.unpack_from(raw, 0)

    mem = memoryview(raw)
    raw_len = len(raw)
    result: dict[str, object] = {}
    sup_19_counts: Counter[bytes] = Counter()

    # Pass 1: regular directory entries. Entries with an unknown type are
    # placeholders; their real entry lives elsewhere in the file.
    for i in range(num_elements):
        entry_offset = data_offset + i * elem_size
        if entry_offset + 28 > raw_len:
            break

        fields = _ENTRY_STRUCT.unpack_from(raw, entry_offset)
        if _is_valid_type(fields[2]):
            key, value = _decode_entry(mem, entry_offset, fields)
            result[key] = value
        else:
            sup_19_counts[fields[0]] += 1

    # Pass 2: find the real entries for the placeholders (same heuristic as
    # the original: first `count` occurrences of the name with a valid type).
    for name_bytes, count in sup_19_counts.items():
        found = 0
        pos = raw.find(name_bytes, 1)
        while pos != -1 and found < count:
            if pos + _ENTRY_STRUCT.size <= raw_len:
                fields = _ENTRY_STRUCT.unpack_from(raw, pos)
                if _is_valid_type(fields[2]):
                    key, value = _decode_entry(mem, pos, fields)
                    result[key] = value
                    found += 1
            pos = raw.find(name_bytes, pos + 1)

    return result


def _decode_element(data: memoryview, elem_type: int, n_elem: int) -> ElementValue | None:
    """Decode an ABIF element payload directly into NumPy arrays or primitive types."""
    if not data:
        return None
    match (elem_type, n_elem):
        # Byte / int8
        case (1, _):
            return (
                int.from_bytes(data, 'big', signed=True)
                if n_elem == 1
                else np.frombuffer(data, dtype='i')
            )
        # ASCII String
        case (2, _):
            return bytes(data).decode('utf-8', errors='replace')
        # Word / uint16
        case (3, _):
            return np.frombuffer(data, dtype='H')
        # Short / int16
        case (4, _) | (1024, _):
            return np.frombuffer(data, dtype='>i2')
        # Long / int32
        case (5, 1):
            return int.from_bytes(data, 'big', signed=True)
        # Long / int32
        case (5, _):
            return np.frombuffer(data, dtype='>i4')
        # Float / float32
        case (7, _):
            return np.frombuffer(data, dtype='>f4')
        # Double / float64
        case (8, _):
            return np.frombuffer(data, dtype='>f8')
        # Boolean
        case (13, _):
            return 'true' if data[0] == 1 else 'false'
        # pString / CString
        case (18, _) | (19, _):
            return bytes(data).decode('utf-8', errors='replace').rstrip('\x00')

    return bytes(data)


def _ensure_array(val: ElementValue | None) -> np.ndarray:
    """Safely convert raw bytes, memoryviews, or existing sequences into 1D int32 NumPy arrays."""
    if val is None:
        return np.array([], dtype=np.int32)
    if isinstance(val, (bytes, memoryview)):
        return np.frombuffer(val, dtype='h').astype(np.int32)
    return np.asarray(val, dtype=np.int32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_hid(path: PathLike) -> dict[str, object] | None:
    """Parse a complete HID or FSA file and return all decoded elements."""
    try:
        with open(path, 'rb') as f:
            return parse_abif_bytes(f.read())
    except PermissionError:
        logger.debug('Permission denied for file {}', path)
        return None


def get_peak_data(
    path: PathLike,
    scaling_strategy: ScalingStrategy,
    data_loading_strategy: str = 'superior',
) -> np.ndarray | None:
    """Extract dye channel data from an HID or FSA file.

    Args:
        path: Path to the HID/FSA file.
        scaling_strategy: Strategy containing STRKit channel configuration.
        data_loading_strategy: Data loading strategy: 'raw', 'analyzed', or 'superior'.

    Returns:
        Array of shape (num_channels, N) containing RFU values for each dye channel,
        or None on failure.
    """
    if data_loading_strategy not in ('raw', 'analyzed', 'superior'):
        raise ValueError(
            f"strategy must be 'raw', 'analyzed', or 'superior', got '{data_loading_strategy}'"
        )

    data = parse_hid(path)
    if data is None:
        return None

    try:
        if data_loading_strategy in ('raw', 'superior'):
            columns = scaling_strategy.kit.hid_file_data_columns_raw
        else:  # analyzed
            columns = scaling_strategy.kit.hid_file_data_columns_analyzed

        if columns is None:
            raise ValueError(
                f'Kit {scaling_strategy.kit.name} does not support data loading strategy {data_loading_strategy}. '
                f'Please specify the hid_file_data_columns in the STRKit configuration.'
            )

        # Build 2D dye matrix safely regardless of input trace encoding
        dyes = np.array([_ensure_array(data[col]) for col in columns], dtype=np.int32)

    except KeyError:
        data_keys = [k for k in data if k.startswith('DATA')]
        logger.warning(
            'Missing {} DATA columns in {}, found: {}', data_loading_strategy, path, data_keys
        )
        return None

    if data_loading_strategy == 'superior':
        baseline = baseline_superior(dyes)
        dyes = dyes - baseline

    iinfo = np.iinfo(np.int16)
    return np.clip(dyes, iinfo.min, iinfo.max).astype(np.int16)
