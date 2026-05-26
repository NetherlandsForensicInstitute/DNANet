"""HID (ABIF) binary file parser.

This module parses HID files — the binary format used by Applied Biosystems
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

from typing import TYPE_CHECKING, Union, Mapping, Sequence
from collections import Counter
from dataclasses import dataclass

import numpy as np
import construct
from loguru import logger

from dnanet.data.preprocessing.baseline import baseline_superior


if TYPE_CHECKING:
    from dnanet.core.types import PathLike
    from dnanet.data.strategies.scaling import ScalingStrategy


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ElementValue = Union[int, Sequence[int], str]


@dataclass(frozen=True)
class HIDElement:
    """A single directory element from an HID/ABIF file."""

    name: str
    element_type: int
    element_size: int
    num_elements: int
    data_offset: int
    data_handle: int | None = None
    data_size: int | None = None
    tag_number: int | None = None
    abif: str | None = None


# ---------------------------------------------------------------------------
# Low-level byte readers
# ---------------------------------------------------------------------------


def _read_string(raw: bytes) -> str:
    return raw.decode(encoding='utf-8')


def _read_signed(raw: bytes) -> int:
    return int.from_bytes(raw, byteorder='big', signed=True)


def _read_unsigned(raw: bytes) -> int:
    return int.from_bytes(raw, byteorder='big', signed=False)


# ---------------------------------------------------------------------------
# Parsing layers
# ---------------------------------------------------------------------------


def parse_hid_header(raw: bytes) -> HIDElement:
    """Parse the ABIF file header."""
    return HIDElement(
        abif=_read_string(raw[0:4]),
        name=_read_string(raw[6:10]),
        element_type=_read_signed(raw[14:16]),
        element_size=_read_signed(raw[16:18]),
        num_elements=_read_signed(raw[18:22]),
        data_offset=_read_signed(raw[26:30]),
        data_handle=_read_signed(raw[30:34]),
    )


def parse_hid_directory(raw: bytes, header: HIDElement) -> list[HIDElement]:
    """Parse all directory entries from the HID file."""
    directory: list[HIDElement] = []
    sup_19: list[HIDElement] = []
    sup_19_raw: Mapping[str, int] = Counter()

    for i in range(header.num_elements):
        start = i * header.element_size + header.data_offset
        entry = raw[start : start + header.element_size + 1]

        elem_type = _read_signed(entry[8:10])
        if elem_type <= 19 or elem_type == 1024:
            directory.append(
                HIDElement(
                    name=_read_string(entry[0:4]),
                    tag_number=_read_signed(entry[4:8]),
                    element_type=elem_type,
                    element_size=_read_signed(entry[10:12]),
                    num_elements=_read_signed(entry[12:16]),
                    data_size=_read_signed(entry[16:20]),
                    data_offset=_read_signed(entry[20:24]),
                )
            )
        else:
            sup_19_raw[_read_string(entry[0:4])] += 1

    # Second pass: decode the more complex element types by scanning for
    # their name bytes in the raw data.
    for sup_name, count in sup_19_raw.items():
        positions: list[int] = []
        pos = 0
        while True:
            pos = raw.find(sup_name.encode(), pos + 1)
            if pos == -1:
                break
            positions.append(pos)

        found = 0
        for idx in positions:
            if _is_valid_sup_19(idx, raw, sup_name) and found < count:
                sup_19.append(
                    HIDElement(
                        name=sup_name,
                        tag_number=_read_signed(raw[idx + 4 : idx + 8]),
                        element_type=_read_signed(raw[idx + 8 : idx + 10]),
                        element_size=_read_signed(raw[idx + 10 : idx + 12]),
                        num_elements=_read_signed(raw[idx + 12 : idx + 16]),
                        data_size=_read_signed(raw[idx + 16 : idx + 20]),
                        data_offset=_read_signed(raw[idx + 20 : idx + 24]),
                    )
                )
                found += 1

    return directory + sup_19


def parse_hid_data(
    raw: bytes, header: HIDElement, directory: list[HIDElement]
) -> dict[str, ElementValue | None]:
    """Decode all data elements from the HID file."""
    result: dict[str, ElementValue | None] = {}

    for i, elem in enumerate(directory):
        if elem.data_size and elem.data_size > 4:
            offset = elem.data_offset
        else:
            offset = i * header.element_size + header.data_offset + 20

        data_slice = raw[offset : offset + elem.num_elements * elem.element_size]
        key = f'{elem.name}_{elem.tag_number}'

        try:
            result[key] = _parse_element(data_slice, elem.element_type, elem.num_elements)
        except KeyError:
            result[key] = None
        except UnicodeError:
            result[key] = None

    return result


def _parse_element(data: bytes, elem_type: int, n_elements: int) -> ElementValue:
    """Decode a single element's bytes based on its type code."""
    if elem_type == 1:
        if n_elements == 1:
            return int.from_bytes(data, byteorder='big', signed=True)
        elif n_elements > 0:
            return np.frombuffer(data, dtype=construct.Int32sb.fmtstr)
    elif elem_type == 2:
        return data.decode('utf-8')
    elif elem_type in (3, 4):
        dtype = construct.Int16ub.fmtstr if elem_type == 3 else construct.Int16sb.fmtstr
        return np.frombuffer(data, dtype=dtype)
    elif elem_type == 5:
        if n_elements == 1:
            return int.from_bytes(data, byteorder='big', signed=True)
        elif n_elements > 0:
            return np.frombuffer(data, dtype='>i', count=n_elements)
    elif elem_type == 13:
        value = int.from_bytes(data, byteorder='big', signed=False)
        return 'true' if value == 1 else 'false'
    elif elem_type == 19:
        return data.decode('utf-8')
    elif elem_type == 1024:
        return data

    raise KeyError(f'Parsing not implemented, element type {elem_type}')


def _is_valid_sup_19(position: int, raw: bytes, name: str) -> bool:
    """Check whether a byte position is a valid sup-19 directory entry."""
    p = position
    return (
        _read_signed(raw[p + 8 : p + 10]) <= 19 or _read_signed(raw[p + 8 : p + 10]) == 1024
    ) and _read_string(raw[p : p + 4]) == name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_hid(path: PathLike) -> dict[str, ElementValue | None] | None:
    """Parse a complete HID file and return all decoded elements.

    Args:
        path: Path to the HID file.

    Returns:
        Dictionary mapping element keys (e.g. ``"DATA_1"``) to their decoded
        values, or ``None`` if the file cannot be read.
    """
    try:
        with open(path, 'rb') as f:
            raw = f.read()
        if raw[:4] != b'ABIF':
            if raw[:7] == b'version':
                logger.error('Git LFS stub detected for {}: run `git lfs pull` to fetch the real file', path)
            else:
                logger.error('Not a valid HID file (missing ABIF magic bytes): {}', path)
            return None
        header = parse_hid_header(raw)
        directory = parse_hid_directory(raw, header)
        return parse_hid_data(raw, header, directory)
    except PermissionError:
        logger.debug('Permission denied for file {}', path)
        return None


def get_peak_data(
    path: PathLike, scaling_strategy: ScalingStrategy, data_loading_strategy: str = 'superior'
) -> np.ndarray | None:
    """Extract dye channel data from an HID file.

    Args:
        path: Path to the HID file.
        scaling_strategy: Scaling strategy to use for scaling.
        data_loading_strategy: Data loading strategy — one of:
            - ``"raw"``: Raw unprocessed signal.
            - ``"analyzed"``: Pre-analyzed signal from the instrument.
            - ``"superior"``: Raw signal with superior baseline subtraction.

    Returns:
        Array of shape ``(6, N)`` containing RFU values for each dye channel
        (5 fluorescence dyes + 1 size standard), or ``None`` on failure.
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
        dyes = np.array([data[col] for col in columns], dtype=np.int32)
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
