import logging
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Mapping, Optional, Sequence, Union

import construct
import numpy as np

from DNAnet.typing import PathLike


ElementValue = Union[int, Sequence[int], str]


LOGGER = logging.getLogger("dnanet")


@dataclass(frozen=True)
class HIDElement:
    """
    Dataclass to support the xml elements of a HID file
    """
    name: str
    element_type: int
    element_size: int
    num_elements: int
    data_offset: int
    data_handle: int = None
    data_size: int = None
    tag_number: int = None
    abif: str = None


def _read_string(raw_input: bytes) -> str:
    """
    Reads string from bytes object

    :param raw_input: raw bytes object
    :returns: parsed string
    """
    return raw_input.decode(encoding="utf-8")


def _read_signed_integer(raw_input: bytes) -> int:
    """
    Reads signed integer from bytes object

    :param raw_input: raw bytes object
    :returns: parsed integer
    """
    return int.from_bytes(raw_input, byteorder="big", signed=True)


def _read_unsigned_integer(raw_input) -> int:
    """
    Reads unsigned integer from bytes object

    :param raw_input: raw bytes object
    :returns: parsed integer
    """
    return int.from_bytes(raw_input, byteorder="big", signed=False)


def parse_hid_header(raw_data: bytes) -> HIDElement:
    """
    Parses header information of HID file

    :param raw_data: raw hid file
    :returns: header information
    """
    return \
        HIDElement(
            abif=_read_string(raw_data[0:4]),
            name=_read_string(raw_data[6:10]),
            element_type=_read_signed_integer(raw_data[14:16]),
            element_size=_read_signed_integer(raw_data[16:18]),
            num_elements=_read_signed_integer(raw_data[18:22]),
            data_offset=_read_signed_integer(raw_data[26:30]),
            data_handle=_read_signed_integer(raw_data[30:34])
        )


@lru_cache(maxsize=None)
def parse_hid_directory(raw_data: bytes,
                        header_dict: HIDElement) -> \
        List[HIDElement]:
    """
    Parse directory of HID file

    :param raw_data: raw hid file
    :param header_dict: header of HID file
    :returns: all parsed hid elements
    """
    hid_directory = []
    sup_19 = []
    sup_19_raw: Mapping[str, int] = Counter()

    n_elements: int = header_dict.num_elements
    # first pass at trying to decode data
    for i in range(n_elements):
        deb = i * header_dict.element_size + header_dict.data_offset
        directory_entry = raw_data[deb:(deb + header_dict.element_size + 1)]

        # extract and decode data for the easier cases
        if _read_signed_integer(directory_entry[8:10]) <= 19 \
                or _read_signed_integer(directory_entry[8:10]) == 1024:
            hid_directory.append(HIDElement(
                name=_read_string(directory_entry[0:4]),
                tag_number=_read_signed_integer(directory_entry[4:8]),
                element_type=_read_signed_integer(directory_entry[8:10]),
                element_size=_read_signed_integer(directory_entry[10:12]),
                num_elements=_read_signed_integer(directory_entry[12:16]),
                data_size=_read_signed_integer(directory_entry[16:20]),
                data_offset=_read_signed_integer(directory_entry[20:24])))

        # make note of the more complicated cases
        else:
            sup_19_raw[_read_string(directory_entry[0:4])] += 1

    for sup_19_name, n_occurrences in sup_19_raw.items():
        # find potential data positions for each name / type of data
        positions = []
        last_position = 0
        while last_position != -1:
            last_position = \
                raw_data.find(sup_19_name.encode(), last_position + 1)
            positions.append(last_position)

        # loop over the positions to decode data
        occurrence = 0
        for idx in positions:
            if _is_valid_sup_19(idx, raw_data, sup_19_name):
                # only add data if value count hasn't been reached
                if occurrence < n_occurrences:
                    sup_19.append(
                        HIDElement(name=sup_19_name,
                                   tag_number=_read_signed_integer(
                                       raw_data[(idx + 4):(idx + 8)]),
                                   element_type=_read_signed_integer(
                                       raw_data[(idx + 8):(idx + 10)]),
                                   element_size=_read_signed_integer(
                                       raw_data[(idx + 10):(idx + 12)]),
                                   num_elements=_read_signed_integer(
                                       raw_data[(idx + 12):(idx + 16)]),
                                   data_size=_read_signed_integer(
                                       raw_data[(idx + 16):(idx + 20)]),
                                   data_offset=_read_signed_integer(
                                       raw_data[(idx + 20):(idx + 24)])))
                    occurrence += 1
                else:
                    break

    return hid_directory + sup_19


def parse_hid_data(raw_data: bytes,
                   hid_header: HIDElement,
                   hid_directory: List[HIDElement]) -> \
        Mapping[str, Optional[ElementValue]]:
    """
    Reads HID data

    :param raw_data: raw HID data to be parsed
    :param hid_header: current HID header
    :param hid_directory: current HID directory
    :returns: all elements (its name and the actual value) in the hid file
    """
    hid_data = {}

    for i, hid_element in enumerate(hid_directory):
        # determine input data range
        if hid_element.data_size > 4:
            deb_int = hid_element.data_offset
        else:
            deb = i * hid_header.element_size + hid_header.data_offset
            deb_int = deb + 20

        # decode data
        element_type = hid_element.element_type
        n_elements = hid_element.num_elements
        element_size = hid_element.element_size

        data_slice = raw_data[deb_int:int(deb_int + n_elements * element_size)]
        key = f'{hid_element.name}_{hid_element.tag_number}'
        try:
            hid_data[key] = parse_element(data_slice, element_type, n_elements)
        except KeyError as e:
            if 'Parsing not implemented, element type' in str(e):
                hid_data[key] = None
            else:
                raise
        except UnicodeError:
            hid_data[key] = None

    return hid_data


def parse_element(data_slice: bytes,
                  element_type: int,
                  n_elements: int) \
        -> ElementValue:
    """
    Parses an element from the raw data

    :param data_slice: slice of data to be parsed
    :param element_type: element type that is parsed
    :param n_elements: number of elements
    :returns: value of the element
    """
    if element_type == 1:
        if n_elements == 1:
            return int.from_bytes(data_slice, byteorder='big', signed=True)
        elif n_elements > 0:
            return np.frombuffer(data_slice, dtype=construct.Int32sb.fmtstr)
    elif element_type == 2:
        return data_slice.decode("utf-8")

    elif element_type in (3, 4):
        dtype = construct.Int16ub.fmtstr if element_type == 3 else construct.Int16sb.fmtstr
        return np.frombuffer(data_slice, dtype=dtype)
    elif element_type == 5:
        if n_elements == 1:
            return int.from_bytes(data_slice, byteorder='big', signed=True)
        elif n_elements > 0:
            return np.frombuffer(data_slice, dtype=">i", count=n_elements)
    elif element_type == 13:
        value = int.from_bytes(data_slice, byteorder='big', signed=False)
        return "true" if value == 1 else "false"

    elif element_type == 19:
        return data_slice.decode("utf-8")
    elif element_type == 1024:
        return data_slice

    raise KeyError(f'Parsing not implemented, element type {element_type}')


def parse_hid(filename: PathLike) -> Optional[Mapping[str, Optional[ElementValue]]]:
    """
    Reads a HID-file

    :param filename: filename of HID file
    :returns: parsed HID data
    """
    try:
        with open(filename, "rb") as f:
            rawdata = f.read()

        hid_header = parse_hid_header(rawdata)
        hid_directory = parse_hid_directory(rawdata, hid_header)
        hid_data = parse_hid_data(rawdata, hid_header, hid_directory)
    except PermissionError:
        LOGGER.debug(f'Permission denied for file {filename}')
        return None
    return hid_data


def get_peak_data(hid_file: PathLike) -> Optional[np.ndarray]:
    """
    Retrieve peak data from HID file. The data per dye
    can be stored in different columns (e.g. the first dye can be stored
    in DATA_9 or in DATA_1). The data of the size standard can
    be stored in either DATA_205 or DATA_105.

    :param hid_file: path to hid file
    :returns: RFU for each dye in a numpy array, or None if problems occurd with reading
    """
    try:
        data = parse_hid(hid_file)
        data_colnames = [name for name in data if name.startswith("DATA")]
    except TypeError:
        LOGGER.debug(f'Could not parse hid file {hid_file}')
        return None

    try:
        if 'DATA_9' in data_colnames:
            dye1 = data['DATA_9']
            dye2 = data['DATA_10']
            dye3 = data['DATA_11']
            dye4 = data['DATA_12']
            dye5 = data['DATA_206']
            size_standard = data['DATA_205']
        else:  # if 'DATA_1' in data_colnames:
            dye1 = data['DATA_1']
            dye2 = data['DATA_2']
            dye3 = data['DATA_3']
            dye4 = data['DATA_4']
            dye5 = data['DATA_106']

            if 'DATA_205' in data_colnames:
                size_standard = data['DATA_205']
            else:
                size_standard = data['DATA_105']
        dyes = [dye1, dye2, dye3, dye4, dye5, size_standard]
    except KeyError:
        LOGGER.debug(f'Could not find valid combination of DATA elements for {hid_file}, '
                     f'found {data_colnames}')
        return None

    dyes = np.array(dyes, dtype=np.int16)
    return dyes


def _is_valid_sup_19(position: int, raw_data, sup_19_name: str) -> bool:
    """
    Checks if the raw data is a valid sup_19

    :param position: index to retrieve bytes from
    :param raw_data: raw data from HID file
    :param sup_19_name: name of sup_19
    :returns: if it is a valid sup_19
    """
    p = position
    return \
        (_read_signed_integer(raw_data[(p + 8):(p + 10)]) <= 19
         or _read_signed_integer(raw_data[(p + 8):(p + 10)]) == 1024) \
        and _read_string(raw_data[p:(p + 4)]) == sup_19_name
