from pathlib import Path
from typing import List


# get all the HID files from the hid_files_path directory
def find_files_by_suffix(root_dir, suffix) -> List[Path]:
    """
    Recursively find all files in root_dir that end with the given suffix.
    """
    return list(Path(root_dir).rglob(f'*{suffix}'))