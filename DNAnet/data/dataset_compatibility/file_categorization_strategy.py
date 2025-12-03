from typing import Protocol
import re
from typing import Literal
from typing import List, Dict
from collections import defaultdict
from pathlib import Path


FileCategory = Literal["sample", "ladder", "control", "unknown"]


class FileCategorizationStrategy(Protocol):
    def __call__(self, file_name: str) -> FileCategory : ...

    def extract_contributor_ids(self, file_name: str) -> list[int]:
        """
        Extracts contributor IDs from the file name.
        Returns a list of integers (IDs), or an empty list if not found.
        """
        ...  # Default implementation can be provided if needed


class NFIFileCategorizer(FileCategorizationStrategy):
    def __call__(self, file_name: str) -> FileCategory:
        OTHER_KITS = ("ppy23", "minifiler", "hdplex")

        fname = file_name.lower()
        # Ladder files
        if "ladder" in fname and not any(kit in fname for kit in OTHER_KITS):
            return "ladder"
        # Controls/blanks (implement your own logic or import from utils)
        if (
            "blanco" in fname
            or "ladder" in fname
            or "pocon" in fname
            or "controle" in fname
            or fname.startswith('a')
        ):
            return "control"
        # Valid sample HID file (using is_rd_hid_filename logic)
        if len(re.findall(r'\d[ABCDEF]\d', file_name[:3])) > 0:
            return "sample"
        # Unknown or unhandled
        return "unknown"


class ProvedItFileCategorizer(FileCategorizationStrategy):
    def __call__(self, file_name: str) -> FileCategory:
        fname = file_name
        # Ladder files
        if "Ladder" in fname:
            return "ladder"
        # Controls/blanks (implement your own logic or import from utils)
        if "LEA" in fname:
            return "control"
        # Valid sample HID file
        if len(self.extract_contributor_ids(fname)) > 0:
            return "sample"
        # Unknown or unhandled
        return "unknown"
    

    # TODO: CHECK WHY IT IS STATIC METHOD
    @staticmethod
    def extract_contributor_ids(file_name: str):
        """
        Extracts all contributor IDs from a ProvedIt filename.
        Returns a list of integers (IDs), or an empty list if not found.
        """
        # Example: F07_RD14-0003-30_31_32_33_34-1;1;1;1;1-M3e-0.075GF-Q2.0_06.5sec.hid
        match = re.search(r"RD14-0003-([\d_]+)-", file_name)
        if match:
            ids = match.group(1).split("_")
            return [int(i) for i in ids if i.isdigit()]
        return []
    

class SyntheticFileCategorizer(FileCategorizationStrategy):
    def __call__(self, file_name: str) -> FileCategory:
        return "sample"

    def extract_contributor_ids(self, file_name: str) -> List[int]:
        raise NotImplementedError

    


def categorize_files(
    files: List, 
    strategy: FileCategorizationStrategy
) -> Dict[FileCategory, List[Path]]:
    """
    Categorizes a list of files based on the provided categorization strategy.
    Args:
        files (List): List of file objects to categorize.
        strategy (FileCategorizationStrategy): Strategy to use for categorization.
    Returns:
        Dict[FileCategory, List]: Dictionary mapping categories to lists of files.
    """
    categorized = defaultdict(list)
    for f in files:
        category = strategy(f.name)
        categorized[category].append(f)
    return dict(categorized)