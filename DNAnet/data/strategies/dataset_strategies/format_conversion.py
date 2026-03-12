import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

LOGGER = logging.getLogger("dnanet")


def find_genotype_file(dataset_root: str) -> Optional[Path]:
    """
    Find the genotype Excel file in the provided dataset root directory.

    Looks for a file directly under the dataset root whose name ends with
    'genotypes.xlsx' (case-insensitive).

    :param dataset_root: Path to the dataset root directory.
    :return: Path to the genotype file if found, otherwise None.
    :raises FileNotFoundError: If the dataset_root is not a directory.
    """
    root = Path(dataset_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset root '{dataset_root}' does not exist or is not a directory."
        )

    return next(
        (
            p
            for p in root.iterdir()
            if p.is_file() and p.name.lower().endswith("genotypes.xlsx")
        ),
        None,
    )


def individualize_genotypes(
    input_path: str,
    output_dir: str,
    exclude_columns: list = ["Sample ID", "Research ID"],
):
    """
    Convert genotype data from a file (Excel or CSV) into individual sample files.
    Each sample will be saved as a separate CSV file with alleles split into two columns.

    :param input_path: Path to the input file (Excel or CSV).
    :param output_dir: Directory where the output files will be saved (created if missing).
    :param exclude_columns: Columns to exclude from the genotype data. First entry is treated
        as the sample ID column. Defaults to ['Sample ID', 'Research ID'].
    :raises ValueError: If the file extension is not supported.
    """

    ext = os.path.splitext(input_path)[1].lower()
    if ext in (".xls", ".xlsx", ".xlsm", ".xlsb"):
        df = pd.read_excel(input_path)
    elif ext == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError("Unsupported file type. Expected an Excel or CSV file.")

    os.makedirs(output_dir, exist_ok=True)
    if any(Path(output_dir).iterdir()):
        LOGGER.warning(
            "Output directory '%s' is not empty. Skipping conversion to avoid overwriting.",
            output_dir,
        )
        return

    marker_columns = [col for col in df.columns if col not in exclude_columns]

    for _, row in df.iterrows():
        sample_name = row[exclude_columns[0]]
        data = []
        for marker in marker_columns:
            if pd.isna(row[marker]) or row[marker] == "":
                continue
            alleles = str(row[marker]).split(",")
            allele1 = alleles[0] if len(alleles) > 0 else ""
            allele2 = alleles[1] if len(alleles) > 1 else ""
            data.append([sample_name, marker, allele1, allele2])
        new_df = pd.DataFrame(
            data, columns=[exclude_columns[0], "Marker", "Allele1", "Allele2"]
        )
        out_path = os.path.join(output_dir, f"{sample_name}.csv")
        new_df.to_csv(out_path, sep=";", index=False)

    LOGGER.info("Individualized genotype files saved to '%s'.", output_dir)
    return
