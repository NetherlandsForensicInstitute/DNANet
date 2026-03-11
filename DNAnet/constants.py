LABEL_CATEGORIES: dict[str, dict[str, object]] = {
    "Unlabeled (Gray - 0)": {"color": "gray", "alpha": 0.2, "index": "0", "pyval": None},
    "Allele (Green - 1)": {"color": "green", "alpha": 0.4, "index": "1", "pyval": "Allele"},
    "Stutter (Yellow - 2)": {"color": "yellow", "alpha": 0.4, "index": "2", "pyval": "Stutter"},
    "Pull Up (Blue - 3)": {"color": "blue", "alpha": 0.4, "index": "3", "pyval": "PullUp"},
    "Bleed Through (Red - 4)": {"color": "red", "alpha": 0.4, "index": "4",
                                "pyval": "BleedThrough"},
    "Spike (Cyan - 5)": {"color": "cyan", "alpha": 0.4, "index": "5", "pyval": "Spike"},
    "Dye Blob (Purple - 6)": {"color": "purple", "alpha": 0.4, "index": "6", "pyval": "DyeBlob"},
    "Artefact (Pink - 7)": {"color": "pink", "alpha": 0.4, "index": "7", "pyval": "Artefact"},
    "Unclear (Orange - 8)": {"color": "tab:orange", "alpha": 0.4, "index": "8", "pyval": "Unclear"},
    "Shoulder (Olive - 9)": {"color": "tab:olive", "alpha": 0.4, "index": "9", "pyval": "Shoulder"},
    "Foreign DNA (Brown - F)": {"color": "tab:brown", "alpha": 0.4, "index": "f",
                                "pyval": "ForeignDNA"},
    "Overloading artefact (Lime - O)": {"color": "lime", "alpha": 0.4, "index": "o",
                                        "pyval": "OverloadingArtifact"},
}

LABELTOOL_VERSION = "0.4"
