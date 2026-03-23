"""CLI entry point for the interactive label tool.

Usage::

    dnanet-label -u Alice -f annotations.csv -d dnanet_rd
    dnanet-label -u Alice -f annotations/ -d dnanet_rd --compare
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import matplotlib
from loguru import logger

from dnanet.tools.labeltool.annotations import AnnotationStore
from dnanet.tools.labeltool.tool import LabelTool
from dnanet.tools.labeltool.visualization import plot_profile_interactive


def main() -> None:
    """Entry point for ``dnanet-label`` console script."""
    parser = argparse.ArgumentParser(
        description="Interactive EPG annotation tool.",
    )
    parser.add_argument(
        "-u", "--user",
        type=str,
        default="Computer",
        help="Annotator identifier (used for saving/filtering annotations).",
    )
    parser.add_argument(
        "-f", "--filepath",
        type=str,
        default="annotations.csv",
        help="CSV file (or folder of CSVs when comparing) for annotations.",
    )
    parser.add_argument(
        "-d", "--data-config",
        type=str,
        required=True,
        help="Hydra data config name (e.g. 'dnanet_rd', 'provedit').",
    )
    parser.add_argument(
        "-c", "--compare",
        action="store_true",
        default=False,
        help="Compare mode: show all annotators' spans stacked (non-interactive).",
    )
    parser.add_argument(
        "-k", "--kit",
        type=str,
        default=None,
        help="Kit name override (e.g. 'PPF6C', 'GLOBALFILER'). "
             "Auto-detected from data config if not provided.",
    )
    args = parser.parse_args()

    # Use TkAgg backend for interactive mode
    if not args.compare:
        matplotlib.use("TkAgg")

    # Suppress default matplotlib key bindings that interfere
    matplotlib.rcParams["keymap.quit"] = []
    matplotlib.rcParams["keymap.zoom"] = []
    matplotlib.rcParams["keymap.fullscreen"] = []

    load_label_tool(
        user=args.user,
        label_file_path=args.filepath,
        data_config=args.data_config,
        compare_mode=args.compare,
    )


def load_label_tool(
    user: str,
    label_file_path: str,
    data_config: str,
    compare_mode: bool = False,
) -> None:
    """Load data and launch the label tool or compare view.

    Args:
        user: Annotator identifier.
        label_file_path: Path to CSV file or folder.
        data_config: Hydra data config name.
        compare_mode: If True, show all annotators non-interactively.
    """
    if compare_mode:
        logger.info("Running in compare mode (non-interactive)")

    store = AnnotationStore(label_file_path)

    # Ensure the file exists if we're not in compare mode
    if not compare_mode:
        store.ensure_file()

    # Load existing annotations
    label_path = Path(label_file_path)
    if label_path.is_dir() and not compare_mode:
        raise ValueError("Folder input requires --compare mode")

    entries_by_profile = None
    if label_path.exists():
        entries_by_profile = store.load_spans_by_profile(
            user=None if compare_mode else user,
        )

    # Load dataset via Hydra compose API
    profile_data = _load_dataset_from_config_name(data_config)
    profile_data = sorted(profile_data, key=lambda x: x.path.stem.split("/")[-1])

    # Build interactive callback or None for compare mode
    interactive = None
    if not compare_mode:
        interactive = partial(
            LabelTool,
            annotation_store=store,
            user=user,
        )

    plot_profile_interactive(
        profile_data,
        interactive=interactive,
        plot_allele_bins=True,
        spans_by_profile=entries_by_profile,
    )


def _load_dataset_from_config_name(data_config: str):
    """Build a Hydra DictConfig from a data config name and load the dataset.

    Uses Hydra's Compose API to resolve the config file (e.g. ``"dnanet_rd"``
    becomes ``conf/data/dnanet_rd.yaml``), then delegates to
    :func:`~dnanet.data.loading.load_dataset`.

    Args:
        data_config: Name of the data config group (e.g. ``"dnanet_rd"``,
            ``"provedit"``), or a path to a YAML file.
    """
    from hydra import compose, initialize_config_dir

    from dnanet.data.loading import load_dataset

    # Locate the conf/ directory relative to the workspace root.
    # dnanet.cli uses Path(__file__).parents[2] from src/dnanet/cli.py;
    # we replicate that by going up from the dnanet package location.
    import dnanet
    workspace = Path(dnanet.__file__).parents[2]
    conf_dir = str(workspace / "conf")

    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        cfg = compose(config_name="config", overrides=[f"data={data_config}"])

    return load_dataset(cfg.data)


if __name__ == "__main__":
    main()
