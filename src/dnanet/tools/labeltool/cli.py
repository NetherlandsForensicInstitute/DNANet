"""CLI entry point for the interactive label tool.

Usage::

    dnanet-label -u Alice -f annotations.csv -d dnanet_rd
    dnanet-label -u Alice -f annotations/ -d dnanet_rd --compare
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from pathlib import Path
from functools import partial

import matplotlib
from hydra import compose, initialize_config_dir
from loguru import logger
from hydra.utils import instantiate

from dnanet.tools.labeltool.tool import LabelTool
from dnanet.tools.labeltool.annotations import AnnotationStore
from dnanet.tools.labeltool.visualization import plot_profile_interactive


if TYPE_CHECKING:
    from omegaconf import DictConfig

    from dnanet.data import HIDDataset


def main() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(WORKSPACE_FOLDER / "conf")):
        cfg = compose(
            config_name="config",
            overrides=[
                "+tools=labeltool",
                *sys.argv[1:]

            ],
        )
    """Entry point for ``dnanet-label`` console script."""
    # instantiate dataset
    data_cfg = cfg.get('data')
    dataset: HIDDataset = instantiate(data_cfg.dataset)
    dataset._transform = None

    labeltool_cfs = cfg.get('tools')

    # Use TkAgg backend for interactive mode
    if not labeltool_cfs.compare:
        matplotlib.use("TkAgg")

    # Suppress default matplotlib key bindings that interfere
    matplotlib.rcParams["keymap.quit"] = []
    matplotlib.rcParams["keymap.zoom"] = []
    matplotlib.rcParams["keymap.fullscreen"] = []

    params = labeltool_cfs.params if "params" in labeltool_cfs else {}

    load_label_tool(
        user=labeltool_cfs.user,
        label_file_path=labeltool_cfs.filepath,
        profile_data=dataset,
        compare_mode=labeltool_cfs.compare,
        params=params,
    )


WORKSPACE_FOLDER = Path(__file__).parents[4]

def load_label_tool(
    user: str,
    label_file_path: str,
    profile_data: HIDDataset,
    params: DictConfig,
    compare_mode: bool = False,

) -> None:
    """Load data and launch the label tool or compare view.

    Args:
        user: Annotator identifier.
        label_file_path: Path to CSV file or folder.
        profile_data: List of images in HID format
        compare_mode: If True, show all annotators non-interactively.
        params: Additional label tool parameters
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
        spans_by_profile=entries_by_profile,
        **params,
    )


if __name__ == "__main__":
    main()
