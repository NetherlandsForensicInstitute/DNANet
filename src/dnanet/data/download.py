"""Download datasets from the Hugging Face Hub when they are missing locally."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path

from loguru import logger


if TYPE_CHECKING:
    from dnanet.core.types import PathLike


def _is_populated(root: Path) -> bool:
    """Return True if ``root`` is an existing directory with at least one entry."""
    return root.is_dir() and any(root.iterdir())


def ensure_dataset(root: PathLike, repo_id: str, revision: str | None = None) -> Path:
    """Make sure the dataset at ``root`` exists, downloading it from the Hub if not.

    The Hub repository is expected to hold the dataset in a top-level folder
    named like ``root`` (e.g. ``2p_5p_Dataset_NFI/``), so the snapshot is
    downloaded into ``root.parent``. Nothing is downloaded when ``root`` already
    contains files.

    Args:
        root: Expected local dataset directory.
        repo_id: Hugging Face dataset repository, e.g. ``org/name``.
        revision: Optional branch, tag or commit to download.

    Returns:
        The dataset root as a ``Path``.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
        FileNotFoundError: If the download did not produce ``root``.
    """
    root = Path(root)
    if _is_populated(root):
        return root

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            f'Dataset not found at {root} and huggingface_hub is not installed. '
            "Install it with `pip install 'dnanet[hub]'` or place the data manually."
        ) from exc

    logger.info(f'Dataset not found at {root}, downloading {repo_id} from Hugging Face')
    kwargs: dict[str, Any] = {
        'repo_id': repo_id,
        'repo_type': 'dataset',
        'revision': revision,
        'local_dir': root.parent,
        'allow_patterns': [f'{root.name}/*'],
    }
    snapshot_download(**kwargs)

    if not _is_populated(root):
        raise FileNotFoundError(
            f'Downloaded {repo_id} but {root} is missing or empty; '
            f'expected a top-level {root.name!r} folder in the repository.'
        )
    return root
