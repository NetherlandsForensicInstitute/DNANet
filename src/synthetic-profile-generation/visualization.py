import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


DYE_LABELS = ['Blue', 'Green', 'Black', 'Red', 'Purple', 'Orange']
DYE_COLORS = ['blue', 'green', 'black', 'red', 'purple', 'orange']


def _squeeze_epg(epg: np.ndarray) -> np.ndarray:
    arr = np.squeeze(epg)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def _plot_epg_pair(
    epg_without: np.ndarray, epg_with: np.ndarray, title: str = 'EPG comparison'
) -> None:
    arr_without = _squeeze_epg(epg_without)
    arr_with = _squeeze_epg(epg_with)
    n_channels = min(arr_without.shape[0], arr_with.shape[0], len(DYE_LABELS))

    fig, axes = plt.subplots(n_channels, 2, figsize=(14, 2.8 * n_channels), sharex='col')
    if n_channels == 1:
        axes = np.array([axes])  # ensure 2D indexing

    for i in range(n_channels):
        axes[i, 0].plot(arr_without[i], color=DYE_COLORS[i])
        axes[i, 0].set_ylabel(DYE_LABELS[i], rotation=0, labelpad=30, fontsize=10)
        axes[i, 0].set_title('Without generator', fontsize=11)
        axes[i, 0].grid(True)
        axes[i, 0].set_xlim(0, arr_without[i].shape[0])

        axes[i, 1].plot(arr_with[i], color=DYE_COLORS[i])
        axes[i, 1].set_title('With generator', fontsize=11)
        axes[i, 1].grid(True)
        axes[i, 1].set_xlim(0, arr_with[i].shape[0])

    axes[-1, 0].set_xlabel('Size bins / Data Points')
    axes[-1, 1].set_xlabel('Size bins / Data Points')
    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def _list_files(epg_dir: Path, limit: int = 10) -> None:
    epg_files = sorted((epg_dir / 'epgs').glob('*.npy'))
    print(f'Listing first {min(limit, len(epg_files))} files in {epg_dir / "epgs"}:')
    for f in epg_files[:limit]:
        print(f'  {f.name}')
    if not epg_files:
        print('  (no .npy files found)')


def main(args: argparse.Namespace) -> None:
    """Main runner for the visualization program."""
    base_dir = Path(__file__).resolve().parent
    with_dir = base_dir / 'generated_epgs' / 'with_generator' / 'epgs'
    without_dir = base_dir / 'generated_epgs' / 'without_generator' / 'epgs'

    _list_files(with_dir)

    file_name = args.file
    if file_name is None:
        with_files = sorted(with_dir.glob('*.npy'))
        if not with_files:
            raise FileNotFoundError(f'No .npy files found in {with_dir}; provide --file explicitly.')
        file_name = with_files[0].name
        print(f'No --file provided; using first file found: {file_name}')

    epg_with_path = with_dir / file_name
    epg_without_path = without_dir / file_name
    if not epg_with_path.exists():
        raise FileNotFoundError(f'With-generator file not found: {epg_with_path}')
    if not epg_without_path.exists():
        raise FileNotFoundError(f'Without-generator file not found: {epg_without_path}')

    epg_with = np.load(epg_with_path)
    epg_without = np.load(epg_without_path)

    _plot_epg_pair(epg_without, epg_with, title=file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize generated vs non-generated EPGs side by side.'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='EPG filename to plot (must exist in both with/without generator epgs dirs).',
    )
    main(parser.parse_args())
