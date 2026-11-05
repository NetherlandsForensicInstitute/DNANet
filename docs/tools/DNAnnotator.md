# Label Tool

The **Label Tool** is an interactive matplotlib-based annotation UI for DNA
electropherogram (EPG) profiles. It allows forensic analysts to visually
annotate peaks with category labels (Allele, Stutter, PullUp, etc.) and
persists annotations to CSV files.

## Quick Start

```bash
pip install -e ".[tools]"

# Annotate NFI R&D profiles (creates annotations.csv if it does not exist)
dnanet-label tools.user=Alice tools.filepath=annotations.csv

# Annotate another dataset
dnanet-label data=provedit tools.user=Alice tools.filepath=annotations.csv

# Compare annotations from multiple annotators (folder of CSVs, read-only)
dnanet-label tools.filepath=annotations/ tools.compare=true
```

## Calling the tool

`dnanet-label` is a [Hydra](https://hydra.cc) application. Arguments are
`key=value` overrides on the master config plus `conf/tools/labeltool.yaml`,
not `--flags`.

| Override | Default | Description |
|----------|---------|-------------|
| `data=<name>` | `dnanet_rd` | Dataset config from `conf/data/` whose profiles are shown. The dataset is downloaded first if missing (see [Installation](../guides/installation.md)) |
| `data.dataset.limit=<n>` | `null` | Only load the first `n` profiles |
| `tools.user=<name>` | `Computer` | Annotator name stored in the CSV and used to filter existing annotations |
| `tools.filepath=<path>` | `annotations.csv` | CSV file to read and write. In compare mode, a folder of CSVs |
| `tools.compare=true` | `false` | Non-interactive view with all annotators stacked per dye channel |
| `tools.params.min_rfu_peak_detection=<rfu>` | `200` | RFU threshold for automatic peak pre-marking; `null` disables it |
| `tools.params.plot_allele_bins=<bool>` | `true` | Draw allele bin positions under each dye channel |

Notes:

- Profiles open one after another. Closing a window (or pressing **q**)
  saves the annotations of that profile; **q** also stops the session.
- Annotations are saved per `user` + profile. Re-annotating a profile
  replaces your previous entries only, other annotators are untouched.
- Profiles that already have annotations from `tools.user` are reloaded
  instead of auto-detected, so a session can be resumed.
- A folder as `tools.filepath` is only valid with `tools.compare=true`.
- Interactive mode uses the `TkAgg` matplotlib backend, so Python needs
  Tk (`tkinter`) support. Compare mode does not.

## Controls

| Action | Control |
|--------|---------|
| Draw new span | **Ctrl + Left-click drag** |
| Delete span | Hover + **Delete** key |
| Change category of span | Hover + **category key** (0-9, f, o) |
| Select category for new spans | Click **radio button** panel or press category key |
| Zoom | Standard matplotlib zoom (toolbar) |
| Save and quit | **q** |
| Save and next profile | Close the window |

## Category Keys

| Key | Category | Color |
|-----|----------|-------|
| 0 | Unlabeled | Gray |
| 1 | Allele | Green |
| 2 | Stutter | Yellow |
| 3 | PullUp | Blue |
| 4 | BleedThrough | Red |
| 5 | Spike | Cyan |
| 6 | DyeBlob | Purple |
| 7 | Artefact | Pink |
| 8 | Unclear | Orange |
| 9 | Shoulder | Olive |
| f | ForeignDNA | Brown |
| o | OverloadingArtefact | Lime |

## Architecture

```
tools/labeltool/
├── __init__.py          # Package exports + main() entry point
├── annotations.py       # AnnotationStore (CSV Repository pattern)
├── cli.py               # CLI argument parsing + orchestration
├── interactivity.py     # Abstract base class for interactive plots
├── tool.py              # LabelTool class (event handlers + UI)
└── visualization.py     # Profile plotting with span management
```

### AnnotationStore

The `AnnotationStore` class implements the **Repository pattern** for
annotation persistence:

```python
from dnanet.tools.labeltool.annotations import AnnotationStore

store = AnnotationStore('annotations.csv')

# Read all annotations grouped by profile
spans = store.load_spans_by_profile(user='Alice')

# Save annotations (replaces previous entries for same user+profile)
store.save_spans('sample1', spans_list, user='Alice', dye_names=dye_names)
```

### CSV Format

| Column | Description |
|--------|-------------|
| `user` | Annotator identifier |
| `date` | Timestamp |
| `profile` | Sample/profile name |
| `dye` | Dye channel color name |
| `x0` | Start scan-point index |
| `x1` | End scan-point index |
| `peak_idx` | Peak apex index (or -1) |
| `category` | Label category name |
| `version` | Annotation format version |

### LabelTool

The `LabelTool` class extends `Interactivity` (Template Method pattern)
and uses matplotlib's event system (Observer pattern) for interactive
behavior:

- **`_on_press`**: Starts span drawing on Ctrl+Left-click
- **`_on_motion`**: Updates drag preview and hover detection
- **`_on_release`**: Finalizes the new span
- **`_on_key_press`**: Handles category shortcuts and deletion
- **`_on_close`**: Auto-saves to CSV via AnnotationStore

## Scan-to-Base-Pair Conversion

The tool converts between scan-point indices and base-pair values using
a linear mapping:

```python
from dnanet.tools.labeltool.tool import scan_to_bp, bp_to_scan

bp = scan_to_bp(2048)  # ≈ 270 bp (midpoint)
scan = bp_to_scan(200)  # ≈ 1349 scan points
```

## Automatic Peak Detection

When no previous annotations exist, the tool automatically detects peaks
above a configurable RFU threshold and creates initial unlabeled spans:

```python
from dnanet.tools.labeltool.visualization import get_peaks

peaks = get_peaks(signal_array, min_rfu=200)
# Returns: [[dye0_peaks], [dye1_peaks], ...]
# Each peak: (start_idx, end_idx, peak_idx)
```

## Compare Mode

Compare mode shows multiple annotators' annotations stacked vertically
within each dye channel, allowing visual comparison without interaction:

```bash
dnanet-label data=dnanet_rd tools.filepath=annotations_folder/ tools.compare=true
```

In compare mode, each annotator's spans are rendered as colored rectangles
stacked within the dye channel, with the annotator name shown in the title.
