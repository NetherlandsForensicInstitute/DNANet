# Label Tool

The **Label Tool** is an interactive matplotlib-based annotation UI for DNA
electropherogram (EPG) profiles. It allows forensic analysts to visually
annotate peaks with category labels (Allele, Stutter, PullUp, etc.) and
persists annotations to CSV files.

## Quick Start

```bash
# Annotate NFI R&D profiles
dnanet-label -u Alice -f annotations.csv -d dnanet_rd

# Compare annotations from multiple annotators
dnanet-label -f annotations_folder/ -d dnanet_rd --compare
```

## Controls

| Action | Control |
|--------|---------|
| Draw new span | **Ctrl + Left-click drag** |
| Delete span | Hover + **Delete** key |
| Change category | Hover + **number key** (0-9, f, o) |
| Select category | Click **radio button** panel |
| Zoom | Standard matplotlib zoom (toolbar) |
| Save | Automatic on **window close** |

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

store = AnnotationStore("annotations.csv")

# Read all annotations grouped by profile
spans = store.load_spans_by_profile(user="Alice")

# Save annotations (replaces previous entries for same user+profile)
store.save_spans("sample1", spans_list, user="Alice", dye_names=dye_names)
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

bp = scan_to_bp(2048)   # ≈ 270 bp (midpoint)
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
dnanet-label -f annotations_folder/ -d dnanet_rd --compare
```

In compare mode, each annotator's spans are rendered as colored rectangles
stacked within the dye channel, with the annotator name shown in the title.
