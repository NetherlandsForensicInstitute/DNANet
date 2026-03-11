from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Get the parent directory of the script folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator

import csv
from matplotlib.collections import PolyCollection
from matplotlib.widgets import RadioButtons, MultiCursor
import matplotlib as mpl

from DNAnet.evaluation.interactivity import Interactivity
from DNAnet.evaluation.visualizations import DNA_CHANNELS, bp_to_scan, plot_profile, scan_to_bp
from config_io import load_dataset
from DNAnet.constants import LABELTOOL_VERSION, LABEL_CATEGORIES

# Use Tk backend (required if you rely on get_tk_widget focus behavior)
matplotlib.use("TkAgg")

# turn these off as they interfere
matplotlib.rcParams['keymap.quit'] = []
matplotlib.rcParams['keymap.zoom'] = []
matplotlib.rcParams['keymap.fullscreen'] = []


class LabelTool(Interactivity):
    def __init__(
            self, figure: Figure, axes: Sequence[Axes],
            spans: Optional[List[Dict[str, object]]] = None,
            label_file_path: str = None,
            user: str = None
    ) -> None:
        super().__init__(figure, axes, spans)
        self.cursor: Optional[MultiCursor] = None

        # Drag state for new span creation
        self.ctrl_press: bool = False
        self.start_x: Optional[float] = None
        self.temp_poly: Optional[PolyCollection] = None  # temporary axvspan during drag

        # Hover state
        self.hover_span: Optional[Dict[str, object]] = None
        self._hover_edge_on: bool = False

        # Categories & current selection
        self.categories = LABEL_CATEGORIES
        self.current_category: str = list(self.categories)[0]

        # Selection state for keyboard navigation
        self.active_span = None  # dict like your spans

        # line styles
        self._ACTIVE_EDGE = dict(color="blue", linewidth=2)
        self._HOVER_EDGE = dict(color="black", linewidth=1.5)

        # Only connect on one axes (they all share x), or connect to all
        self.axs[-1].callbacks.connect('xlim_changed', self.on_xlim_changed)

        self.label_file_path = label_file_path
        self.user = user

    def on_xlim_changed(self, ax):
        self.update_xticks()

    def update_xticks(self):
        # Use x-limits from shared axis (any will do)
        scan_min, scan_max = self.axs[-1].get_xlim()
        bp_min = scan_to_bp(scan_min)
        bp_max = scan_to_bp(scan_max)
        bp_range = bp_max - bp_min

        # Pick a good step size for bp ticks
        ideal_num_ticks = 8
        rough_step = bp_range / ideal_num_ticks

        def nice_step(val):
            steps = [1, 2, 5, 10, 25, 50, 100, 250, 500]
            return min(steps, key=lambda s: abs(s - val))

        step = nice_step(rough_step)
        bp_start = step * np.floor(bp_min / step)
        bp_ticks = np.arange(bp_start, bp_max + step, step)
        scan_ticks = bp_to_scan(bp_ticks)

        self.axs[-1].xaxis.set_major_locator(FixedLocator(scan_ticks))

    def activate_interactivity(self):
        mpl.rcParams['keymap.back'] = ['alt+left']
        mpl.rcParams['keymap.forward'] = ['alt+right']

        self.category_radio_buttons()

        # Event wiring
        self.figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.figure.canvas.mpl_connect("button_release_event", self.on_release)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.figure.canvas.mpl_connect("close_event", self.replace_annotations_csv)

        # Crosshair across axes #AnnotatedCursor
        self.cursor = MultiCursor(
            self.figure.canvas, self.axs, color="k", lw=0.5, horizOn=True, vertOn=True
        )

    # --------------------------- UI SETUP ------------------------ #

    def category_radio_buttons(self):
        labels = list(self.categories.keys())
        # Initial generous box; you can shrink this if desired
        plt.subplots_adjust(left=0.25)  # Leave room for the radio panel
        self.radio_ax = plt.axes([0.02, 0.75, 0.15, 0.2])
        self.radio = RadioButtons(self.radio_ax, labels)
        # Select default "Unlabeled"
        self.radio.set_active(labels.index(self.current_category))
        self.radio.activecolor = self.categories[self.current_category][
            "color"]  # type: ignore[index]
        self.radio.on_clicked(self.select_category)
        self._focus_canvas()

    # ------------------------------ Event hooks ---------------------------- #

    def on_press(self, event) -> None:
        """Mouse press: start drawing (Ctrl+Left) or remove span (Right)."""
        self._focus_canvas()
        if event.inaxes not in self.axs:
            return
        axis: Axes = event.inaxes  # type: ignore[assignment]

        # Right-click: remove span under cursor (if any)
        if event.button == 3:
            print('Right clicked')
            return

        # Ctrl + Left: begin drawing a new span (as PolyCollection)
        ctrl_down = ('control' in getattr(event, 'modifiers', set())) or (
                event.key == 'control')
        if event.button == 1 and ctrl_down:
            self.ctrl_press = True
            self.start_x = event.xdata
            # preview with current category color but lighter alpha
            props = self.categories[self.current_category]
            self.temp_poly = axis.axvspan(self.start_x, self.start_x,  # zero width initially
                                          color=props['color'], alpha=0.2)
            self.figure.canvas.draw_idle()

    def on_motion(self, event) -> None:
        """Mouse motion: update temp rect during drag; track hover span otherwise."""
        # If we're dragging a new span, update its preview width
        if (
                self.ctrl_press
                and self.start_x is not None
                and self.temp_poly is not None
                and event.inaxes in self.axs
        ):
            ax = event.inaxes
            end_x = event.xdata
            # rebuild the temp axvspan with new bounds
            self.temp_poly.remove()
            props = self.categories[self.current_category]
            self.temp_poly = ax.axvspan(self.start_x, end_x, color=props['color'], alpha=0.2)
            self.figure.canvas.draw_idle()
            return

        # Hover detection: only if cursor is over one of our axes
        if event.inaxes not in self.axs:
            # If we left the axes, clear any hover outline
            if self.hover_span is not None and self._hover_edge_on:
                artist: PolyCollection = self.hover_span["artist"]  # type: ignore[assignment]
                artist.set_edgecolor(None)
                artist.set_linewidth(0)
                self._hover_edge_on = False
                self.figure.canvas.draw_idle()
            self.hover_span = None
            return

        # Determine which (if any) span is under the cursor; last wins if overlapping
        hovered: Optional[Dict[str, object]] = None
        for s in self.spans:
            artist: PolyCollection = s["artist"]  # type: ignore[assignment]
            hit, _ = artist.contains(event)
            if hit and s["ax"] is event.inaxes:
                hovered = s

        # Update hover outline only if changed
        if hovered is not self.hover_span:
            if self.hover_span is not None and self._hover_edge_on:
                prev_artist: PolyCollection = self.hover_span["artist"]  # type: ignore[assignment]
                prev_artist.set_edgecolor(None)
                prev_artist.set_linewidth(0)
                self._hover_edge_on = False

            self.hover_span = hovered

            if self.hover_span is not None:
                new_artist: PolyCollection = self.hover_span["artist"]  # type: ignore[assignment]
                new_artist.set_edgecolor("black")
                new_artist.set_linewidth(1.5)
                self._hover_edge_on = True

            self.figure.canvas.draw_idle()

    def on_release(self, event):
        if not self.ctrl_press or self.start_x is None or event.inaxes not in self.axs:
            return

        ax = event.inaxes
        self.ctrl_press = False

        x0, x1 = sorted([self.start_x, event.xdata])
        props = self.categories[self.current_category]

        if self.temp_poly is not None:
            # Rebuild once more to be sure bounds are final, then set final alpha
            self.temp_poly.remove()
            artist = ax.axvspan(x0, x1, color=props['color'], alpha=props['alpha'])
            span_dict = {'artist': artist, 'ax': ax, 'category': props.get('pyval', None),
                         'x0': x0, 'x1': x1, "peak_idx": -1}
            self.spans.append(span_dict)
            self._set_active_span(span_dict)  # optional: make it active
            self.temp_poly = None

        self.start_x = None
        self.figure.canvas.draw_idle()

    def on_key_press(self, event) -> None:
        print(event.key)
        # if we pressed delete, remove the span
        if event.key == 'delete':
            if self.hover_span is not None:
                artist: PolyCollection = self.hover_span["artist"]  # type: ignore[assignment]
                artist.remove()
                self.spans.remove(self.hover_span)
                self.hover_span = None
                self._hover_edge_on = False
                self.figure.canvas.draw_idle()
                print(f"Removed span.")

        # Keyboard: 0/1/2/3/4 set the category of the *hovered* span.
        for name, props in self.categories.items():
            if props["index"] == event.key:  # type: ignore[index]
                # Update current category (UI feedback & for next new span)
                self.current_category = name
                self.radio.set_active(list(self.categories.keys()).index(name))
                self.radio.activecolor = props["color"]  # type: ignore[index]
                print(f"[Key {event.key}] Selected category: {name}")

                # If hovering a span, recolor it immediately
                if self.hover_span is not None:
                    artist: PolyCollection = self.hover_span["artist"]  # type: ignore[assignment]
                    artist.set_color(props["color"])  # type: ignore[index]
                    artist.set_alpha(props["alpha"])  # type: ignore[index]
                    self.hover_span["category"] = props.get("pyval", None)
                    self.figure.canvas.draw_idle()
                    print("Updated hovered span category.")
                return

    def replace_annotations_csv(self, event) -> None:
        """Save spans in tabular (CSV) format, replacing old ones for the same profile and user."""

        # Get current context
        profile_name = self.figure.get_suptitle().strip()
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Prepare new rows to write
        new_rows = []
        for span in self.spans:
            dye_idx=np.where(self.axs == span["ax"])[0][0]
            dye_name = DNA_CHANNELS[dye_idx]
            new_rows.append([
                self.user, date, profile_name, dye_name,
                int(span['x0']),
                int(span['x1']),
                int(span['peak_idx']),
                span["category"],
                LABELTOOL_VERSION
            ])

        # File path
        label_file = Path(self.label_file_path)

        # Header for the CSV
        header = ["user", "date", "profile", "dye", "x0", "x1", "peak_idx", "category", "version"]

        # Read existing entries (if file exists), filter out matching profile+user
        existing_rows = []

        removed = 0
        if label_file.exists():
            with open(label_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not (row['profile'] == profile_name and row['user'] == self.user):
                        # Keep this row
                        existing_rows.append([
                            row['user'], row['date'], row['profile'], row['dye'],
                            row['x0'], row['x1'], row['peak_idx'], row['category'],
                            row['version'] if 'version' in row else '0.1'
                        ])
                    else:
                        removed += 1

        # Combine filtered old rows with new rows
        all_rows = existing_rows + new_rows

        # Write everything back to the file (overwrite)
        with open(label_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(all_rows)

        print(
            f"Saved {len(new_rows)} annotations to {self.label_file_path}, removed {removed} old "
            f"ones.")

    # --------------------------- Utility methods ---------------------------- #

    def _focus_canvas(self) -> None:
        """Try to give keyboard focus back to the canvas (backend-safe)."""
        # TkAgg: canvas has get_tk_widget(); others may expose manager.window
        canvas = self.figure.canvas
        try:
            if hasattr(canvas, "get_tk_widget"):
                canvas.get_tk_widget().focus_set()  # type: ignore[attr-defined]
            elif getattr(canvas, "manager", None) and hasattr(canvas.manager, "window"):
                # Fallback for some GUI backends (Qt/others)
                try:
                    canvas.manager.window.focus_force()  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            pass  # Focusing is best-effort only

    def set_span_category(self, span_dict: Dict[str, object], category_name: str) -> None:
        """
        Apply category color/alpha to a span and store its semantic value.

        Parameters
        ----------
        span_dict : dict
            Span record containing 'artist', 'ax', and 'category'.
        category_name : str
            One of the keys in self.categories.
        """
        props = self.categories[category_name]
        artist: PolyCollection = span_dict["artist"]  # type: ignore[assignment]
        artist.set_color(props["color"])  # type: ignore[index]
        artist.set_alpha(props["alpha"])  # type: ignore[index]
        span_dict["category"] = props["pyval"]  # type: ignore[index]
        self.figure.canvas.draw_idle()

    def select_category(self, label: str) -> None:
        """
        RadioButtons callback: update current category used for new spans.

        Also re-focus the figure canvas so numeric key presses work immediately.
        """
        self.current_category = label
        self.radio.activecolor = self.categories[self.current_category][
            "color"]  # type: ignore[index]
        print(f"Selected category: {self.current_category}")
        self._focus_canvas()

    @staticmethod
    def _clear_edge(span_dict):
        """Remove any outline on a span."""
        artist = span_dict["artist"]
        artist.set_edgecolor(None)
        artist.set_linewidth(0)

    def _set_active_span(self, span_dict):
        """Visually mark active span, clear previous active."""
        # clear previous active
        if self.active_span is not None and self.active_span is not span_dict:
            self._clear_edge(self.active_span)
        # also clear hover outline to avoid double styling
        if self.hover_span is not None and self.hover_span is not span_dict:
            self._clear_edge(self.hover_span)

        self.active_span = span_dict
        self.hover_span = span_dict  # so number keys recolor this one
        a = span_dict["artist"]
        a.set_edgecolor(self._ACTIVE_EDGE["color"])
        a.set_linewidth(self._ACTIVE_EDGE["linewidth"])
        if self.figure:
            self.figure.canvas.draw_idle()


def load_label_tool(user: str,
                    label_file_or_folder_path: str,
                    data_config: str,
                    compare_mode: bool = False) -> None:
    """
    Runs a label tool to create scan point annotations for epgs and compare annotations.

    Args:
        user: The identifier for the current annotator. Ignored when comparing
        label_file_or_folder_path: the path of the csv file or folder containing the csv files.
        compare_mode: when true, loads all annotations from all users in the given file or folder,
        and shows them stacked in non-interactive mode

    """
    if compare_mode:
        print("Running the tool to compare annotations, non-interactively!")
    # create file with header if it does not yet exist. Otherwise, read it in.
    label_file = Path(label_file_or_folder_path)

    # Header for new file
    header = ["user", "date", "profile", "dye", "x0", "x1", "peak_idx", "category"]
    entries_by_profile = None

    existing_entries = []  # no data yet
    csv_files = []
    if not label_file.exists():
        print('No annotations found, creating a new file')
        # create a file
        with open(label_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    elif label_file.suffix == '.csv':
        print('Reading annotations from file')
        csv_files = [label_file]
    elif label_file.is_dir():
        print('Reading from directory')
        if not compare_mode:
            raise ValueError('if input is a folder, we expect compare mode!')
        csv_files = label_file.rglob('*.csv')
    else:
        raise ValueError(f'Cannot parse label file {label_file}')

    if csv_files:
        for csv_file in csv_files:
            with open(csv_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                existing_entries += list(reader)  # list of dicts, each row is a dict
        # map the categories to colours
        label_to_coloralpha = {}
        for name, props in LABEL_CATEGORIES.items():
            label_to_coloralpha[props['pyval']] = [props['color'], props['alpha']]

        # turn these into spans

        # Group by 'profile'
        entries_by_profile = defaultdict(list)
        for row in existing_entries:
            # only consider the annotations you made, unless we're comparing
            if not compare_mode and row['user'] != user:
                continue
            profile = row['profile']

            # Convert fields to appropriate types
            try:
                x0 = int(row['x0'])
                x1 = int(row['x1'])
                peak_idx = int(row['peak_idx'])
            except (ValueError, TypeError):
                x0 = x1 = None  # or raise/log as needed

            obj = {
                'annotator': row['user'],
                'artist': None,
                'ax': None,  # Placeholder
                'dye': row['dye'],
                'category': row['category'],
                'x0': x0,
                'x1': x1,
                'peak_idx': peak_idx,
                # pass the color/alpha from here so artists can be made when axes are known
                'color': label_to_coloralpha[row['category']][0] if row['category'] else 'gray',
                'alpha': label_to_coloralpha[row['category']][1] if row['category'] else 0.2,
            }

            entries_by_profile[profile].append(obj)
    profile_data = load_dataset(data_config)
    # order alphabetically
    profile_data = sorted(profile_data, key=lambda x: x.path.stem.split('/')[-1])
    interactive = partial(LabelTool, label_file_path=label_file_or_folder_path, user=user) \
        if not compare_mode else None
    plot_profile(
        profile_data,
        interactive=interactive,
        plot_allele_bins=True,
        spans_by_profile=entries_by_profile
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tool to annotate EPGs."
    )
    parser.add_argument(
        "-u",
        "--user",
        type=str,
        required=False,
        default="Computer",
        help="The name of the current user. Used to save annotations, and filter existing "
             "annotations when labelling. Parameter is ignored when comparing annotations.",
    )
    parser.add_argument(
        "-f",
        "--filepath",
        type=str,
        required=False,
        default="annotations.csv",
        help="The csv file from and to which to write annotations, or the folder name in which to "
             "find the csv's when comparing.",
    )
    parser.add_argument(
        '-d',
        '--data-config',
        type=str,
        required=True,
        help="The path to a .yaml file (or the basename without an extension "
             "if located in `./config/data/`) containing the configuration "
             "for the dataset to label"
    )
    parser.add_argument(
        "-c",
        "--compare",
        action='store_true',
        default=False,
        help="If true, read in annotations from all users and show them stacked, "
             "in non-interactive mode",
    )
    args = parser.parse_args()
    load_label_tool(user=args.user,
                    label_file_or_folder_path=args.filepath,
                    compare_mode=args.compare,
                    data_config=args.data_config)
