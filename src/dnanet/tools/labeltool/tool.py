"""Interactive label tool for DNA electropherogram annotation.

Provides a matplotlib-based interactive UI for annotating peaks in EPG
profiles with category labels (Allele, Stutter, PullUp, etc.).

Controls:
    - **Ctrl+Left-click drag**: Draw a new span in the current category.
    - **Hover**: Highlights the span under the cursor.
    - **Number keys (0-9, f, o)**: Change the category of the hovered span.
    - **Delete**: Remove the hovered span.
    - **q**: Quit the application.
    - **Close window**: Auto-saves annotations to CSV.

Design pattern: **Observer** (matplotlib event system)
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import matplotlib as mpl
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator
from matplotlib.widgets import MultiCursor, RadioButtons
from matplotlib.collections import PolyCollection

from dnanet.core.constants import LabelCategory
from dnanet.tools.labeltool.annotations import AnnotationStore
from dnanet.tools.labeltool.interactivity import Interactivity


# Linear scan-to-bp conversion constants (default for 4096-point profiles)
# TODO: Convert using scaling strategy
SLOPE = (475.0 - 65.0) / 4096.0
OFFSET = 65.0


def scan_to_bp(x: float) -> float:
    """Convert scan-point index to base-pair value."""
    return OFFSET + SLOPE * x


def bp_to_scan(bp: float | np.ndarray) -> float | np.ndarray:
    """Convert base-pair value to scan-point index."""
    return (bp - OFFSET) / SLOPE


class LabelTool(Interactivity):
    """Interactive annotation tool for EPG profiles.

    Inherits from :class:`Interactivity` and wires up matplotlib events
    for span creation, deletion, category assignment, and CSV persistence.

    Args:
        figure: The matplotlib Figure to annotate.
        axes: Sequence of Axes (one per dye channel).
        spans: Optional pre-existing spans.
        annotation_store: Persistence backend for saving annotations.
        user: Annotator identifier.
        dye_names: Ordered dye channel color names.
    """

    def __init__(
        self,
        figure: Figure,
        axes: Sequence[Axes],
        spans: list[dict[str, Any]] | None = None,
        *,
        annotation_store: AnnotationStore | None = None,
        user: str = "Computer",
        dye_names: list[str] | None = None,
    ) -> None:
        super().__init__(figure, axes, spans)
        self.cursor: MultiCursor | None = None
        self.annotation_store = annotation_store
        self.user = user
        self.dye_names = dye_names or [
            "blue", "green", "yellow", "red", "purple", "orange",
        ]

        # Drag state for new span creation
        self.ctrl_press: bool = False
        self.start_x: float | None = None
        self.temp_poly: PolyCollection | None = None

        # Hover state
        self.hover_span: dict[str, Any] | None = None
        self._hover_edge_on: bool = False

        # Categories — keyed by radio-button label (e.g. "Allele (Green - 1)")
        # to match the original LABEL_CATEGORIES dict format.
        self._categories = {cat.radio_label: cat for cat in LabelCategory}
        self.current_category: str = LabelCategory.UNLABELED.radio_label

        # Selection state
        self.active_span: dict[str, Any] | None = None

        # Line styles
        self._ACTIVE_EDGE = {"color": "blue", "linewidth": 2}
        self._HOVER_EDGE = {"color": "black", "linewidth": 1.5}

        # Set when the user presses 'q' — signals the plot loop to stop.
        self.quit_requested: bool = False

        # Dynamic x-tick updates when zooming
        self.axs[-1].callbacks.connect("xlim_changed", self._on_xlim_changed)

    # ------------------------------------------------------------------
    # Interactivity implementation
    # ------------------------------------------------------------------

    def activate_interactivity(self) -> None:
        """Wire up all matplotlib event handlers."""
        mpl.rcParams["keymap.back"] = ["alt+left"]
        mpl.rcParams["keymap.forward"] = ["alt+right"]

        self._create_radio_buttons()

        self.figure.canvas.mpl_connect("button_press_event", self._on_press)
        self.figure.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.figure.canvas.mpl_connect("button_release_event", self._on_release)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.figure.canvas.mpl_connect("close_event", self._on_close)

        # Crosshair cursor
        self.cursor = MultiCursor(
            self.figure.canvas, self.axs, color="k", lw=0.5,
            horizOn=True, vertOn=True,
        )

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _create_radio_buttons(self) -> None:
        """Create the category radio button panel with colored labels."""
        import matplotlib.pyplot as plt

        labels = list(self._categories.keys())
        plt.subplots_adjust(left=0.25)
        self.radio_ax = plt.axes([0.02, 0.65, 0.18, 0.30])
        self.radio = RadioButtons(self.radio_ax, labels)
        self.radio.set_active(labels.index(self.current_category))

        # Color each radio button label with its category color
        for label_text, cat in zip(self.radio.labels, LabelCategory):
            label_text.set_color(cat.color)
            label_text.set_fontsize(8)

        cat = self._categories[self.current_category]
        self.radio.activecolor = cat.color
        self.radio.on_clicked(self._select_category)
        self._focus_canvas()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_press(self, event: Any) -> None:
        """Mouse press: Ctrl+Left starts drawing, Right logs click."""
        self._focus_canvas()
        if event.inaxes not in self.axs:
            return

        # Right-click: reset hovered span to Unlabeled
        if event.button == 3:
            if self.hover_span is not None:
                unlabeled = LabelCategory.UNLABELED
                artist = self.hover_span["artist"]
                artist.set_color(unlabeled.color)
                artist.set_alpha(unlabeled.alpha)
                self.hover_span["category"] = None
                self.figure.canvas.draw_idle()
                logger.debug("Unlabeled span via right-click")
            return

        # Ctrl + Left: begin new span
        modifiers = getattr(event, "modifiers", None) or set()
        ctrl_down = "control" in modifiers or getattr(event, "key", None) == "control"
        if event.button == 1 and ctrl_down:
            self.ctrl_press = True
            self.start_x = event.xdata
            cat = self._categories[self.current_category]
            self.temp_poly = event.inaxes.axvspan(
                self.start_x, self.start_x,
                color=cat.color, alpha=0.2,
            )
            self.figure.canvas.draw_idle()

    def _on_motion(self, event: Any) -> None:
        """Mouse motion: update drag preview or track hover."""
        # During drag: update preview span
        if (
            self.ctrl_press
            and self.start_x is not None
            and self.temp_poly is not None
            and event.inaxes in self.axs
        ):
            self.temp_poly.remove()
            cat = self._categories[self.current_category]
            self.temp_poly = event.inaxes.axvspan(
                self.start_x, event.xdata,
                color=cat.color, alpha=0.2,
            )
            self.figure.canvas.draw_idle()
            return

        # Hover detection
        if event.inaxes not in self.axs:
            if self.hover_span is not None and self._hover_edge_on:
                artist: PolyCollection = self.hover_span["artist"]
                artist.set_edgecolor(None)
                artist.set_linewidth(0)
                self._hover_edge_on = False
                self.figure.canvas.draw_idle()
            self.hover_span = None
            return

        hovered: dict[str, Any] | None = None
        for s in self.spans:
            artist = s["artist"]
            if artist is not None:
                hit, _ = artist.contains(event)
                if hit and s["ax"] is event.inaxes:
                    hovered = s

        if hovered is not self.hover_span:
            if self.hover_span is not None and self._hover_edge_on:
                prev: PolyCollection = self.hover_span["artist"]
                prev.set_edgecolor(None)
                prev.set_linewidth(0)
                self._hover_edge_on = False

            self.hover_span = hovered

            if self.hover_span is not None:
                new_artist: PolyCollection = self.hover_span["artist"]
                new_artist.set_edgecolor("black")
                new_artist.set_linewidth(1.5)
                self._hover_edge_on = True

            self.figure.canvas.draw_idle()

    def _on_release(self, event: Any) -> None:
        """Mouse release: finalize new span."""
        if not self.ctrl_press or self.start_x is None or event.inaxes not in self.axs:
            return

        self.ctrl_press = False
        x0, x1 = sorted([self.start_x, event.xdata])
        cat = self._categories[self.current_category]

        if self.temp_poly is not None:
            self.temp_poly.remove()
            artist = event.inaxes.axvspan(x0, x1, color=cat.color, alpha=cat.alpha)
            span_dict = {
                "artist": artist,
                "ax": event.inaxes,
                # Store pyval-style category (None for Unlabeled, "Allele" etc.)
                "category": cat.label_name or None,
                "x0": x0,
                "x1": x1,
                "peak_idx": -1,
            }
            self.spans.append(span_dict)
            self._set_active_span(span_dict)
            self.temp_poly = None

        self.start_x = None
        self.figure.canvas.draw_idle()

    def _on_key_press(self, event: Any) -> None:
        """Key press: delete span, change category, or quit."""
        logger.debug("Key: {}", event.key)

        # Quit — save annotations, then close the figure and signal exit.
        if event.key == "q":
            import matplotlib.pyplot as plt
            self.quit_requested = True
            self._on_close(event)  # save annotations before closing
            plt.close(self.figure)
            return

        # Delete hovered span
        if event.key == "delete":
            if self.hover_span is not None:
                artist: PolyCollection = self.hover_span["artist"]
                artist.remove()
                self.spans.remove(self.hover_span)
                self.hover_span = None
                self._hover_edge_on = False
                self.figure.canvas.draw_idle()
                logger.debug("Removed span")
            return

        # Number/letter keys: change category of hovered span or selection
        for name, cat in self._categories.items():
            if cat.shortcut == event.key:
                self.current_category = name
                self.radio.set_active(list(self._categories.keys()).index(name))
                self.radio.activecolor = cat.color
                logger.debug("[Key {}] Selected category: {}", event.key, name)

                if self.hover_span is not None:
                    artist = self.hover_span["artist"]
                    artist.set_color(cat.color)
                    artist.set_alpha(cat.alpha)
                    # Store pyval-style category (None for Unlabeled)
                    self.hover_span["category"] = cat.label_name or None
                    self.figure.canvas.draw_idle()
                    logger.debug("Updated hovered span category")
                return

    def _on_close(self, event: Any) -> None:
        """Window close: save annotations (only once)."""
        if self.annotation_store is None:
            return
        if getattr(self, "_saved", False):
            return
        self._saved = True

        profile_name = self.figure.get_suptitle().strip()
        self.annotation_store.save_spans(
            profile_name,
            self.spans,
            user=self.user,
            dye_names=self.dye_names,
            axes=np.array(self.axs),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _on_xlim_changed(self, ax: Axes) -> None:
        """Dynamically update x-tick labels when zooming."""
        self._update_xticks()

    def _update_xticks(self) -> None:
        """Recompute base-pair tick labels for current view."""
        scan_min, scan_max = self.axs[-1].get_xlim()
        bp_min = scan_to_bp(scan_min)
        bp_max = scan_to_bp(scan_max)
        bp_range = bp_max - bp_min

        ideal_num_ticks = 8
        rough_step = bp_range / ideal_num_ticks
        steps = [1, 2, 5, 10, 25, 50, 100, 250, 500]
        step = min(steps, key=lambda s: abs(s - rough_step))

        bp_start = step * np.floor(bp_min / step)
        bp_ticks = np.arange(bp_start, bp_max + step, step)
        scan_ticks = bp_to_scan(bp_ticks)

        self.axs[-1].xaxis.set_major_locator(FixedLocator(scan_ticks))

    def _select_category(self, label: str) -> None:
        """RadioButtons callback."""
        self.current_category = label
        cat = self._categories[self.current_category]
        self.radio.activecolor = cat.color
        logger.debug("Selected category: {}", self.current_category)
        self._focus_canvas()

    def _focus_canvas(self) -> None:
        """Give keyboard focus back to the canvas (best-effort)."""
        canvas = self.figure.canvas
        try:
            if hasattr(canvas, "get_tk_widget"):
                canvas.get_tk_widget().focus_set()
            elif getattr(canvas, "manager", None) and hasattr(canvas.manager, "window"):
                canvas.manager.window.focus_force()
        except Exception:
            pass

    def _set_active_span(self, span_dict: dict[str, Any]) -> None:
        """Visually mark a span as active."""
        if self.active_span is not None and self.active_span is not span_dict:
            self._clear_edge(self.active_span)
        if self.hover_span is not None and self.hover_span is not span_dict:
            self._clear_edge(self.hover_span)

        self.active_span = span_dict
        self.hover_span = span_dict
        a = span_dict["artist"]
        a.set_edgecolor(self._ACTIVE_EDGE["color"])
        a.set_linewidth(self._ACTIVE_EDGE["linewidth"])
        self.figure.canvas.draw_idle()

    @staticmethod
    def _clear_edge(span_dict: dict[str, Any]) -> None:
        """Remove any outline on a span."""
        artist = span_dict["artist"]
        artist.set_edgecolor(None)
        artist.set_linewidth(0)
