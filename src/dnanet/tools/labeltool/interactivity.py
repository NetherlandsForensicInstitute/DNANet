"""Abstract base class for interactive matplotlib visualizations.

Design pattern: **Template Method**
    ``Interactivity`` defines the skeleton of interactive behavior
    (figure, axes, spans). Subclasses implement ``activate_interactivity``
    to wire up event handlers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from matplotlib.axes import Axes
from matplotlib.figure import Figure


class Interactivity(ABC):
    """Base class for interactive matplotlib visualizations.

    Args:
        figure: The matplotlib Figure.
        axes: Sequence of Axes (one per dye channel).
        spans: Optional list of span dictionaries from previous annotations.
    """

    def __init__(
        self,
        figure: Figure,
        axes: Sequence[Axes],
        spans: list[dict[str, Any]] | None = None,
    ) -> None:
        self.figure = figure
        self.axs = axes
        self.spans: list[dict[str, Any]] = spans if spans else []

    @abstractmethod
    def activate_interactivity(self) -> None:
        """Wire up matplotlib event handlers for interactive behavior."""
