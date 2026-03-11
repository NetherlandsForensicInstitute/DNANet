from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Sequence

from matplotlib.axes import Axes
from matplotlib.figure import Figure


class Interactivity(ABC):
    def __init__(
            self, figure: Figure, axes: Sequence[Axes],
            spans: Optional[List[Dict[str, object]]] = None
    ) -> None:
        self.figure = figure
        self.axs = axes

        if not spans:
            spans = list()
        self.spans = spans


    @abstractmethod
    def activate_interactivity(self) -> None:
        pass
