"""MapTile — full-size map canvas tile (bottom of center splitter)."""
from __future__ import annotations

from PySide6.QtWidgets import QStackedLayout, QWidget


class MapTile(QWidget):
    """Center-bottom tile: full-bleed map canvas + transparent HUD overlay.

    Where section lines are drawn and spatial context lives. Mirrors SectionTile:
    the map canvas fills the tile and a MapHUDLayer (edge rulers + scale bar) is
    composited on top, so the map and section share the same canvas idiom.
    """

    def __init__(self, map_view: QWidget, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 180)
        self.setStyleSheet("background: #0e1014;")

        stacked = QStackedLayout(self)
        stacked.setStackingMode(QStackedLayout.StackingMode.StackAll)

        map_view.setParent(self)
        stacked.addWidget(map_view)
        self.canvas = map_view   # alias

        from section_tool.hud.map_hud_layer import MapHUDLayer
        self.hud = MapHUDLayer(self, map_view)
        stacked.addWidget(self.hud)
        self.hud.raise_()
