"""MapTile — full-size map canvas tile (bottom of center splitter)."""
from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget


class MapTile(QWidget):
    """Center-bottom tile: full-size map canvas.

    Where section lines are drawn and spatial context lives.
    Replaces the minimap thumbnail used in the old game-UI layout.
    """

    def __init__(self, map_view: QWidget, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 180)
        self.setStyleSheet("background: #0e1014;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        map_view.setParent(self)
        layout.addWidget(map_view)
        self.canvas = map_view   # alias
