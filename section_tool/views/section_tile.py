"""SectionTile — hosts the section canvas and its HUD overlay."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QStackedLayout, QWidget


class SectionTile(QWidget):
    """Center-top tile: section canvas (matplotlib) + transparent HUD overlay.

    The HUD includes depth ruler, formation strip, scale bar, and tool
    indicator.  The map inset is NOT here — the map tile replaces it.
    The command palette is a child of this tile (not of HUDLayer) so it
    can still receive mouse events despite HUDLayer being transparent.
    """

    def __init__(self, section_view: QWidget, state, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background: #0e1014;")

        stacked = QStackedLayout(self)
        stacked.setStackingMode(QStackedLayout.StackingMode.StackAll)

        # Section canvas fills the tile
        section_view.setParent(self)
        stacked.addWidget(section_view)
        self.canvas = section_view   # alias

        # HUD layer — transparent for mouse events; no map inset
        from section_tool.hud.hud_layer import HUDLayer
        self.hud = HUDLayer(self, state=state, show_minimap=False)
        stacked.addWidget(self.hud)

        # Command palette as a non-layout child so it receives mouse events
        from section_tool.hud.command_palette import CommandPalette
        self.command_palette = CommandPalette(self)
        self.hud.command_palette = self.command_palette  # API compat
        self.command_palette.hide()
