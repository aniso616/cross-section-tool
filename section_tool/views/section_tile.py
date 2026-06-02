"""SectionTile — hosts the section canvas and its HUD overlay."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QStackedLayout, QWidget


class ToolHUD(QWidget):
    """Small semi-transparent bar at top-centre of section canvas — active tool + hint."""

    _HINTS = {
        "select":          "Click to select  ·  Double-click for node editing",
        "node_edit":       "Click node to select  ·  Drag to move  ·  Del to remove",
        "horizon_pick":    "Click to place pick  ·  Right-click or Esc to end",
        "fault_pick":      "Click to place pick  ·  Right-click or Esc to end",
        "polygon":         "Click vertices  ·  Right-click to close",
        "measure":         "Click two points to measure",
        "pan":             "Drag to pan  ·  Scroll to zoom",
        "new_section":     "Click endpoints  ·  Double-click or Enter to finish",
        # Construction tools (parameters set in the bar above the canvas)
        "extend":          "Click a pick endpoint, then the target line",
        "trim":            "Click the keep-side of a line, then the cutting line",
        "parallel":        "Click a reference horizon, then click to place the copy",
        "dip_constrained": "Click anchor, then extent  ·  set Dip angle above",
        "kink_band":       "Click backlimb horizon, then axial trace  ·  set dips above",
    }

    # Initial key legend, shown before any tool is activated.
    _LEGEND = ("V Select · A Nodes · H Horizon · F Fault · G Polygon · M Measure"
               "    E Extend · T Trim · P Parallel · D Dip · K Kink")

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedHeight(28)
        self.setStyleSheet("""
            QWidget {
                background: rgba(20, 20, 28, 195);
                border-radius: 4px;
            }
            QLabel { background: transparent; }
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 2, 10, 2)
        layout.setSpacing(10)

        self._tool_lbl = QLabel("● Select")
        self._tool_lbl.setStyleSheet(
            "color: #4A9EFF; font-weight: bold; font-size: 9pt;")
        layout.addWidget(self._tool_lbl)

        self._hint_lbl = QLabel(self._LEGEND)
        self._hint_lbl.setStyleSheet("color: #707880; font-size: 8pt;")
        layout.addWidget(self._hint_lbl)
        self.show()

    def set_tool(self, tool_id: str | None) -> None:
        _names = {
            "select":          "● Select",
            "node_edit":       "● Nodes",
            "horizon_pick":    "● Horizon Pick",
            "fault_pick":      "● Fault Pick",
            "polygon":         "● Polygon",
            "measure":         "● Measure",
            "pan":             "● Pan",
            "new_section":     "● Draw Section",
            "h_ref":           "● H-Ref",
            "v_ref":           "● V-Ref",
            "a_ref":           "● A-Ref",
            "extend":          "● Extend",
            "trim":            "● Trim",
            "parallel":        "● Parallel",
            "dip_constrained": "● Dip-Constrained",
            "kink_band":       "● Kink Band",
        }
        tool = tool_id or "select"
        self._tool_lbl.setText(_names.get(tool, f"● {tool}"))
        # Fall back to the key legend when a tool has no specific hint.
        self._hint_lbl.setText(self._HINTS.get(tool, self._LEGEND))
        self.adjustSize()
        self._reposition()

    def _reposition(self) -> None:
        parent = self.parent()
        if parent:
            pw = parent.width()
            self.setFixedWidth(min(600, pw - 40))
            self.move((pw - self.width()) // 2, 8)


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

        # Tool HUD bar — top-centre, shows active tool + hint
        self.tool_hud = ToolHUD(self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "tool_hud"):
            self.tool_hud._reposition()
