"""StatusStrip — bottom status bar: coordinates, active tool, hint."""
from __future__ import annotations

from PySide6.QtWidgets import QLabel, QStatusBar


_STYLE = """
    QStatusBar {
        background: #1a1c20;
        border-top: 1px solid #2a2d33;
        color: #707880;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 11px;
        padding: 0 8px;
    }
    QLabel { color: #707880; background: transparent; }
"""


class StatusStrip(QStatusBar):
    """Bottom status bar.

    Left:   cursor coordinates
    Center: active tool indicator
    Right:  contextual hint
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(_STYLE)
        self.setSizeGripEnabled(False)
        self.setFixedHeight(24)

        self._coords = QLabel("x: —  |  depth: —  |  elev: —")
        self._tool   = QLabel("")
        self._hint   = QLabel("Draw a section line on the map to get started")

        self._tool.setStyleSheet("color: #8090a8;")
        self._hint.setStyleSheet("color: #505860;")

        self.addWidget(self._coords)
        self.addPermanentWidget(self._tool)
        self.addPermanentWidget(self._hint)

    def update_coords(self, x_m: float, depth_m: float, elev_m: float):
        self._coords.setText(
            f"x: {x_m:,.0f} m  |  "
            f"depth: {depth_m:,.0f} m  |  "
            f"elev: {elev_m:+,.0f} m asl"
        )

    def set_tool(self, tool: str | None):
        # Driven by AppState.tool_changed, so use the AppState-keyed labels
        # (covers construction tools, which bypass the ToolManager).
        from section_tool.interaction.tool_manager import APPSTATE_TOOL_LABELS
        self._tool.setText(APPSTATE_TOOL_LABELS.get(tool, ""))

    def set_hint(self, text: str):
        self._hint.setText(text)

    def clear_coords(self):
        self._coords.setText("x: —  |  depth: —  |  elev: —")
