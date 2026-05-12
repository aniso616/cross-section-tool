from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

# (tool_id, icon, label, tooltip) | str (group header) | None (separator)

_TOOL_DEFS: list[tuple[str, str, str, str] | str | None] = [
    "Navigate",
    ("select",      "↖", "Select",
     "Select Object  (V)\n"
     "Click an object to select it. Double-click to enter node-edit mode.\n"
     "Drag to move the selected object.  Delete to remove it.\n"
     "Hold Space to temporarily pan."),
    ("node_edit",   "↗", "Nodes",
     "Direct Select / Node Edit  (A)\n"
     "Click a node to select it; drag to move.  Delete to remove it.\n"
     "Click between nodes on a line to insert a new node.\n"
     "Escape to exit node mode."),
    ("pan",         "⊕", "Pan",
     "Pan  (H)\n"
     "Left-drag or middle-drag to pan the view.\n"
     "Hold Space from any tool to pan temporarily."),
    ("zoom",        "⊙", "Zoom",
     "Zoom  (Z) · Shift+Z to fit\n"
     "Scroll wheel zooms in/out centred on the cursor.\n"
     "Shift+Z resets view to fit all data."),
    None,
    "Draw",
    ("new_section", "╱", "Section",
     "Draw Section  (S)\n"
     "Click nodes on the map to draw a section trace.\n"
     "Double-click or Enter to finish.  Escape to cancel.\n"
     "Section → New Section (east-west default) for a quick default."),
    None,
    "Interpret",
    ("horizon_pick","─", "Horizon",
     "Horizon Pick  (P)\n"
     "Click on the section view to add picks to the active horizon.\n"
     "Right-click or Escape to end.  Double-click to place final pick."),
    ("fault_pick",  "╲", "Fault",
     "Fault Pick  (F)\n"
     "Click to add picks to the active fault.  Right-click to end."),
    ("polygon",     "▭", "Polygon",
     "Polygon  (G)\n"
     "Click to place vertices; right-click to close the polygon."),
    None,
    "Construct",
    ("h_ref",       "═", "H-Ref",
     "Horizontal Reference Line  (R)\n"
     "Click on the section to place a horizontal guide at that depth."),
    ("v_ref",       "‖", "V-Ref",
     "Vertical Reference Line  (R twice)\n"
     "Click to place a vertical guide at that distance."),
    ("a_ref",       "╱", "A-Ref",
     "Angled Reference Line  (R thrice)\n"
     "Click to set anchor; move to set direction; click to confirm."),
    ("extend",      "→|", "Extend",
     "Extend  —  click an endpoint then click the target line."),
    ("trim",        "|←", "Trim",
     "Trim  —  click a pick line then click the cutting line."),
    ("parallel",    "‗", "Parallel",
     "Parallel  —  click source line then click to place parallel copy."),
    None,
    "Tools",
    ("measure",     "↔", "Measure",
     "Measure  (M)\n"
     "Measure distances, depth differences, and angles on the section or map."),
]

_TOOL_IDS: list[str] = [t[0] for t in _TOOL_DEFS if isinstance(t, tuple)]

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

_BTN_STYLE = """
QPushButton {{
    background: transparent;
    border: none;
    border-radius: 5px;
    color: {fg};
    font-size: 14px;
    padding: 0px;
    margin: 0px;
}}
QPushButton:hover {{
    background: rgba(0, 0, 0, 0.12);
}}
QPushButton:checked {{
    background: #1f77b4;
    color: white;
    border-radius: 5px;
}}
"""

_CATEGORY_STYLE = (
    "QLabel { color: #888888; font-size: 7pt; font-weight: bold; "
    "padding: 5px 0 1px 4px; }"
)

_LABEL_STYLE = (
    "QLabel { color: #555555; font-size: 7pt; padding: 0; margin: 0; }"
)


# ---------------------------------------------------------------------------
# Tool button widget
# ---------------------------------------------------------------------------

class _ToolButton(QWidget):
    """40×36 icon button + 12px short-label = ~50px total height."""

    clicked = Signal()

    def __init__(self, tool_id: str, icon: str, label: str,
                 tooltip: str, parent=None) -> None:
        super().__init__(parent)
        self.tool_id = tool_id
        self.setFixedWidth(52)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(2, 0, 2, 2)
        vbox.setSpacing(0)

        self._btn = QPushButton(icon)
        self._btn.setFixedSize(44, 36)
        self._btn.setCheckable(True)
        self._btn.setFlat(True)
        self._btn.setToolTip(tooltip)
        self._btn.setStyleSheet(_BTN_STYLE.format(fg="#222222"))
        self._btn.clicked.connect(self.clicked)
        vbox.addWidget(self._btn, alignment=Qt.AlignmentFlag.AlignHCenter)

        lbl = QLabel(label)
        lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        lbl.setStyleSheet(_LABEL_STYLE)
        vbox.addWidget(lbl, alignment=Qt.AlignmentFlag.AlignHCenter)

    def isChecked(self) -> bool:
        return self._btn.isChecked()

    def set_checked(self, val: bool) -> None:
        self._btn.setChecked(val)


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

class ToolPalette(QWidget):
    """Vertical tool palette with grouped icon buttons.

    Signals
    -------
    tool_changed(str)  — tool_id of the newly-active tool.
    """

    tool_changed = Signal(str)
    DEFAULT_TOOL = "select"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._buttons: dict[str, _ToolButton] = {}
        self._active_tool: str = ""
        self._setup_ui()
        self._activate(self.DEFAULT_TOOL, emit=False)

    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setFixedWidth(56)
        self.setObjectName("ToolPalette")
        self.setStyleSheet(
            "QWidget#ToolPalette { background: #f0f0f0; "
            "border-right: 1px solid #c8c8c8; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(1)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        for item in _TOOL_DEFS:
            if item is None:
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.HLine)
                sep.setStyleSheet("QFrame { color: #d0d0d0; margin: 2px 6px; }")
                sep.setFixedHeight(4)
                layout.addWidget(sep)
                continue

            if isinstance(item, str):
                lbl = QLabel(item)
                lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
                lbl.setStyleSheet(_CATEGORY_STYLE)
                layout.addWidget(lbl)
                line = QFrame()
                line.setFrameShape(QFrame.Shape.HLine)
                line.setStyleSheet("QFrame { color: #d8d8d8; margin: 0 4px; }")
                line.setFixedHeight(3)
                layout.addWidget(line)
                continue

            tool_id, icon, label, tooltip = item
            tbtn = _ToolButton(tool_id, icon, label, tooltip)
            tbtn.clicked.connect(
                lambda tid=tool_id: self._on_button_clicked(tid)
            )
            layout.addWidget(tbtn, alignment=Qt.AlignmentFlag.AlignHCenter)
            self._buttons[tool_id] = tbtn

        layout.addStretch()

    # ------------------------------------------------------------------

    @property
    def active_tool(self) -> str:
        return self._active_tool

    @property
    def tool_ids(self) -> list[str]:
        return list(_TOOL_IDS)

    def set_active_tool(self, tool_id: str) -> None:
        if tool_id not in self._buttons:
            return
        self._activate(tool_id, emit=True)

    def _activate(self, tool_id: str, *, emit: bool) -> None:
        changed = self._active_tool != tool_id
        self._active_tool = tool_id
        for tid, tbtn in self._buttons.items():
            tbtn.set_checked(tid == tool_id)
        if emit and changed:
            self.tool_changed.emit(tool_id)

    def _on_button_clicked(self, tool_id: str) -> None:
        self._activate(tool_id, emit=True)
