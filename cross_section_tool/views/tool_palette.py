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

# Each entry is one of:
#   (tool_id, icon, label, tooltip)  — a tool button
#   str                              — a category header ("Navigate" etc.)
#   None                             — a thin separator line

_TOOL_DEFS: list[tuple[str, str, str, str] | str | None] = [
    "Navigate",
    ("select",       "↖", "Select",
     "Select  (V)\n"
     "Click to select an object, double-click to edit its nodes.\n"
     "Drag a selected object to move it."),
    ("pan",          "⊕", "Pan",
     "Pan  (H)\n"
     "Left-drag or middle-drag to pan the view."),
    ("zoom",         "⊙", "Zoom",
     "Zoom  (Z)\n"
     "Scroll wheel zooms in/out centred on the cursor."),
    None,
    "Draw",
    ("new_section",  "╱", "Section",
     "Draw Section  (S)\n"
     "Click nodes on the map to draw a section trace.\n"
     "Double-click or press Enter to finish.  Escape to cancel."),
    ("edit_nodes",   "◉", "Nodes",
     "Edit Nodes  (E)\n"
     "Select, move, insert or delete section nodes."),
    None,
    "Interpret",
    ("horizon_pick", "─", "Horizon",
     "Horizon Pick  (P)\n"
     "Click on the section view to place horizon picks.\n"
     "Right-click or double-click to end the pick sequence."),
    ("fault_pick",   "╲", "Fault",
     "Fault Pick  (F)\n"
     "Draw a fault trace on the section view.\n"
     "Right-click or double-click to end."),
    ("polygon",      "▭", "Polygon",
     "Polygon  (G)\n"
     "Draw a filled polygon — right-click to close."),
    None,
    "Construct",
    ("h_ref",        "═", "H-Ref",
     "Horizontal Reference Line\n"
     "Click on the section view to place a horizontal guide at that depth."),
    ("v_ref",        "‖", "V-Ref",
     "Vertical Reference Line\n"
     "Click on the section view to place a vertical guide at that distance."),
    ("a_ref",        "╱", "A-Ref",
     "Angled Reference Line\n"
     "Click to set anchor, click again to set direction (angle shown in status bar)."),
    ("extend",       "→|", "Extend",
     "Extend  —  click an endpoint, then click the target line to extend to it."),
    ("trim",         "|←", "Trim",
     "Trim  —  click a pick line, then the cutting line."),
    ("parallel",     "‗", "Parallel",
     "Parallel  —  click a source line, then place a parallel copy at click position."),
    None,
    "Tools",
    ("measure",      "↔", "Measure",
     "Measure  (M)\n"
     "Measure distances along the section or on the map."),
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
    font-size: 20px;
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
    """40×40 icon button + 12px short-label row = ~52px total height."""

    clicked = Signal()

    def __init__(self, tool_id: str, icon: str, label: str,
                 tooltip: str, parent=None) -> None:
        super().__init__(parent)
        self.tool_id = tool_id
        self.setFixedWidth(48)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(4, 0, 4, 2)
        vbox.setSpacing(0)

        self._btn = QPushButton(icon)
        self._btn.setFixedSize(40, 40)
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
    """Vertical tool palette — 40×40 icons with labels, grouped by category.

    Signals
    -------
    tool_changed(str)
        The tool_id of the newly-active tool.
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
                sep.setStyleSheet(
                    "QFrame { color: #d0d0d0; margin: 2px 6px; }"
                )
                sep.setFixedHeight(4)
                layout.addWidget(sep)
                continue

            if isinstance(item, str):
                lbl = QLabel(item)
                lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
                lbl.setStyleSheet(_CATEGORY_STYLE)
                layout.addWidget(lbl)
                # thin underline
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
    # Public API
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

    # ------------------------------------------------------------------

    def _activate(self, tool_id: str, *, emit: bool) -> None:
        changed = self._active_tool != tool_id
        self._active_tool = tool_id
        for tid, tbtn in self._buttons.items():
            tbtn.set_checked(tid == tool_id)
        if emit and changed:
            self.tool_changed.emit(tool_id)

    def _on_button_clicked(self, tool_id: str) -> None:
        self._activate(tool_id, emit=True)
