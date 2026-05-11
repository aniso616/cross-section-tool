from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

# (tool_id, unicode_icon, short_label, tooltip_text)
# None inserts a group separator; str inserts a category header label.

_TOOL_DEFS: list[tuple[str, str, str, str] | str | None] = [
    "NAV",
    ("select",       "↖", "Sel",
     "Select (V)\nSelect and edit objects — click to select, double-click to edit nodes"),
    ("pan",          "⊕", "Pan",
     "Pan (H)\nMiddle-drag or left-drag to pan the active view"),
    ("zoom",         "⊙", "Zoom",
     "Zoom (Z)\nScroll wheel to zoom in / out centred on cursor"),
    None,
    "DRAW",
    ("new_section",  "╱", "Sec",
     "New Section (S)\nDraw a section trace on the map — click nodes, Enter to finish"),
    ("edit_nodes",   "◉", "Nod",
     "Edit Nodes (E)\nSelect, move, insert, or delete section nodes"),
    None,
    "PICK",
    ("horizon_pick", "─", "Hrz",
     "Horizon Pick (P)\nLeft-click on the section view to place horizon picks\n"
     "Click a horizon in the panel first to select the target"),
    ("fault_pick",   "╲", "Flt",
     "Fault Pick (F)\nDraw a fault trace on the section view"),
    ("polygon",      "▭", "Ply",
     "Polygon (G)\nDraw a filled polygon on the section view — right-click to close"),
    None,
    "TOOL",
    ("measure",      "↔", "Msr",
     "Measure (M)\nMeasure distances along the section or on the map"),
]

_TOOL_IDS: list[str] = [t[0] for t in _TOOL_DEFS
                         if isinstance(t, tuple)]

_BTN_ICON_STYLE = """
    QPushButton {{
        background: transparent;
        border: none;
        border-radius: 4px;
        color: {fg};
        font-size: 14px;
        padding: 0px;
        margin: 0px;
    }}
    QPushButton:hover {{
        background: rgba(0, 0, 0, 0.10);
    }}
    QPushButton:checked {{
        background: #1f77b4;
        color: white;
        border-radius: 4px;
    }}
"""

_CAT_STYLE = (
    "QLabel { color: #999999; font-size: 7px; font-weight: bold; "
    "padding: 3px 0 0 0; margin: 0; }"
)
_SHORT_STYLE = (
    "QLabel { color: #666666; font-size: 7px; padding: 0; margin: 0; }"
)


class _ToolButton(QWidget):
    """36×46 widget: icon button (36×30) + centred short label (×16)."""

    clicked = Signal()

    def __init__(self, tool_id: str, icon: str, short: str,
                 tooltip: str, parent=None) -> None:
        super().__init__(parent)
        self.tool_id = tool_id
        self.setFixedWidth(36)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(1, 0, 1, 2)
        vbox.setSpacing(0)

        icon_font = QFont()
        icon_font.setPointSize(13)

        self._btn = QPushButton(icon)
        self._btn.setFixedSize(36, 30)
        self._btn.setCheckable(True)
        self._btn.setFlat(True)
        self._btn.setFont(icon_font)
        self._btn.setToolTip(tooltip)
        self._btn.setStyleSheet(_BTN_ICON_STYLE.format(fg="#444444"))
        self._btn.clicked.connect(self.clicked)
        vbox.addWidget(self._btn, alignment=Qt.AlignmentFlag.AlignHCenter)

        lbl = QLabel(short)
        lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        lbl.setStyleSheet(_SHORT_STYLE)
        vbox.addWidget(lbl, alignment=Qt.AlignmentFlag.AlignHCenter)

    @property
    def is_checked(self) -> bool:
        return self._btn.isChecked()

    def set_checked(self, val: bool) -> None:
        self._btn.setChecked(val)

    def isChecked(self) -> bool:
        return self._btn.isChecked()


class ToolPalette(QWidget):
    """Vertical tool palette with grouped icons and short labels.

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
        self.setFixedWidth(40)
        self.setObjectName("ToolPalette")
        self.setStyleSheet(
            "QWidget#ToolPalette { background: #efefef; "
            "border-right: 1px solid #c8c8c8; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 4)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        for item in _TOOL_DEFS:
            if item is None:
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.HLine)
                sep.setStyleSheet("color: #c8c8c8; margin: 2px 4px;")
                sep.setFixedHeight(5)
                layout.addWidget(sep)
                continue

            if isinstance(item, str):
                # Category label
                lbl = QLabel(item)
                lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
                lbl.setStyleSheet(_CAT_STYLE)
                layout.addWidget(lbl)
                continue

            tool_id, icon, short, tooltip = item
            tbtn = _ToolButton(tool_id, icon, short, tooltip)
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
