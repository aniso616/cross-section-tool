from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

# (tool_id, unicode_icon, tooltip_text) — None inserts a visual separator
_TOOL_DEFS: list[tuple[str, str, str] | None] = [
    ("select",       "↖",  "Select / edit objects  (V)"),
    ("pan",          "⊕",  "Pan view  (Space)"),
    ("zoom",         "⊙",  "Zoom — left: in, right: out  (Z)"),
    None,
    ("new_section",  "╱",  "New Section — draw on map  (N)"),
    ("edit_nodes",   "◉",  "Edit section nodes  (E)"),
    None,
    ("horizon_pick", "─",  "Horizon Pick  (H)"),
    ("fault_pick",   "╲",  "Fault Pick  (F)"),
    ("polygon",      "▭",  "Polygon  (G)"),
    None,
    ("measure",      "↔",  "Measure distance  (M)"),
]

_TOOL_IDS: list[str] = [t[0] for t in _TOOL_DEFS if t is not None]

_BTN_STYLE = """
    QPushButton {{
        background: transparent;
        border: none;
        border-radius: 4px;
        color: {fg};
        font-size: 15px;
        padding: 0px;
    }}
    QPushButton:hover {{
        background: rgba(0, 0, 0, 0.10);
    }}
    QPushButton:checked {{
        background: #1f77b4;
        color: white;
    }}
"""


class ToolPalette(QWidget):
    """Vertical icon-strip tool palette, QGIS / Illustrator style.

    Exactly one tool is active at a time.  Activating a tool via
    :meth:`set_active_tool` or clicking a button emits :attr:`tool_changed`.

    Signals
    -------
    tool_changed(str)
        The tool_id of the newly-active tool.
    """

    tool_changed = Signal(str)

    DEFAULT_TOOL = "select"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._buttons: dict[str, QPushButton] = {}
        self._active_tool: str = ""
        self._setup_ui()
        # Activate default without emitting (first activation sets state only)
        self._activate(self.DEFAULT_TOOL, emit=False)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setFixedWidth(40)
        self.setObjectName("ToolPalette")
        self.setStyleSheet(
            "QWidget#ToolPalette { background: #efefef; border-right: 1px solid #c8c8c8; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 6, 2, 6)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        font = QFont()
        font.setPointSize(13)

        for item in _TOOL_DEFS:
            if item is None:
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.HLine)
                sep.setStyleSheet("color: #c0c0c0; margin: 3px 4px;")
                sep.setFixedHeight(6)
                layout.addWidget(sep)
                continue

            tool_id, icon, tooltip = item
            btn = QPushButton(icon)
            btn.setFixedSize(36, 36)
            btn.setCheckable(True)
            btn.setFlat(True)
            btn.setFont(font)
            btn.setToolTip(tooltip)
            btn.setStyleSheet(_BTN_STYLE.format(fg="#444444"))
            btn.clicked.connect(
                lambda _checked, tid=tool_id: self._on_button_clicked(tid)
            )
            layout.addWidget(btn)
            self._buttons[tool_id] = btn

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
        """Activate *tool_id*, deactivating all others.  Emits :attr:`tool_changed`."""
        if tool_id not in self._buttons:
            return
        self._activate(tool_id, emit=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _activate(self, tool_id: str, *, emit: bool) -> None:
        changed = self._active_tool != tool_id
        self._active_tool = tool_id
        for tid, btn in self._buttons.items():
            btn.setChecked(tid == tool_id)
        if emit and changed:
            self.tool_changed.emit(tool_id)

    def _on_button_clicked(self, tool_id: str) -> None:
        self._activate(tool_id, emit=True)
