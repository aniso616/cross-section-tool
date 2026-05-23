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
    "Nav",
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
    "Pick",
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
    "Build",
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
    ("dip_constrained", "⧊", "Dip",
     "Dip-Constrained  (D) — click anchor then click extent.\n"
     "Creates a horizon segment locked to the configured dip angle."),
    ("kink_band",   "⋀", "Kink",
     "Kink Band  (K) — click backlimb horizon then click axial trace.\n"
     "Creates a forelimb extension using kink-band kinematics."),
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
    border-radius: 4px;
    color: #CCCCCC;
    font-size: 15px;
    padding: 0px;
    margin: 0px;
}}
QPushButton:hover {{
    background: #444444;
    color: white;
}}
QPushButton:checked {{
    background: #2563EB;
    color: white;
    border-radius: 4px;
    border: 1px solid #1D4ED8;
}}
QPushButton:disabled {{
    color: #555555;
    background: transparent;
}}
"""

_CATEGORY_STYLE = (
    "QLabel { color: #777777; font-size: 7pt; font-weight: bold; "
    "letter-spacing: 1px; padding: 12px 0 4px 4px; }"
)

_LABEL_STYLE = (
    "QLabel { color: #888888; font-size: 7pt; padding: 0; margin: 0; }"
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
        self._base_tooltip = tooltip
        self._label_text   = label
        self.setFixedWidth(52)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(2, 0, 2, 2)
        vbox.setSpacing(0)

        self._btn = QPushButton(icon)
        self._btn.setFixedSize(44, 36)
        self._btn.setCheckable(True)
        self._btn.setFlat(True)
        self._btn.setToolTip(tooltip)
        self._btn.setStyleSheet(_BTN_STYLE)
        self._btn.clicked.connect(self.clicked)
        vbox.addWidget(self._btn, alignment=Qt.AlignmentFlag.AlignHCenter)

        self._lbl = QLabel(label)
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._lbl.setStyleSheet(_LABEL_STYLE)
        vbox.addWidget(self._lbl, alignment=Qt.AlignmentFlag.AlignHCenter)

    def isChecked(self) -> bool:
        return self._btn.isChecked()

    def set_checked(self, val: bool) -> None:
        self._btn.setChecked(val)

    def set_available(self, available: bool, reason: str = "") -> None:
        """Grey out (disabled) or restore the button."""
        self._btn.setEnabled(available)
        if available:
            self._btn.setToolTip(self._base_tooltip)
            self._lbl.setStyleSheet(_LABEL_STYLE)
        else:
            tip = reason or "Not available"
            self._btn.setToolTip(tip)
            self._lbl.setStyleSheet(
                "QLabel { color: #505050; font-size: 7pt; padding: 0; margin: 0; }"
            )


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
            "QWidget#ToolPalette { background: #363636; "
            "border-right: 2px solid #4A4A4A; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(1)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        first_group = True
        for item in _TOOL_DEFS:
            if item is None:
                continue   # explicit separators removed; groups already spaced by labels

            if isinstance(item, str):
                # Thin rule above every group except the first
                if not first_group:
                    line = QFrame()
                    line.setFrameShape(QFrame.Shape.HLine)
                    line.setStyleSheet(
                        "QFrame { color: #444; margin: 0 6px; }")
                    line.setFixedHeight(1)
                    layout.addWidget(line)
                first_group = False

                lbl = QLabel(item.upper())
                lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
                lbl.setStyleSheet(_CATEGORY_STYLE)
                layout.addWidget(lbl)
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

    # ------------------------------------------------------------------
    # Context-sensitive availability
    # ------------------------------------------------------------------

    # Tools that need an active section to be useful
    _NEEDS_SECTION: frozenset[str] = frozenset({
        "horizon_pick", "fault_pick", "polygon",
        "h_ref", "v_ref", "a_ref",
    })
    # Tools that additionally need at least one pick object
    _NEEDS_PICKS: frozenset[str] = frozenset({
        "extend", "trim", "parallel",
    })

    def update_tool_availability(self, has_section: bool,
                                  has_picks: bool) -> None:
        """Enable / disable tool buttons based on current project state."""
        for tool_id, tbtn in self._buttons.items():
            if tool_id in self._NEEDS_PICKS:
                if not has_picks:
                    tbtn.set_available(False, "Create a horizon or fault first")
                    continue
            if tool_id in self._NEEDS_SECTION:
                if not has_section:
                    tbtn.set_available(False, "Load a section first")
                    continue
            tbtn.set_available(True)
