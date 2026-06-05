"""FloatingToolPalette — a persistent, draggable tool overlay on the section.

A curated "building" toolset that doubles as a fast tool-switcher. It is a
frameless overlay *child widget* (deliberately NOT a QDockWidget — we moved off
docks) parented over the section canvas.

It owns NO activation logic: clicking a button emits ``tool_activation_requested``
which the app routes through the one true path (ToolPalette.set_active_tool →
ToolPalette.tool_changed → _on_tool_changed → AppState.set_active_tool). Its
highlight subscribes to ``AppState.tool_changed`` so it agrees with the rail and
hotkeys no matter how the tool changed.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFrame, QGraphicsOpacityEffect, QGridLayout,
                               QVBoxLayout, QWidget)

from section_tool.style import C_RULE
from section_tool.views.tool_palette import _TOOL_DEFS, _ToolButton, _rgba

# The curated building set — a single editable list of (tool_id, hotkey_letter).
# Tune freely; ids and hotkeys are the real ones (see app._register_shortcuts).
# Reference lines (h_ref/v_ref/a_ref) are intentionally omitted; they would slot
# in with the same (id, "R") pattern but share the single R-cycle hotkey.
BUILDING_TOOLS: list[tuple[str, str]] = [
    ("select",          "V"), ("node_edit",      "A"),
    ("horizon_pick",    "H"), ("fault_pick",     "F"),
    ("polygon",         "G"), ("extend",         "E"),
    ("trim",            "T"), ("parallel",       "P"),
    ("dip_constrained", "D"), ("kink_band",      "K"),
]

# tool_id -> (icon, label, tooltip), reused from the rail so icons stay in sync.
_DEF = {t[0]: (t[1], t[2], t[3]) for t in _TOOL_DEFS if isinstance(t, tuple)}

_REST_OPACITY = 0.6     # recede at rest; full on hover (brighten, don't vanish)
_GRIP_H       = 14


class FloatingToolPalette(QWidget):
    """Frameless overlay palette: 2-column grid of building tools + drag grip."""

    tool_activation_requested = Signal(str)   # tool_id — routed via the rail path

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("FloatingToolPalette")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"QWidget#FloatingToolPalette {{ background: {_rgba((20, 22, 28), 238)}; "
            f"border: 1px solid {_rgba(C_RULE)}; border-radius: 6px; }}"
        )
        self._buttons: dict[str, _ToolButton] = {}
        self._build_ui()

        # Resting translucency; brightened to full on hover via enter/leaveEvent.
        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(_REST_OPACITY)
        self.setGraphicsEffect(self._opacity)

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Thin drag grip across the top (drag logic added in Phase 2).
        self.grip = QFrame(self)
        self.grip.setObjectName("PaletteGrip")
        self.grip.setFixedHeight(_GRIP_H)
        self.grip.setCursor(Qt.CursorShape.SizeAllCursor)
        self.grip.setStyleSheet(
            f"QFrame#PaletteGrip {{ background: {_rgba(C_RULE, 130)}; "
            "border-top-left-radius: 6px; border-top-right-radius: 6px; }"
            f"QFrame#PaletteGrip:hover {{ background: {_rgba(C_RULE, 200)}; }}"
        )
        outer.addWidget(self.grip)

        grid = QGridLayout()
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setSpacing(2)
        for i, (tid, key) in enumerate(BUILDING_TOOLS):
            icon, _label, tip = _DEF.get(tid, ("?", tid, tid))
            btn = _ToolButton(tid, icon, key, tip)   # short-label IS the hotkey
            btn.clicked.connect(lambda t=tid: self.tool_activation_requested.emit(t))
            self._buttons[tid] = btn
            grid.addWidget(btn, i // 2, i % 2, alignment=Qt.AlignmentFlag.AlignHCenter)
        outer.addLayout(grid)

    # ------------------------------------------------------------------

    def set_active(self, tool_id: str) -> None:
        """Reflect the active tool — driven by AppState.tool_changed, so the
        highlight agrees with the rail and hotkeys regardless of source."""
        for tid, btn in self._buttons.items():
            btn.set_checked(tid == tool_id)

    def enterEvent(self, event) -> None:
        self._opacity.setOpacity(1.0)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._opacity.setOpacity(_REST_OPACITY)
        super().leaveEvent(event)
