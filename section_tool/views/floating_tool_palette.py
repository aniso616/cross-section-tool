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

from PySide6.QtCore import QEvent, QPoint, QSettings, Qt, Signal
from PySide6.QtWidgets import (QFrame, QGraphicsOpacityEffect, QGridLayout,
                               QLayout, QVBoxLayout, QWidget)

from section_tool.style import C_RULE
from section_tool.views.tool_palette import (
    _TOOL_DEFS, _ToolButton, _rgba, TOOL_HOTKEYS, compose_tooltip)

# The curated building set — a single editable list of tool_ids. The hotkey
# letter shown on each button and in its tooltip is pulled from TOOL_HOTKEYS
# (the single source of truth), so the palette can't drift from the bindings.
# Reference lines (h_ref/v_ref/a_ref) are intentionally omitted; they share the
# single R-cycle hotkey.
BUILDING_TOOLS: list[str] = [
    "select", "node_edit",
    "horizon_pick", "fault_pick",
    "polygon", "extend",
    "trim", "parallel",
    "dip_constrained", "kink_band",
]

# tool_id -> (icon, label, tooltip), reused from the rail so icons stay in sync.
_DEF = {t[0]: (t[1], t[2], t[3]) for t in _TOOL_DEFS if isinstance(t, tuple)}

_REST_OPACITY = 0.6     # recede at rest; full on hover (brighten, don't vanish)
_GRIP_H       = 14
_SETTINGS_ORG = "Geoscience"
_SETTINGS_APP = "CrossSectionTool"
_POS_KEY      = "floating_palette/pos"


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
        self._drag_press: QPoint | None = None
        self._drag_origin: QPoint | None = None
        self._build_ui()

        # Resting translucency; brightened to full on hover via enter/leaveEvent.
        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(_REST_OPACITY)
        self.setGraphicsEffect(self._opacity)

        # Watch the parent so a resize can never strand the palette off-canvas.
        if parent is not None:
            parent.installEventFilter(self)

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        # As a free overlay child (not in a parent layout) the palette won't
        # auto-fit its contents; SetFixedSize makes it size exactly to the grid.
        outer.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

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
        for i, tid in enumerate(BUILDING_TOOLS):
            icon, _label, tip = _DEF.get(tid, ("?", tid, tid))
            key = TOOL_HOTKEYS.get(tid, "")           # single source of truth
            btn = _ToolButton(tid, icon, key,         # short-label IS the hotkey
                              compose_tooltip(tid, tip))
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

    # ---- drag (by the grip) -------------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and \
                self.grip.geometry().contains(event.position().toPoint()):
            self._drag_press  = event.globalPosition().toPoint()
            self._drag_origin = self.pos()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_press is not None:
            delta = event.globalPosition().toPoint() - self._drag_press
            self._move_clamped(self._drag_origin + delta)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_press is not None:
            self._drag_press = None
            self._save_position()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ---- position: clamp / persist / restore --------------------------

    def _move_clamped(self, pt: QPoint) -> None:
        """Move so the palette stays fully inside the parent (never stranded)."""
        par = self.parentWidget()
        if par is None:
            self.move(pt)
            return
        max_x = max(0, par.width() - self.width())
        max_y = max(0, par.height() - self.height())
        self.move(min(max(pt.x(), 0), max_x), min(max(pt.y(), 0), max_y))

    def _save_position(self) -> None:
        QSettings(_SETTINGS_ORG, _SETTINGS_APP).setValue(_POS_KEY, self.pos())

    def restore_position(self, default_pt: QPoint) -> None:
        """Restore the saved position, but only if it lands inside the current
        parent bounds; otherwise fall back to *default_pt*. Guards against the
        off-screen-dock failure that motivated moving off docks entirely."""
        val = QSettings(_SETTINGS_ORG, _SETTINGS_APP).value(_POS_KEY)
        pt = val if isinstance(val, QPoint) else None
        if pt is None or not self._point_usable(pt):
            pt = default_pt
        self._move_clamped(pt)

    def _point_usable(self, pt: QPoint) -> bool:
        """The saved top-left must sit within the parent with the grip reachable."""
        par = self.parentWidget()
        if par is None or par.width() <= 0 or par.height() <= 0:
            return False
        return (0 <= pt.x() <= max(0, par.width() - 40)
                and 0 <= pt.y() <= max(0, par.height() - _GRIP_H))

    def eventFilter(self, obj, event) -> bool:
        if obj is self.parentWidget() and event.type() == QEvent.Type.Resize:
            self._move_clamped(self.pos())   # re-clamp so a shrink can't strand it
        return super().eventFilter(obj, event)
