from PySide6.QtCore import QEvent, QObject, Qt, Signal


TOOL_KEYS = {
    Qt.Key.Key_V: "select",
    Qt.Key.Key_A: "node_edit",
    Qt.Key.Key_H: "horizon",
    Qt.Key.Key_F: "fault",
    Qt.Key.Key_W: "pick",      # well-top pick (moved off T, which is now Trim)
    Qt.Key.Key_G: "polygon",
    Qt.Key.Key_M: "measure",
}

TOOL_LABELS = {
    "select":     "",          # default; no indicator when in select mode
    "node_edit":  "● Nodes",
    "horizon":    "● Horizon",
    "fault":      "● Fault",
    "pick":       "● Pick",
    "polygon":    "● Polygon",
    "measure":    "● Measure",
    None:         "",
}

# AppState tool-id → on-screen label. The status strip and corner indicator are
# driven by AppState.tool_changed (not ToolManager.tool_changed), so this map —
# unlike TOOL_LABELS above, which is keyed by ToolManager ids — must cover the
# construction tools, which bypass the ToolManager and activate via the palette.
APPSTATE_TOOL_LABELS = {
    "select":          "",
    "node_edit":       "● Nodes",
    "horizon_pick":    "● Horizon",
    "fault_pick":      "● Fault",
    "polygon":         "● Polygon",
    "measure":         "● Measure",
    "pan":             "● Pan",
    "zoom":            "● Zoom",
    "new_section":     "● Section",
    "h_ref":           "● H-Ref",
    "v_ref":           "● V-Ref",
    "a_ref":           "● A-Ref",
    "extend":          "● Extend",
    "trim":            "● Trim",
    "parallel":        "● Parallel",
    "dip_constrained": "● Dip-Constrained",
    "kink_band":       "● Kink Band",
    None:              "",
}


class ToolManager(QObject):
    tool_changed = Signal(object)  # str or None

    def __init__(self):
        super().__init__()
        self._active = None

    @property
    def active(self):
        return self._active

    def handle_key(self, key) -> bool:
        """Returns True if the key was consumed."""
        if key == Qt.Key.Key_Escape:
            self._set(None)
            return True
        if key in TOOL_KEYS:
            tool = TOOL_KEYS[key]
            self._set(None if tool == self._active else tool)
            return True
        return False

    def _set(self, tool):
        if tool != self._active:
            self._active = tool
            self.tool_changed.emit(tool)

    def reset(self) -> None:
        """Clear the active tool WITHOUT emitting — used to resync after an
        external tool change (e.g. Escape routed through the palette), so the
        next tool key doesn't toggle off a stale active tool."""
        self._active = None


class ToolKeyFilter(QObject):
    """Event filter that routes tool-key presses through ToolManager.

    Install on the canvas widget so tool keys are handled before matplotlib.
    H/F/T/A are consumed (matplotlib ignores them).
    Escape is NOT consumed (matplotlib still handles pick/polygon cancellation).
    Slash and Ctrl+K invoke the command palette.
    """

    palette_invoke_requested = Signal()

    def __init__(self, tool_manager: ToolManager, parent=None):
        super().__init__(parent)
        self._mgr = tool_manager

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
            key = event.key()

            # Escape: handle in tool manager but let matplotlib also see it
            if key == Qt.Key.Key_Escape:
                self._mgr.handle_key(key)
                return False

            # Command palette invocation
            if key == Qt.Key.Key_Slash:
                self.palette_invoke_requested.emit()
                return True
            if (key == Qt.Key.Key_K
                    and event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self.palette_invoke_requested.emit()
                return True

            # Tool keys — consume so matplotlib doesn't see them
            if self._mgr.handle_key(key):
                return True

        return False
