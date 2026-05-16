from PySide6.QtCore import QEvent, QObject, Qt, Signal


TOOL_KEYS = {
    Qt.Key.Key_H: "horizon",
    Qt.Key.Key_F: "fault",
    Qt.Key.Key_T: "pick",
    Qt.Key.Key_A: "annotation",
}

TOOL_LABELS = {
    "horizon":    "● Horizon",
    "fault":      "● Fault",
    "pick":       "● Pick",
    "annotation": "● Note",
    None:         "",
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
