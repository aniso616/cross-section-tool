"""ToolIndicator — minimal active-tool label, top-left corner."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from section_tool.style import C_READ, C_BG, C_RULE


class ToolIndicator(QWidget):
    """Active tool label.  Present only when a special tool is active."""

    C_TEXT = QColor(*C_READ)
    C_BG_  = QColor(*C_BG)
    C_RULE_= QColor(*C_RULE)

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._label = ""
        self._font  = QFont("Inter", 10)
        self.hide()

    def set_tool(self, tool):
        # Driven by AppState.tool_changed, so use the AppState-keyed labels
        # (covers construction tools, which bypass the ToolManager).
        from section_tool.interaction.tool_manager import APPSTATE_TOOL_LABELS
        self._label = APPSTATE_TOOL_LABELS.get(tool, "")
        if self._label:
            self.show()
        else:
            self.hide()
        self.update()

    def paintEvent(self, event):
        if not self._label:
            return
        p  = QPainter(self)
        p.setFont(self._font)
        fm  = p.fontMetrics()
        tw  = fm.horizontalAdvance(self._label)
        th  = fm.height()
        pad = 5

        bw = tw + pad * 2
        bh = th + pad * 2
        p.fillRect(0, 0, bw, bh, self.C_BG_)
        p.setPen(QPen(self.C_RULE_, 1))
        p.drawRect(0, 0, bw - 1, bh - 1)
        p.setPen(self.C_TEXT)
        p.drawText(pad, pad, tw, th, Qt.AlignmentFlag.AlignLeft, self._label)
