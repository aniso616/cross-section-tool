from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel


class ToolIndicator(QLabel):
    """Single active-tool pill shown top-left of the canvas.  Hidden when idle."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setStyleSheet("""
            QLabel {
                color: rgba(120, 180, 255, 230);
                font-size: 12px;
                font-family: 'Inter', 'Helvetica Neue', sans-serif;
                background-color: rgba(18, 18, 30, 200);
                border: 1px solid rgba(80, 130, 200, 160);
                border-radius: 5px;
                padding: 3px 10px;
            }
        """)
        self.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.hide()

    def set_tool(self, tool):
        from section_tool.interaction.tool_manager import TOOL_LABELS
        label = TOOL_LABELS.get(tool, "")
        if label:
            self.setText(label)
            self.show()
        else:
            self.hide()
