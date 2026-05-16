from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel


class ToolIndicator(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet("""
            color: rgba(120, 180, 255, 230);
            font-size: 12px;
            font-family: 'Inter', 'Helvetica Neue', sans-serif;
            background: rgba(18, 18, 26, 170);
            border: 1px solid rgba(80, 130, 200, 140);
            border-radius: 5px;
            padding: 3px 10px;
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
