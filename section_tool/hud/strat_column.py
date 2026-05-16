from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QWidget


class StratColumn(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setStyleSheet("background-color: #16161C;")
        self._formations = []

    def update_formations(self, formations):
        self._formations = formations
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(22, 22, 28))
        p.setPen(QColor(100, 100, 115))
        p.setFont(QFont("Courier New", 7))
        if not self._formations:
            p.drawText(
                self.rect().adjusted(2, 4, -2, -4),
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
                "Strat",
            )
