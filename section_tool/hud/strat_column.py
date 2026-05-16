from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QWidget


class StratColumn(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._formations = []

    def update_formations(self, formations):
        self._formations = formations
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(18, 18, 24, 170))
        if not self._formations:
            p.setPen(QColor(100, 100, 115))
            p.setFont(QFont("Courier New", 8))
            p.drawText(self.rect().adjusted(2, 4, -2, -4),
                       Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
                       "Strat")
