from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QWidget


class Minimap(QWidget):
    section_line_moved = Signal(object)

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self._label  = "Map"
        self._pixmap = None

    def set_label(self, label: str):
        self._label = label
        self.update()

    def update_content(self, pixmap):
        self._pixmap = pixmap
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(1, 1, -1, -1)
        p.setBrush(QColor(18, 18, 22, 200))
        p.setPen(QColor(65, 65, 88, 180))
        p.drawRoundedRect(r, 6, 6)
        if self._pixmap:
            p.drawPixmap(4, 4, self.width() - 8,
                         self.height() - 22, self._pixmap)
        p.setPen(QColor(130, 130, 145))
        p.setFont(QFont("Inter", 9))
        p.drawText(r.adjusted(0, 0, 0, -2),
                   Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
                   self._label)
