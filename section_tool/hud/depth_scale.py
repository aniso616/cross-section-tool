from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QWidget


class DepthScale(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        # Opaque, not translucent — we paint our own solid dark background.
        self.setStyleSheet("background-color: #16161C;")
        self._z_min = 0.0
        self._z_max = 5000.0

    def update_range(self, z_min: float, z_max: float):
        self._z_min = z_min
        self._z_max = z_max
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(22, 22, 28))
        span = self._z_max - self._z_min
        if span <= 0:
            return
        p.setPen(QColor(140, 140, 158))
        p.setFont(QFont("Courier New", 7))
        h = max(self.height(), 1)
        n = 6
        for i in range(n + 1):
            frac = i / n
            depth = self._z_min + frac * span
            y = int(frac * h)
            label = f"{depth:,.0f}"
            p.drawText(2, min(y + 9, h - 2), label)
            p.drawLine(self.width() - 4, y, self.width() - 1, y)
