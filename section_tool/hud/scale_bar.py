"""ScaleBar — bottom-edge horizontal scale, map convention."""
import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from section_tool.style import C_RULE, C_LABEL


class ScaleBar(QWidget):
    """One ruled line, endpoint coordinates, centre distance badge."""

    HEIGHT = 26
    C_LINE_ = QColor(*C_RULE)
    C_LBL_  = QColor(*C_LABEL)

    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedHeight(self.HEIGHT)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._x_min = 0.0
        self._x_max = 10000.0
        self._font  = QFont("JetBrains Mono", 9)

    def set_range(self, x_min: float, x_max: float):
        self._x_min = x_min
        self._x_max = x_max
        self.update()

    def paintEvent(self, event):
        p  = QPainter(self)
        p.setFont(self._font)
        fm = p.fontMetrics()
        w  = self.width()
        by = self.HEIGHT // 2

        p.setPen(QPen(self.C_LINE_, 1))
        p.drawLine(0, by, w, by)
        p.drawLine(0,     by - 4, 0,     by + 4)
        p.drawLine(w - 1, by - 4, w - 1, by + 4)

        p.setPen(self.C_LBL_)
        p.drawText(4, 0, 100, self.HEIGHT,
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   self._format_coord(self._x_min))
        p.drawText(w - 104, 0, 100, self.HEIGHT,
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                   self._format_coord(self._x_max))

        extent   = self._x_max - self._x_min
        bar_dist = self._round_distance(extent * 0.35)
        badge    = f"─  {self._format_distance(bar_dist)}  ─"
        bw       = fm.horizontalAdvance(badge)
        p.drawText((w - bw) // 2, 0, bw, self.HEIGHT,
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   badge)

    @staticmethod
    def _round_distance(d: float) -> float:
        if d <= 0:
            return 1.0
        mag = 10 ** math.floor(math.log10(d))
        for f in (1, 2, 2.5, 5, 10):
            c = mag * f
            if c >= d:
                return c
        return mag * 10

    @staticmethod
    def _format_distance(d: float) -> str:
        if d >= 1000:
            v = d / 1000
            return f"{v:.0f} km" if v == int(v) else f"{v:.1f} km"
        return f"{int(d)} m"

    @staticmethod
    def _format_coord(x: float) -> str:
        if abs(x) >= 1000:
            v = x / 1000
            return f"{v:.1f} km"
        return f"{int(x)} m"
