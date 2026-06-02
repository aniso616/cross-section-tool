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

    def __init__(self, parent, *, compact: bool = False):
        super().__init__(parent)
        self.setFixedHeight(self.HEIGHT)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._x_min = 0.0
        self._x_max = 10000.0
        self._compact = compact          # corner geographic bar (map) vs full strip (section)
        self._mpp = None                 # metres per pixel, for compact mode
        self._font  = QFont("JetBrains Mono", 9)

    def set_range(self, x_min: float, x_max: float):
        self._x_min = x_min
        self._x_max = x_max
        self.update()

    def set_compact_scale(self, x_min: float, x_max: float, canvas_px_width: float):
        """Drive the compact corner bar: a pixel-accurate geographic scale.

        *canvas_px_width* is the full map canvas width in logical pixels; the
        E-extent (x_max - x_min) spans it, giving metres-per-pixel.
        """
        self._x_min = x_min
        self._x_max = x_max
        self._mpp = (x_max - x_min) / canvas_px_width if canvas_px_width else None
        self.update()

    def paintEvent(self, event):
        if self._compact:
            self._paint_compact()
            return
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

    def _paint_compact(self):
        """Bottom-right geographic scale bar: ruled line of a round distance."""
        if self._mpp is None or self._mpp <= 0:
            return
        p  = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        p.setFont(self._font)
        fm = p.fontMetrics()
        w  = self.width()

        target_px = w * 0.7                      # aim for ~70% of the widget width
        bar_len_m = self._round_distance(target_px * self._mpp)
        bar_px    = int(bar_len_m / self._mpp)
        bar_px    = min(bar_px, w - 8)

        x2 = w - 4
        x1 = x2 - bar_px
        by = self.HEIGHT - 7                     # line near the bottom

        p.setPen(QPen(self.C_LINE_, 1))
        p.drawLine(x1, by, x2, by)
        p.drawLine(x1, by - 4, x1, by + 4)
        p.drawLine(x2, by - 4, x2, by + 4)

        label = self._format_distance(bar_len_m)
        lw    = fm.horizontalAdvance(label)
        p.setPen(self.C_LBL_)
        p.drawText(x1 + (bar_px - lw) // 2, 0, lw, by - 4,
                   Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft, label)

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
