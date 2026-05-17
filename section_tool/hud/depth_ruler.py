"""DepthRuler — wireline-log-style depth track with cursor line."""
import math

from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QWidget

from section_tool.style import C_RULE, C_LABEL, C_READ, C_BG


class DepthRuler(QWidget):
    """Left-edge depth track — tick marks, labels, cursor line."""

    WIDTH          = 52
    MAJOR_INTERVAL = 500
    MINOR_INTERVAL = 100
    MAJOR_TICK_W   = 16
    MINOR_TICK_W   = 6

    C_RULE_   = QColor(*C_RULE)
    C_LABEL_  = QColor(*C_LABEL)
    C_CURSOR  = QColor(*C_READ)
    C_BG_LBL  = QColor(*C_BG)

    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedWidth(self.WIDTH)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._depth_min    = 0.0
        self._depth_max    = 5000.0
        self._cursor_depth = None
        self._formations: list = []   # FormationBand list for color chaser
        self._font = QFont("JetBrains Mono", 9)

    def set_view_range(self, depth_min: float, depth_max: float):
        self._depth_min = depth_min
        self._depth_max = depth_max
        self.update()

    def set_formations(self, bands: list) -> None:
        """Update formation color bands.  bands = list of FormationBand namedtuples."""
        self._formations = bands
        self.update()

    def set_cursor_depth(self, depth_m):
        self._cursor_depth = depth_m
        self.update()

    def _depth_to_y(self, depth_m: float) -> int:
        rng = self._depth_max - self._depth_min or 1
        return int((depth_m - self._depth_min) / rng * self.height())

    @staticmethod
    def _first_tick(depth_min: float, interval: float) -> float:
        return math.ceil(depth_min / interval) * interval

    def paintEvent(self, event):
        p  = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        h  = self.height()
        rx = self.WIDTH - 1
        BAND_W = 8   # formation color strip width on left edge

        # Formation color chaser: thin colored strip on left edge
        for band in self._formations:
            y1 = self._depth_to_y(band.top_m)
            y2 = self._depth_to_y(band.base_m)
            if y2 <= y1:
                continue
            r, g, b = band.color[:3]
            p.fillRect(0, y1, BAND_W, y2 - y1, QColor(r, g, b, 200))

        p.setFont(self._font)
        fm = QFontMetrics(self._font)

        # Vertical rule line
        p.setPen(QPen(self.C_RULE_, 1))
        p.drawLine(rx, 0, rx, h)

        # Minor ticks
        d = self._first_tick(self._depth_min, self.MINOR_INTERVAL)
        while d <= self._depth_max:
            y = self._depth_to_y(d)
            p.drawLine(rx - self.MINOR_TICK_W, y, rx, y)
            d += self.MINOR_INTERVAL

        # Major ticks + labels
        d = self._first_tick(self._depth_min, self.MAJOR_INTERVAL)
        while d <= self._depth_max:
            y     = self._depth_to_y(d)
            label = f"{int(d)}"
            lw    = fm.horizontalAdvance(label)
            lx    = rx - self.MAJOR_TICK_W - lw - 4

            p.setPen(QPen(self.C_RULE_, 1))
            p.drawLine(rx - self.MAJOR_TICK_W, y, rx, y)

            p.setPen(self.C_LABEL_)
            p.drawText(lx, y - fm.ascent() // 2,
                       lw, fm.height(), Qt.AlignmentFlag.AlignLeft, label)
            d += self.MAJOR_INTERVAL

        # Cursor line + floating label
        if self._cursor_depth is not None:
            y     = self._depth_to_y(self._cursor_depth)
            label = f"{self._cursor_depth:,.0f} m"
            lw    = fm.horizontalAdvance(label)
            lh    = fm.height()
            pad   = 3

            p.setPen(QPen(self.C_CURSOR, 1))
            p.drawLine(0, y, rx, y)

            bg = QRect(
                rx - self.MAJOR_TICK_W - lw - pad * 2 - 2,
                y - lh // 2 - pad,
                lw + pad * 2,
                lh + pad * 2,
            )
            p.fillRect(bg, self.C_BG_LBL)
            p.setPen(self.C_CURSOR)
            p.drawText(bg, Qt.AlignmentFlag.AlignCenter, label)
