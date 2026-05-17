"""FormationStrip — right-edge stratigraphic column and formation badge."""
from __future__ import annotations

from typing import NamedTuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from section_tool.style import C_RULE, C_BG


class FormationBand(NamedTuple):
    """Pre-computed depth band for one formation on this section."""
    top_m:  float
    base_m: float
    name:   str
    color:  tuple   # (R, G, B)


class FormationStrip(QWidget):
    """Colored formation bands, current-formation badge at top."""

    WIDTH   = 60
    BADGE_H = 22

    C_RULE_       = QColor(*C_RULE)
    C_NAME        = QColor(230, 230, 230, 200)
    C_BADGE_TEXT  = QColor(240, 240, 240, 230)
    C_EMPTY_BG    = QColor(*C_BG)

    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedWidth(self.WIDTH)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._bands        = []   # list[FormationBand]
        self._depth_min    = 0.0
        self._depth_max    = 5000.0
        self._cursor_depth = None
        self._font_name    = QFont("Inter", 8)
        self._font_badge   = QFont("Inter", 9)

    def set_stratigraphy(self, bands: list[FormationBand],
                         depth_min: float, depth_max: float):
        self._bands     = bands
        self._depth_min = depth_min
        self._depth_max = depth_max
        self.update()

    def set_cursor_depth(self, depth_m):
        self._cursor_depth = depth_m
        self.update()

    def _depth_to_y(self, depth_m: float) -> int:
        rng = self._depth_max - self._depth_min or 1
        return self.BADGE_H + int(
            (depth_m - self._depth_min) / rng * (self.height() - self.BADGE_H)
        )

    def _band_at_depth(self, depth_m: float) -> FormationBand | None:
        for b in self._bands:
            if b.top_m <= depth_m < b.base_m:
                return b
        return None

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        h, w = self.height(), self.WIDTH

        # Formation bands
        for band in self._bands:
            y1 = max(self.BADGE_H, self._depth_to_y(band.top_m))
            y2 = min(h, self._depth_to_y(band.base_m))
            if y2 <= y1:
                continue
            r, g, b = band.color[:3]
            p.fillRect(0, y1, w, y2 - y1, QColor(r, g, b, 175))
            if y2 - y1 > 16:
                p.setPen(self.C_NAME)
                p.setFont(self._font_name)
                fm = p.fontMetrics()
                p.drawText(0, y1, w, y2 - y1,
                           Qt.AlignmentFlag.AlignCenter,
                           fm.elidedText(band.name, Qt.TextElideMode.ElideRight,
                                         w - 6))

        # Left rule line (below badge)
        p.setPen(QPen(self.C_RULE_, 1))
        p.drawLine(0, self.BADGE_H, 0, h)

        # Badge — current formation at cursor depth
        current = (self._band_at_depth(self._cursor_depth)
                   if self._cursor_depth is not None else None)
        if current:
            r, g, b = current.color[:3]
            p.fillRect(0, 0, w, self.BADGE_H, QColor(r, g, b, 220))
            p.setPen(self.C_BADGE_TEXT)
            p.setFont(self._font_badge)
            fm = p.fontMetrics()
            p.drawText(0, 0, w, self.BADGE_H,
                       Qt.AlignmentFlag.AlignCenter,
                       fm.elidedText(current.name, Qt.TextElideMode.ElideRight,
                                     w - 6))
        else:
            p.fillRect(0, 0, w, self.BADGE_H, self.C_EMPTY_BG)

        # Rule line separating badge from column
        p.setPen(QPen(self.C_RULE_, 1))
        p.drawLine(0, self.BADGE_H, w, self.BADGE_H)
