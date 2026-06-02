"""AxisRuler — generic edge ruler (ticks, labels, optional cursor line).

Orientation- and quantity-agnostic: it knows nothing about what it measures.
Configure it with an orientation, a value→string formatter, a pixel direction,
and either fixed tick intervals or automatic "nice" intervals. The section
depth ruler (vertical, fixed 500/100 m) and the map E/N rulers
(horizontal/vertical, auto intervals) are all instances of this one widget, so
their typography and tick treatment stay seamless by construction.

Tokens and font come from :mod:`section_tool.style` — no hardcoded colors.
"""
from __future__ import annotations

import math
from typing import Callable, Literal

from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QWidget

from section_tool.style import C_RULE, C_LABEL, C_READ, C_BG


def nice_interval(span: float, target_ticks: int = 5) -> float:
    """Round ``span / target_ticks`` to a 1 / 2 / 2.5 / 5 × 10ⁿ value."""
    if span <= 0:
        return 1.0
    raw = span / max(target_ticks, 1)
    mag = 10 ** math.floor(math.log10(raw))
    for f in (1, 2, 2.5, 5, 10):
        if mag * f >= raw:
            return mag * f
    return mag * 10


class AxisRuler(QWidget):
    """One edge ruler: rule line, minor/major ticks + labels, cursor readout."""

    THICKNESS  = 52     # px: widget width if vertical, height if horizontal
    MAJOR_TICK = 16
    MINOR_TICK = 6

    C_RULE_  = QColor(*C_RULE)
    C_LABEL_ = QColor(*C_LABEL)
    C_CURSOR = QColor(*C_READ)
    C_BG_LBL = QColor(*C_BG)

    def __init__(
        self,
        parent,
        *,
        orientation: Literal["vertical", "horizontal"] = "vertical",
        formatter: Callable[[float], str] | None = None,
        cursor_formatter: Callable[[float], str] | None = None,
        inverted: bool = False,
        major_interval: float | None = None,
        minor_interval: float | None = None,
        target_major_ticks: int = 5,
        thickness: int | None = None,
    ) -> None:
        super().__init__(parent)
        self._orient = orientation
        self._fmt = formatter or (lambda v: f"{v:,.0f}")
        self._cursor_fmt = cursor_formatter or self._fmt
        self._inverted = inverted
        self._major_interval = major_interval     # None → auto nice interval
        self._minor_interval = minor_interval
        self._target = target_major_ticks
        self._lo = 0.0
        self._hi = 1000.0
        self._cursor: float | None = None
        self._font = QFont("JetBrains Mono", 9)

        self._thickness = thickness if thickness is not None else self.THICKNESS
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        if orientation == "vertical":
            self.setFixedWidth(self._thickness)
        else:
            self.setFixedHeight(self._thickness)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_view_range(self, lo: float, hi: float) -> None:
        self._lo, self._hi = float(lo), float(hi)
        self.update()

    def set_cursor_value(self, v: float | None) -> None:
        self._cursor = v
        self.update()

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _value_to_px(self, v: float, length: int) -> int:
        rng = self._hi - self._lo or 1.0
        frac = (v - self._lo) / rng
        if self._inverted:
            frac = 1.0 - frac
        return int(frac * length)

    def _intervals(self) -> tuple[float, float]:
        if self._major_interval is not None:
            major = self._major_interval
            minor = self._minor_interval or major / 5
        else:
            major = nice_interval(self._hi - self._lo, self._target)
            minor = major / 5
        return major, minor

    @staticmethod
    def _first_tick(lo: float, interval: float) -> float:
        return math.ceil(lo / interval) * interval

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        p.setFont(self._font)
        fm = QFontMetrics(self._font)
        self._paint_underlay(p, fm)          # subclass hook (e.g. formation chaser)
        if self._orient == "vertical":
            self._paint_vertical(p, fm)
        else:
            self._paint_horizontal(p, fm)

    def _paint_underlay(self, p: QPainter, fm: QFontMetrics) -> None:
        """Hook for subclasses to draw behind the ticks. Default: nothing."""

    def _paint_vertical(self, p: QPainter, fm: QFontMetrics) -> None:
        h     = self.height()
        rx    = self.width() - 1            # rule on canvas-adjacent (right) edge
        major, minor = self._intervals()

        p.setPen(QPen(self.C_RULE_, 1))
        p.drawLine(rx, 0, rx, h)

        d = self._first_tick(self._lo, minor)
        while d <= self._hi:
            y = self._value_to_px(d, h)
            p.drawLine(rx - self.MINOR_TICK, y, rx, y)
            d += minor

        d = self._first_tick(self._lo, major)
        while d <= self._hi:
            y     = self._value_to_px(d, h)
            label = self._fmt(d)
            lw    = fm.horizontalAdvance(label)
            p.setPen(QPen(self.C_RULE_, 1))
            p.drawLine(rx - self.MAJOR_TICK, y, rx, y)
            p.setPen(self.C_LABEL_)
            p.drawText(rx - self.MAJOR_TICK - lw - 4, y - fm.ascent() // 2,
                       lw, fm.height(), Qt.AlignmentFlag.AlignLeft, label)
            d += major

        if self._cursor is not None:
            y     = self._value_to_px(self._cursor, h)
            label = self._cursor_fmt(self._cursor)
            lw    = fm.horizontalAdvance(label)
            lh    = fm.height()
            pad   = 3
            p.setPen(QPen(self.C_CURSOR, 1))
            p.drawLine(0, y, rx, y)
            bg = QRect(rx - self.MAJOR_TICK - lw - pad * 2 - 2, y - lh // 2 - pad,
                       lw + pad * 2, lh + pad * 2)
            p.fillRect(bg, self.C_BG_LBL)
            p.setPen(self.C_CURSOR)
            p.drawText(bg, Qt.AlignmentFlag.AlignCenter, label)

    def _paint_horizontal(self, p: QPainter, fm: QFontMetrics) -> None:
        w     = self.width()
        ty    = 0                           # rule on canvas-adjacent (top) edge
        major, minor = self._intervals()

        p.setPen(QPen(self.C_RULE_, 1))
        p.drawLine(0, ty, w, ty)

        d = self._first_tick(self._lo, minor)
        while d <= self._hi:
            x = self._value_to_px(d, w)
            p.drawLine(x, ty, x, ty + self.MINOR_TICK)
            d += minor

        d = self._first_tick(self._lo, major)
        while d <= self._hi:
            x     = self._value_to_px(d, w)
            label = self._fmt(d)
            lw    = fm.horizontalAdvance(label)
            p.setPen(QPen(self.C_RULE_, 1))
            p.drawLine(x, ty, x, ty + self.MAJOR_TICK)
            p.setPen(self.C_LABEL_)
            p.drawText(x - lw // 2, ty + self.MAJOR_TICK + 2,
                       lw, fm.height(), Qt.AlignmentFlag.AlignHCenter, label)
            d += major

        if self._cursor is not None:
            x     = self._value_to_px(self._cursor, w)
            label = self._cursor_fmt(self._cursor)
            lw    = fm.horizontalAdvance(label)
            lh    = fm.height()
            pad   = 3
            p.setPen(QPen(self.C_CURSOR, 1))
            p.drawLine(x, ty, x, self.height())
            bg = QRect(x - lw // 2 - pad, ty + self.MAJOR_TICK + 1,
                       lw + pad * 2, lh + pad * 2)
            p.fillRect(bg, self.C_BG_LBL)
            p.setPen(self.C_CURSOR)
            p.drawText(bg, Qt.AlignmentFlag.AlignCenter, label)
