"""DepthRuler — the section's left-edge depth track.

Thin shim over :class:`~section_tool.hud.axis_ruler.AxisRuler`: a vertical ruler
with fixed 500 / 100 m intervals and depth/elevation formatting, plus the
section-specific formation color-chaser strip on the left edge. All generic
ruler behaviour (ticks, labels, cursor line) now lives in AxisRuler, shared with
the map's E/N rulers.
"""
from PySide6.QtGui import QColor, QFontMetrics, QPainter

from section_tool.hud.axis_ruler import AxisRuler


class DepthRuler(AxisRuler):
    """Vertical AxisRuler + formation color-chaser (section depth track)."""

    WIDTH          = AxisRuler.THICKNESS   # back-compat constant
    MAJOR_INTERVAL = 500
    MINOR_INTERVAL = 100
    BAND_W         = 8                     # formation color strip width (px)

    def __init__(self, parent):
        super().__init__(
            parent,
            orientation="vertical",
            formatter=lambda d: f"{int(d)}",
            cursor_formatter=lambda d: f"{d:,.0f} m",
            inverted=False,                # depth increases downward (lo at top)
            major_interval=self.MAJOR_INTERVAL,
            minor_interval=self.MINOR_INTERVAL,
        )
        self._formations: list = []        # FormationBand list for color chaser

    # ------------------------------------------------------------------
    # Depth-specific API (preserved for existing callers)
    # ------------------------------------------------------------------

    def set_formations(self, bands: list) -> None:
        """Update formation color bands.  bands = list of FormationBand."""
        self._formations = bands
        self.update()

    def set_cursor_depth(self, depth_m) -> None:
        self.set_cursor_value(depth_m)

    # ------------------------------------------------------------------
    # Formation color chaser — painted behind the ticks
    # ------------------------------------------------------------------

    def _paint_underlay(self, p: QPainter, fm: QFontMetrics) -> None:
        h = self.height()
        for band in self._formations:
            y1 = self._value_to_px(band.top_m, h)
            y2 = self._value_to_px(band.base_m, h)
            if y2 <= y1:
                continue
            r, g, b = band.color[:3]
            p.fillRect(0, y1, self.BAND_W, y2 - y1, QColor(r, g, b, 200))
