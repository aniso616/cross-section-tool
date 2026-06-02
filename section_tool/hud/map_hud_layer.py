"""MapHUDLayer — transparent edge-reference overlay for the map canvas.

Mirrors the section's HUDLayer but for plan view: a left-edge Northing ruler, a
bottom-edge Easting ruler, and a corner geographic scale bar. Reuses the shared
AxisRuler and ScaleBar so the map and section stay seamless by construction
(same typography, tokens, tick treatment). Transparent to mouse events — all
interaction passes through to the map canvas beneath.

The rulers span the full canvas dimension (not an inset sub-range) so their
ticks align exactly with the data columns/rows of the full-bleed map.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from section_tool.hud.axis_ruler import AxisRuler
from section_tool.hud.scale_bar import ScaleBar


def _fmt_coord(v: float) -> str:
    """Comma-grouped integer metres, matching the map readout."""
    return f"{v:,.0f}"


class MapHUDLayer(QWidget):
    """Edge rulers + scale bar overlaid on the full-bleed map canvas."""

    N_RULER_W = 84                      # wide enough for 7-digit comma northings
    E_RULER_H = AxisRuler.THICKNESS     # 52
    SB_W      = 150
    SB_H      = ScaleBar.HEIGHT         # 26
    M         = 10

    def __init__(self, parent, map_view):
        super().__init__(parent)
        self._map = map_view
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")

        self.n_ruler = AxisRuler(self, orientation="vertical", inverted=True,
                                 formatter=_fmt_coord, thickness=self.N_RULER_W)
        self.e_ruler = AxisRuler(self, orientation="horizontal",
                                 formatter=_fmt_coord)
        self.scale_bar = ScaleBar(self, compact=True)
        self.scale_bar.setFixedWidth(self.SB_W)

        # Reflow whenever the canvas redraws — draw_event fires after matplotlib
        # has applied the equal-aspect/datalim adjustment, so the limits we read
        # are the ones actually on screen (render, resize, pan, zoom all covered).
        map_view.canvas.mpl_connect("draw_event", lambda _evt: self.refresh())

    # ------------------------------------------------------------------

    def resizeEvent(self, event):
        self._layout_children()
        super().resizeEvent(event)

    def _layout_children(self):
        w, h = self.width(), self.height()
        nw, eh = self.N_RULER_W, self.E_RULER_H
        # Full-span rulers so ticks align with the full-bleed data; they overlap
        # only in the bottom-left corner (empty padding region in practice).
        self.n_ruler.setGeometry(0, 0, nw, h)
        self.e_ruler.setGeometry(0, h - eh, w, eh)
        self.scale_bar.setGeometry(w - self.SB_W - self.M, h - eh - self.SB_H,
                                   self.SB_W, self.SB_H)

    # ------------------------------------------------------------------

    def refresh(self):
        """Pull the current map extent and push it into the rulers + scale bar."""
        ax = self._map.axes
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        self.e_ruler.set_view_range(x_min, x_max)
        self.n_ruler.set_view_range(y_min, y_max)
        self.scale_bar.set_compact_scale(
            x_min, x_max, max(self._map.canvas.width(), 1))
