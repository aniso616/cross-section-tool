"""MapInset — lower-left map inset with section-line overlay."""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QPushButton, QWidget

from section_tool.style import BG_CANVAS, C_RULE, C_LABEL, C_DIM


class MapInset(QWidget):
    """Small map inset — drafting-log aesthetic.  Positioned by HUDLayer."""

    W, H = 200, 160

    C_BORDER_  = QColor(*C_RULE)
    C_BG_      = QColor(14, 16, 20, 220)
    C_SECTION  = QColor(200, 210, 220, 220)
    C_LABEL_   = QColor(*C_LABEL)
    C_NO_DATA  = QColor(*C_DIM)

    section_line_moved = Signal(object)

    def __init__(self, parent, state):
        super().__init__(parent)
        self._state = state
        self.setFixedSize(self.W, self.H)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._font = QFont("JetBrains Mono", 8)
        self._crosshair_x: float | None = None
        self._crosshair_y: float | None = None
        self._crosshair_artists: list = []
        self._collapsed = False
        self._expanded_h = self.H

        # Collapse/expand button in top-right corner
        self._btn = QPushButton("—", self)
        self._btn.setFixedSize(16, 14)
        self._btn.setStyleSheet("""
            QPushButton {
                background: rgba(50,50,60,200);
                color: #aaaaaa;
                border: none;
                border-radius: 2px;
                font-size: 9px;
                padding: 0;
            }
            QPushButton:hover { background: rgba(80,80,100,220); color: #dddddd; }
        """)
        self._btn.move(self.W - 18, 2)
        self._btn.clicked.connect(self._toggle_collapse)
        self._btn.raise_()

        # Own matplotlib figure for map content
        self._fig = Figure(figsize=(self.W / 96, self.H / 96), dpi=96)
        self._fig.patch.set_facecolor(BG_CANVAS)
        self._fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        self._ax = self._fig.add_subplot(111)
        self._configure_ax()
        self._mpl = FigureCanvasQTAgg(self._fig)
        self._mpl.setParent(self)
        self._mpl.resize(self.W, self.H)

        self._dirty = True
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._render)

    def _configure_ax(self):
        self._ax.set_facecolor(BG_CANVAS)
        for spine in self._ax.spines.values():
            spine.set_visible(False)
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._ax.xaxis.set_visible(False)
        self._ax.yaxis.set_visible(False)

    def _toggle_collapse(self) -> None:
        self._collapsed = not self._collapsed
        if self._collapsed:
            self._mpl.setVisible(False)
            self.setFixedSize(self.W, 18)
            self._btn.setText("◻")
        else:
            self._mpl.setVisible(True)
            self.setFixedSize(self.W, self._expanded_h)
            self._btn.setText("—")
        self._btn.move(self.W - 18, 2)

    def schedule_update(self):
        self._dirty = True
        if not self._timer.isActive():
            self._timer.start()

    def update_crosshair(self, map_x: float, map_y: float) -> None:
        """Draw a subtle crosshair at geographic position without a full re-render."""
        self._crosshair_x = map_x
        self._crosshair_y = map_y
        for a in self._crosshair_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._crosshair_artists.clear()
        try:
            xlim = self._ax.get_xlim()
            ylim = self._ax.get_ylim()
            xs = (xlim[1] - xlim[0]) * 0.025  # ±2.5% crosshair arm
            ys = (ylim[1] - ylim[0]) * 0.025

            # Solid short crosshair arms at the point
            h_arm, = self._ax.plot(
                [map_x - xs, map_x + xs], [map_y, map_y],
                color="#FF6666", lw=1.5, solid_capstyle="round", zorder=21)
            v_arm, = self._ax.plot(
                [map_x, map_x], [map_y - ys, map_y + ys],
                color="#FF6666", lw=1.5, solid_capstyle="round", zorder=21)
            # Faint dashed lines extending to edges
            h_dash, = self._ax.plot(
                [xlim[0], xlim[1]], [map_y, map_y],
                color="#FF6666", lw=0.5, ls="--", alpha=0.35, zorder=19)
            v_dash, = self._ax.plot(
                [map_x, map_x], [ylim[0], ylim[1]],
                color="#FF6666", lw=0.5, ls="--", alpha=0.35, zorder=19)
            # Tiny dot
            dot, = self._ax.plot(map_x, map_y, "o",
                                 color="#FF6666", ms=3, zorder=22)
            self._crosshair_artists = [h_arm, v_arm, h_dash, v_dash, dot]
        except Exception:
            pass
        self._mpl.draw_idle()

    def clear_crosshair(self) -> None:
        for a in self._crosshair_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._crosshair_artists.clear()
        self._mpl.draw_idle()

    def paintEvent(self, event):
        # Draw 1px border last (on top of matplotlib content)
        super().paintEvent(event)
        p = QPainter(self)
        p.setPen(QPen(self.C_BORDER_, 1))
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))

        if not self._dirty and not self._timer.isActive():
            return
        # Label bottom-left
        p.setFont(self._font)
        p.setPen(self.C_LABEL_)
        p.drawText(self.rect().adjusted(4, 0, 0, -3),
                   Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
                   "Map")

    def _render(self):
        if not self._dirty:
            return
        self._dirty = False
        self._ax.clear()
        self._configure_ax()

        proj    = self._state.project
        active  = self._state.active_section
        any_data = False

        # Vector overlays (shapefiles etc.) — drawn first, behind everything
        for lyr in self._state.get_vector_layers():
            col      = lyr.get("color", "#FFAA00")
            features = lyr.get("features", [])
            for feat in features:
                geom   = feat.get("geometry") or {}
                gtype  = geom.get("type", "")
                coords = geom.get("coordinates", [])
                try:
                    if gtype in ("LineString", "3D LineString"):
                        xs = [c[0] for c in coords]; ys = [c[1] for c in coords]
                        self._ax.plot(xs, ys, color=col, lw=0.6, alpha=0.6, zorder=1)
                    elif gtype in ("MultiLineString", "3D MultiLineString"):
                        for ln in coords:
                            xs = [c[0] for c in ln]; ys = [c[1] for c in ln]
                            self._ax.plot(xs, ys, color=col, lw=0.6, alpha=0.6, zorder=1)
                    elif gtype in ("Polygon", "3D Polygon"):
                        ring = coords[0]
                        xs = [c[0] for c in ring]; ys = [c[1] for c in ring]
                        self._ax.fill(xs, ys, color=col, alpha=0.08, zorder=0)
                        self._ax.plot(xs, ys, color=col, lw=0.5, alpha=0.6, zorder=1)
                    elif gtype in ("MultiPolygon", "3D MultiPolygon"):
                        for poly in coords:
                            ring = poly[0]
                            xs = [c[0] for c in ring]; ys = [c[1] for c in ring]
                            self._ax.fill(xs, ys, color=col, alpha=0.08, zorder=0)
                            self._ax.plot(xs, ys, color=col, lw=0.5, alpha=0.6, zorder=1)
                    elif gtype in ("Point", "3D Point"):
                        self._ax.plot(coords[0], coords[1], "o",
                                      color=col, ms=2, zorder=2)
                    elif gtype in ("MultiPoint", "3D MultiPoint"):
                        for pt in coords:
                            self._ax.plot(pt[0], pt[1], "o",
                                          color=col, ms=2, zorder=2)
                except (IndexError, TypeError):
                    continue
            any_data = True

        # Seismic extents
        for ref in proj.seismic_refs:
            xmn, xmx = ref.extent_x_min, ref.extent_x_max
            ymn, ymx = ref.extent_y_min, ref.extent_y_max
            if xmx != xmn:
                xs = [xmn, xmx, xmx, xmn, xmn]
                ys = [ymn, ymn, ymx, ymx, ymn]
                self._ax.fill(xs, ys, alpha=0.10, color="#FF8C00")
                self._ax.plot(xs, ys, color="#FF8C00", lw=0.5,
                              linestyle="--", alpha=0.5)
                any_data = True

        # Section traces
        for sec in proj.sections:
            nodes = sec.nodes
            if len(nodes) < 2:
                continue
            is_active = (sec is active)
            col = "#B0C8E0" if is_active else "#444455"
            lw  = 1.5       if is_active else 0.7
            self._ax.plot(nodes[:, 0], nodes[:, 1],
                          color=col, lw=lw, solid_capstyle="round")
            # Endpoint open circles
            for x, y in [nodes[0], nodes[-1]]:
                self._ax.plot(x, y, "o", color=col,
                              markersize=3 if is_active else 2,
                              markerfacecolor="none", markeredgewidth=0.8)
            any_data = True

        # Well collars
        for well in proj.wells:
            sx, sy = well.deviation.surface_x, well.deviation.surface_y
            wc = getattr(well, "color", "#CC7744")
            self._ax.plot(sx, sy, "^", color=wc, markersize=3, zorder=5)
            any_data = True

        if not any_data:
            self._ax.text(0.5, 0.5, "No data", ha="center", va="center",
                          fontsize=7, color="#505060",
                          transform=self._ax.transAxes)
        else:
            self._ax.autoscale(enable=True, tight=True)
            self._ax.margins(0.1)

        self._mpl.draw_idle()
        self.update()
        # Re-draw crosshair on top after full re-render clears it
        if self._crosshair_x is not None:
            self.update_crosshair(self._crosshair_x, self._crosshair_y)
