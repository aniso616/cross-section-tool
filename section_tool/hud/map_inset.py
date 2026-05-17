"""MapInset — lower-left map inset with section-line overlay."""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

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

    def schedule_update(self):
        self._dirty = True
        if not self._timer.isActive():
            self._timer.start()

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
            self._ax.plot(sx, sy, "^", color="#CC7744", markersize=3, zorder=5)
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
