"""MinimapOverlay — small floating map view overlaid on the section canvas."""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, QObject, QTimer, Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget


class _ResizeFilter(QObject):
    """Repositions *target* whenever *watched* emits a resize event."""

    def __init__(self, target: "MinimapOverlay", parent=None):
        super().__init__(parent)
        self._target = target

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Resize:
            self._target.update_position()
        return False


class MinimapOverlay(QWidget):
    """Small matplotlib map overlay, game-HUD style, on the section canvas.

    Renders section traces, well collars, and seismic extents.
    Parent should be the SectionView widget so the overlay moves with it.
    """

    _MARGIN   = 10
    _W        = 220
    _H        = 180

    def __init__(self, parent_section_view: QWidget, state):
        super().__init__(parent_section_view)
        self._state = state
        self.setFixedSize(self._W, self._H)
        self.setWindowFlags(Qt.WindowType.SubWindow)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setStyleSheet("""
            MinimapOverlay {
                background-color: #1A1A20;
                border: 1px solid #3A3A50;
                border-radius: 4px;
            }
        """)

        # Matplotlib figure (small DPI keeps it crisp)
        self._fig = Figure(figsize=(self._W / 96, self._H / 96), dpi=96)
        self._fig.patch.set_facecolor("#1A1A20")
        self._fig.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.08)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor("#1A1A20")
        self._mpl_canvas = FigureCanvasQTAgg(self._fig)
        self._mpl_canvas.setParent(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self._mpl_canvas)

        # Throttled render: update no more than once per second
        self._dirty = True
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(800)
        self._timer.timeout.connect(self._do_render)

        # Event filter so we reposition when the parent resizes
        self._resize_filter = _ResizeFilter(self, self)
        parent_section_view.installEventFilter(self._resize_filter)

        self.update_position()
        self.show()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_position(self):
        parent = self.parent()
        if parent is None:
            return
        m = self._MARGIN
        self.move(m, parent.height() - self._H - m)
        self.raise_()

    def schedule_update(self):
        self._dirty = True
        if not self._timer.isActive():
            self._timer.start()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _do_render(self):
        if not self._dirty:
            return
        self._dirty = False
        self._ax.clear()
        self._ax.set_facecolor("#1A1A20")
        for spine in self._ax.spines.values():
            spine.set_color("#3A3A50")
        self._ax.tick_params(colors="#666680", labelsize=5, which="both")

        proj = self._state.project
        any_data = False

        # Seismic extents
        for ref in proj.seismic_refs:
            xmn, xmx = ref.extent_x_min, ref.extent_x_max
            ymn, ymx = ref.extent_y_min, ref.extent_y_max
            if xmx != xmn:
                xs = [xmn, xmx, xmx, xmn, xmn]
                ys = [ymn, ymn, ymx, ymx, ymn]
                self._ax.fill(xs, ys, alpha=0.12, color="#FF8C00")
                self._ax.plot(xs, ys, color="#FF8C00", linewidth=0.6,
                              linestyle="--", alpha=0.6)
                any_data = True

        # Section traces
        active = self._state.active_section
        for sec in proj.sections:
            nodes = sec.nodes
            if len(nodes) < 2:
                continue
            xs = nodes[:, 0]
            ys = nodes[:, 1]
            is_active = (sec is active)
            color = "#4A9EFF" if is_active else "#555568"
            lw    = 2.0      if is_active else 0.8
            self._ax.plot(xs, ys, color=color, linewidth=lw, solid_capstyle="round")
            # Endpoint markers
            self._ax.plot(xs[[0, -1]], ys[[0, -1]], "o",
                          color=color, markersize=3 if is_active else 2)
            any_data = True

        # Well collars
        for well in proj.wells:
            sx = well.deviation.surface_x
            sy = well.deviation.surface_y
            self._ax.plot(sx, sy, "^", color="#FF8C42", markersize=4, zorder=5)
            any_data = True

        if not any_data:
            self._ax.text(
                0.5, 0.5, "No data",
                ha="center", va="center", fontsize=7,
                color="#555568", transform=self._ax.transAxes,
            )

        # Label
        self._ax.set_title("Map", fontsize=6, color="#888898", pad=2)
        self._mpl_canvas.draw_idle()
