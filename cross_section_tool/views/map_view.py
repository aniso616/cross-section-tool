from __future__ import annotations

import math

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from cross_section_tool.app_state import AppState
from cross_section_tool.core.section import Section


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HIT_PX = 10              # node / line pick tolerance in display pixels
_ACTIVE_COLOR = "#1f77b4"
_INACTIVE_COLOR = "#999999"
_ACTIVE_LW = 2.0
_INACTIVE_LW = 1.0
_NODE_MS = 7              # marker size (points)
_WELL_COLOR = "#8B4513"
_SURFACE_COLOR = "darkorange"
_SEISMIC_COLOR = "#888888"


class MapView(QWidget):
    """Plan-view (map) display of sections, wells, and data extents.

    Renders all project sections as polylines, active section highlighted.
    Supports dragging section nodes with the mouse, which updates the
    :class:`AppState` on release.

    Parameters
    ----------
    state:
        Central application state.

    Signals
    -------
    section_node_moved(int, int, float, float)
        Emitted after a drag-release with ``(section_index, node_index, x, y)``.
        The :class:`AppState` has already been updated by the time this fires.
    """

    section_node_moved = Signal(int, int, float, float)

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._drag: dict | None = None  # keys: sec_idx, node_idx, section_copy
        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self._fig = Figure(figsize=(8, 6), tight_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        self._canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self._canvas.mpl_connect("button_release_event", self._on_canvas_release)

    def _connect_signals(self) -> None:
        s = self._state
        s.project_changed.connect(self.render)
        s.section_added.connect(self._on_sections_changed)
        s.section_removed.connect(self._on_sections_changed)
        s.section_modified.connect(self._on_sections_changed)
        s.active_section_changed.connect(self._on_sections_changed)
        s.well_added.connect(self._on_wells_changed)
        s.well_removed.connect(self._on_wells_changed)
        s.well_modified.connect(self._on_wells_changed)
        s.surface_added.connect(self._on_surfaces_changed)
        s.surface_removed.connect(self._on_surfaces_changed)
        s.surface_modified.connect(self._on_surfaces_changed)
        s.seismic_ref_added.connect(self._on_seismic_changed)
        s.seismic_ref_removed.connect(self._on_seismic_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def figure(self) -> Figure:
        return self._fig

    @property
    def canvas(self) -> FigureCanvasQTAgg:
        return self._canvas

    @property
    def axes(self):
        return self._ax

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, *_args) -> None:
        """Full redraw of the map view."""
        self._ax.clear()
        self._ax.set_xlabel("Easting (m)")
        self._ax.set_ylabel("Northing (m)")
        self._ax.set_aspect("equal", adjustable="datalim")

        self._render_seismic_coverage()
        self._render_surfaces()
        self._render_sections()
        self._render_wells()
        self._canvas.draw_idle()

    def _render_sections(self) -> None:
        active = self._state.active_section
        for i, section in enumerate(self._state.project.sections):
            # During a drag, substitute the live-updated copy
            if self._drag and self._drag["sec_idx"] == i:
                display_sec = self._drag["section_copy"]
            else:
                display_sec = section

            is_active = (display_sec is section and section is active) or (
                self._drag
                and self._drag["sec_idx"] == i
                and active is self._state.project.sections[i]
            )
            color = _ACTIVE_COLOR if is_active else _INACTIVE_COLOR
            lw = _ACTIVE_LW if is_active else _INACTIVE_LW

            nodes = display_sec.nodes
            self._ax.plot(
                nodes[:, 0], nodes[:, 1],
                color=color,
                linewidth=lw,
                zorder=3,
            )
            self._ax.plot(
                nodes[:, 0], nodes[:, 1],
                marker="o",
                markersize=_NODE_MS,
                color=color,
                linestyle="none",
                zorder=4,
            )
            # Label at midpoint
            mid = len(nodes) // 2
            label = display_sec.name or f"Section {i}"
            self._ax.text(
                nodes[mid, 0], nodes[mid, 1],
                f" {label}",
                fontsize=7,
                color=color,
                va="bottom",
                zorder=5,
            )

    def _render_wells(self) -> None:
        for well in self._state.project.wells:
            self._ax.scatter(
                well.x, well.y,
                marker="^",
                s=50,
                color=_WELL_COLOR,
                zorder=5,
            )
            self._ax.text(
                well.x, well.y,
                f" {well.name}",
                fontsize=7,
                color=_WELL_COLOR,
                va="bottom",
                zorder=5,
            )

    def _render_surfaces(self) -> None:
        for surf in self._state.project.surfaces:
            xmin, xmax, ymin, ymax = surf.extent()
            w = xmax - xmin
            h = ymax - ymin
            if w <= 0 or h <= 0:
                continue
            rect = Rectangle(
                (xmin, ymin), w, h,
                fill=False,
                edgecolor=_SURFACE_COLOR,
                linewidth=1.5,
                linestyle="--",
                zorder=2,
            )
            self._ax.add_patch(rect)
            self._ax.text(
                xmin, ymax,
                f" {surf.name}",
                fontsize=6,
                color=_SURFACE_COLOR,
                va="bottom",
                zorder=5,
            )

    def _render_seismic_coverage(self) -> None:
        """Plot trace positions for any SEG-Y already in the AppState (no loading)."""
        # Seismic data is large — we only show traces if a SeismicDataset has
        # already been loaded and cached by the SectionView.  The MapView never
        # triggers a file load; it just visualises whatever's already in memory.
        # Since MapView doesn't hold the seismic cache, we skip loading entirely
        # and rely on the SectionView's cache being passed in via future integration.
        # For now, seismic coverage is not rendered (placeholder for integration).
        pass

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def _pixel_threshold(self) -> float:
        """Return the hit-test threshold in data units (adapts to zoom)."""
        try:
            inv = self._ax.transData.inverted()
            p0 = inv.transform([0.0, 0.0])
            p1 = inv.transform([float(_HIT_PX), float(_HIT_PX)])
            dx = abs(float(p1[0]) - float(p0[0]))
            dy = abs(float(p1[1]) - float(p0[1]))
            return max(dx, dy)
        except Exception:
            return float("inf")

    def _find_nearest_node(
        self, x: float, y: float
    ) -> tuple[int, int] | None:
        """Return (section_index, node_index) of the nearest node within threshold."""
        threshold = self._pixel_threshold()
        best: tuple[int, int] | None = None
        best_dist = float("inf")
        for i, section in enumerate(self._state.project.sections):
            nodes = section.nodes
            for j in range(len(nodes)):
                d = math.hypot(x - nodes[j, 0], y - nodes[j, 1])
                if d < threshold and d < best_dist:
                    best_dist = d
                    best = (i, j)
        return best

    def _find_nearest_section(self, x: float, y: float) -> int | None:
        """Return the index of the nearest section within threshold, or None."""
        threshold = self._pixel_threshold()
        best_idx: int | None = None
        best_dist = float("inf")
        for i, section in enumerate(self._state.project.sections):
            d = _min_dist_to_polyline(x, y, section.nodes)
            if d < threshold and d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def _on_canvas_press(self, event) -> None:
        if event.button != 1 or event.inaxes is not self._ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x, y = float(event.xdata), float(event.ydata)

        # --- Try to start a node drag ---
        hit = self._find_nearest_node(x, y)
        if hit is not None:
            sec_idx, node_idx = hit
            import copy
            sec_copy = copy.deepcopy(self._state.project.sections[sec_idx])
            self._drag = {
                "sec_idx": sec_idx,
                "node_idx": node_idx,
                "section_copy": sec_copy,
            }
            return

        # --- Otherwise select the nearest section ---
        sec_idx = self._find_nearest_section(x, y)
        if sec_idx is not None:
            self._state.set_active_section(
                self._state.project.sections[sec_idx]
            )

    def _on_canvas_motion(self, event) -> None:
        if self._drag is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        self._drag["section_copy"].move_node(self._drag["node_idx"], x, y)
        self.render()

    def _on_canvas_release(self, event) -> None:
        if self._drag is None:
            return
        if event.button != 1:
            return

        sec_idx = self._drag["sec_idx"]
        node_idx = self._drag["node_idx"]
        sec_copy = self._drag["section_copy"]
        self._drag = None

        self._state.update_section(sec_idx, sec_copy)
        new_pos = sec_copy.nodes[node_idx]
        self.section_node_moved.emit(
            sec_idx, node_idx, float(new_pos[0]), float(new_pos[1])
        )

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_sections_changed(self, *_args) -> None:
        self.render()

    def _on_wells_changed(self, *_args) -> None:
        self.render()

    def _on_surfaces_changed(self, *_args) -> None:
        self.render()

    def _on_seismic_changed(self, *_args) -> None:
        self.render()


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------

def _min_dist_to_polyline(x: float, y: float, nodes: np.ndarray) -> float:
    """Minimum Euclidean distance from (x, y) to the nearest point on *nodes*."""
    min_d = float("inf")
    for i in range(len(nodes) - 1):
        ax, ay = float(nodes[i, 0]), float(nodes[i, 1])
        bx, by = float(nodes[i + 1, 0]), float(nodes[i + 1, 1])
        dx, dy = bx - ax, by - ay
        seg_len2 = dx * dx + dy * dy
        if seg_len2 == 0.0:
            d = math.hypot(x - ax, y - ay)
        else:
            t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / seg_len2))
            d = math.hypot(x - ax - t * dx, y - ay - t * dy)
        if d < min_d:
            min_d = d
    return min_d
