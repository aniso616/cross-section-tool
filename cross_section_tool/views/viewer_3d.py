from __future__ import annotations

import numpy as np
import pyvista as pv
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from cross_section_tool.app_state import AppState
from cross_section_tool.core.section import Section
from cross_section_tool.core.surfaces import HorizonPick, Surface
from cross_section_tool.core.wells import Well

_DEFAULT_DEPTH = 5000.0  # fallback maximum depth (m) when no data determines it

# Cycling surface colours (matplotlib tab10 palette)
_SURFACE_COLORS = [
    "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
]


# ---------------------------------------------------------------------------
# Mesh builders — pure functions, no Qt dependency, fully testable
# ---------------------------------------------------------------------------

def build_section_curtain(
    section: Section,
    max_depth: float = _DEFAULT_DEPTH,
) -> pv.PolyData | None:
    """Return a vertical curtain :class:`pv.PolyData` for *section*.

    The curtain spans from z = 0 (ground surface) to z = *-max_depth*.
    Quad faces are created for every segment of the section polyline.
    Returns ``None`` when the section has fewer than 2 nodes.

    Coordinate convention used throughout this module:
    x = easting, y = northing, z = elevation (negative = below surface).
    """
    nodes = section.nodes
    n = len(nodes)
    if n < 2:
        return None

    top = np.column_stack([nodes[:, 0], nodes[:, 1], np.zeros(n)])
    bot = np.column_stack([nodes[:, 0], nodes[:, 1], np.full(n, -max_depth)])
    pts = np.vstack([top, bot])  # top nodes 0..n-1, bottom nodes n..2n-1

    faces: list[int] = []
    for i in range(n - 1):
        tl, tr = i, i + 1
        bl, br = i + n, i + 1 + n
        faces.extend([4, tl, tr, br, bl])

    return pv.PolyData(pts, faces=np.array(faces, dtype=int))


def build_surface_mesh(surface: Surface) -> pv.DataSet | None:
    """Return a PyVista mesh for *surface*.

    * Grid surfaces → :class:`pv.StructuredGrid` (exact topology).
    * Scattered surfaces → :class:`pv.PolyData` triangulated with
      Delaunay 2D (projected onto the XY plane).

    Returns ``None`` on failure or if the surface has fewer than 3 points.
    """
    if len(surface._x) < 3:
        return None
    try:
        if surface._is_grid:
            xx, yy = np.meshgrid(surface._grid_x, surface._grid_y)
            zz = -surface._grid_z  # depth -> elevation
            return pv.StructuredGrid(xx, yy, zz)
        else:
            pts = np.column_stack([surface._x, surface._y, -surface._z])
            cloud = pv.PolyData(pts)
            mesh = cloud.delaunay_2d()
            return mesh if mesh.n_points > 0 else None
    except Exception:
        return None


def build_well_track(well: Well) -> pv.PolyData | None:
    """Return a polyline :class:`pv.PolyData` for the well deviation track.

    Points are ``(x_easting, y_northing, -tvd)`` at each deviation station.
    Returns ``None`` when the track has fewer than 2 stations.
    """
    dev = well.deviation
    pts = np.column_stack([dev._x, dev._y, -dev._tvd])
    if len(pts) < 2:
        return None
    n = len(pts)
    lines = np.empty(n + 1, dtype=np.intp)
    lines[0] = n
    lines[1:] = np.arange(n)
    return pv.PolyData(pts, lines=lines)


def build_horizon_pick_3d(
    pick: HorizonPick,
    section: Section,
) -> pv.PolyData | None:
    """Return a 3D line for *pick* on *section*, using only that section's picks."""
    d_sec, z_sec = pick.picks_for_section(section.name)
    if len(d_sec) < 2:
        return None
    total = section.total_length()
    distances = np.linspace(0.0, total, 200)
    depths = np.interp(distances, d_sec, z_sec, left=np.nan, right=np.nan)
    valid = ~np.isnan(depths)
    if not np.any(valid):
        return None

    d_valid = distances[valid]
    z_valid = depths[valid]
    pts = []
    for d, z in zip(d_valid, z_valid):
        x, y = section.section_to_map(float(d))
        pts.append([x, y, -float(z)])

    if len(pts) < 2:
        return None

    pts_arr = np.array(pts)
    n = len(pts_arr)
    lines = np.empty(n + 1, dtype=np.intp)
    lines[0] = n
    lines[1:] = np.arange(n)
    return pv.PolyData(pts_arr, lines=lines)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class Viewer3D(QWidget):
    """PyVista-based 3D perspective viewer embedded in a PySide6 widget.

    Renders section curtains, surface meshes, well deviation tracks,
    formation top spheres, and horizon pick lines (for the active section).

    All rendering uses a z = elevation convention (positive up).  Depths
    are negated on entry so that deeper features appear lower in the scene.

    Parameters
    ----------
    state:
        Central application state.
    """

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._plotter = None  # lazy — created only when user clicks Enable
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(100)  # 10fps max for 3D (heavier than 2D)
        self._redraw_timer.timeout.connect(self.render)
        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._placeholder = QLabel("3D View — click Enable to activate")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._enable_btn = QPushButton("Enable 3D View")
        self._enable_btn.clicked.connect(self._init_plotter)
        self._layout.addWidget(self._placeholder)
        self._layout.addWidget(self._enable_btn)

    def _init_plotter(self) -> None:
        """Create the PyVista Qt widget — only on explicit user request."""
        try:
            from pyvistaqt import QtInteractor
            self._plotter = QtInteractor(self)
            self._layout.removeWidget(self._placeholder)
            self._layout.removeWidget(self._enable_btn)
            self._placeholder.deleteLater()
            self._enable_btn.deleteLater()
            self._layout.addWidget(self._plotter)
            self._plotter.show()
            self.render()
        except Exception as e:
            self._placeholder.setText(f"3D View failed to initialize: {e}")

    def _connect_signals(self) -> None:
        s = self._state
        s.project_changed.connect(self.request_render)
        s.active_section_changed.connect(self._on_sections_changed)
        s.section_added.connect(self._on_sections_changed)
        s.section_removed.connect(self._on_sections_changed)
        s.section_modified.connect(self._on_sections_changed)
        s.surface_added.connect(self._on_surfaces_changed)
        s.surface_removed.connect(self._on_surfaces_changed)
        s.surface_modified.connect(self._on_surfaces_changed)
        s.horizon_pick_added.connect(self._on_picks_changed)
        s.horizon_pick_removed.connect(self._on_picks_changed)
        s.horizon_pick_modified.connect(self._on_picks_changed)
        s.well_added.connect(self._on_wells_changed)
        s.well_removed.connect(self._on_wells_changed)
        s.well_modified.connect(self._on_wells_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def plotter(self):
        return self._plotter

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def request_render(self, *_args) -> None:
        """Schedule a throttled render (100ms debounce)."""
        if not self._redraw_timer.isActive():
            self._redraw_timer.start()

    def render(self, *_args) -> None:
        """Full redraw: clear the scene and re-add all meshes."""
        if self._plotter is None:
            return
        try:
            self._plotter.clear()
        except Exception:
            return

        max_depth = self._compute_max_depth()

        self._render_surfaces()
        self._render_sections(max_depth)
        self._render_horizon_picks()
        self._render_wells(max_depth)

        self._plotter.reset_camera()
        try:
            self._plotter.render()
        except Exception:
            pass

    def _safe_add_mesh(self, mesh, **kwargs) -> None:
        """Add *mesh* to the plotter, silently skipping None or empty meshes."""
        if mesh is None:
            return
        if hasattr(mesh, "n_points") and mesh.n_points == 0:
            return
        try:
            self._plotter.add_mesh(mesh, **kwargs)
        except Exception:
            pass

    def _render_sections(self, max_depth: float) -> None:
        active = self._state.active_section
        for section in self._state.project.sections:
            mesh = build_section_curtain(section, max_depth)
            is_active = section is active
            self._safe_add_mesh(
                mesh,
                color="#1f77b4" if is_active else "#aaaaaa",
                opacity=0.35 if is_active else 0.15,
                show_edges=True,
                edge_color="grey",
            )

    def _render_surfaces(self) -> None:
        for i, surf in enumerate(self._state.project.surfaces):
            self._safe_add_mesh(
                build_surface_mesh(surf),
                color=_SURFACE_COLORS[i % len(_SURFACE_COLORS)],
                opacity=0.6,
                show_edges=False,
            )

    def _render_wells(self, max_depth: float) -> None:
        collar_r = max(20.0, max_depth * 0.005)
        top_r = max(15.0, max_depth * 0.004)

        for well in self._state.project.wells:
            self._safe_add_mesh(build_well_track(well), color="#8B4513", line_width=3)
            self._safe_add_mesh(
                pv.Sphere(radius=collar_r, center=(well.x, well.y, 0.0)),
                color="#8B4513",
            )
            for _top_name, md in well.formation_tops.items():
                try:
                    x, y, tvd = well.deviation.xyz_at_md(md)
                    self._safe_add_mesh(
                        pv.Sphere(radius=top_r, center=(x, y, -tvd)),
                        color="green",
                    )
                except Exception:
                    pass

    def _render_horizon_picks(self) -> None:
        """Phase 1: Show picks from ALL sections in the 3D view."""
        for section in self._state.project.sections:
            for pick in self._state.project.horizon_picks:
                if pick.n_picks == 0:
                    continue
                # Only render if this section has picks for this horizon
                if pick.n_picks_for_section(section.name) == 0:
                    continue
                self._safe_add_mesh(
                    build_horizon_pick_3d(pick, section),
                    color=pick.color,
                    line_width=2,
                )

    # ------------------------------------------------------------------
    # Depth computation
    # ------------------------------------------------------------------

    def _compute_max_depth(self) -> float:
        """Return the maximum depth present in the current project data."""
        depths = [_DEFAULT_DEPTH]
        for well in self._state.project.wells:
            depths.append(well.deviation.max_tvd)
        for surf in self._state.project.surfaces:
            valid = surf._z[~np.isnan(surf._z)]
            if len(valid) > 0:
                depths.append(float(valid.max()))
        for pick in self._state.project.horizon_picks:
            valid = pick.depths[~np.isnan(pick.depths)]
            if len(valid) > 0:
                depths.append(float(valid.max()))
        return float(max(depths))

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_sections_changed(self, *_args) -> None:
        self.request_render()

    def _on_surfaces_changed(self, *_args) -> None:
        self.request_render()

    def _on_picks_changed(self, *_args) -> None:
        self.request_render()

    def _on_wells_changed(self, *_args) -> None:
        self.request_render()
