"""Section view — 2D cross-section display with picking, faults, and polygons."""
from __future__ import annotations

import copy
import math
from typing import Literal

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cross_section_tool.app_state import AppState
from cross_section_tool.core.section import Section
from cross_section_tool.core.surfaces import HorizonPick
from cross_section_tool.io.project import SeismicRef
from cross_section_tool.io.segy import SeismicDataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PICK_HIT_PX  = 10      # pick-point hit-test radius in screen pixels
_PICK_DRAG_PX = 3       # minimum movement (px) before drag activates
_DEFAULT_DEPTH = 5000.0  # default y-axis range when no data loaded (m)

# Pick-point visual states: (radius_pt, face, edge, ew)
_PP_NORMAL   = (5,  "white",   "#555", 0.8)
_PP_HOVER    = (7,  "#ffffaa", "#555", 0.8)
_PP_SELECTED = (7,  "#ff7f0e", "white", 1.5)
_PP_DRAG     = (7,  "red",     "white", 1.5)


class SectionView(QWidget):
    """Matplotlib-based 2D cross-section display.

    Coordinate conventions:
    * X axis — distance along section (0 → total_length), left to right.
    * Y axis — depth / TWT, **inverted** (0 at top, increases downward).

    Object layers (back → front): seismic | grid | polygon fills | fault lines
    | horizon lines | polygon outlines | wells | rubber-band preview.

    Signals
    -------
    polygon_vertex_added(float, float)  — distance, depth added during drawing
    polygon_finished(object)            — committed SectionPolygon
    """

    polygon_vertex_added = Signal(float, float)
    polygon_finished     = Signal(object)

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state = state

        # ---- seismic cache ----
        self._seismic_cache: dict[str, SeismicDataset] = {}

        # ---- active tool flags (set by MainWindow._on_tool_changed) ----
        self._picking_active:  bool = False   # horizon_pick tool
        self._fault_picking:   bool = False   # fault_pick tool
        self._polygon_drawing: bool = False

        # ---- polygon in-progress ----
        self._polygon_vertices: list[tuple[float, float]] = []

        # ---- display mode ----
        self._display_mode: Literal["variable_density", "wiggle"] = "variable_density"

        # ---- pick-node interaction ----
        # _pick_ref: (category, obj_idx, pt_idx) — "Horizons"|"Faults"
        self._pick_hover:    tuple[str, int, int] | None = None
        self._pick_selected: tuple[str, int, int] | None = None
        self._pick_drag:     bool                        = False
        self._pick_press_px: tuple[float, float] | None = None
        self._pick_copy:     HorizonPick | None          = None

        # ---- pan state ----
        self._sv_pan_anchor: tuple[float, float] | None = None
        self._sv_pan_xlim0:  tuple[float, float] | None = None
        self._sv_pan_ylim0:  tuple[float, float] | None = None
        self._sv_pan_inv     = None

        # ---- rubber-band cursor position ----
        self._cursor_data: tuple[float, float] | None = None

        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self._fig    = Figure(figsize=(10, 6), tight_layout=True)
        self._ax     = self._fig.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._fig)

        # Hidden toolbar — kept for zoom stack; NOT in the layout.
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._toolbar.hide()

        # Header bar: section name + VE spinbox + VE lock
        self._header = QWidget()
        self._header.setFixedHeight(28)
        self._header.setStyleSheet("background: #f5f5f5; border-bottom: 1px solid #ddd;")
        hl = QHBoxLayout(self._header)
        hl.setContentsMargins(8, 2, 8, 2)
        self._section_name_label = QLabel("— no section —")
        self._section_name_label.setStyleSheet("color: #444; font-size: 9px;")
        hl.addWidget(self._section_name_label)
        hl.addStretch()
        hl.addWidget(QLabel("VE:"))
        self._ve_spin = QDoubleSpinBox()
        self._ve_spin.setRange(0.1, 20.0)
        self._ve_spin.setSingleStep(0.5)
        self._ve_spin.setValue(1.0)
        self._ve_spin.setFixedWidth(60)
        self._ve_spin.setToolTip(
            "Vertical exaggeration (1.0 = true scale)\n"
            "Higher values stretch depth axis, steepening apparent dips."
        )
        self._ve_spin.valueChanged.connect(self._on_ve_changed)
        hl.addWidget(self._ve_spin)
        self._ve_lock_btn = QPushButton("\U0001F512")   # 🔒
        self._ve_lock_btn.setCheckable(True)
        self._ve_lock_btn.setFixedSize(24, 22)
        self._ve_lock_btn.setToolTip(
            "Lock VE — when checked, the same vertical exaggeration\n"
            "applies to all sections (switching sections keeps this value)."
        )
        self._ve_lock_btn.setStyleSheet(
            "QPushButton { border: 1px solid #bbb; border-radius: 3px; font-size: 11px; }"
            "QPushButton:checked { background: #d0e8ff; border-color: #5599cc; }"
        )
        hl.addWidget(self._ve_lock_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._header)
        layout.addWidget(self._canvas, stretch=1)

        # Matplotlib events
        self._canvas.mpl_connect("button_press_event",   self._on_sv_press)
        self._canvas.mpl_connect("motion_notify_event",  self._on_sv_motion)
        self._canvas.mpl_connect("button_release_event", self._on_sv_release)
        self._canvas.mpl_connect("scroll_event",         self._on_scroll_sv)
        self._canvas.mpl_connect("key_press_event",      self._on_sv_key)
        self._canvas.mpl_connect("resize_event",         lambda _: self._canvas.draw_idle())
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _connect_signals(self) -> None:
        s = self._state
        s.active_section_changed.connect(self._on_active_section_changed)
        s.active_pick_target_changed.connect(self._on_data_changed)
        s.project_changed.connect(self.render)
        s.horizon_pick_added.connect(self._on_data_changed)
        s.horizon_pick_removed.connect(self._on_data_changed)
        s.horizon_pick_modified.connect(self._on_data_changed)
        s.fault_pick_added.connect(self._on_data_changed)
        s.fault_pick_removed.connect(self._on_data_changed)
        s.fault_pick_modified.connect(self._on_data_changed)
        s.well_added.connect(self._on_data_changed)
        s.well_removed.connect(self._on_data_changed)
        s.well_modified.connect(self._on_data_changed)
        s.surface_added.connect(self._on_data_changed)
        s.surface_removed.connect(self._on_data_changed)
        s.surface_modified.connect(self._on_data_changed)
        s.seismic_ref_added.connect(self._on_seismic_refs_changed)
        s.seismic_ref_removed.connect(self._on_seismic_refs_changed)
        s.polygon_added.connect(self._on_data_changed)
        s.polygon_removed.connect(self._on_data_changed)
        s.polygon_modified.connect(self._on_data_changed)

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

    @property
    def display_mode(self) -> str:
        return self._display_mode

    def set_display_mode(self, mode: Literal["variable_density", "wiggle"]) -> None:
        self._display_mode = mode
        self.render()

    def set_picking_active(self, active: bool) -> None:
        """Enable/disable horizon pick mode."""
        self._picking_active   = active
        self._fault_picking    = False if active else self._fault_picking
        self._polygon_drawing  = False if active else self._polygon_drawing
        if active:
            self._polygon_vertices.clear()

    def set_fault_picking(self, active: bool) -> None:
        """Enable/disable fault pick mode."""
        self._fault_picking    = active
        self._picking_active   = False if active else self._picking_active
        self._polygon_drawing  = False if active else self._polygon_drawing

    def set_polygon_drawing(self, active: bool) -> None:
        """Enable/disable polygon drawing mode."""
        self._polygon_drawing  = active
        self._picking_active   = False if active else self._picking_active
        self._fault_picking    = False if active else self._fault_picking
        self._polygon_vertices.clear()
        self.render()

    def finish_polygon(self) -> None:
        """Close and commit the in-progress polygon."""
        if len(self._polygon_vertices) >= 3:
            from cross_section_tool.core.polygons import SectionPolygon
            poly = SectionPolygon(
                self._polygon_vertices,
                name=f"Polygon {len(self._state.project.polygons) + 1}",
            )
            self._polygon_vertices.clear()
            self.polygon_finished.emit(poly)
        else:
            self._polygon_vertices.clear()
        self.render()

    def clear_seismic_cache(self) -> None:
        self._seismic_cache.clear()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, *_args) -> None:
        """Full redraw of the active section."""
        self._ax.clear()
        section = self._state.active_section
        if section is None:
            self._section_name_label.setText("— no section —")
            self._ve_spin.setEnabled(False)
            self._canvas.draw_idle()
            return

        self._section_name_label.setText(section.name or "Unnamed section")
        self._ve_spin.setEnabled(True)
        # Sync spinbox to section only when VE is not locked
        if not self._ve_lock_btn.isChecked():
            self._ve_spin.blockSignals(True)
            self._ve_spin.setValue(section.vertical_exaggeration)
            self._ve_spin.blockSignals(False)

        self._setup_axes(section)
        self._render_seismic(section)
        self._render_grid(section)
        self._render_section_ends(section)
        self._render_polygons(section)
        self._render_surfaces(section)
        self._render_faults(section)
        self._render_horizons(section)
        self._render_wells(section)
        self._render_rubber_band(section)
        self._render_polygon_in_progress()
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Axis setup
    # ------------------------------------------------------------------

    def _setup_axes(self, section: Section) -> None:
        total = section.total_length()
        ve    = section.vertical_exaggeration

        # X axis
        self._ax.set_xlim(0.0, max(total, 1.0))

        # Y axis — depth down, inverted
        max_d = self._compute_max_depth(section)
        y_range = max_d / max(ve, 0.01)
        self._ax.set_ylim(y_range, 0.0)   # inverted: 0 at top

        # Labels
        if section.depth_domain == "twt":
            ylabel = "TWT (ms)"
            xlabel = "Distance (m)"
        else:
            ylabel = f"Depth ({section.depth_units})"
            xlabel = f"Distance ({section.depth_units})"

        self._ax.set_xlabel(xlabel, fontsize=8)
        self._ax.set_ylabel(ylabel, fontsize=8)
        self._ax.tick_params(labelsize=7)

    def _compute_max_depth(self, section: Section) -> float:
        """Best estimate of maximum depth from loaded data."""
        candidates = [_DEFAULT_DEPTH]
        for hp in self._state.project.horizon_picks:
            v = hp.depths[~np.isnan(hp.depths)]
            if len(v):
                candidates.append(float(v.max()))
        for fp in self._state.project.fault_picks:
            v = fp.depths[~np.isnan(fp.depths)]
            if len(v):
                candidates.append(float(v.max()))
        for well in self._state.project.wells:
            candidates.append(well.deviation.max_tvd)
        for ref in self._state.project.seismic_refs:
            ds = self._seismic_cache.get(ref.path)
            if ds is not None:
                candidates.append(float(ds.samples[-1]))
        return max(candidates)

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------

    def _render_grid(self, section: Section) -> None:
        xl  = self._ax.get_xlim()
        yl  = self._ax.get_ylim()
        span = max(abs(xl[1] - xl[0]), abs(yl[1] - yl[0]))
        interval = _nice_interval(span / 5)
        grid_kw = dict(color="#e0e0e0", linewidth=0.5, linestyle="--", zorder=0)
        xs = np.arange(math.floor(xl[0] / interval) * interval, xl[1] + interval, interval)
        ys = np.arange(math.floor(min(yl) / interval) * interval, max(yl) + interval, interval)
        for x in xs:
            self._ax.axvline(x, **grid_kw)
        for y in ys:
            self._ax.axhline(y, **grid_kw)
        self._ax.xaxis.set_major_locator(MultipleLocator(interval))
        self._ax.yaxis.set_major_locator(MultipleLocator(interval))
        self._ax.ticklabel_format(style="plain", axis="both")

    def _render_section_ends(self, section: Section) -> None:
        """Draw vertical end-cap lines at x=0 and x=total_length."""
        total = section.total_length()
        yl    = self._ax.get_ylim()
        ylo, yhi = min(yl), max(yl)
        kw = dict(color="#666666", linewidth=1.5, alpha=0.7, zorder=2,
                  solid_capstyle="butt")
        self._ax.plot([0, 0],         [ylo, yhi], **kw)
        self._ax.plot([total, total], [ylo, yhi], **kw)

    # ------------------------------------------------------------------
    # Object renderers
    # ------------------------------------------------------------------

    @staticmethod
    def _mpl_linestyle(style: str) -> str:
        return {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}.get(style, "-")

    def _is_active_pick(self, category: str, obj_idx: int) -> bool:
        return (self._state.active_pick_category == category and
                self._state.active_pick_index == obj_idx)

    def _render_horizons(self, section: Section) -> None:
        for obj_idx, hp in enumerate(self._state.project.horizon_picks):
            if hp.n_picks == 0:
                continue
            lw = getattr(hp, "line_width", 1.5)
            ls = self._mpl_linestyle(getattr(hp, "line_style", "solid"))
            is_active = self._is_active_pick("Horizons", obj_idx)
            self._ax.plot(
                hp.distances, hp.depths,
                color=hp.color, linewidth=lw * 1.6 if is_active else lw,
                linestyle=ls, zorder=3 if not is_active else 4,
                label=hp.name or "_nolegend_",
            )
            for pt_idx in range(hp.n_picks):
                d, z = hp.distances[pt_idx], hp.depths[pt_idx]
                ms, fc, ec, ew = self._pick_point_style("Horizons", obj_idx, pt_idx)
                self._ax.plot(d, z, "o",
                              markersize=ms, markerfacecolor=fc,
                              markeredgecolor=ec, markeredgewidth=ew, zorder=5)

    def _render_faults(self, section: Section) -> None:
        for obj_idx, fp in enumerate(self._state.project.fault_picks):
            if fp.n_picks == 0:
                continue
            lw = getattr(fp, "line_width", 1.5)
            ls = self._mpl_linestyle(getattr(fp, "line_style", "dashed"))
            is_active = self._is_active_pick("Faults", obj_idx)
            self._ax.plot(
                fp.distances, fp.depths,
                color=fp.color, linewidth=lw * 1.6 if is_active else lw,
                linestyle=ls, zorder=3 if not is_active else 4,
                label=fp.name or "_nolegend_",
            )
            for pt_idx in range(fp.n_picks):
                d, z = fp.distances[pt_idx], fp.depths[pt_idx]
                ms, fc, ec, ew = self._pick_point_style("Faults", obj_idx, pt_idx)
                self._ax.plot(d, z, "D",
                              markersize=ms, markerfacecolor=fc,
                              markeredgecolor=ec, markeredgewidth=ew, zorder=5)

    def _pick_point_style(
        self, category: str, obj_idx: int, pt_idx: int
    ) -> tuple:
        ref = (category, obj_idx, pt_idx)
        if self._pick_drag and self._pick_selected == ref:
            return _PP_DRAG
        if self._pick_selected == ref:
            return _PP_SELECTED
        if self._pick_hover == ref:
            return _PP_HOVER
        return _PP_NORMAL

    def _render_seismic(self, section: Section) -> None:
        for ref in self._state.project.seismic_refs:
            ds = self._get_or_load_seismic(ref)
            if ds is None or ds.n_traces == 0:
                continue
            distances, data, _ = ds.traces_sorted_by_section(section)
            if len(distances) < 2:
                continue
            vmax = float(np.percentile(np.abs(data), 95)) or 1.0
            if self._display_mode == "variable_density":
                self._ax.imshow(
                    data.T,
                    aspect="auto",
                    extent=[distances[0], distances[-1], ds.samples[-1], ds.samples[0]],
                    origin="upper",
                    cmap="seismic",
                    vmin=-vmax, vmax=vmax,
                    interpolation="bilinear",
                )
                self._ax.set_ylim(ds.samples[-1], ds.samples[0])
            else:
                self._render_wiggle(distances, data, ds.samples)

    def _render_wiggle(self, distances, data, samples) -> None:
        if len(distances) < 2:
            return
        spacing = (distances[-1] - distances[0]) / max(len(distances) - 1, 1)
        scale   = spacing * 0.8 / (np.percentile(np.abs(data), 95) or 1.0)
        for dist, trace in zip(distances, data):
            self._ax.plot(dist + trace * scale, samples, "k-", linewidth=0.3)
            pos = np.where(trace > 0, trace, 0)
            self._ax.fill_betweenx(samples, dist, dist + pos * scale,
                                   color="k", alpha=0.7)

    def _render_surfaces(self, section: Section) -> None:
        for surf in self._state.project.surfaces:
            distances, z_values = surf.profile_along_section(section, n_samples=300)
            valid = ~np.isnan(z_values)
            if not np.any(valid):
                continue
            self._ax.plot(distances[valid], z_values[valid],
                          color="darkorange", linewidth=1.5, linestyle="--",
                          alpha=0.85, zorder=3)

    def _render_polygons(self, section: Section) -> None:
        from matplotlib.patches import Polygon as MplPolygon
        for poly in self._state.project.polygons:
            verts = poly.vertices
            if len(verts) < 3:
                continue
            patch = MplPolygon(verts, closed=True,
                               facecolor=poly.fill_color, alpha=poly.fill_alpha,
                               edgecolor=poly.edge_color, linewidth=poly.edge_width, zorder=2)
            self._ax.add_patch(patch)
            if poly.name:
                cx, cy = float(verts[:, 0].mean()), float(verts[:, 1].mean())
                self._ax.text(cx, cy, poly.name, fontsize=6,
                              ha="center", va="center", zorder=3)

    def _render_polygon_in_progress(self) -> None:
        if not self._polygon_drawing or not self._polygon_vertices:
            return
        xs = [v[0] for v in self._polygon_vertices]
        ys = [v[1] for v in self._polygon_vertices]
        self._ax.plot(xs, ys, "o-", color="#9467bd", linewidth=1.5,
                      markersize=5, zorder=10)
        if len(xs) >= 2:
            self._ax.plot([xs[-1], xs[0]], [ys[-1], ys[0]],
                          "--", color="#9467bd", linewidth=1.0, alpha=0.5, zorder=10)

    def _render_rubber_band(self, section: Section) -> None:
        """V-shaped dashed ghost line connecting cursor to its neighbouring picks."""
        if not (self._picking_active or self._fault_picking):
            return
        if self._cursor_data is None:
            return
        cat = self._state.active_pick_category
        idx = self._state.active_pick_index
        if cat is None or idx is None:
            return
        picks = (self._state.project.horizon_picks if cat == "Horizons"
                 else self._state.project.fault_picks)
        if idx >= len(picks):
            return
        hp = picks[idx]
        if hp.n_picks == 0:
            return
        cx, cy   = self._cursor_data
        d, z     = hp.distances, hp.depths
        color    = hp.color
        rb_kw    = dict(linestyle="--", color=color, linewidth=1.0, alpha=0.6, zorder=8)
        left     = d < cx
        right    = d > cx
        if left.any():
            li = int(np.where(left)[0][-1])
            self._ax.plot([d[li], cx], [z[li], cy], **rb_kw)
        if right.any():
            ri = int(np.where(right)[0][0])
            self._ax.plot([cx, d[ri]], [cy, z[ri]], **rb_kw)

    def _render_wells(self, section: Section) -> None:
        for well in self._state.project.wells:
            distances, tvds = well.section_track(section)
            collar_dist, _ = well.project_to_section(section)
            self._ax.plot(distances, tvds, color="#8B4513", linewidth=2.0,
                          solid_capstyle="round", zorder=4)
            if len(tvds) > 0:
                self._ax.annotate(
                    well.name, xy=(collar_dist, tvds[0]),
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=7, color="#8B4513", zorder=5,
                )
            for top_name in well.formation_tops:
                try:
                    td, tz = well.formation_top_in_section(top_name, section)
                except KeyError:
                    continue
                self._ax.plot(td, tz, marker="<", markersize=6,
                              color="green", zorder=5)
                self._ax.text(td + section.total_length() * 0.005, tz,
                              top_name, fontsize=6, color="green", va="center", zorder=5)

    # ------------------------------------------------------------------
    # Pick-node interaction helpers
    # ------------------------------------------------------------------

    def _to_screen_px_sv(self, xd: float, yd: float) -> tuple[float, float]:
        pt = self._ax.transData.transform([[xd, yd]])
        return float(pt[0, 0]), float(pt[0, 1])

    def _find_nearest_pick_px(
        self, event_x: float, event_y: float
    ) -> tuple[str, int, int] | None:
        """Return (category, obj_idx, pt_idx) for the nearest pick within threshold."""
        ex, ey = self._to_screen_px_sv(event_x, event_y)
        best = None
        best_dist = float("inf")

        def _check(category, picks):
            nonlocal best, best_dist
            for oi, hp in enumerate(picks):
                for pi in range(hp.n_picks):
                    nx, ny = self._to_screen_px_sv(hp.distances[pi], hp.depths[pi])
                    d = math.hypot(ex - nx, ey - ny)
                    if d <= _PICK_HIT_PX and d < best_dist:
                        best_dist = d
                        best = (category, oi, pi)

        _check("Horizons", self._state.project.horizon_picks)
        _check("Faults",   self._state.project.fault_picks)
        return best

    def _get_active_pick_last_point(
        self,
    ) -> tuple[HorizonPick | None, float | None, float | None]:
        """Return (pick_object, last_distance, last_depth) for the active pick target."""
        cat = self._state.active_pick_category
        idx = self._state.active_pick_index
        if cat is None or idx is None:
            return None, None, None
        if cat == "Horizons":
            picks = self._state.project.horizon_picks
        elif cat == "Faults":
            picks = self._state.project.fault_picks
        else:
            return None, None, None
        if idx >= len(picks):
            return None, None, None
        hp = picks[idx]
        if hp.n_picks == 0:
            return hp, None, None
        # Nearest existing pick to current cursor (for rubber band)
        if self._cursor_data is not None:
            cx = self._cursor_data[0]
            dist_to_cursor = np.abs(hp.distances - cx)
            nearest = int(np.argmin(dist_to_cursor))
            return hp, float(hp.distances[nearest]), float(hp.depths[nearest])
        return hp, float(hp.distances[-1]), float(hp.depths[-1])

    def _add_pick_to_active_target(self, x: float, y: float) -> None:
        """Insert a pick point into the currently active horizon/fault."""
        cat = self._state.active_pick_category
        idx = self._state.active_pick_index
        if cat is None or idx is None:
            return
        if cat == "Horizons":
            picks = self._state.project.horizon_picks
            if idx >= len(picks):
                return
            hp = copy.deepcopy(picks[idx])
            hp.insert_pick(x, y)
            self._state.update_horizon_pick(idx, hp)
        elif cat == "Faults":
            picks = self._state.project.fault_picks
            if idx >= len(picks):
                return
            fp = copy.deepcopy(picks[idx])
            fp.insert_pick(x, y)
            self._state.update_fault_pick(idx, fp)

    # ------------------------------------------------------------------
    # Seismic cache
    # ------------------------------------------------------------------

    def _get_or_load_seismic(self, ref: SeismicRef) -> SeismicDataset | None:
        if ref.path not in self._seismic_cache:
            try:
                self._seismic_cache[ref.path] = ref.load()
            except Exception:
                return None
        return self._seismic_cache.get(ref.path)

    # ------------------------------------------------------------------
    # VE spinbox
    # ------------------------------------------------------------------

    def _on_ve_changed(self, value: float) -> None:
        if self._ve_lock_btn.isChecked():
            # Apply to every section
            for i, sec in enumerate(self._state.project.sections):
                if abs(getattr(sec, "vertical_exaggeration", 1.0) - value) > 0.001:
                    sec_copy = copy.deepcopy(sec)
                    sec_copy.vertical_exaggeration = value
                    self._state.update_section(i, sec_copy)
        else:
            section = self._state.active_section
            if section is None:
                return
            idx = self._state.project.sections.index(section)
            sec_copy = copy.deepcopy(section)
            sec_copy.vertical_exaggeration = value
            self._state.update_section(idx, sec_copy)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    # Alias kept for backward compatibility with tests
    def _on_canvas_click(self, event) -> None:
        self._on_sv_press(event)

    def _on_sv_press(self, event) -> None:
        if event.inaxes is not self._ax:
            return
        try:
            ex, ey = getattr(event, "x", None), getattr(event, "y", None)
            if ex is not None and ey is not None:
                xy = self._ax.transData.inverted().transform([[ex, ey]])[0]
                x, y = float(xy[0]), float(xy[1])
            elif event.xdata is not None and event.ydata is not None:
                x, y = float(event.xdata), float(event.ydata)
            else:
                return
        except Exception:
            if event.xdata is not None and event.ydata is not None:
                x, y = float(event.xdata), float(event.ydata)
            else:
                return

        tool = self._state.active_tool

        # ---- Pan ----
        if event.button == 1 and tool == "pan":
            self._sv_pan_anchor = (event.x, event.y)
            self._sv_pan_xlim0  = self._ax.get_xlim()
            self._sv_pan_ylim0  = self._ax.get_ylim()
            self._sv_pan_inv    = self._ax.transData.inverted()
            return

        # ---- Horizon / fault pick ----
        if event.button == 1 and (self._picking_active or self._fault_picking):
            self._add_pick_to_active_target(x, y)
            return

        # ---- Polygon drawing ----
        if event.button == 1 and self._polygon_drawing:
            self._polygon_vertices.append((x, y))
            self.polygon_vertex_added.emit(x, y)
            self.render()
            return
        if event.button == 3 and self._polygon_drawing:
            self.finish_polygon()
            return

        # ---- Pick-node select / drag (select / edit_nodes tool) ----
        if event.button == 1 and tool in ("select", "edit_nodes"):
            hit = self._find_nearest_pick_px(x, y)
            if hit is not None:
                self._pick_selected = hit
                self._pick_drag     = False
                self._pick_press_px = (event.x, event.y)
                cat, oi, pi = hit
                picks = (self._state.project.horizon_picks if cat == "Horizons"
                         else self._state.project.fault_picks)
                self._pick_copy = copy.deepcopy(picks[oi])
                self.render()
                return
            # Click empty space → deselect
            if self._pick_selected is not None:
                self._pick_selected = None
                self._pick_copy     = None
                self._pick_drag     = False
                self.render()

        # ---- Right-click context menu on pick point ----
        if event.button == 3 and tool in ("select", "edit_nodes"):
            hit = self._find_nearest_pick_px(x, y)
            if hit is not None:
                self._show_pick_context_menu(hit, event)

    def _on_sv_motion(self, event) -> None:
        # Track cursor for rubber band (always)
        if event.xdata is not None and event.ydata is not None:
            self._cursor_data = (float(event.xdata), float(event.ydata))
        else:
            self._cursor_data = None

        # ---- Pan ----
        if self._sv_pan_anchor is not None:
            d0 = self._sv_pan_inv.transform(self._sv_pan_anchor)
            d1 = self._sv_pan_inv.transform([event.x, event.y])
            self._ax.set_xlim(self._sv_pan_xlim0[0] + d0[0] - d1[0],
                              self._sv_pan_xlim0[1] + d0[0] - d1[0])
            self._ax.set_ylim(self._sv_pan_ylim0[0] + d0[1] - d1[1],
                              self._sv_pan_ylim0[1] + d0[1] - d1[1])
            self._canvas.draw_idle()
            return

        # ---- Drag a pick point ----
        if self._pick_selected is not None and self._pick_press_px is not None:
            try:
                xy = self._ax.transData.inverted().transform([[event.x, event.y]])[0]
                x, y = float(xy[0]), float(xy[1])
            except Exception:
                return
            dx = math.hypot(event.x - self._pick_press_px[0],
                            event.y - self._pick_press_px[1])
            if not self._pick_drag and dx < _PICK_DRAG_PX:
                # Redraw rubber band while considering a drag
                if self._picking_active or self._fault_picking:
                    self.render()
                return
            self._pick_drag = True
            cat, oi, pi = self._pick_selected
            # Update pick in the copy
            self._pick_copy._distances[pi] = x
            self._pick_copy._depths[pi]    = y
            # Re-sort (pick may have moved past a neighbour)
            order = np.argsort(self._pick_copy._distances, kind="stable")
            self._pick_copy._distances = self._pick_copy._distances[order]
            self._pick_copy._depths    = self._pick_copy._depths[order]
            # Update selected index after sort
            new_pi = int(np.where(order == pi)[0][0])
            self._pick_selected = (cat, oi, new_pi)
            self.render()
            return

        # ---- Hover for pick-node select tool ----
        tool = self._state.active_tool
        if tool in ("select", "edit_nodes"):
            if event.xdata is not None:
                new_hover = self._find_nearest_pick_px(
                    float(event.xdata), float(event.ydata)
                )
                if new_hover != self._pick_hover:
                    self._pick_hover = new_hover
                    self.render()
                    return

        # ---- Redraw rubber band if picking ----
        if (self._picking_active or self._fault_picking) and self._cursor_data is not None:
            self.render()

    def _on_sv_release(self, event) -> None:
        # End pan
        if event.button == 1:
            self._sv_pan_anchor = None

        # Commit drag
        if self._pick_drag and self._pick_selected is not None:
            cat, oi, _ = self._pick_selected
            if cat == "Horizons":
                self._state.update_horizon_pick(oi, self._pick_copy)
            else:
                self._state.update_fault_pick(oi, self._pick_copy)
            self._pick_drag  = False
            self._pick_copy  = None
            self._pick_press_px = None

    def _on_scroll_sv(self, event) -> None:
        if self._state.active_tool != "zoom":
            return
        if event.inaxes is not self._ax:
            return
        factor = 0.85 if (getattr(event, "step", 0) > 0 or event.button == "up") else 1.0 / 0.85
        cx = event.xdata if event.xdata is not None else sum(self._ax.get_xlim()) / 2
        cy = event.ydata if event.ydata is not None else sum(self._ax.get_ylim()) / 2
        xl, yl = self._ax.get_xlim(), self._ax.get_ylim()
        self._ax.set_xlim([cx + (x - cx) * factor for x in xl])
        self._ax.set_ylim([cy + (y - cy) * factor for y in yl])
        self._canvas.draw_idle()

    def _on_sv_key(self, event) -> None:
        if event.key == "escape":
            if self._pick_drag:
                # Cancel drag: restore original
                self._pick_drag     = False
                self._pick_copy     = None
                self._pick_press_px = None
                self.render()
            elif self._pick_selected is not None:
                self._pick_selected = None
                self._pick_copy     = None
                self.render()
        elif event.key == "delete":
            if self._pick_selected is not None and not self._pick_drag:
                self._delete_selected_pick()

    def _on_active_section_changed(self, section) -> None:
        self.render()

    def _on_data_changed(self, *_args) -> None:
        self.render()

    def _on_seismic_refs_changed(self, *_args) -> None:
        self._seismic_cache.clear()
        self.render()

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------

    def _show_pick_context_menu(
        self, pick_ref: tuple[str, int, int], event
    ) -> None:
        menu = QMenu(self)
        del_act = menu.addAction("Delete Pick")
        pos = self._canvas.mapToGlobal(
            self._canvas.rect().topLeft()
        )
        # Approximate screen position
        from PySide6.QtCore import QPoint
        screen_pos = self._canvas.mapToGlobal(
            QPoint(int(event.x), self._canvas.height() - int(event.y))
        )
        chosen = menu.exec(screen_pos)
        if chosen is del_act:
            self._pick_selected = pick_ref
            self._delete_selected_pick()

    # ------------------------------------------------------------------
    # Pick deletion
    # ------------------------------------------------------------------

    def _delete_selected_pick(self) -> None:
        if self._pick_selected is None:
            return
        cat, oi, pi = self._pick_selected
        self._pick_selected = None
        self._pick_copy     = None

        if cat == "Horizons":
            picks = self._state.project.horizon_picks
            if oi >= len(picks):
                return
            hp = copy.deepcopy(picks[oi])
            if hp.n_picks <= 1:
                return  # keep at least 1 point
            hp.delete_pick(pi)
            self._state.update_horizon_pick(oi, hp)
        elif cat == "Faults":
            picks = self._state.project.fault_picks
            if oi >= len(picks):
                return
            fp = copy.deepcopy(picks[oi])
            if fp.n_picks <= 1:
                return
            fp.delete_pick(pi)
            self._state.update_fault_pick(oi, fp)


# ---------------------------------------------------------------------------
# Grid interval helper (reused from map_view logic)
# ---------------------------------------------------------------------------

def _nice_interval(raw: float) -> float:
    if raw <= 0:
        return 1.0
    import math as _math
    exp  = _math.floor(_math.log10(raw))
    base = 10 ** exp
    for step in (1.0, 2.0, 5.0, 10.0):
        candidate = step * base
        if candidate >= raw:
            return candidate
    return 10.0 * base
