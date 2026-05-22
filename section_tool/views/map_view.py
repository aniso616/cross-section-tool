from __future__ import annotations

import copy
import math

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from section_tool.app_state import AppState
from section_tool.core.section import Section


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NODE_HIT_PX  = 12    # node selection/hover radius in screen pixels
_LINE_HIT_PX  = 8     # section-line selection tolerance in screen pixels
_DRAG_MIN_PX  = 3     # pixels of movement before drag activates

# Section line style
_ACTIVE_COLOR   = "#2563EB"
_INACTIVE_COLOR = "#94A3B8"
_ACTIVE_LW      = 2.0
_INACTIVE_LW    = 1.2

# Node visual states: (radius_pt, face, edge, edge_width)
_NODE_NORMAL   = (6,  "white",  "#444444", 1.0)
_NODE_HOVER    = (9,  "yellow", "#444444", 1.0)
_NODE_SELECTED = (9,  "#ff7f0e", "white",  1.5)
_NODE_DRAG     = (9,  "red",    "white",   1.5)

_WELL_COLOR    = "#8B4513"
_SURFACE_COLOR = "darkorange"

# Tools that activate node editing (A-key tool only now)
_EDIT_TOOLS = ("node_edit",)
_SELECT_TOOLS = ("select", "node_edit")  # both can select sections/objects


class MapView(QWidget):
    """Plan-view (map) display of sections, wells, and data extents.

    Manages a four-state node interaction model:
    unselected → hover (cursor in range) → selected (click) → dragging (drag).

    Signals
    -------
    section_node_moved(int, int, float, float)
        After a committed drag: (section_index, node_index, x, y).
    status_message(str)
        Coordinate string during drag; empty string when drag ends.
    """

    section_node_moved = Signal(int, int, float, float)
    status_message     = Signal(str)
    cursor_map_pos     = Signal(float, float)   # map x,y on hover

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state = state

        # ---- node state machine ----
        self._hover_node:    tuple[int, int] | None = None
        self._selected_node: tuple[int, int] | None = None
        self._mouse_pressed: bool                   = False
        self._press_px:      tuple[float, float] | None = None
        self._drag_active:   bool                   = False
        self._drag_section_copy: Section | None     = None

        # ---- Phase 1: undo for node deletion ----
        self._last_delete_for_undo: dict | None = None

        # ---- Phase 7: interactive section drawing ----
        self._new_sec_nodes: list[tuple[float, float]] = []
        self._new_sec_cursor: tuple[float, float] | None = None

        # ---- place-well mode ----
        self._placing_well_index: int | None = None

        # ---- pan state ----
        self._pan_anchor:  tuple[float, float] | None = None  # display px
        self._pan_xlim0:   tuple[float, float] | None = None
        self._pan_ylim0:   tuple[float, float] | None = None
        self._pan_inv      = None   # inverse transform captured at press

        # ---- last right-click map position ----
        self._rclick_xy: tuple[float, float] | None = None
        self._show_grid: bool = False

        # ---- render throttle and re-entry guard ----
        self._is_rendering = False
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(50)   # max 20 redraws/sec from signals
        self._redraw_timer.timeout.connect(self.render)

        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        from section_tool.style import BG_CANVAS as CANVAS_BG
        self._fig    = Figure(figsize=(8, 6), facecolor=CANVAS_BG, tight_layout=True)
        self._ax     = self._fig.add_subplot(111)
        self._ax.set_facecolor(CANVAS_BG)
        self._canvas = FigureCanvasQTAgg(self._fig)

        # Create toolbar (for its zoom-stack internals) but keep it hidden.
        # It is NOT in the layout — the tool palette owns all interaction.
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._toolbar.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._canvas)   # canvas only — toolbar hidden

        # Matplotlib events
        self._canvas.mpl_connect("button_press_event",   self._on_canvas_press)
        self._canvas.mpl_connect("motion_notify_event",  self._on_canvas_motion)
        self._canvas.mpl_connect("button_release_event", self._on_canvas_release)
        self._canvas.mpl_connect("scroll_event",         self._on_scroll)
        self._canvas.mpl_connect("key_press_event",      self._on_key_press)
        self._canvas.mpl_connect("resize_event",         lambda _: self._canvas.draw_idle())

        # Accept keyboard focus so Delete / Escape work
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _connect_signals(self) -> None:
        s = self._state
        s.project_changed.connect(self.request_render)
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
        # Clear hover/selection when active tool changes
        s.tool_changed.connect(self._on_tool_changed)

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

    def set_grid_visible(self, visible: bool) -> None:
        self._show_grid = visible
        self.render()

    def _render_vector_layers(self) -> None:
        """Render imported vector layers (shapefiles, geopackages, GeoJSON)."""
        layers = self._state.get_vector_layers()
        for lyr in layers:
            color    = lyr.get("color", "#FFAA00")
            features = lyr.get("features", [])
            for feat in features:
                geom   = feat.get("geometry") or {}
                gtype  = geom.get("type", "")
                coords = geom.get("coordinates", [])
                try:
                    if gtype in ("LineString", "3D LineString"):
                        xs = [c[0] for c in coords]
                        ys = [c[1] for c in coords]
                        self._ax.plot(xs, ys, color=color, lw=0.9,
                                      alpha=0.75, zorder=4)

                    elif gtype in ("MultiLineString", "3D MultiLineString"):
                        for line in coords:
                            xs = [c[0] for c in line]
                            ys = [c[1] for c in line]
                            self._ax.plot(xs, ys, color=color, lw=0.9,
                                          alpha=0.75, zorder=4)

                    elif gtype in ("Polygon", "3D Polygon"):
                        ring = coords[0]
                        xs = [c[0] for c in ring]
                        ys = [c[1] for c in ring]
                        self._ax.fill(xs, ys, color=color, alpha=0.12, zorder=3)
                        self._ax.plot(xs, ys, color=color, lw=0.7,
                                      alpha=0.75, zorder=4)

                    elif gtype in ("MultiPolygon", "3D MultiPolygon"):
                        for polygon in coords:
                            ring = polygon[0]
                            xs = [c[0] for c in ring]
                            ys = [c[1] for c in ring]
                            self._ax.fill(xs, ys, color=color, alpha=0.12, zorder=3)
                            self._ax.plot(xs, ys, color=color, lw=0.7,
                                          alpha=0.75, zorder=4)

                    elif gtype in ("Point", "3D Point"):
                        self._ax.plot(coords[0], coords[1], "o",
                                      color=color, ms=3, zorder=5)

                    elif gtype in ("MultiPoint", "3D MultiPoint"):
                        for pt in coords:
                            self._ax.plot(pt[0], pt[1], "o",
                                          color=color, ms=3, zorder=5)
                except (IndexError, TypeError):
                    continue

    def show_cursor_crosshair(self, map_x: float, map_y: float) -> None:
        """Show a crosshair at geographic position without a full re-render."""
        for a in getattr(self, "_crosshair_artists", []):
            try:
                a.remove()
            except Exception:
                pass
        try:
            vl  = self._ax.axvline(map_x, color="#FF4444", linewidth=0.8, alpha=0.7, zorder=20)
            hl  = self._ax.axhline(map_y, color="#FF4444", linewidth=0.8, alpha=0.7, zorder=20)
            dot = self._ax.plot(map_x, map_y, "o", color="#FF4444",
                                markersize=5, zorder=21)[0]
            self._crosshair_artists = [vl, hl, dot]
        except Exception:
            self._crosshair_artists = []
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Public API for MainWindow
    # ------------------------------------------------------------------

    @property
    def map_center(self) -> tuple[float, float]:
        """Current view center in map coordinates."""
        xl = self._ax.get_xlim()
        yl = self._ax.get_ylim()
        return (xl[0] + xl[1]) / 2, (yl[0] + yl[1]) / 2

    def zoom_to_all_data(self) -> None:
        """Zoom map to fit all loaded data with 15% padding."""
        self._apply_map_limits()
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def request_render(self, *_args) -> None:
        """Schedule a render on the next idle cycle (debounced for signal bursts)."""
        if not self._redraw_timer.isActive():
            self._redraw_timer.start()

    def render(self, *_args) -> None:
        """Full redraw of the map view."""
        if self._is_rendering:
            return
        # Guard against degenerate canvas size (causes AGG MemoryError)
        if self._canvas.width() < 4 or self._canvas.height() < 4:
            return
        self._is_rendering = True
        try:
            self._render_impl()
        finally:
            self._is_rendering = False

    @staticmethod
    def _configure_axes(ax) -> None:
        from section_tool.style import BG_CANVAS
        ax.set_facecolor(BG_CANVAS)
        ax.figure.patch.set_facecolor(BG_CANVAS)
        for spine in ax.spines.values():
            spine.set_color("#444455")
        ax.tick_params(colors="#AAAAAA", which="both", labelsize=7)
        ax.xaxis.label.set_color("#CCCCCC")
        ax.yaxis.label.set_color("#CCCCCC")

    def _render_impl(self) -> None:
        """Internal render body — called only from render() with re-entry guard held."""

        self._ax.clear()
        self._crosshair_artists = []   # reset after clear
        self._configure_axes(self._ax)

        self._render_vector_layers()
        self._render_seismic_coverage()
        self._render_surfaces()
        self._render_aoi()
        self._render_sections()
        self._render_wells()
        self._render_new_section_preview()

        # Set limits from data bounding box with 15% padding per axis independently.
        # No equal-aspect forced — the map fills its panel naturally.
        self._apply_map_limits()

        self._render_graticule()
        self._canvas.draw_idle()

    def _render_new_section_preview(self) -> None:
        """Phase 7: draw the in-progress section trace."""
        nodes = self._new_sec_nodes
        if not nodes:
            return
        xs = [n[0] for n in nodes]
        ys = [n[1] for n in nodes]
        self._ax.plot(xs, ys, "o--", color="#ff7f0e", lw=1.5,
                      markersize=5, zorder=8)
        # Rubber-band to cursor
        if self._new_sec_cursor is not None:
            cx, cy = self._new_sec_cursor
            self._ax.plot([xs[-1], cx], [ys[-1], cy],
                          ":", color="#ff7f0e", lw=1.2, alpha=0.6, zorder=8)

    def _apply_map_limits(self) -> None:
        """Set xlim/ylim from data bounding box with 15% padding per axis."""
        proj = self._state.project
        all_x: list[float] = []
        all_y: list[float] = []

        for sec in proj.sections:
            all_x.extend(sec.nodes[:, 0].tolist())
            all_y.extend(sec.nodes[:, 1].tolist())
        for well in proj.wells:
            if well.x != 0.0 or well.y != 0.0:
                all_x.append(well.x)
                all_y.append(well.y)
        for surf in proj.surfaces:
            try:
                xmn, ymn, xmx, ymx = surf.bounds()
                if xmx > xmn and ymx > ymn:
                    all_x.extend([xmn, xmx])
                    all_y.extend([ymn, ymx])
            except Exception:
                pass
        for ref in proj.seismic_refs:
            if ref.extent_x_max != ref.extent_x_min:
                all_x.extend([ref.extent_x_min, ref.extent_x_max])
                all_y.extend([ref.extent_y_min, ref.extent_y_max])

        if not all_x:
            self._ax.set_xlim(-500, 10500)
            self._ax.set_ylim(-500, 10500)
            return

        xmn, xmx = min(all_x), max(all_x)
        ymn, ymx = min(all_y), max(all_y)
        xpad = max((xmx - xmn) * 0.15, 500.0)
        ypad = max((ymx - ymn) * 0.15, 500.0)
        self._ax.set_xlim(xmn - xpad, xmx + xpad)
        self._ax.set_ylim(ymn - ypad, ymx + ypad)

    def _render_graticule(self) -> None:
        _LABEL_COLOR = "#606870"
        xmin, xmax = self._ax.get_xlim()
        ymin, ymax = self._ax.get_ylim()
        span_x = xmax - xmin
        span_y = ymax - ymin
        if span_x <= 0 or span_y <= 0:
            self._ax.set_xlabel("Easting (m)", color=_LABEL_COLOR)
            self._ax.set_ylabel("Northing (m)", color=_LABEL_COLOR)
            return

        x_interval = _nice_interval(span_x / 5)
        y_interval = _nice_interval(span_y / 5)
        xs = np.arange(math.floor(xmin / x_interval) * x_interval, xmax + x_interval, x_interval)
        ys = np.arange(math.floor(ymin / y_interval) * y_interval, ymax + y_interval, y_interval)

        if self._show_grid:
            grid_kw = dict(color="#252832", linewidth=0.6, linestyle="--", zorder=0)
            if len(xs) <= 500:
                for x in xs:
                    self._ax.axvline(x, **grid_kw)
            if len(ys) <= 500:
                for y in ys:
                    self._ax.axhline(y, **grid_kw)

        self._ax.set_xlabel("Easting (m)", color=_LABEL_COLOR)
        self._ax.set_ylabel("Northing (m)", color=_LABEL_COLOR)
        from matplotlib.ticker import MultipleLocator, FuncFormatter
        self._ax.xaxis.set_major_locator(MultipleLocator(x_interval))
        self._ax.yaxis.set_major_locator(MultipleLocator(y_interval))
        self._ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        self._ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        self._ax.tick_params(colors=_LABEL_COLOR, which="both", labelsize=8)
        self._render_scale_bar(xmin, xmax, ymin, ymax)

    def _render_scale_bar(self, xmin, xmax, ymin, ymax) -> None:
        """Draw a scale bar in the bottom-right corner."""
        span = xmax - xmin
        # Pick a round bar length: roughly 15% of view width
        raw = span * 0.15
        exp = math.floor(math.log10(max(raw, 1)))
        base = 10 ** exp
        for step in (1, 2, 5, 10):
            bar_len = step * base
            if bar_len >= raw:
                break

        # Position: 5% from right, 7% from bottom
        bx1 = xmax - span * 0.07 - bar_len
        bx2 = bx1 + bar_len
        by  = ymin + (ymax - ymin) * 0.05

        self._ax.plot([bx1, bx2], [by, by], color="black", linewidth=2.5, zorder=12,
                      solid_capstyle="butt")
        # End ticks
        tick_h = (ymax - ymin) * 0.01
        for bx in (bx1, bx2):
            self._ax.plot([bx, bx], [by - tick_h, by + tick_h],
                          color="black", linewidth=2.0, zorder=12)
        # Label
        if bar_len >= 1000:
            label = f"{bar_len / 1000:.0f} km"
        else:
            label = f"{bar_len:.0f} m"
        self._ax.text((bx1 + bx2) / 2, by + tick_h * 2.5, label,
                      ha="center", va="bottom", fontsize=7, zorder=12,
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    def _render_aoi(self) -> None:
        """Draw the project AOI polygon outline on the map."""
        aoi = getattr(self._state.project, "aoi", None)
        if aoi is None:
            return
        try:
            from shapely.geometry import mapping
            poly = aoi.polygon
            coords = list(poly.exterior.coords)
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            self._ax.fill(xs, ys, alpha=0.06, color="#44AAFF", zorder=1)
            self._ax.plot(xs, ys, color="#44AAFF", linewidth=1.2,
                          linestyle="-", alpha=0.8, zorder=3)
            self._ax.text(xs[0], ys[0], f" {aoi.name}",
                          fontsize=6, color="#44AAFF", va="bottom", zorder=5)
        except Exception:
            pass

    def _render_sections(self) -> None:
        active = self._state.active_section
        for i, section in enumerate(self._state.project.sections):
            # Use drag copy when dragging this section
            if (self._drag_active
                    and self._selected_node is not None
                    and self._selected_node[0] == i):
                display_sec = self._drag_section_copy
            else:
                display_sec = section

            is_active = (section is active) or (
                self._drag_active
                and self._selected_node is not None
                and self._selected_node[0] == i
                and active is section
            )
            color = _ACTIVE_COLOR if is_active else _INACTIVE_COLOR
            lw    = _ACTIVE_LW    if is_active else _INACTIVE_LW
            nodes = display_sec.nodes

            # Section polyline
            self._ax.plot(nodes[:, 0], nodes[:, 1],
                          color=color, linewidth=lw, zorder=3)

            # All nodes — default state (small, white-filled)
            ms, fc, ec, ew = _NODE_NORMAL
            self._ax.plot(nodes[:, 0], nodes[:, 1],
                          linestyle="none", marker="o",
                          markersize=ms, markerfacecolor=fc,
                          markeredgecolor=ec, markeredgewidth=ew, zorder=4)

            # Special-state overlay for one node (drag > selected > hover)
            spec_j    = None
            spec_style = None
            if self._drag_active and self._selected_node and self._selected_node[0] == i:
                spec_j     = self._selected_node[1]
                spec_style = _NODE_DRAG
            elif self._selected_node and self._selected_node[0] == i:
                spec_j     = self._selected_node[1]
                spec_style = _NODE_SELECTED
            elif (not self._drag_active
                  and self._hover_node and self._hover_node[0] == i
                  and not (self._selected_node and self._selected_node[0] == i)):
                spec_j     = self._hover_node[1]
                spec_style = _NODE_HOVER

            if spec_j is not None:
                ms, fc, ec, ew = spec_style
                n = nodes[spec_j]
                self._ax.plot(n[0], n[1],
                              linestyle="none", marker="o",
                              markersize=ms, markerfacecolor=fc,
                              markeredgecolor=ec, markeredgewidth=ew, zorder=6)

            # Section label
            mid   = len(nodes) // 2
            label = display_sec.name or f"Section {i}"
            self._ax.text(nodes[mid, 0], nodes[mid, 1], f" {label}",
                          fontsize=7, color=color, va="bottom", zorder=5)

    def _render_wells(self) -> None:
        for well in self._state.project.wells:
            self._ax.scatter(well.x, well.y, marker="^", s=50,
                             color=_WELL_COLOR, zorder=5)
            self._ax.text(well.x, well.y, f" {well.name}",
                          fontsize=7, color=_WELL_COLOR, va="bottom", zorder=5)

    def _render_surfaces(self) -> None:
        for surf in self._state.get_visible_surfaces():
            color = surf.display_color
            xmin, ymin, xmax, ymax = surf.bounds()
            if xmin == xmax or ymin == ymax:
                continue
            # Contour map clipped to visible extent
            try:
                xl = self._ax.get_xlim()
                yl = self._ax.get_ylim()
                gx0 = max(xl[0], xmin); gx1 = min(xl[1], xmax)
                gy0 = max(yl[0], ymin); gy1 = min(yl[1], ymax)
                if gx1 > gx0 and gy1 > gy0:
                    nx, ny = 60, 60
                    xs_g = np.linspace(gx0, gx1, nx)
                    ys_g = np.linspace(gy0, gy1, ny)
                    xx, yy = np.meshgrid(xs_g, ys_g)
                    zz = surf.sample_many(xx.ravel(), yy.ravel()).reshape(ny, nx)
                    if np.isfinite(zz).sum() > 20:
                        self._ax.contour(xx, yy, zz, levels=6,
                                         colors=[color], linewidths=0.7,
                                         alpha=0.7, zorder=2)
            except Exception:
                pass
            rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                              fill=False, edgecolor=color,
                              linewidth=0.8, linestyle="--", alpha=0.5, zorder=2)
            self._ax.add_patch(rect)
            self._ax.text(xmin, ymax, f" {surf.name}",
                          fontsize=6, color=color, va="bottom", zorder=5)

    def _render_seismic_coverage(self) -> None:
        """Draw each SEG-Y survey's spatial extent as a semi-transparent box."""
        for ref in self._state.project.seismic_refs:
            xmn, xmx = ref.extent_x_min, ref.extent_x_max
            ymn, ymx = ref.extent_y_min, ref.extent_y_max
            if xmn == xmx == ymn == ymx == 0.0:
                continue
            xs = [xmn, xmx, xmx, xmn, xmn]
            ys = [ymn, ymn, ymx, ymx, ymn]
            self._ax.fill(xs, ys, alpha=0.15, color="orange", zorder=1)
            self._ax.plot(xs, ys, color="orange", linewidth=1.5,
                          linestyle="--", zorder=2)
            cx = (xmn + xmx) / 2
            cy = (ymn + ymx) / 2
            w_km = (xmx - xmn) / 1000
            h_km = (ymx - ymn) / 1000
            label = ref.name
            if w_km > 0 and h_km > 0:
                label += f"\n{w_km:.0f}×{h_km:.0f} km"
            self._ax.text(cx, cy, label, fontsize=6, color="darkorange",
                          ha="center", va="center", zorder=3,
                          bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                    ec="none", alpha=0.6))

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _to_screen_px(self, xdata: float, ydata: float) -> tuple[float, float]:
        pt = self._ax.transData.transform([[xdata, ydata]])
        return float(pt[0, 0]), float(pt[0, 1])

    def _expand_view_for_point(self, x: float, y: float) -> None:
        """Expand axes limits when a drag moves a node outside the current view."""
        xl = list(self._ax.get_xlim())
        yl = list(self._ax.get_ylim())
        mx = max(abs(xl[1] - xl[0]) * 0.08, 50.0)
        my = max(abs(yl[1] - yl[0]) * 0.08, 50.0)
        changed = False
        if   x < xl[0]: xl[0] = x - mx; changed = True
        elif x > xl[1]: xl[1] = x + mx; changed = True
        y_lo, y_hi = min(yl), max(yl)
        if   y < y_lo: y_lo = y - my; changed = True
        elif y > y_hi: y_hi = y + my; changed = True
        if changed:
            self._ax.set_xlim(xl)
            self._ax.set_ylim(y_lo, y_hi)
            self._canvas.draw_idle()

    def _pixel_threshold(self, pixels: float = _LINE_HIT_PX) -> float:
        """Return *pixels* screen pixels expressed in data units (adapts to zoom)."""
        try:
            inv = self._ax.transData.inverted()
            p0  = inv.transform([0.0, 0.0])
            p1  = inv.transform([float(pixels), float(pixels)])
            return max(abs(float(p1[0]) - float(p0[0])),
                       abs(float(p1[1]) - float(p0[1])))
        except Exception:
            return float("inf")

    def _find_nearest_node(self, x: float, y: float) -> tuple[int, int] | None:
        """Data-space node hit-test using _NODE_HIT_PX threshold."""
        threshold = self._pixel_threshold(_NODE_HIT_PX)
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
        threshold = self._pixel_threshold()
        best_idx: int | None = None
        best_dist = float("inf")
        for i, section in enumerate(self._state.project.sections):
            d = _min_dist_to_polyline(x, y, section.nodes)
            if d < threshold and d < best_dist:
                best_dist = d
                best_idx  = i
        return best_idx

    # ------------------------------------------------------------------
    # Pan helpers
    # ------------------------------------------------------------------

    def _start_pan(self, event) -> None:
        self._pan_anchor = (event.x, event.y)
        self._pan_xlim0  = self._ax.get_xlim()
        self._pan_ylim0  = self._ax.get_ylim()
        self._pan_inv    = self._ax.transData.inverted()

    def _continue_pan(self, event) -> None:
        if self._pan_anchor is None:
            return
        d0 = self._pan_inv.transform(self._pan_anchor)
        d1 = self._pan_inv.transform([event.x, event.y])
        dx, dy = d0[0] - d1[0], d0[1] - d1[1]
        self._ax.set_xlim(self._pan_xlim0[0] + dx, self._pan_xlim0[1] + dx)
        self._ax.set_ylim(self._pan_ylim0[0] + dy, self._pan_ylim0[1] + dy)
        self._canvas.draw_idle()

    def _end_pan(self) -> None:
        self._pan_anchor = None

    # ------------------------------------------------------------------
    # Mouse event handlers
    # ------------------------------------------------------------------

    def _on_canvas_press(self, event) -> None:
        if event.inaxes is not self._ax:
            return

        # Place-well mode takes priority over all other tools
        if self._placing_well_index is not None and event.button == 1:
            if event.xdata is not None and event.ydata is not None:
                self._place_well_click(float(event.xdata), float(event.ydata))
            return

        # Right-click: context menu
        if event.button == 3:
            if event.xdata is not None and event.ydata is not None:
                self._rclick_xy = (float(event.xdata), float(event.ydata))
            self._show_map_context_menu()
            return

        # Double-click: zoom 2× centered on cursor (Shift = zoom out)
        if getattr(event, "dblclick", False) and event.button == 1:
            if event.xdata is not None and event.ydata is not None:
                from PySide6.QtWidgets import QApplication as _QApp
                from PySide6.QtCore import Qt as _Qt
                shift_held = bool(
                    _QApp.keyboardModifiers() & _Qt.KeyboardModifier.ShiftModifier
                )
                factor = 2.0 if shift_held else 0.5
                cx, cy = float(event.xdata), float(event.ydata)
                xl, yl = self._ax.get_xlim(), self._ax.get_ylim()
                relx = (cx - xl[0]) / max(xl[1] - xl[0], 1e-9)
                rely = (cy - yl[0]) / max(yl[1] - yl[0], 1e-9)
                nw = (xl[1] - xl[0]) * factor
                nh = (yl[1] - yl[0]) * factor
                self._ax.set_xlim(cx - nw * relx, cx + nw * (1 - relx))
                self._ax.set_ylim(cy - nh * rely, cy + nh * (1 - rely))
                self._canvas.draw_idle()
            return

        tool = self._state.active_tool

        # Middle button always pans
        if event.button == 2:
            self._start_pan(event)
            return

        if event.button == 1:
            # ---- Interactive section drawing (Phase 7) ----
            if tool == "new_section":
                x, y = event.xdata, event.ydata
                if x is None or y is None:
                    return
                is_dbl = getattr(event, "dblclick", False)
                if is_dbl and len(self._new_sec_nodes) >= 1:
                    # Double-click: add final node and finish
                    self._new_sec_nodes.append((float(x), float(y)))
                    self._finish_new_section()
                else:
                    self._new_sec_nodes.append((float(x), float(y)))
                    self.render()
                return

            if tool == "pan":
                self._start_pan(event)
                return

            if tool in _SELECT_TOOLS:
                x, y = event.xdata, event.ydata
                if x is None or y is None:
                    return

                # A-tool (node_edit): node selection has priority
                if tool in _EDIT_TOOLS:
                    self._mouse_pressed = True
                    px = getattr(event, "x", None)
                    py = getattr(event, "y", None)
                    self._press_px = (float(px), float(py)) if (px is not None and py is not None) else None
                    hit = self._find_nearest_node(x, y)
                    if hit is not None:
                        self._selected_node     = hit
                        self._drag_section_copy = copy.deepcopy(
                            self._state.project.sections[hit[0]]
                        )
                        self._drag_active = False
                        self.render()
                        return

                # Both V and A: section line selection
                sec_idx = self._find_nearest_section(x, y)
                if sec_idx is not None:
                    self._state.set_active_section(
                        self._state.project.sections[sec_idx]
                    )
                    self._selected_node     = None
                    self._drag_section_copy = None
                    self._drag_active       = False
                    return

                # Empty space → deselect node
                if self._selected_node is not None:
                    self._selected_node     = None
                    self._drag_section_copy = None
                    self._drag_active       = False
                    self.render()

    def _on_canvas_motion(self, event) -> None:
        # ---- New section rubber-band ----
        if self._new_sec_nodes and self._state.active_tool == "new_section":
            if event.xdata is not None and event.ydata is not None:
                self._new_sec_cursor = (float(event.xdata), float(event.ydata))
            else:
                self._new_sec_cursor = None
            self.render()
            return

        # ---- Pan ----
        if self._pan_anchor is not None:
            self._continue_pan(event)
            return

        # ---- Drag node ----
        if self._mouse_pressed and self._selected_node is not None:
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                return
            px = getattr(event, "x", None)
            py = getattr(event, "y", None)

            if not self._drag_active:
                if (px is not None and py is not None
                        and self._press_px is not None):
                    ppx, ppy = self._press_px
                    if math.hypot(float(px) - ppx, float(py) - ppy) < _DRAG_MIN_PX:
                        return
                self._drag_active = True

            self._drag_section_copy.move_node(self._selected_node[1], x, y)
            self.status_message.emit(f"E: {x:.0f}  N: {y:.0f}")
            self.render()
            # Expand view if node dragged outside current limits
            self._expand_view_for_point(x, y)
            return

        # ---- Coordinate readout + nearest-well distance ----
        if event.xdata is not None and event.ydata is not None and not self._drag_active:
            mx, my = float(event.xdata), float(event.ydata)
            self.cursor_map_pos.emit(mx, my)   # bidirectional crosshair
            msg = f"E: {mx:,.0f}   N: {my:,.0f}"
            wells = self._state.project.wells
            if wells:
                nearest = min(wells, key=lambda w: math.hypot(w.x - mx, w.y - my))
                d = math.hypot(nearest.x - mx, nearest.y - my)
                if d < 50_000:
                    msg += f"   ·   {d:,.0f} m from {nearest.name}"
            self.status_message.emit(msg)

        # ---- Hover (tool-dependent, no button held) ----
        tool = self._state.active_tool
        if tool in _EDIT_TOOLS and not self._mouse_pressed:
            self._update_hover(event)

    def _on_canvas_release(self, event) -> None:
        if event.button in (1, 2):
            # End pan (left button in pan-tool, or middle button)
            self._end_pan()

            # Commit drag
            self._mouse_pressed = False
            self._press_px      = None

            if self._drag_active:
                sec_idx, node_idx = self._selected_node
                sec_copy          = self._drag_section_copy

                self._drag_active       = False
                self._drag_section_copy = None
                self.status_message.emit("")

                self._state.update_section(sec_idx, sec_copy)
                new_pos = sec_copy.nodes[node_idx]
                self.section_node_moved.emit(
                    sec_idx, node_idx, float(new_pos[0]), float(new_pos[1])
                )

    def _on_scroll(self, event) -> None:
        if event.inaxes is not self._ax:
            return
        # 1.3× per tick — zoom in on "up", out on "down", centered on cursor
        zoom_in = getattr(event, "step", 0) > 0 or event.button == "up"
        factor = 1.0 / 1.3 if zoom_in else 1.3
        cx = event.xdata if event.xdata is not None else sum(self._ax.get_xlim()) / 2
        cy = event.ydata if event.ydata is not None else sum(self._ax.get_ylim()) / 2
        xl = self._ax.get_xlim()
        yl = self._ax.get_ylim()
        relx = (cx - xl[0]) / max(xl[1] - xl[0], 1e-9)
        rely = (cy - yl[0]) / max(yl[1] - yl[0], 1e-9)
        new_w = (xl[1] - xl[0]) * factor
        new_h = (yl[1] - yl[0]) * factor
        self._ax.set_xlim(cx - new_w * relx, cx + new_w * (1 - relx))
        self._ax.set_ylim(cy - new_h * rely, cy + new_h * (1 - rely))
        self._canvas.draw_idle()

    def _on_key_press(self, event) -> None:
        if event.key == "escape":
            if self._new_sec_nodes:
                self._new_sec_nodes.clear()
                self._new_sec_cursor = None
                self.render()
                return
            if self._drag_active:
                self._drag_active       = False
                self._drag_section_copy = copy.deepcopy(
                    self._state.project.sections[self._selected_node[0]]
                )
                self.status_message.emit("")
                self.render()
            elif self._selected_node is not None:
                self._selected_node     = None
                self._hover_node        = None
                self._drag_section_copy = None
                self.render()
        elif event.key == "delete":
            if self._selected_node is not None and not self._drag_active:
                self._delete_selected_node()
        elif event.key == "ctrl+z":
            self._state.undo()
        elif event.key in ("ctrl+shift+z", "ctrl+y"):
            self._state.redo()
        elif event.key == "enter":
            if self._new_sec_nodes and self._state.active_tool == "new_section":
                self._finish_new_section()

    # ------------------------------------------------------------------
    # Hover state
    # ------------------------------------------------------------------

    def _update_hover(self, event) -> None:
        if event.xdata is None or event.ydata is None:
            new_hover = None
        else:
            new_hover = self._find_nearest_node(
                float(event.xdata), float(event.ydata)
            )

        if new_hover != self._hover_node:
            self._hover_node = new_hover
            if new_hover is not None:
                self._canvas.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self._canvas.unsetCursor()
            self.render()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def _delete_selected_node(self) -> None:
        sec_idx, node_idx = self._selected_node
        section = self._state.project.sections[sec_idx]
        if section.n_nodes <= 2:
            return  # must keep at least 2 nodes
        # Store for Ctrl+Z undo
        self._last_delete_for_undo = {
            "sec_idx":   sec_idx,
            "node_idx":  node_idx,
            "x":         float(section.nodes[node_idx, 0]),
            "y":         float(section.nodes[node_idx, 1]),
        }
        sec_copy = copy.deepcopy(section)
        sec_copy.delete_node(node_idx)
        self._selected_node     = None
        self._drag_section_copy = None
        self._state.update_section(sec_idx, sec_copy)

    def _undo_last_delete(self) -> None:
        """Phase 1: restore the last deleted section node."""
        u = self._last_delete_for_undo
        if u is None:
            return
        self._last_delete_for_undo = None
        sec_idx = u["sec_idx"]
        proj = self._state.project
        if sec_idx >= len(proj.sections):
            return
        sec_copy = copy.deepcopy(proj.sections[sec_idx])
        # Re-insert at original position
        nodes = sec_copy.nodes
        new_nodes = np.insert(nodes, u["node_idx"], [u["x"], u["y"]], axis=0)
        import section_tool.core.section as _sec_mod
        restored = _sec_mod.Section(
            new_nodes,
            name=sec_copy.name,
            depth_domain=sec_copy.depth_domain,
            depth_units=sec_copy.depth_units,
            vertical_exaggeration=sec_copy.vertical_exaggeration,
            crs_epsg=sec_copy.crs_epsg,
        )
        self._state.update_section(sec_idx, restored)

    # ------------------------------------------------------------------
    # Section drawing helpers
    # ------------------------------------------------------------------

    def _finish_new_section(self) -> None:
        """Phase 7: commit the in-progress section trace to AppState."""
        nodes = self._new_sec_nodes
        self._new_sec_nodes = []
        self._new_sec_cursor = None
        if len(nodes) < 2:
            self.render()
            return
        from section_tool.core.section import Section as _Sec
        n_existing = len(self._state.project.sections)
        sec = _Sec(
            nodes,
            name=f"Section {n_existing + 1}",
            crs_epsg=self._state.project.crs_epsg,
        )
        self._state.add_section(sec)
        self._state.set_active_section(sec)
        # Return to select tool after drawing
        self._state.set_active_tool("select")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def apply_tool_cursor(self, tool_id: str) -> None:
        """Phase 5: set an appropriate cursor for the active tool."""
        from PySide6.QtCore import Qt as _Qt
        _map = {
            "select":       _Qt.CursorShape.ArrowCursor,
            "node_edit":    _Qt.CursorShape.CrossCursor,
            "pan":          _Qt.CursorShape.OpenHandCursor,
            "zoom":         _Qt.CursorShape.SizeAllCursor,
            "new_section":  _Qt.CursorShape.CrossCursor,
            "horizon_pick": _Qt.CursorShape.CrossCursor,
            "fault_pick":   _Qt.CursorShape.CrossCursor,
            "polygon":      _Qt.CursorShape.CrossCursor,
        }
        shape = _map.get(tool_id, _Qt.CursorShape.ArrowCursor)
        self._canvas.setCursor(_Qt.CursorShape(shape))

    def _on_sections_changed(self, *_args) -> None:
        self.request_render()

    def _on_wells_changed(self, *_args) -> None:
        self.request_render()

    def _on_surfaces_changed(self, *_args) -> None:
        self.request_render()

    def _on_seismic_changed(self, *_args) -> None:
        self.request_render()

    # ------------------------------------------------------------------
    # Place-well mode
    # ------------------------------------------------------------------

    def start_place_well(self, well_index: int) -> None:
        """Enter place-well mode: next left-click positions the well at that index."""
        self._placing_well_index = well_index
        self._canvas.setCursor(Qt.CursorShape.CrossCursor)
        wells = self._state.project.wells
        name = wells[well_index].name if well_index < len(wells) else "well"
        self.status_message.emit(f"Click on the map to place well '{name}'")

    def _place_well_click(self, x: float, y: float) -> None:
        import copy
        idx = self._placing_well_index
        self._placing_well_index = None
        wells = self._state.project.wells
        if idx is None or idx >= len(wells):
            self._canvas.setCursor(Qt.CursorShape.ArrowCursor)
            self.status_message.emit("")
            return
        wc = copy.copy(wells[idx])
        wc.x = x
        wc.y = y
        wc.deviation = wc.deviation.__class__.vertical(x, y)
        self._state.update_well(idx, wc)
        self._canvas.setCursor(Qt.CursorShape.ArrowCursor)
        self.status_message.emit("")

    def _show_map_context_menu(self) -> None:
        """Right-click context menu on the map canvas."""
        from PySide6.QtWidgets import QMenu
        from PySide6.QtGui import QCursor
        menu = QMenu(self)

        zoom_all = menu.addAction("Zoom to All Data")
        zoom_all.triggered.connect(self.zoom_to_all_data)

        menu.addSeparator()

        if self._rclick_xy is not None:
            rx, ry = self._rclick_xy
            sec_here = menu.addAction(f"Create E–W Section Here")
            sec_here.triggered.connect(
                lambda _, x=rx, y=ry: self._create_section_at(x, y, "ew"))
            sec_here_ns = menu.addAction(f"Create N–S Section Here")
            sec_here_ns.triggered.connect(
                lambda _, x=rx, y=ry: self._create_section_at(x, y, "ns"))

        # Create section through nearest well
        wells = self._state.project.wells
        if wells and self._rclick_xy is not None:
            rx, ry = self._rclick_xy
            nearest_well = min(
                wells, key=lambda w: math.hypot(w.x - rx, w.y - ry)
            )
            thru_well = menu.addAction(
                f"Create Section Through '{nearest_well.name}'"
            )
            thru_well.triggered.connect(
                lambda _, w=nearest_well: self._create_section_at(w.x, w.y, "ew"))

        menu.popup(QCursor.pos())

    def _create_section_at(self, cx: float, cy: float, orientation: str) -> None:
        """Create a 10km section centered at (cx, cy) in the given orientation."""
        import numpy as np
        from section_tool.core.section import Section
        half = 5_000.0
        if orientation == "ew":
            nodes = np.array([[cx - half, cy], [cx + half, cy]])
            name = f"E-W Section"
        else:
            nodes = np.array([[cx, cy - half], [cx, cy + half]])
            name = f"N-S Section"
        n = len(self._state.project.sections)
        sec = Section(nodes, name=f"{name} {n + 1}",
                      crs_epsg=self._state.project.crs_epsg)
        self._state.add_section(sec)
        self._state.set_active_section(sec)

    def _on_tool_changed(self, tool_id: str) -> None:
        # Cancel section drawing when switching away
        if tool_id != "new_section" and self._new_sec_nodes:
            self._new_sec_nodes.clear()
            self._new_sec_cursor = None
            self.render()
        # Clear editing state when leaving an edit tool
        if tool_id not in _EDIT_TOOLS:
            if self._drag_active:
                self._drag_active       = False
                self._drag_section_copy = None
                self.status_message.emit("")
            self._selected_node = None
            self._hover_node    = None
            self._canvas.unsetCursor()
            self.render()


# ---------------------------------------------------------------------------
# Grid / graticule helpers
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


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------

def _min_dist_to_polyline(x: float, y: float, nodes: np.ndarray) -> float:
    min_d = float("inf")
    for i in range(len(nodes) - 1):
        ax, ay = float(nodes[i, 0]), float(nodes[i, 1])
        bx, by = float(nodes[i + 1, 0]), float(nodes[i + 1, 1])
        dx, dy  = bx - ax, by - ay
        seg_len2 = dx * dx + dy * dy
        if seg_len2 == 0.0:
            d = math.hypot(x - ax, y - ay)
        else:
            t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / seg_len2))
            d = math.hypot(x - ax - t * dx, y - ay - t * dy)
        if d < min_d:
            min_d = d
    return min_d
