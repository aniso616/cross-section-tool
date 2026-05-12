from __future__ import annotations

import copy
import math

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from cross_section_tool.app_state import AppState
from cross_section_tool.core.section import Section


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NODE_HIT_PX  = 12    # node selection/hover radius in screen pixels
_LINE_HIT_PX  = 8     # section-line selection tolerance in screen pixels
_DRAG_MIN_PX  = 3     # pixels of movement before drag activates

# Section line style
_ACTIVE_COLOR   = "#1f77b4"
_INACTIVE_COLOR = "#999999"
_ACTIVE_LW      = 2.0
_INACTIVE_LW    = 1.0

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

        # ---- pan state ----
        self._pan_anchor:  tuple[float, float] | None = None  # display px
        self._pan_xlim0:   tuple[float, float] | None = None
        self._pan_ylim0:   tuple[float, float] | None = None
        self._pan_inv      = None   # inverse transform captured at press

        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self._fig    = Figure(figsize=(8, 6), tight_layout=True)
        self._ax     = self._fig.add_subplot(111)
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

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, *_args) -> None:
        """Full redraw of the map view."""
        self._ax.clear()
        self._ax.set_aspect("equal", adjustable="datalim")

        self._render_seismic_coverage()
        self._render_surfaces()
        self._render_sections()
        self._render_wells()
        self._render_new_section_preview()
        self._render_graticule()

        # Sensible default extent when nothing is loaded
        proj = self._state.project
        if not proj.sections and not proj.wells and not proj.surfaces:
            self._ax.set_xlim(-500, 10500)
            self._ax.set_ylim(-500, 10500)

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

    def _render_graticule(self) -> None:
        xmin, xmax = self._ax.get_xlim()
        ymin, ymax = self._ax.get_ylim()
        span_x = xmax - xmin
        span_y = ymax - ymin
        if span_x <= 0 or span_y <= 0:
            self._ax.set_xlabel("Easting (m)")
            self._ax.set_ylabel("Northing (m)")
            return

        interval = _nice_interval(max(span_x, span_y) / 5)
        xs = np.arange(math.floor(xmin / interval) * interval, xmax + interval, interval)
        ys = np.arange(math.floor(ymin / interval) * interval, ymax + interval, interval)
        grid_kw = dict(color="#cccccc", linewidth=0.5, linestyle="--", zorder=0)
        for x in xs:
            self._ax.axvline(x, **grid_kw)
        for y in ys:
            self._ax.axhline(y, **grid_kw)

        self._ax.set_xlabel("Easting (m)")
        self._ax.set_ylabel("Northing (m)")
        from matplotlib.ticker import MultipleLocator
        self._ax.xaxis.set_major_locator(MultipleLocator(interval))
        self._ax.yaxis.set_major_locator(MultipleLocator(interval))
        self._ax.ticklabel_format(style="plain", axis="both")

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
        for surf in self._state.project.surfaces:
            xmin, xmax, ymin, ymax = surf.extent()
            w, h = xmax - xmin, ymax - ymin
            if w <= 0 or h <= 0:
                continue
            rect = Rectangle((xmin, ymin), w, h,
                              fill=False, edgecolor=_SURFACE_COLOR,
                              linewidth=1.5, linestyle="--", zorder=2)
            self._ax.add_patch(rect)
            self._ax.text(xmin, ymax, f" {surf.name}",
                          fontsize=6, color=_SURFACE_COLOR, va="bottom", zorder=5)

    def _render_seismic_coverage(self) -> None:
        pass  # placeholder for future integration

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
        factor = 0.85 if (getattr(event, "step", 0) > 0 or event.button == "up") else 1.0 / 0.85
        cx = event.xdata if event.xdata is not None else sum(self._ax.get_xlim()) / 2
        cy = event.ydata if event.ydata is not None else sum(self._ax.get_ylim()) / 2
        xl = self._ax.get_xlim()
        yl = self._ax.get_ylim()
        self._ax.set_xlim([cx + (x - cx) * factor for x in xl])
        self._ax.set_ylim([cy + (y - cy) * factor for y in yl])
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
        import cross_section_tool.core.section as _sec_mod
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
        from cross_section_tool.core.section import Section as _Sec
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
            "zoom":         _Qt.CursorShape.SizeFCursor,
            "new_section":  _Qt.CursorShape.CrossCursor,
            "horizon_pick": _Qt.CursorShape.CrossCursor,
            "fault_pick":   _Qt.CursorShape.CrossCursor,
            "polygon":      _Qt.CursorShape.CrossCursor,
        }
        shape = _map.get(tool_id, _Qt.CursorShape.ArrowCursor)
        self._canvas.setCursor(_Qt.CursorShape(shape))

    def _on_sections_changed(self, *_args) -> None:
        self.render()

    def _on_wells_changed(self, *_args) -> None:
        self.render()

    def _on_surfaces_changed(self, *_args) -> None:
        self.render()

    def _on_seismic_changed(self, *_args) -> None:
        self.render()

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
