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
    QComboBox,
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

_PICK_HIT_PX      = 10    # pick-point hit-test radius in screen pixels
_LINE_HIT_PX      = 8     # pick-line hit-test tolerance in screen pixels
_PICK_DRAG_PX     = 3     # minimum movement before drag activates
_OBJ_DRAG_PX      = 3     # minimum movement before object-move activates
_SNAP_THRESHOLD   = 15    # snap radius in screen pixels
_DEFAULT_DEPTH    = 5000.0

_DEPTH_UNITS = ["m", "ft", "km", "mi", "ms", "s"]

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
    pick_ended           = Signal()       # emitted when pick sequence ends
    node_selected        = Signal(str, int, int)   # Phase 3: (cat, obj_idx, pt_idx)

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state = state

        # ---- seismic cache ----
        self._seismic_cache: dict[str, SeismicDataset] = {}

        # ---- active tool flags (set by MainWindow._on_tool_changed) ----
        self._picking_active:  bool = False   # horizon_pick tool
        self._fault_picking:   bool = False   # fault_pick tool
        self._polygon_drawing: bool = False
        # Phase 2: reference-line placement tool
        self._ref_line_tool:   str | None = None   # "h_ref"|"v_ref"|"a_ref"|None
        # Phase 2: A-Ref two-click anchor
        self._aref_anchor:     tuple[float, float] | None = None
        # Phase 3: stub tools
        self._construct_tool:  str | None = None   # "extend"|"trim"|"parallel"

        # ---- polygon in-progress ----
        self._polygon_vertices: list[tuple[float, float]] = []

        # ---- display mode ----
        self._display_mode: Literal["variable_density", "wiggle"] = "variable_density"

        # ---- Phase 2: object selection state machine ----
        # mode: "idle" → "object_selected" → "edit_mode"
        self._sv_mode:         str                        = "idle"
        self._selected_object: tuple[str, int] | None    = None   # (cat, obj_idx)
        self._hover_object:    tuple[str, int] | None    = None
        # Phase 3: whole-object drag
        self._object_drag_active:   bool                       = False
        self._object_drag_press_pt: tuple[float, float] | None = None
        self._object_drag_origin:   HorizonPick | None         = None

        # ---- pick-node interaction (edit mode only) ----
        # _pick_ref: (category, obj_idx, FULL_ARRAY_pt_idx)
        self._pick_hover:    tuple[str, int, int] | None = None
        self._pick_selected: tuple[str, int, int] | None = None
        self._pick_drag:     bool                        = False
        self._pick_press_px: tuple[float, float] | None = None
        self._pick_copy:     HorizonPick | None          = None

        # ---- Phase 5: snapping ----
        self._snap_active: bool                       = True
        self._snap_point:  tuple[float, float] | None = None

        # drag preview (set during motion, consumed on release)
        self._object_drag_preview = None

        # Phase 1: single-level undo for pick deletion
        self._last_delete_for_undo: dict | None = None

        # FIX 2: track whether axis limits have been initialised for the
        # current section; reset to False whenever the section changes so
        # the first render gets default limits, subsequent renders preserve
        # the user's zoom/pan state.
        self._ax_limits_set: bool = False

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
        self._fig    = Figure(figsize=(10, 6), facecolor="white")
        self._ax     = self._fig.add_subplot(111)
        self._ax.set_facecolor("white")
        self._canvas = FigureCanvasQTAgg(self._fig)

        # Hidden toolbar — kept for zoom stack; NOT in the layout.
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._toolbar.hide()

        # Header bar: [section name] [depth-units combo] [VE spinbox] [VE lock]
        self._header = QWidget()
        self._header.setFixedHeight(28)
        self._header.setStyleSheet("background: #f5f5f5; border-bottom: 1px solid #ddd;")
        hl = QHBoxLayout(self._header)
        hl.setContentsMargins(8, 2, 8, 2)
        hl.setSpacing(6)
        self._section_name_label = QLabel("— no section —")
        self._section_name_label.setStyleSheet("color: #444; font-size: 9px;")
        hl.addWidget(self._section_name_label)
        hl.addStretch()

        # Depth units combo
        hl.addWidget(QLabel("Units:"))
        self._depth_units_combo = QComboBox()
        self._depth_units_combo.setFixedWidth(52)
        self._depth_units_combo.setToolTip("Depth / time axis units")
        for u in _DEPTH_UNITS:
            self._depth_units_combo.addItem(u)
        self._depth_units_combo.currentIndexChanged.connect(self._on_depth_units_changed)
        hl.addWidget(self._depth_units_combo)

        # VE spinbox
        hl.addWidget(QLabel("VE:"))
        self._ve_spin = QDoubleSpinBox()
        self._ve_spin.setRange(0.5, 20.0)
        self._ve_spin.setSingleStep(0.5)
        self._ve_spin.setValue(1.0)
        self._ve_spin.setFixedWidth(60)
        self._ve_spin.setDecimals(1)
        self._ve_spin.setToolTip(
            "Vertical exaggeration (1.0 = true scale)\n"
            "Higher values stretch depth axis, steepening apparent dips."
        )
        self._ve_spin.valueChanged.connect(self._on_ve_changed)
        hl.addWidget(self._ve_spin)

        # VE lock — icon toggles between 🔒 and 🔓
        self._ve_lock_btn = QPushButton("\U0001F513")   # 🔓 (unlocked default)
        self._ve_lock_btn.setCheckable(True)
        self._ve_lock_btn.setFixedSize(24, 22)
        self._ve_lock_btn.setToolTip(
            "Lock VE: when locked, the same vertical exaggeration applies\n"
            "to all sections and is preserved when switching between them.\n"
            "Scroll or click to unlock."
        )
        self._ve_lock_btn.setStyleSheet(
            "QPushButton { border: 1px solid #bbb; border-radius: 3px; font-size: 11px; }"
            "QPushButton:checked { background: #d0e8ff; border-color: #5599cc; }"
        )
        self._ve_lock_btn.toggled.connect(
            lambda locked: self._ve_lock_btn.setText("\U0001F512" if locked else "\U0001F513")
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
        s.reference_line_added.connect(self._on_data_changed)
        s.reference_line_removed.connect(self._on_data_changed)
        s.reference_line_modified.connect(self._on_data_changed)

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

    def render_to_figure(
        self, width_inches: float = 10.0, height_inches: float = 6.0,
        dpi: int = 150
    ):
        """Phase 8: render the current section to an independent Matplotlib Figure.

        Returns a :class:`matplotlib.figure.Figure` suitable for saving to
        PNG/SVG/PDF without affecting the on-screen display.
        """
        from matplotlib.figure import Figure as _Figure
        fig = _Figure(figsize=(width_inches, height_inches))
        ax  = fig.add_subplot(111)
        section = self._state.active_section
        if section is None:
            return fig
        old_ax, self._ax = self._ax, ax
        try:
            self._setup_axes(section)
            if self._ax_limits_set:
                self._ax.set_xlim(old_ax.get_xlim())
                self._ax.set_ylim(old_ax.get_ylim())
            self._render_seismic(section)
            self._render_grid(section)
            self._render_section_ends(section)
            self._render_reference_lines(section)
            self._render_polygons(section)
            self._render_surfaces(section)
            self._render_faults(section)
            self._render_horizons(section)
            self._render_wells(section)
            self._render_annotations(section)
        finally:
            self._ax = old_ax
        return fig

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
            "h_ref":        _Qt.CursorShape.CrossCursor,
            "v_ref":        _Qt.CursorShape.CrossCursor,
            "a_ref":        _Qt.CursorShape.CrossCursor,
            "measure":      _Qt.CursorShape.CrossCursor,
        }
        shape = _map.get(tool_id, _Qt.CursorShape.ArrowCursor)
        self._canvas.setCursor(_Qt.CursorShape(shape))

    def set_ref_line_tool(self, tool_id: str) -> None:
        """Phase 2+3: activate/deactivate reference-line and construct tools."""
        ref_tools     = {"h_ref", "v_ref", "a_ref"}
        construct_tools = {"extend", "trim", "parallel"}
        self._ref_line_tool  = tool_id if tool_id in ref_tools      else None
        self._construct_tool = tool_id if tool_id in construct_tools else None
        self._aref_anchor    = None  # reset any in-progress A-Ref

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
        # FIX 2: save user's zoom/pan limits before the clear wipes them
        if self._ax_limits_set:
            _saved_xl = self._ax.get_xlim()
            _saved_yl = self._ax.get_ylim()
        else:
            _saved_xl = _saved_yl = None

        self._ax.clear()
        section = self._state.active_section
        if section is None:
            self._section_name_label.setText("— no section —")
            self._ve_spin.setEnabled(False)
            self._depth_units_combo.setEnabled(False)
            self._ax_limits_set = False
            self._canvas.draw_idle()
            return

        self._section_name_label.setText(section.name or "Unnamed section")
        self._ve_spin.setEnabled(True)
        self._depth_units_combo.setEnabled(True)
        # Sync depth units combo
        u = section.depth_units if section.depth_units in _DEPTH_UNITS else "m"
        self._depth_units_combo.blockSignals(True)
        self._depth_units_combo.setCurrentIndex(_DEPTH_UNITS.index(u))
        self._depth_units_combo.blockSignals(False)
        # Sync VE spinbox only when not locked
        if not self._ve_lock_btn.isChecked():
            self._ve_spin.blockSignals(True)
            self._ve_spin.setValue(section.vertical_exaggeration)
            self._ve_spin.blockSignals(False)

        self._setup_axes(section)          # sets default limits
        # FIX 2: restore user zoom/pan if limits were already customised
        if _saved_xl is not None:
            self._ax.set_xlim(_saved_xl)
            self._ax.set_ylim(_saved_yl)
        else:
            self._ax_limits_set = True     # default limits now active

        self._render_seismic(section)
        self._render_grid(section)
        self._render_section_ends(section)
        self._render_reference_lines(section)
        self._render_polygons(section)
        self._render_surfaces(section)
        self._render_faults(section)
        self._render_horizons(section)
        self._render_wells(section)
        self._render_rubber_band(section)
        self._render_snap_indicator()
        self._render_polygon_in_progress()
        self._render_annotations(section)
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
        self._fig.subplots_adjust(left=0.10, right=0.97, top=0.97, bottom=0.09)

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

    def _render_annotations(self, section: Section) -> None:
        """Phase 6: draw text annotations (and optional leader lines)."""
        for ann in self._state.project.annotations:
            if ann.section_name and ann.section_name != section.name:
                continue
            r, g, b = ann.color
            color = "#{:02x}{:02x}{:02x}".format(r, g, b)
            px, pz = ann.position
            if ann.anchor_point is not None:
                ax_, az_ = ann.anchor_point
                self._ax.annotate(
                    ann.text,
                    xy=(ax_, az_), xytext=(px, pz),
                    fontsize=ann.font_size, color=color,
                    rotation=ann.rotation_degrees,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
                    zorder=11,
                )
            else:
                self._ax.text(
                    px, pz, ann.text,
                    fontsize=ann.font_size, color=color,
                    rotation=ann.rotation_degrees, zorder=11,
                )

    def _render_section_ends(self, section: Section) -> None:
        """Draw vertical end-cap lines at x=0 and x=total_length."""
        total = section.total_length()
        yl    = self._ax.get_ylim()
        ylo, yhi = min(yl), max(yl)
        kw = dict(color="#666666", linewidth=1.5, alpha=0.7, zorder=2,
                  solid_capstyle="butt")
        self._ax.plot([0, 0],         [ylo, yhi], **kw)
        self._ax.plot([total, total], [ylo, yhi], **kw)

    def _render_reference_lines(self, section: Section) -> None:
        """Phase 2/4: horizontal, vertical, and angled construction lines."""
        xl = self._ax.get_xlim()
        yl = self._ax.get_ylim()
        ylo, yhi = min(yl), max(yl)
        kw = dict(color="#aaaaaa", linewidth=0.8, linestyle=(0, (6, 4)), zorder=1)
        for rl in self._state.project.reference_lines:
            if not rl.visible:
                continue
            label = rl.name or ""
            if rl.kind == "horizontal":
                self._ax.axhline(rl.value, **kw)
                if label:
                    self._ax.text(xl[1], rl.value, f" {label}", fontsize=6,
                                  color="#999", va="center", ha="right", zorder=1)
            elif rl.kind == "vertical":
                self._ax.axvline(rl.value, **kw)
                if label:
                    self._ax.text(rl.value, ylo, f" {label}", fontsize=6,
                                  color="#999", va="bottom", ha="left",
                                  rotation=90, zorder=1)
            elif rl.kind == "angled":
                # Extend far beyond view in both directions, clip to axes
                ang = math.radians(rl.angle_deg)
                far = max(abs(xl[1] - xl[0]), abs(yhi - ylo)) * 10
                dx  = math.cos(ang) * far
                dy  = -math.sin(ang) * far   # depth increases downward
                self._ax.plot(
                    [rl.anchor_x - dx, rl.anchor_x + dx],
                    [rl.anchor_y - dy, rl.anchor_y + dy],
                    **kw,
                )
                if label:
                    self._ax.text(rl.anchor_x, rl.anchor_y, f" {label}",
                                  fontsize=6, color="#999", zorder=1)

        # A-Ref rubber band (anchor set, cursor pending)
        if self._ref_line_tool == "a_ref" and self._aref_anchor and self._cursor_data:
            ax_, ay_ = self._aref_anchor
            cx, cy   = self._cursor_data
            dx, dy   = cx - ax_, cy - ay_
            ang_d    = math.degrees(math.atan2(-dy, dx))
            self._ax.plot([ax_, cx], [ay_, cy],
                          color="#888", lw=1.0, linestyle="--", zorder=9)
            self._ax.text(cx, cy, f"  {ang_d:.0f}°", fontsize=7,
                          color="#555", zorder=9)

    def _render_snap_indicator(self) -> None:
        """Phase 5: small crosshair at the snapped cursor position."""
        if self._snap_point is None:
            return
        sx, sy = self._snap_point
        s = 6   # half-size in screen pixels
        try:
            inv = self._ax.transData.inverted()
            p0  = inv.transform([0, 0])
            p1  = inv.transform([s, s])
            dx  = abs(float(p1[0]) - float(p0[0]))
            dy  = abs(float(p1[1]) - float(p0[1]))
        except Exception:
            return
        self._ax.plot([sx - dx, sx + dx], [sy, sy],
                      color="#ff8800", lw=1.2, zorder=12)
        self._ax.plot([sx, sx], [sy - dy, sy + dy],
                      color="#ff8800", lw=1.2, zorder=12)

    # ------------------------------------------------------------------
    # Object renderers
    # ------------------------------------------------------------------

    @staticmethod
    def _mpl_linestyle(style: str) -> str:
        return {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}.get(style, "-")

    def _is_active_pick(self, category: str, obj_idx: int) -> bool:
        return (self._state.active_pick_category == category and
                self._state.active_pick_index == obj_idx)

    def _render_pick_object(
        self, category: str, obj_idx: int, hp: HorizonPick,
        section: Section, marker: str, default_ls: str,
    ) -> None:
        """Shared renderer for horizons and faults."""
        # Phase 3: use drag preview if available
        preview = getattr(self, "_object_drag_preview", None)
        if (preview is not None
                and preview[0] == category and preview[1] == obj_idx):
            hp = preview[2]
        # Phase 1: only picks belonging to this section (+ global picks)
        sec_idxs = hp.section_indices(section.name)
        d_sec = hp._distances[sec_idxs]
        z_sec = hp._depths[sec_idxs]
        if len(d_sec) == 0:
            return

        lw       = getattr(hp, "line_width", 1.5)
        ct       = getattr(hp, "contact_type", "conformable") if category == "Horizons" else None
        ft       = getattr(hp, "fault_type", "normal")        if category == "Faults"   else None
        # Phase A/B: contact/fault type overrides line style for non-conformable
        decorated = (
            (ct is not None and ct != "conformable" and ct != "marker_bed")
            or (ft is not None and ft != "strike_slip")
        )
        if ct == "marker_bed":
            ls = (0, (8, 4))        # custom dash pattern
        elif decorated:
            ls = "-"                # decorations handle the style
        else:
            ls = self._mpl_linestyle(getattr(hp, "line_style", default_ls))

        is_active   = self._is_active_pick(category, obj_idx)
        is_selected = (self._selected_object == (category, obj_idx))
        is_edit     = (self._sv_mode == "edit_mode" and is_selected)

        render_lw = lw * 1.6 if is_active else lw
        zorder    = 4 if (is_active or is_selected) else 3

        # Selection glow (Phase 2)
        if is_selected:
            self._ax.plot(d_sec, z_sec, color=hp.color,
                          linewidth=render_lw * 3, alpha=0.20,
                          zorder=zorder - 1, solid_capstyle="round")

        # Only draw the main line for non-decorated types (decorated types
        # draw their own lines in _render_line_decoration)
        if not decorated:
            self._ax.plot(d_sec, z_sec, color=hp.color,
                          linewidth=render_lw, linestyle=ls, zorder=zorder)

        # Phase A/B: contact-type / fault-type decorations
        if len(d_sec) >= 2:
            self._render_line_decoration(hp, d_sec, z_sec, category, lw)

        # Phase 2: nodes only in edit mode for this object
        if is_edit:
            for fi_full in sec_idxs:
                d = float(hp._distances[fi_full])
                z = float(hp._depths[fi_full])
                ms, fc, ec, ew = self._pick_point_style(category, obj_idx, fi_full)
                self._ax.plot(d, z, marker,
                              markersize=ms, markerfacecolor=fc,
                              markeredgecolor=ec, markeredgewidth=ew, zorder=5)

    def _render_line_decoration(
        self, hp, d_sec: np.ndarray, z_sec: np.ndarray,
        category: str, base_lw: float
    ) -> None:
        """Phase A/B: draw decorations derived from contact_type / fault_type."""
        ct = getattr(hp, "contact_type", "conformable") if category == "Horizons" \
             else None
        ft = getattr(hp, "fault_type",   "normal") if category == "Faults" \
             else None

        if ct in ("unconformity", "angular_unconformity"):
            xw, yw = _wavy_coords(self._ax, d_sec, z_sec, 3.0, 20.0)
            self._ax.plot(xw, yw, color=hp.color, lw=base_lw, zorder=3)

        elif ct == "disconformity":
            xw, yw = _wavy_coords(self._ax, d_sec, z_sec, 3.0, 20.0)
            self._ax.plot(xw, yw, color=hp.color, lw=base_lw,
                          linestyle="--", zorder=3)

        elif ct == "intrusive_contact":
            ticks = _line_ticks(self._ax, d_sec, z_sec, 30.0, 6.0, 1.0)
            for x0, y0, x1, y1 in ticks:
                self._ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                                  arrowprops=dict(arrowstyle="x", color=hp.color,
                                                  lw=0.8), zorder=4)

        elif ct == "sequence_boundary":
            self._ax.plot(d_sec, z_sec, color=hp.color, lw=2.5, zorder=3)

        elif ct == "maximum_flooding_surface":
            tris = _line_triangles(self._ax, d_sec, z_sec, 40.0, 7.0, -1.0)
            from matplotlib.patches import Polygon as MplPoly
            for verts in tris:
                patch = MplPoly(verts, closed=True,
                                facecolor=hp.color, edgecolor=hp.color,
                                lw=0.5, zorder=4)
                self._ax.add_patch(patch)

        elif ft == "reverse" or ft == "thrust":
            lw_line = base_lw * (1.3 if ft == "thrust" else 1.0)
            self._ax.plot(d_sec, z_sec, color=hp.color, lw=lw_line, zorder=3)
            side = 1.0 if getattr(hp, "dip_direction", "right") == "right" else -1.0
            tris = _line_triangles(self._ax, d_sec, z_sec, 40.0, 8.0, side)
            from matplotlib.patches import Polygon as MplPoly
            for verts in tris:
                patch = MplPoly(verts, closed=True,
                                facecolor=hp.color, edgecolor=hp.color,
                                lw=0.5, zorder=4)
                self._ax.add_patch(patch)

        elif ft in ("normal", "growth_fault"):
            self._ax.plot(d_sec, z_sec, color=hp.color, lw=base_lw, zorder=3)
            side = 1.0 if getattr(hp, "dip_direction", "right") == "right" else -1.0
            ticks = _line_ticks(self._ax, d_sec, z_sec, 40.0, 8.0, side)
            for x0, y0, x1, y1 in ticks:
                self._ax.plot([x0, x1], [y0, y1],
                              color=hp.color, lw=0.9, zorder=4)

        elif ft == "detachment":
            self._ax.plot(d_sec, z_sec, color=hp.color, lw=base_lw * 2, zorder=3)

        # else: conformable / strike_slip / marker_bed — rendered by main plot above

    def _render_horizons(self, section: Section) -> None:
        for obj_idx, hp in enumerate(self._state.project.horizon_picks):
            self._render_pick_object("Horizons", obj_idx, hp, section, "o", "solid")

    def _render_faults(self, section: Section) -> None:
        for obj_idx, fp in enumerate(self._state.project.fault_picks):
            self._render_pick_object("Faults", obj_idx, fp, section, "D", "dashed")

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
            # Resolve formation for lithology pattern (Phase D)
            hatch = None
            formation_name = getattr(poly, "formation", "")
            if formation_name:
                fm = self._state.project.strat_column.get_formation(formation_name)
                if fm is not None and fm.lithology_pattern != "none":
                    raw_hatch = _LITHOLOGY_HATCH.get(fm.primary_lithology)
                    if raw_hatch:
                        # Scale density: default pattern_scale=1.0 → hatch twice
                        reps = max(1, round(1.0 / max(fm.pattern_scale, 0.1)))
                        hatch = raw_hatch * reps

            patch = MplPolygon(verts, closed=True,
                               facecolor=poly.fill_color, alpha=poly.fill_alpha,
                               edgecolor=poly.edge_color, linewidth=poly.edge_width,
                               hatch=hatch, zorder=2)
            self._ax.add_patch(patch)
            # Hatch overlay in a separate transparent patch so hatch color = black
            if hatch:
                hatch_patch = MplPolygon(verts, closed=True,
                                         facecolor="none", alpha=0.35,
                                         edgecolor="black", linewidth=0,
                                         hatch=hatch, zorder=2)
                self._ax.add_patch(hatch_patch)
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
        """V-shaped dashed ghost line; shows angle-snap guide when Shift held."""
        if not (self._picking_active or self._fault_picking):
            return
        if self._cursor_data is None:
            return
        # Phase 3: Shift → angle-snap guide
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt as _Qt
        shift_held = bool(
            QApplication.keyboardModifiers() & _Qt.KeyboardModifier.ShiftModifier
        )
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
        # If Shift held, override cursor with angle-snapped position
        if shift_held:
            cx, cy = self._apply_angle_snap(cx, cy)

        left     = d < cx
        right    = d > cx
        if left.any():
            li = int(np.where(left)[0][-1])
            self._ax.plot([d[li], cx], [z[li], cy], **rb_kw)
        if right.any():
            ri = int(np.where(right)[0][0])
            self._ax.plot([cx, d[ri]], [cy, z[ri]], **rb_kw)

        # Phase 3: show angle-snap guide line
        if shift_held:
            _, last_d, last_z = self._get_active_pick_last_point()
            if last_d is not None:
                # Draw full guide line through snap direction
                ang = math.atan2(-(cy - last_z), cx - last_d)
                xl = self._ax.get_xlim()
                far = abs(xl[1] - xl[0])
                self._ax.plot(
                    [last_d - far * math.cos(ang), last_d + far * math.cos(ang)],
                    [last_z + far * math.sin(ang), last_z - far * math.sin(ang)],
                    color=color, lw=0.5, linestyle=":", alpha=0.4, zorder=7,
                )

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
        """Return (category, obj_idx, FULL_pt_idx) for nearest pick in current section."""
        section = self._state.active_section
        sec_name = section.name if section is not None else ""
        ex, ey = self._to_screen_px_sv(event_x, event_y)
        best = None
        best_dist = float("inf")

        def _check(category, picks):
            nonlocal best, best_dist
            for oi, hp in enumerate(picks):
                # Only check picks visible on this section (Phase 1)
                for fi_full in hp.section_indices(sec_name):
                    nx, ny = self._to_screen_px_sv(
                        float(hp._distances[fi_full]), float(hp._depths[fi_full])
                    )
                    d = math.hypot(ex - nx, ey - ny)
                    if d <= _PICK_HIT_PX and d < best_dist:
                        best_dist = d
                        best = (category, oi, int(fi_full))

        _check("Horizons", self._state.project.horizon_picks)
        _check("Faults",   self._state.project.fault_picks)
        return best

    def _find_nearest_pick_line(
        self, event_x: float, event_y: float
    ) -> tuple[str, int] | None:
        """Phase 2: Return (category, obj_idx) of the nearest pick LINE within tolerance."""
        section = self._state.active_section
        if section is None:
            return None
        sec_name = section.name
        ex, ey = self._to_screen_px_sv(event_x, event_y)
        best_cat, best_idx = None, None
        best_dist = float("inf")

        def _check(category, picks):
            nonlocal best_cat, best_idx, best_dist
            for oi, hp in enumerate(picks):
                sec_idxs = hp.section_indices(sec_name)
                if len(sec_idxs) < 2:
                    continue
                d_sec = hp._distances[sec_idxs]
                z_sec = hp._depths[sec_idxs]
                for i in range(len(d_sec) - 1):
                    ax2, ay2 = self._to_screen_px_sv(float(d_sec[i]),   float(z_sec[i]))
                    bx2, by2 = self._to_screen_px_sv(float(d_sec[i+1]), float(z_sec[i+1]))
                    d = _seg_dist(ex, ey, ax2, ay2, bx2, by2)
                    if d <= _LINE_HIT_PX and d < best_dist:
                        best_dist = d
                        best_cat, best_idx = category, oi

        _check("Horizons", self._state.project.horizon_picks)
        _check("Faults",   self._state.project.fault_picks)
        return (best_cat, best_idx) if best_cat is not None else None

    def _compute_snap(self, x: float, y: float) -> tuple[float, float] | None:
        """Phase 5: Return nearest pick point within snap threshold, or None."""
        if not self._snap_active:
            return None
        section = self._state.active_section
        if section is None:
            return None
        sec_name = section.name
        ex, ey = self._to_screen_px_sv(x, y)
        best_dist = float(_SNAP_THRESHOLD)
        best_pt: tuple[float, float] | None = None

        for picks_list in [self._state.project.horizon_picks,
                            self._state.project.fault_picks]:
            for hp in picks_list:
                for fi in hp.section_indices(sec_name):
                    d = float(hp._distances[fi])
                    z = float(hp._depths[fi])
                    nx, ny = self._to_screen_px_sv(d, z)
                    dist = math.hypot(ex - nx, ey - ny)
                    if dist < best_dist:
                        best_dist = dist
                        best_pt = (d, z)
        return best_pt

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
        """Insert a pick point tagged with current section name."""
        cat = self._state.active_pick_category
        idx = self._state.active_pick_index
        if cat is None or idx is None:
            return
        section = self._state.active_section
        sec_name = section.name if section is not None else ""
        proj = self._state.project
        picks = proj.horizon_picks if cat == "Horizons" else proj.fault_picks
        if idx >= len(picks):
            return
        hp_before = copy.deepcopy(picks[idx])
        hp_after  = copy.deepcopy(hp_before)
        hp_after.insert_pick(x, y, sec_name)

        def _do():
            if cat == "Horizons":
                self._state.update_horizon_pick(idx, copy.deepcopy(hp_after))
            else:
                self._state.update_fault_pick(idx, copy.deepcopy(hp_after))
        def _undo():
            if cat == "Horizons":
                self._state.update_horizon_pick(idx, copy.deepcopy(hp_before))
            else:
                self._state.update_fault_pick(idx, copy.deepcopy(hp_before))

        _do()
        self._state.record_command(
            f"Add pick to {hp_before.name or cat}", undo=_undo, redo=_do
        )

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

    def _apply_angle_snap(self, x: float, y: float) -> tuple[float, float]:
        """Phase 3: constrain (x,y) to 15° increments from the last pick."""
        _, last_d, last_z = self._get_active_pick_last_point()
        if last_d is None:
            return x, y
        dx = x - last_d
        dy = y - last_z
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            return x, y
        ang = math.degrees(math.atan2(-dy, dx))  # note: depth inverted
        snapped = round(ang / 15.0) * 15.0
        rad = math.radians(snapped)
        return last_d + dist * math.cos(rad), last_z - dist * math.sin(rad)

    def _place_reference_line(self, x: float, y: float) -> None:
        """Phase 2: place a reference line at the clicked position."""
        from cross_section_tool.core.reference_line import ReferenceLine
        tool = self._ref_line_tool
        if tool == "h_ref":
            rl = ReferenceLine(kind="horizontal", value=y,
                               name=f"H {y:.0f}")
            self._state.add_reference_line(rl)
        elif tool == "v_ref":
            rl = ReferenceLine(kind="vertical", value=x,
                               name=f"V {x:.0f}")
            self._state.add_reference_line(rl)
        elif tool == "a_ref":
            if self._aref_anchor is None:
                self._aref_anchor = (x, y)
                # Show status hint
                return
            else:
                ax_, ay_ = self._aref_anchor
                dx = x - ax_
                dy = y - ay_
                angle_deg = math.degrees(math.atan2(-dy, dx))
                rl = ReferenceLine(kind="angled", anchor_x=ax_, anchor_y=ay_,
                                   angle_deg=angle_deg,
                                   name=f"{angle_deg:.0f}°")
                self._aref_anchor = None
                self._state.add_reference_line(rl)

    def _end_pick_sequence(self) -> None:
        """FIX 1: finish picking, return to select mode."""
        self._picking_active = False
        self._fault_picking  = False
        self._cursor_data    = None
        self._snap_point     = None
        self.pick_ended.emit()
        self.render()

    def _on_depth_units_changed(self, index: int) -> None:
        section = self._state.active_section
        if section is None:
            return
        units = _DEPTH_UNITS[index]
        if section.depth_units == units:
            return
        idx = self._state.project.sections.index(section)
        sec_copy = copy.deepcopy(section)
        sec_copy.depth_units = units
        self._state.update_section(idx, sec_copy)

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

        # Apply snap (Phase 5)
        if self._snap_point is not None:
            x, y = self._snap_point

        tool = self._state.active_tool

        # ---- Pan (left+pan-tool or middle button) ----
        if (event.button == 1 and tool == "pan") or event.button == 2:
            self._sv_pan_anchor = (getattr(event, "x", x), getattr(event, "y", y))
            self._sv_pan_xlim0  = self._ax.get_xlim()
            self._sv_pan_ylim0  = self._ax.get_ylim()
            self._sv_pan_inv    = self._ax.transData.inverted()
            return

        # ---- FIX 1: right-click ends pick sequence ----
        if event.button == 3 and (self._picking_active or self._fault_picking):
            self._end_pick_sequence()
            return

        # ---- Picking / polygon (tool-active modes) ----
        if event.button == 1 and (self._picking_active or self._fault_picking):
            is_dbl = getattr(event, "dblclick", False)
            # Phase 3: Shift → 15° angle snap from last pick
            from PySide6.QtWidgets import QApplication
            from PySide6.QtCore import Qt as _Qt
            if QApplication.keyboardModifiers() & _Qt.KeyboardModifier.ShiftModifier:
                x, y = self._apply_angle_snap(x, y)
            self._add_pick_to_active_target(x, y)
            if is_dbl:
                self._end_pick_sequence()
            return

        if event.button == 1 and self._polygon_drawing:
            self._polygon_vertices.append((x, y))
            self.polygon_vertex_added.emit(x, y)
            self.render()
            return
        if event.button == 3 and self._polygon_drawing:
            self.finish_polygon()
            return

        # ---- Phase 2: reference line placement ----
        if event.button == 1 and self._ref_line_tool:
            self._place_reference_line(x, y)
            return

        # ---- Phase 3: construct tool stubs ----
        if event.button == 1 and self._construct_tool:
            from cross_section_tool.app_state import AppState as _AS
            self._state.set_active_tool("select")
            return

        # ---- Phase 3 polish: node hit test has priority in ALL modes ----
        if event.button == 1 and tool in ("select", "node_edit"):
            is_dbl = getattr(event, "dblclick", False)

            # Check for nearby pick node FIRST (any mode)
            hit_node = self._find_nearest_pick_px(x, y)
            if hit_node is not None:
                self._pick_selected = hit_node
                self._pick_drag     = False
                self._pick_press_px = (getattr(event, "x", x), getattr(event, "y", y))
                cat, oi, pi = hit_node
                picks = (self._state.project.horizon_picks if cat == "Horizons"
                         else self._state.project.fault_picks)
                self._pick_copy = copy.deepcopy(picks[oi])
                self._sv_mode = "edit_mode"
                self._selected_object = (cat, oi)
                self.node_selected.emit(cat, oi, pi)
                self.render()
                return

            if self._sv_mode == "edit_mode":
                # In edit mode but no nearby node: click on line → insert; empty → exit
                # Click near line (not a node) → insert pick there
                if self._selected_object is not None:
                    cat, oi = self._selected_object
                    hit_line = self._find_nearest_pick_line(x, y)
                    if hit_line == self._selected_object:
                        # Insert new node at click position
                        picks = (self._state.project.horizon_picks if cat == "Horizons"
                                 else self._state.project.fault_picks)
                        section = self._state.active_section
                        sec_name = section.name if section else ""
                        hp = copy.deepcopy(picks[oi])
                        hp.insert_pick(x, y, sec_name)
                        if cat == "Horizons":
                            self._state.update_horizon_pick(oi, hp)
                        else:
                            self._state.update_fault_pick(oi, hp)
                        return
                # Click on empty space → exit edit mode
                self._sv_mode = "object_selected"
                self._pick_selected = None
                self._pick_copy = None
                self.render()
                return

            # mode == "idle" or "object_selected"
            if is_dbl and self._selected_object is not None:
                # Double-click on selected object → enter edit mode
                hit_line = self._find_nearest_pick_line(x, y)
                if hit_line == self._selected_object:
                    self._sv_mode = "edit_mode"
                    self._pick_selected = None
                    self.render()
                    return

            hit_line = self._find_nearest_pick_line(x, y)
            if hit_line is not None:
                prev = self._selected_object
                self._selected_object = hit_line
                self._sv_mode = "object_selected"
                self._pick_selected = None
                # Save state for potential object-drag
                self._object_drag_press_pt = (x, y)
                cat, oi = hit_line
                picks = (self._state.project.horizon_picks if cat == "Horizons"
                         else self._state.project.fault_picks)
                self._object_drag_origin = copy.deepcopy(picks[oi])
                self._object_drag_active = False
                if prev != hit_line:
                    self.render()
                return

            # Click empty space → deselect
            if self._sv_mode != "idle" or self._selected_object is not None:
                self._sv_mode = "idle"
                self._selected_object = None
                self._pick_selected = None
                self._pick_copy = None
                self._object_drag_active = False
                self.render()

        # ---- Right-click context on pick node (edit mode only) ----
        if event.button == 3 and tool in ("select", "node_edit"):
            if self._sv_mode == "edit_mode":
                hit = self._find_nearest_pick_px(x, y)
                if hit is not None:
                    self._show_pick_context_menu(hit, event)

    def _on_sv_motion(self, event) -> None:
        # Track cursor + compute snap (Phase 5)
        if event.xdata is not None and event.ydata is not None:
            cx, cy = float(event.xdata), float(event.ydata)
            self._cursor_data = (cx, cy)
            self._snap_point  = self._compute_snap(cx, cy)
        else:
            self._cursor_data = None
            self._snap_point  = None

        # ---- Pan ----
        if self._sv_pan_anchor is not None:
            try:
                d0 = self._sv_pan_inv.transform(self._sv_pan_anchor)
                d1 = self._sv_pan_inv.transform([event.x, event.y])
                self._ax.set_xlim(self._sv_pan_xlim0[0] + d0[0] - d1[0],
                                  self._sv_pan_xlim0[1] + d0[0] - d1[0])
                self._ax.set_ylim(self._sv_pan_ylim0[0] + d0[1] - d1[1],
                                  self._sv_pan_ylim0[1] + d0[1] - d1[1])
            except Exception:
                pass
            self._canvas.draw_idle()
            return

        # ---- Phase 3: Drag a pick node (edit mode) ----
        if self._pick_selected is not None and self._pick_press_px is not None:
            try:
                xy = self._ax.transData.inverted().transform([[event.x, event.y]])[0]
                x, y = float(xy[0]), float(xy[1])
            except Exception:
                return
            dx = math.hypot(event.x - self._pick_press_px[0],
                            event.y - self._pick_press_px[1])
            if not self._pick_drag and dx < _PICK_DRAG_PX:
                return
            self._pick_drag = True
            cat, oi, pi = self._pick_selected
            self._pick_copy._distances[pi] = x
            self._pick_copy._depths[pi]    = y
            order = np.argsort(self._pick_copy._distances, kind="stable")
            for _attr in ("_distances", "_depths", "_section_names",
                          "_confidence", "_quality", "_note"):
                arr = getattr(self._pick_copy, _attr, None)
                if arr is not None and len(arr) == len(order):
                    setattr(self._pick_copy, _attr, arr[order])
            new_pi = int(np.where(order == pi)[0][0])
            self._pick_selected = (cat, oi, new_pi)
            self.render()
            return

        # ---- Phase 3: Drag entire object (object_selected mode) ----
        if (self._sv_mode == "object_selected"
                and self._selected_object is not None
                and self._object_drag_press_pt is not None
                and self._object_drag_origin is not None):
            x_cur = getattr(event, "xdata", None)
            y_cur = getattr(event, "ydata", None)
            if x_cur is None or y_cur is None:
                return
            dx = float(x_cur) - self._object_drag_press_pt[0]
            dy = float(y_cur) - self._object_drag_press_pt[1]
            if not self._object_drag_active:
                if math.hypot(dx, dy) < _OBJ_DRAG_PX:
                    return
                self._object_drag_active = True
            section = self._state.active_section
            sec_name = section.name if section else ""
            cat, oi = self._selected_object
            # Build preview copy
            preview = copy.deepcopy(self._object_drag_origin)
            sec_idxs = preview.section_indices(sec_name)
            preview._distances[sec_idxs] += dx
            preview._depths[sec_idxs]    += dy
            # Stash for render (rendered in _render_pick_object via the stored copy)
            self._object_drag_preview = (cat, oi, preview)
            self.render()
            return

        # ---- Hover: pick-node in ALL modes (Phase 3 polish) ----
        tool = self._state.active_tool
        if tool in ("select", "node_edit"):
            if event.xdata is not None:
                new_hover = self._find_nearest_pick_px(
                    float(event.xdata), float(event.ydata)
                )
                if new_hover != self._pick_hover:
                    self._pick_hover = new_hover
                    if new_hover is not None:
                        self._canvas.setCursor(Qt.CursorShape.SizeAllCursor)
                    else:
                        self._canvas.unsetCursor()
                    self.render()
                    return

        # ---- Hover: object line ----
        if tool in ("select", "node_edit") and self._sv_mode != "edit_mode":
            if event.xdata is not None:
                new_obj = self._find_nearest_pick_line(
                    float(event.xdata), float(event.ydata)
                )
                if new_obj != self._hover_object:
                    self._hover_object = new_obj
                    self.render()
                    return

        # ---- Rubber band / snap / ref-line preview ----
        if (self._picking_active or self._fault_picking
                or self._snap_point is not None
                or self._ref_line_tool == "a_ref"):
            self.render()

    def _on_sv_release(self, event) -> None:
        if event.button in (1, 2):
            self._sv_pan_anchor = None

        # Commit node drag
        if self._pick_drag and self._pick_selected is not None:
            cat, oi, _ = self._pick_selected
            if cat == "Horizons":
                self._state.update_horizon_pick(oi, self._pick_copy)
            else:
                self._state.update_fault_pick(oi, self._pick_copy)
            self._pick_drag     = False
            self._pick_copy     = None
            self._pick_press_px = None

        # Commit object drag (Phase 3)
        if self._object_drag_active:
            preview_data = getattr(self, "_object_drag_preview", None)
            if preview_data is not None:
                cat, oi, preview = preview_data
                if cat == "Horizons":
                    self._state.update_horizon_pick(oi, preview)
                else:
                    self._state.update_fault_pick(oi, preview)
            self._object_drag_active  = False
            self._object_drag_preview = None
            self._object_drag_press_pt = None
        elif event.button == 1:
            # Clear drag prep (no actual drag occurred)
            self._object_drag_press_pt = None

    def _on_scroll_sv(self, event) -> None:
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
                self._pick_drag     = False
                self._pick_copy     = None
                self._pick_press_px = None
                self.render()
            elif self._pick_selected is not None:
                self._pick_selected = None
                self._pick_copy     = None
                self.render()
            elif self._sv_mode == "edit_mode":
                self._sv_mode = "object_selected"
                self.render()
            elif self._sv_mode == "object_selected":
                self._sv_mode = "idle"
                self._selected_object = None
                self.render()
        elif event.key == "delete":
            if self._pick_selected is not None and not self._pick_drag:
                self._delete_selected_pick()
            elif self._selected_object is not None and self._sv_mode in (
                "object_selected", "edit_mode"
            ):
                self._delete_selected_object_with_confirm()
        elif event.key == "ctrl+z":
            self._state.undo()
        elif event.key in ("ctrl+shift+z", "ctrl+y"):
            self._state.redo()

    def _on_context_action(self, action: str) -> None:
        """Phase 4: handle actions from the context toolbar."""
        if action == "end_pick":
            self._end_pick_sequence()
        elif action == "close_polygon":
            self.finish_polygon()
        elif action == "cancel_polygon":
            self._polygon_vertices.clear()
            self.set_polygon_drawing(False)
        elif action == "delete_node":
            if self._pick_selected and not self._pick_drag:
                self._delete_selected_pick()
        elif action == "delete_object":
            if self._selected_object:
                self._delete_selected_object_with_confirm()
        elif action == "new_horizon":
            # Signal up to app — can't create directly from here without dialog
            self._state.set_active_tool("select")
        elif action == "new_fault":
            self._state.set_active_tool("select")

    def _on_active_section_changed(self, section) -> None:
        self._ax_limits_set = False   # FIX 2: new section gets default limits
        if section is not None and self._ve_lock_btn.isChecked():
            locked_ve = self._ve_spin.value()
            if abs(getattr(section, "vertical_exaggeration", 1.0) - locked_ve) > 0.001:
                idx = self._state.project.sections.index(section)
                sec_copy = copy.deepcopy(section)
                sec_copy.vertical_exaggeration = locked_ve
                self._state.update_section(idx, sec_copy)
                return  # update_section triggers re-render
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
        props_act = menu.addAction("Properties…")
        menu.addSeparator()
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
        if chosen is props_act:
            self._show_pick_properties(pick_ref)
        elif chosen is del_act:
            self._pick_selected = pick_ref
            self._delete_selected_pick()

    # ------------------------------------------------------------------
    # Object deletion (Phase 3)
    # ------------------------------------------------------------------

    def _delete_selected_object_with_confirm(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        if self._selected_object is None:
            return
        cat, oi = self._selected_object
        proj = self._state.project
        picks = proj.horizon_picks if cat == "Horizons" else proj.fault_picks
        if oi >= len(picks):
            return
        name = picks[oi].name or f"{cat[:-1]} {oi + 1}"
        reply = QMessageBox.question(
            self, "Delete Object",
            f"Delete '{name}'? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._sv_mode = "idle"
        self._selected_object = None
        self._pick_selected = None
        self._pick_copy = None
        if cat == "Horizons":
            self._state.remove_horizon_pick(picks[oi])
        else:
            self._state.remove_fault_pick(picks[oi])

    def _show_pick_properties(self, pick_ref: tuple[str, int, int]) -> None:
        """Phase 3: edit per-point confidence, quality, note."""
        from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QComboBox,
                                       QDoubleSpinBox, QFormLayout, QLineEdit,
                                       QVBoxLayout)
        cat, oi, pi = pick_ref
        proj = self._state.project
        picks = proj.horizon_picks if cat == "Horizons" else proj.fault_picks
        if oi >= len(picks):
            return
        hp = picks[oi]
        if pi >= hp.n_picks:
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Pick Properties")
        form = QFormLayout()
        conf_spin = QDoubleSpinBox(); conf_spin.setRange(0, 1); conf_spin.setSingleStep(0.1)
        conf_spin.setDecimals(2); conf_spin.setValue(float(hp._confidence[pi]))
        qual_combo = QComboBox()
        for q in ("picked", "interpolated", "projected", "inferred"):
            qual_combo.addItem(q)
        cur_q = str(hp._quality[pi])
        idx_q = ["picked", "interpolated", "projected", "inferred"].index(cur_q) \
                if cur_q in ["picked", "interpolated", "projected", "inferred"] else 0
        qual_combo.setCurrentIndex(idx_q)
        note_edit = QLineEdit(str(hp._note[pi]))
        form.addRow("Confidence:", conf_spin)
        form.addRow("Quality:", qual_combo)
        form.addRow("Note:", note_edit)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        vb = QVBoxLayout(dlg); vb.addLayout(form); vb.addWidget(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        hp2 = copy.deepcopy(hp)
        hp2._confidence[pi] = conf_spin.value()
        hp2._quality[pi]    = qual_combo.currentText()
        hp2._note[pi]       = note_edit.text()
        if cat == "Horizons":
            self._state.update_horizon_pick(oi, hp2)
        else:
            self._state.update_fault_pick(oi, hp2)

    # ------------------------------------------------------------------
    # Pick deletion
    # ------------------------------------------------------------------

    def _delete_selected_pick(self) -> None:
        if self._pick_selected is None:
            return
        cat, oi, pi = self._pick_selected
        self._pick_selected = None
        self._pick_copy     = None

        proj = self._state.project
        picks = proj.horizon_picks if cat == "Horizons" else proj.fault_picks
        if oi >= len(picks):
            return
        hp = copy.deepcopy(picks[oi])

        # Store for Ctrl+Z undo
        self._last_delete_for_undo = {
            "type": "pick", "category": cat, "obj_idx": oi,
            "distance":     float(hp._distances[pi]),
            "depth":        float(hp._depths[pi]),
            "section_name": str(hp._section_names[pi]),
        }

        if hp.n_picks <= 1:
            # Allow object to become empty (Phase 1 spec)
            hp._distances     = np.array([], dtype=float)
            hp._depths        = np.array([], dtype=float)
            hp._section_names = np.array([], dtype=object)
        else:
            hp.delete_pick(pi)

        hp_before = copy.deepcopy(picks[oi])  # save for undo (before mutation)
        if cat == "Horizons":
            self._state.update_horizon_pick(oi, hp)
        else:
            self._state.update_fault_pick(oi, hp)
        # Record undo
        _oi, _cat = oi, cat
        _saved = hp_before
        def _undo_del():
            if _cat == "Horizons":
                self._state.update_horizon_pick(_oi, copy.deepcopy(_saved))
            else:
                self._state.update_fault_pick(_oi, copy.deepcopy(_saved))
        self._state.record_command(f"Delete pick from {hp_before.name}", undo=_undo_del)

    def _undo_last_delete(self) -> None:
        """Phase 1: restore the last deleted pick point."""
        u = self._last_delete_for_undo
        if u is None:
            return
        self._last_delete_for_undo = None
        cat, oi = u["category"], u["obj_idx"]
        proj = self._state.project
        picks = proj.horizon_picks if cat == "Horizons" else proj.fault_picks
        if oi >= len(picks):
            return
        hp = copy.deepcopy(picks[oi])
        hp.insert_pick(u["distance"], u["depth"], u["section_name"])
        if cat == "Horizons":
            self._state.update_horizon_pick(oi, hp)
        else:
            self._state.update_fault_pick(oi, hp)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Lithology pattern → matplotlib hatch mapping (Phase D)
# ---------------------------------------------------------------------------

_LITHOLOGY_HATCH: dict[str, str] = {
    "sandstone":   "....",
    "shale":       "----",
    "siltstone":   ".-.-",
    "limestone":   "++++",
    "dolomite":    "xxxx",
    "conglomerate":"oooo",
    "coal":        "////",
    "salt":        "****",
    "basement":    "++++",
    "volcanic":    "////",
}


# ---------------------------------------------------------------------------
# Line-decoration helpers (Phase A / B)
# ---------------------------------------------------------------------------

def _wavy_coords(
    ax,
    xs: np.ndarray,
    ys: np.ndarray,
    amplitude_px: float = 3.0,
    wavelength_px: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sinusoidal perpendicular offset of a polyline (screen-pixel units)."""
    if len(xs) < 2:
        return xs, ys
    pts = np.column_stack([xs, ys])
    try:
        pts_px = ax.transData.transform(pts)
    except Exception:
        return xs, ys
    diffs = np.diff(pts_px, axis=0)
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])
    total = arc[-1]
    if total < 1.0:
        return xs, ys
    n = max(60, int(total))
    t = np.linspace(0.0, total, n)
    xp = np.interp(t, arc, pts_px[:, 0])
    yp = np.interp(t, arc, pts_px[:, 1])
    dxt = np.gradient(xp, t)
    dyt = np.gradient(yp, t)
    nrm = np.hypot(dxt, dyt)
    nrm[nrm < 1e-12] = 1.0
    nx, ny = -dyt / nrm, dxt / nrm
    wave = amplitude_px * np.sin(2.0 * math.pi * t / wavelength_px)
    try:
        wavy_data = ax.transData.inverted().transform(
            np.column_stack([xp + nx * wave, yp + ny * wave])
        )
        return wavy_data[:, 0], wavy_data[:, 1]
    except Exception:
        return xs, ys


def _line_ticks(
    ax,
    xs: np.ndarray,
    ys: np.ndarray,
    spacing_px: float = 40.0,
    length_px: float = 8.0,
    side: float = 1.0,
) -> list[tuple]:
    """Return [(x0,y0,x1,y1), …] tick segments perpendicular to a polyline."""
    if len(xs) < 2:
        return []
    pts_px = ax.transData.transform(np.column_stack([xs, ys]))
    diffs = np.diff(pts_px, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    arc = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = arc[-1]
    result = []
    inv = ax.transData.inverted()
    for t in np.arange(spacing_px / 2.0, total, spacing_px):
        xi = float(np.interp(t, arc, pts_px[:, 0]))
        yi = float(np.interp(t, arc, pts_px[:, 1]))
        si = min(int(np.searchsorted(arc[1:], t)), len(diffs) - 1)
        sl = float(seg_lens[si])
        if sl < 1e-10:
            continue
        dx, dy = diffs[si, 0] / sl, diffs[si, 1] / sl
        nx, ny = -dy * side, dx * side
        try:
            p0 = inv.transform([xi, yi])
            p1 = inv.transform([xi + nx * length_px, yi + ny * length_px])
            result.append((float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1])))
        except Exception:
            pass
    return result


def _line_triangles(
    ax,
    xs: np.ndarray,
    ys: np.ndarray,
    spacing_px: float = 40.0,
    size_px: float = 8.0,
    side: float = 1.0,
) -> list[np.ndarray]:
    """Return list of (3,2) vertex arrays for filled triangles along a polyline."""
    if len(xs) < 2:
        return []
    pts_px = ax.transData.transform(np.column_stack([xs, ys]))
    diffs  = np.diff(pts_px, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    arc  = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = arc[-1]
    result = []
    inv = ax.transData.inverted()
    for t in np.arange(spacing_px / 2.0, total, spacing_px):
        xi = float(np.interp(t, arc, pts_px[:, 0]))
        yi = float(np.interp(t, arc, pts_px[:, 1]))
        si = min(int(np.searchsorted(arc[1:], t)), len(diffs) - 1)
        sl = float(seg_lens[si])
        if sl < 1e-10:
            continue
        dx, dy = diffs[si, 0] / sl, diffs[si, 1] / sl
        nx, ny = -dy * side, dx * side
        h = size_px
        w = size_px * 0.6
        # tip on the line, base on the hanging-wall side
        tip   = [xi, yi]
        base1 = [xi + nx * h - dx * w / 2, yi + ny * h - dy * w / 2]
        base2 = [xi + nx * h + dx * w / 2, yi + ny * h + dy * w / 2]
        try:
            verts = np.array([
                inv.transform(tip),
                inv.transform(base1),
                inv.transform(base2),
            ])
            result.append(verts)
        except Exception:
            pass
    return result


def _seg_dist(px: float, py: float,
              ax: float, ay: float,
              bx: float, by: float) -> float:
    """Screen-pixel distance from (px,py) to segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    if len2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len2))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


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
