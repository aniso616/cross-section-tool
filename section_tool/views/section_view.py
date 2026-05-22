"""Section view — 2D cross-section display with picking, faults, and polygons."""
from __future__ import annotations

import copy
import math
import time
from typing import Literal

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from PySide6.QtCore import Qt, QTimer, Signal
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

from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick
from section_tool.io.project import SeismicRef
from section_tool.io.segy import SeismicDataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PICK_HIT_PX      = 10    # pick-point hit-test radius in screen pixels
_LINE_HIT_PX      = 8     # pick-line hit-test tolerance in screen pixels
_PICK_DRAG_PX     = 3     # minimum movement before drag activates
_OBJ_DRAG_PX      = 3     # minimum movement before object-move activates
_SNAP_THRESHOLD   = 15    # snap radius in screen pixels
_DEFAULT_DEPTH    = 5000.0

_DEPTH_UNITS = ["m", "ft", "km", "mi", "ms", "s", "m+ft"]

# Tools where snap-to-pick-point/intersection is active
_SNAP_TOOLS = frozenset({
    "horizon_pick", "fault_pick", "polygon",
    "extend", "trim", "parallel",
    "node_edit",
})

# Seismic colormap name mapping: SeismicDisplaySettings.colormap → matplotlib
_SEGY_CMAP = {
    "seismic_red_blue": "seismic",
    "grey":             "gray",
    "gray_r":           "gray_r",   # reversed grayscale — industry default
    "RdBu_r":           "RdBu_r",
    "RdYlBu_r":         "RdYlBu_r",
    "bone":             "bone",
    "viridis":          "viridis",
    "inferno":          "inferno",
    "jet":              "jet",
}
_DEFAULT_CMAP = "gray_r"

# Max perpendicular distance for well projection onto section
_WELL_MAX_PERP = 2000.0   # metres


def _fm_color(fm, fallback: str) -> str:
    """Return hex color string for a Formation (or fallback if fm is None)."""
    if fm is None:
        return fallback
    try:
        r, g, b = fm.color
        return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))
    except Exception:
        return fallback


# Pick-point visual states: (radius_pt, face, edge, ew)
_PP_NORMAL   = (5,  "white",   "#555", 0.8)
_PP_HOVER    = (7,  "#ffffaa", "#555", 0.8)
_PP_SELECTED = (7,  "#ff7f0e", "white", 1.5)
_PP_DRAG     = (7,  "red",     "white", 1.5)


class SectionViewState:
    """View-state adapter exposing axes extents and pan/zoom for WASDNavigator."""

    def __init__(self, section_view: "SectionView") -> None:
        self._sv = section_view

    @property
    def x_min(self) -> float:
        return float(self._sv._ax.get_xlim()[0])

    @property
    def x_max(self) -> float:
        return float(self._sv._ax.get_xlim()[1])

    @property
    def z_min(self) -> float:
        # Inverted Y axis: ylim[1] is the shallower (smaller) depth
        return float(self._sv._ax.get_ylim()[1])

    @property
    def z_max(self) -> float:
        # Inverted Y axis: ylim[0] is the deeper (larger) depth
        return float(self._sv._ax.get_ylim()[0])

    def pan(self, dx_m: float, dz_m: float) -> None:
        xl = self._sv._ax.get_xlim()
        yl = self._sv._ax.get_ylim()
        new_xl = (xl[0] + dx_m, xl[1] + dx_m)
        new_yl = (yl[0] + dz_m, yl[1] + dz_m)
        self._sv._ax.set_xlim(new_xl)
        self._sv._ax.set_ylim(new_yl)
        self._sv._saved_xlim   = new_xl
        self._sv._saved_ylim   = new_yl
        self._sv._user_has_zoomed = True
        self._sv._canvas.draw_idle()
        self._sv.request_hud_update()

    def zoom(self, factor: float, center_x_m: float, center_z_m: float) -> None:
        xl = self._sv._ax.get_xlim()
        yl = self._sv._ax.get_ylim()
        x_range = xl[1] - xl[0]
        y_range = yl[0] - yl[1]  # positive (inverted: yl[0] > yl[1])
        if x_range == 0 or y_range == 0:
            return
        new_x_range = x_range * factor
        new_y_range = y_range * factor
        x_frac         = (center_x_m - xl[0]) / x_range
        y_frac_from_top = (center_z_m - yl[1]) / y_range
        new_xl = (center_x_m - new_x_range * x_frac,
                  center_x_m + new_x_range * (1.0 - x_frac))
        new_yl = (center_z_m + new_y_range * (1.0 - y_frac_from_top),  # bottom
                  center_z_m - new_y_range * y_frac_from_top)            # top
        self._sv._ax.set_xlim(new_xl)
        self._sv._ax.set_ylim(new_yl)
        self._sv._saved_xlim   = new_xl
        self._sv._saved_ylim   = new_yl
        self._sv._user_has_zoomed = True
        self._sv._canvas.draw_idle()
        self._sv.request_hud_update()

    def pixel_to_world(self, canvas_pos) -> tuple[float, float]:
        """Convert QPoint canvas position to (x_m, depth_m)."""
        px = canvas_pos.x()
        # Qt: origin top-left; Matplotlib: origin bottom-left
        py = self._sv._canvas.height() - canvas_pos.y()
        try:
            xy = self._sv._ax.transData.inverted().transform([[px, py]])[0]
            return float(xy[0]), float(xy[1])
        except Exception:
            xl = self._sv._ax.get_xlim()
            yl = self._sv._ax.get_ylim()
            return (xl[0] + xl[1]) / 2, (yl[0] + yl[1]) / 2

    def pixels_to_metres(self, n_px: int) -> float:
        """Approximate: n screen pixels → metres in data space."""
        try:
            t  = self._sv._ax.transData.inverted()
            p0 = t.transform([[0, 0]])[0]
            p1 = t.transform([[n_px, 0]])[0]
            return abs(float(p1[0] - p0[0]))
        except Exception:
            return 100.0


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
    coords_updated(float, float)        — x_m, depth_m on mouse move (game UI)
    """

    polygon_vertex_added = Signal(float, float)
    polygon_finished     = Signal(object)
    pick_ended           = Signal()       # emitted when pick sequence ends
    node_selected        = Signal(str, int, int)   # Phase 3: (cat, obj_idx, pt_idx)
    frame_time_ms        = Signal(float)  # Phase 6: FPS display
    coords_updated       = Signal(float, float)    # game HUD coord readout
    view_changed         = Signal()                # after any pan/zoom/render
    cursor_map_pos       = Signal(float, float)    # map x,y as cursor moves on section

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._game_mode: bool = False
        # Debounced view-change signal for HUD updates (pan/zoom via WASD)
        self._hud_timer = QTimer(self)
        self._hud_timer.setSingleShot(True)
        self._hud_timer.setInterval(80)
        self._hud_timer.timeout.connect(self.view_changed.emit)

        # ---- seismic cache (loaded datasets) ----
        self._seismic_cache: dict[str, SeismicDataset] = {}
        # ---- seismic projection cache (expensive per-section computation) ----
        # key: (section_name, ref_path) → (distances, data, perps)
        self._seismic_proj_cache: dict[tuple, tuple] = {}
        # Pending limits: one-shot override for the next render (from scroll/VE)
        self._pending_xlim: tuple | None = None
        self._pending_ylim: tuple | None = None
        # Saved limits: persist across renders so rubber-band / pick renders
        # don't reset the zoom the user set via scroll/pan.
        self._saved_xlim: tuple | None = None
        self._saved_ylim: tuple | None = None
        self._user_has_zoomed: bool = False
        # Artist tracking (populated each render, cleared by ax.clear())
        self._seismic_artists: list = []
        self._overlay_artists: list = []
        # ---- image overlays [(path, section_name, (d0,d1), (z0,z1))] ----
        self._image_overlays: list[tuple[str, str, tuple, tuple]] = []
        # ---- FPS tracking ----
        self._show_fps: bool = False
        self._strat_col_visible: bool = True  # kept for API compat; not used internally
        self._show_grid: bool = False
        # ---- display toggles ----
        self._show_sea_level: bool = True
        # ---- topography per section: {section_name: (distances, elevations)} ----
        self._topography: dict[str, tuple] = {}

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
        self._poly_preflight: dict = {}  # name/formation/color/opacity for next polygon

        # ---- construct tool state machine ----
        self._cst_state: str = "idle"   # "idle" | "source_selected"
        self._cst_source: dict | None = None  # {'cat': ..., 'idx': ..., 'endpoint': ...}
        self._cst_preview_line: tuple | None = None  # for parallel preview

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
        from section_tool.style import BG_CANVAS
        self._fig = Figure(figsize=(10, 6), facecolor=BG_CANVAS)
        # Single axis — strat column is now a HUD QWidget, not a matplotlib subplot
        self._fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        self._ax = self._fig.add_subplot(111)
        self._configure_axes(self._ax)
        self._canvas = FigureCanvasQTAgg(self._fig)

        # Hidden toolbar — kept for zoom stack; NOT in the layout.
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._toolbar.hide()

        # ── Row 1: section name, units, VE ─────────────────────────────
        self._header = QWidget()
        self._header.setFixedHeight(28)
        self._header.setStyleSheet(
            "background: #f5f5f5; border-bottom: 1px solid #ddd; color: #333333;")
        hl = QHBoxLayout(self._header)
        hl.setContentsMargins(8, 2, 8, 2)
        hl.setSpacing(6)
        self._section_name_label = QLabel("— no section —")
        self._section_name_label.setStyleSheet(
            "font-size: 10pt; font-weight: bold; color: #333333;")
        hl.addWidget(self._section_name_label)
        hl.addStretch()

        _units_lbl = QLabel("Units:")
        _units_lbl.setStyleSheet("color: #333333; font-size: 8pt;")
        hl.addWidget(_units_lbl)
        self._depth_units_combo = QComboBox()
        self._depth_units_combo.setFixedWidth(64)
        self._depth_units_combo.setToolTip("Depth / time axis units")
        self._depth_units_combo.setStyleSheet("color: #333333; background: #ffffff;")
        for u in _DEPTH_UNITS:
            self._depth_units_combo.addItem(u)
        self._depth_units_combo.currentIndexChanged.connect(self._on_depth_units_changed)
        hl.addWidget(self._depth_units_combo)

        _ve_lbl = QLabel("VE:")
        _ve_lbl.setStyleSheet("color: #333333; font-size: 8pt;")
        hl.addWidget(_ve_lbl)
        self._ve_spin = QDoubleSpinBox()
        self._ve_spin.setRange(0.5, 50.0)
        self._ve_spin.setSingleStep(0.5)
        self._ve_spin.setValue(1.0)
        self._ve_spin.setFixedWidth(60)
        self._ve_spin.setDecimals(1)
        self._ve_spin.setKeyboardTracking(False)
        self._ve_spin.setStyleSheet(
            "color: #333333; background: #ffffff; border: 1px solid #999999; min-width: 60px;")
        self._ve_spin.setToolTip(
            "Vertical exaggeration (1.0 = true scale)\n"
            "Higher values stretch depth axis, steepening apparent dips."
        )
        self._ve_timer = QTimer(self)
        self._ve_timer.setSingleShot(True)
        self._ve_timer.setInterval(200)
        self._ve_timer.timeout.connect(self._on_ve_changed)
        self._ve_spin.valueChanged.connect(lambda _v: self._ve_timer.start())
        hl.addWidget(self._ve_spin)

        self._ve_lock_btn = QPushButton("\U0001F513")
        self._ve_lock_btn.setCheckable(True)
        self._ve_lock_btn.setFixedSize(24, 22)
        self._ve_lock_btn.setToolTip(
            "Lock VE: when locked, the same vertical exaggeration applies\n"
            "to all sections and is preserved when switching between them."
        )
        self._ve_lock_btn.setStyleSheet(
            "QPushButton { border: 1px solid #bbb; border-radius: 3px; font-size: 11px;"
            " color: #333333; }"
            "QPushButton:checked { background: #d0e8ff; border-color: #5599cc; }"
        )
        self._ve_lock_btn.toggled.connect(
            lambda locked: self._ve_lock_btn.setText("\U0001F512" if locked else "\U0001F513")
        )
        hl.addWidget(self._ve_lock_btn)

        # ── Row 2: seismic controls (hidden when no seismic) ────────────
        self._seismic_row = QWidget()
        self._seismic_row.setFixedHeight(26)
        self._seismic_row.setStyleSheet(
            "background: #eef2f5; border-bottom: 1px solid #ddd; color: #333333;")
        sl = QHBoxLayout(self._seismic_row)
        sl.setContentsMargins(8, 2, 8, 2)
        sl.setSpacing(6)
        _dom_lbl = QLabel("Seismic:")
        _dom_lbl.setStyleSheet("color: #555555; font-size: 8pt; font-weight: bold;")
        sl.addWidget(_dom_lbl)
        self._seismic_domain_combo = QComboBox()
        self._seismic_domain_combo.addItem("Depth – linear stretch", "linear")
        self._seismic_domain_combo.addItem("TWT (native ms)", "native_twt")
        self._seismic_domain_combo.setFixedWidth(160)
        self._seismic_domain_combo.setToolTip(
            "Seismic Y-axis domain:\n"
            "Depth – linear stretch: convert TWT to depth using constant velocity\n"
            "TWT (native): display sample axis in milliseconds as recorded"
        )
        self._seismic_domain_combo.setStyleSheet("color: #333333; background: #ffffff;")
        self._seismic_domain_combo.currentIndexChanged.connect(
            self._on_seismic_domain_changed)
        sl.addWidget(self._seismic_domain_combo)
        _vel_lbl = QLabel("V:")
        _vel_lbl.setStyleSheet("color: #555555; font-size: 8pt;")
        sl.addWidget(_vel_lbl)
        self._seismic_vel_spin = QDoubleSpinBox()
        self._seismic_vel_spin.setRange(500.0, 6000.0)
        self._seismic_vel_spin.setSingleStep(100.0)
        self._seismic_vel_spin.setValue(2000.0)
        self._seismic_vel_spin.setDecimals(0)
        self._seismic_vel_spin.setFixedWidth(76)
        self._seismic_vel_spin.setSuffix(" m/s")
        self._seismic_vel_spin.setToolTip(
            "Constant interval velocity used to convert TWT (ms) to depth (m).\n"
            "depth = twt_ms × V / 2000"
        )
        self._seismic_vel_spin.setStyleSheet(
            "color: #333333; background: #ffffff; border: 1px solid #999999;")
        self._seismic_vel_spin.valueChanged.connect(self._on_seismic_velocity_changed)
        sl.addWidget(self._seismic_vel_spin)
        from PySide6.QtWidgets import QCheckBox
        self._fast_display_cb = QCheckBox("Fast display")
        self._fast_display_cb.setChecked(False)
        self._fast_display_cb.setToolTip(
            "Downsample seismic to screen resolution before rendering.\n"
            "Enable for fast panning; disable for full-detail viewing."
        )
        self._fast_display_cb.setStyleSheet("color: #555555; font-size: 8pt;")
        sl.addWidget(self._fast_display_cb)
        sl.addStretch()
        self._seismic_row.hide()   # shown when seismic refs are present

        # ── Pick mode banner ─────────────────────────────────────────────
        self._pick_banner = QWidget()
        self._pick_banner.setFixedHeight(26)
        self._pick_banner.setObjectName("PickBanner")
        self._pick_banner.setStyleSheet(
            "QWidget#PickBanner { background: #1D4ED8; }"
            "QLabel { color: white; font-size: 8pt; font-weight: bold; background: transparent; }"
            "QPushButton { color: white; background: rgba(255,255,255,0.15); "
            "  border: 1px solid rgba(255,255,255,0.35); border-radius: 3px; "
            "  font-size: 8pt; padding: 1px 8px; }"
            "QPushButton:hover { background: rgba(255,255,255,0.3); }"
        )
        _bl = QHBoxLayout(self._pick_banner)
        _bl.setContentsMargins(10, 2, 10, 2)
        _bl.setSpacing(8)
        self._pick_banner_label = QLabel("Picking: —")
        _bl.addWidget(self._pick_banner_label)
        _bl.addStretch()
        _end_btn = QPushButton("�?� End Picking")
        _end_btn.setToolTip("Finish this pick session  (Right-click or Escape)")
        _end_btn.clicked.connect(self._end_pick_sequence)
        _bl.addWidget(_end_btn)
        self._pick_banner.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._header)
        layout.addWidget(self._seismic_row)
        layout.addWidget(self._pick_banner)
        layout.addWidget(self._canvas, stretch=1)

        # Matplotlib events
        self._canvas.mpl_connect("button_press_event",   self._on_sv_press)
        self._canvas.mpl_connect("motion_notify_event",  self._on_sv_motion)
        self._canvas.mpl_connect("button_release_event", self._on_sv_release)
        self._canvas.mpl_connect("scroll_event",         self._on_scroll_sv)
        self._canvas.mpl_connect("key_press_event",      self._on_sv_key)
        self._canvas.mpl_connect("resize_event",         self._on_sv_resize)
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _connect_signals(self) -> None:
        s = self._state
        s.active_section_changed.connect(self._on_active_section_changed_seismic_invalidate)
        s.active_section_changed.connect(self._on_active_section_changed)
        s.section_modified.connect(lambda *_: self._seismic_proj_cache.clear())
        s.active_pick_target_changed.connect(self._on_data_changed)
        s.active_pick_target_changed.connect(lambda *_: self._update_pick_banner())
        s.project_changed.connect(self.request_render)
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
        s.seismic_extracted.connect(self._on_data_changed)
        s.polygon_added.connect(self._on_data_changed)
        s.polygon_removed.connect(self._on_data_changed)
        s.polygon_modified.connect(self._on_data_changed)
        s.reference_line_added.connect(self._on_data_changed)
        s.reference_line_removed.connect(self._on_data_changed)
        s.reference_line_modified.connect(self._on_data_changed)
        # Topology: redraw intersection markers when graph updates
        s.topology_changed.connect(self._on_data_changed)

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
    def view_state(self) -> "SectionViewState":
        return SectionViewState(self)

    def _blit_overlays(self) -> None:
        """Fast redraw for rubber-band / snap / polygon preview — no seismic re-render.

        Removes only the dynamic overlay artists (rubber band, snap marker, etc.)
        then re-renders overlays and calls draw_idle() — much faster than a full
        render because the seismic imshow, horizon/fault lines, etc. are preserved.
        Falls back to full render if no section is active.
        """
        section = self._state.active_section
        if section is None:
            self.request_render()
            return
        # Remove only previous overlay artists (leaves seismic/horizons intact)
        for a in self._overlay_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._overlay_artists.clear()
        # Re-render lightweight overlays only
        self._render_overlays(section)
        self._canvas.draw_idle()

    def request_hud_update(self) -> None:
        """Debounced trigger for HUD view-state refresh (pan/zoom without full render)."""
        if not self._hud_timer.isActive():
            self._hud_timer.start()

    @staticmethod
    def _configure_axes(ax) -> None:
        """Strip all matplotlib chrome from an axes object."""
        from section_tool.style import BG_CANVAS
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_facecolor(BG_CANVAS)
        ax.figure.patch.set_facecolor(BG_CANVAS)

    def set_game_mode(self, enabled: bool) -> None:
        """Switch to game UI mode: hide all chrome, suppress pick banner."""
        self._game_mode = enabled
        if enabled:
            self._header.hide()
            self._seismic_row.hide()
            self._pick_banner.hide()
        else:
            self._header.show()

    def _surface_elev_at(self, x_m: float) -> float:
        """Return ground-surface elevation (m) at distance x_m along section."""
        section = self._state.active_section
        if section is None:
            return 0.0
        topo = self._topography.get(section.name)
        if topo is None or len(topo[0]) == 0:
            return 0.0
        distances, elevations = topo
        return float(np.interp(x_m, distances, elevations))

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
        old_seismic_artists = self._seismic_artists
        old_overlay_artists = self._overlay_artists
        self._seismic_artists = []
        self._overlay_artists = []
        try:
            self._setup_axes(section)
            if self._ax_limits_set:
                self._ax.set_xlim(old_ax.get_xlim())
                self._ax.set_ylim(old_ax.get_ylim())
            self._render_image_overlays(section)
            self._setup_seismic_artists(section)
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
            self._seismic_artists = old_seismic_artists
            self._overlay_artists = old_overlay_artists
        return fig

    def apply_tool_cursor(self, tool_id: str) -> None:
        """Phase 5: set an appropriate cursor for the active tool."""
        from PySide6.QtCore import Qt as _Qt
        _map = {
            "select":       _Qt.CursorShape.ArrowCursor,
            "node_edit":    _Qt.CursorShape.PointingHandCursor,
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
        self._update_pick_banner()

    def set_fault_picking(self, active: bool) -> None:
        """Enable/disable fault pick mode."""
        self._fault_picking    = active
        self._picking_active   = False if active else self._picking_active
        self._polygon_drawing  = False if active else self._polygon_drawing
        self._update_pick_banner()

    def _update_pick_banner(self) -> None:
        """Show / hide the pick-mode banner and update its label."""
        if self._game_mode:
            return  # banner replaced by HUD tool indicator in game UI
        if self._picking_active or self._fault_picking:
            cat = self._state.active_pick_category
            idx = self._state.active_pick_index
            if cat is not None and idx is not None:
                picks = (self._state.project.horizon_picks if cat == "Horizons"
                         else self._state.project.fault_picks)
                if idx < len(picks):
                    kind = "Horizon" if cat == "Horizons" else "Fault"
                    name = picks[idx].name or f"{kind} {idx + 1}"
                    self._pick_banner_label.setText(
                        f"�?  Picking: {name}  ({kind})  �?"
                    )
            self._pick_banner.show()
        else:
            self._pick_banner.hide()

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
            from section_tool.core.polygons import SectionPolygon
            pf = self._poly_preflight
            name = pf.get("name") or f"Polygon {len(self._state.project.polygons) + 1}"
            color = pf.get("color", "#9467bd")
            opacity = pf.get("opacity", 0.6)
            formation = pf.get("formation", "")
            poly = SectionPolygon(
                vertices=self._polygon_vertices,
                name=name,
                fill_color=color,
                fill_alpha=opacity,
                formation=formation,
            )
            self._polygon_vertices.clear()
            self._poly_preflight = {}
            self.polygon_finished.emit(poly)
        else:
            self._polygon_vertices.clear()
            self._poly_preflight = {}
        self.render()

    def set_polygon_preflight(self, name: str, formation: str,
                              color: str, opacity: float) -> None:
        """Store polygon creation settings for use when drawing completes."""
        self._poly_preflight = dict(
            name=name, formation=formation, color=color, opacity=opacity
        )

    def clear_seismic_cache(self) -> None:
        self._seismic_cache.clear()

    def preload_seismic_ref(self, ref, progress_callback=None) -> None:
        """Eagerly load a SeismicRef into the cache (with optional progress)."""
        if ref.path not in self._seismic_cache:
            ds = ref.load(progress_callback=progress_callback)
            self._seismic_cache[ref.path] = ds

    def show_map_cursor_on_section(self, map_x: float, map_y: float) -> None:
        """Show where a map position falls on this section as a vertical line."""
        section = self._state.active_section
        if section is None or self._ax is None:
            return
        try:
            dist_along, perp_offset = section.project_point(map_x, map_y)
            # Only show if close to the section (within 50% of total length)
            if abs(perp_offset) > max(section.total_length() * 0.5, 2000):
                self._clear_map_cursor()
                return
            if hasattr(self, "_map_cursor_artist") and self._map_cursor_artist:
                try:
                    self._map_cursor_artist.remove()
                except Exception:
                    pass
            yl = self._ax.get_ylim()
            self._map_cursor_artist, = self._ax.plot(
                [dist_along, dist_along], [min(yl), max(yl)],
                color="#FF6666", lw=0.6, ls="--", alpha=0.5, zorder=15,
            )
            self._canvas.draw_idle()
        except Exception:
            pass

    def _clear_map_cursor(self) -> None:
        if hasattr(self, "_map_cursor_artist") and self._map_cursor_artist:
            try:
                self._map_cursor_artist.remove()
            except Exception:
                pass
            self._map_cursor_artist = None
            self._canvas.draw_idle()

    def set_grid_visible(self, visible: bool) -> None:
        self._show_grid = visible
        self.request_render()

    def set_strat_column_visible(self, visible: bool) -> None:
        self._strat_col_visible = visible
        # Formation strip is now a HUD widget; visibility is controlled externally.
        self.render()

    def set_sea_level_visible(self, visible: bool) -> None:
        self._show_sea_level = visible
        self.render()

    def set_topography(self, section_name: str,
                       distances: "np.ndarray", elevations: "np.ndarray") -> None:
        """Register a topography profile for *section_name* and redraw."""
        self._topography[section_name] = (distances, elevations)
        self.render()

    def set_fps_display(self, enabled: bool) -> None:
        self._show_fps = enabled
        if not enabled:
            self.frame_time_ms.emit(-1.0)

    def add_image_overlay(
        self,
        path: str,
        section_name: str,
        dist_range: tuple[float, float],
        depth_range: tuple[float, float],
    ) -> None:
        """Register a raster image to display as a section background."""
        self._image_overlays = [
            o for o in self._image_overlays if o[1] != section_name
        ]
        self._image_overlays.append((path, section_name, dist_range, depth_range))
        self.render()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Seismic image caching helpers
    # ------------------------------------------------------------------

    def _seismic_settings_key(self, section) -> tuple:
        """Stable hash of all settings that affect the seismic image appearance."""
        sds = getattr(section, "seismic_display", None)
        return (
            section.name,
            tuple(ref.path for ref in self._state.project.seismic_refs),
            sds.clip_percentile   if sds else 99.0,
            sds.gain              if sds else 1.0,
            sds.opacity           if sds else 1.0,
            sds.colormap          if sds else _DEFAULT_CMAP,
            sds.show_wiggle       if sds else False,
            sds.stretch_mode      if sds else "linear",
            sds.constant_velocity if sds else 2000.0,
            self._display_mode,
        )

    def _invalidate_seismic_cache(self) -> None:
        self._seismic_proj_cache.clear()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def request_render(self, *_args) -> None:
        """Schedule a render on the next idle cycle (debounced for signal bursts)."""
        if not self._redraw_timer.isActive():
            self._redraw_timer.start()

    def render(self, *_args) -> None:
        """Full redraw of the active section."""
        if self._is_rendering:
            return
        # Guard against degenerate canvas size (avoids AGG MemoryError cascade)
        if self._canvas.width() < 4 or self._canvas.height() < 4:
            return
        self._is_rendering = True
        try:
            self._render_impl()
        finally:
            self._is_rendering = False

    def _render_impl(self) -> None:
        _t0 = time.perf_counter()
        section = self._state.active_section

        if section is None:
            self._seismic_artists.clear()
            self._overlay_artists.clear()
            self._ax.clear()
            self._configure_axes(self._ax)
            self._ax.set_xlim(0, 10000)
            self._ax.set_ylim(5000, 0)   # inverted: 0 at top
            self._section_name_label.setText("— no section —")
            self._seismic_row.hide()
            self._ve_spin.setEnabled(False)
            self._depth_units_combo.setEnabled(False)
            self._ax_limits_set = False
            self._saved_xlim = None
            self._saved_ylim = None
            self._canvas.draw_idle()
            return

        # Sync header widgets
        self._section_name_label.setText(section.name or "Unnamed section")
        self._ve_spin.setEnabled(True)
        self._depth_units_combo.setEnabled(True)
        u = section.depth_units if section.depth_units in _DEPTH_UNITS else "m"
        self._depth_units_combo.blockSignals(True)
        self._depth_units_combo.setCurrentIndex(_DEPTH_UNITS.index(u))
        self._depth_units_combo.blockSignals(False)
        if not self._ve_lock_btn.isChecked():
            self._ve_spin.blockSignals(True)
            self._ve_spin.setValue(section.vertical_exaggeration)
            self._ve_spin.blockSignals(False)

        sds = getattr(section, "seismic_display", None)
        dom_idx = 0 if (sds is None or sds.stretch_mode == "linear") else 1
        vel_val = sds.constant_velocity if sds else 2000.0
        self._seismic_domain_combo.blockSignals(True)
        self._seismic_domain_combo.setCurrentIndex(dom_idx)
        self._seismic_domain_combo.blockSignals(False)
        self._seismic_vel_spin.blockSignals(True)
        self._seismic_vel_spin.setValue(vel_val)
        self._seismic_vel_spin.blockSignals(False)
        self._seismic_vel_spin.setVisible(dom_idx == 0)

        has_seis = bool(self._state.project.seismic_refs
                        or self._state.get_seismic_for_section(section.name)[0] is not None)
        if not self._game_mode:
            self._seismic_row.setVisible(has_seis)
        self._full_render(section)

        if self._show_fps:
            ms = (time.perf_counter() - _t0) * 1000.0
            self.frame_time_ms.emit(ms)

    def _full_render(self, section) -> None:
        """Simple, correct render. No caching tricks — ax.clear() every frame."""
        self._seismic_artists.clear()
        self._overlay_artists.clear()
        self._ax.clear()
        self._configure_axes(self._ax)
        self._setup_axes(section)   # sets default xlim/ylim
        self._render_image_overlays(section)
        self._setup_seismic_artists(section)

        # Apply zoom limits AFTER seismic setup.
        # Priority: pending (new scroll/VE command) > saved (persisted from last zoom)
        # > defaults from _setup_axes (first render, section change).
        if self._pending_xlim is not None:
            self._ax.set_xlim(self._pending_xlim)
            self._saved_xlim = self._pending_xlim   # persist for future renders
            self._pending_xlim = None
        elif self._saved_xlim is not None:
            self._ax.set_xlim(self._saved_xlim)     # restore user's zoom

        if self._pending_ylim is not None:
            self._ax.set_ylim(self._pending_ylim)
            self._saved_ylim = self._pending_ylim   # persist
            self._pending_ylim = None
        elif self._saved_ylim is not None:
            self._ax.set_ylim(self._saved_ylim)     # restore user's zoom

        self._ax_limits_set = True

        # Safety: Y axis must always be inverted (depth=0 at top, increases downward)
        yl = self._ax.get_ylim()
        if yl[0] < yl[1]:                    # wrong orientation — fix it
            self._ax.set_ylim(yl[1], yl[0])
            if self._saved_ylim is not None:
                self._saved_ylim = (self._saved_ylim[1], self._saved_ylim[0])

        self._render_overlays(section)

        # Final safety after overlays: Y axis must stay inverted (depth 0 = top)
        _yl = self._ax.get_ylim()
        if _yl[0] < _yl[1]:
            self._ax.set_ylim(_yl[1], _yl[0])
            if self._saved_ylim is not None:
                self._saved_ylim = (self._saved_ylim[1], self._saved_ylim[0])

        self.view_changed.emit()
        _t_draw = time.perf_counter()
        self._canvas.draw_idle()
        _draw_ms = (time.perf_counter() - _t_draw) * 1000.0
        if _draw_ms > 30:
            ex_data, _ = self._state.get_seismic_for_section(section.name)
            shape_str = str(ex_data.shape) if ex_data is not None else "n/a"
            print(f"DRAW TIME: {_draw_ms:.0f}ms  seismic={'yes' if ex_data is not None else 'no'}"
                  f"  data_shape={shape_str}")

    def _render_depth_scale(self, section: Section) -> None:
        """Draw depth tick marks and labels at the left edge of the section axes."""
        yl = self._ax.get_ylim()
        y_top, y_bot = min(yl), max(yl)
        visible_range = y_bot - y_top
        if visible_range <= 0:
            return
        interval = _nice_interval(visible_range / 6)
        xl = self._ax.get_xlim()
        x_range = xl[1] - xl[0]
        x_pos   = xl[0] + x_range * 0.005
        tick_len = x_range * 0.008
        label_kw = dict(fontsize=7, color="#888888", va="center", ha="left", zorder=15)
        y = math.ceil(y_top / interval) * interval
        while y <= y_bot:
            self._overlay_artists.extend(
                self._ax.plot([x_pos, x_pos + tick_len], [y, y],
                              color="#666666", linewidth=0.7, zorder=15))
            self._overlay_artists.append(
                self._ax.text(x_pos + tick_len * 1.5, y, f"{y:.0f}", **label_kw))
            y += interval

    def _render_distance_scale(self, section: Section) -> None:
        """Draw distance tick marks along the bottom of the section axes."""
        xl = self._ax.get_xlim()
        x_start, x_end = xl[0], xl[1]
        visible_range = x_end - x_start
        if visible_range <= 0:
            return
        interval = _nice_interval(visible_range / 6)
        yl = self._ax.get_ylim()
        y_bot = max(yl)
        y_range = abs(yl[0] - yl[1])
        y_pos    = y_bot - y_range * 0.005
        tick_len = y_range * 0.008
        label_kw = dict(fontsize=7, color="#888888", va="top", ha="center", zorder=15)
        x = math.ceil(x_start / interval) * interval
        while x <= x_end:
            self._overlay_artists.extend(
                self._ax.plot([x, x], [y_pos, y_pos - tick_len],
                              color="#666666", linewidth=0.7, zorder=15))
            label = f"{x/1000:.1f}km" if interval >= 1000 else f"{x:.0f}m"
            self._overlay_artists.append(
                self._ax.text(x, y_pos - tick_len * 1.5, label, **label_kw))
            x += interval

    def _render_overlays(self, section) -> None:
        """Render all lightweight overlay layers, tracking artists for next-frame removal."""
        self._render_strat_column_chaser(section)
        if self._show_grid:
            self._render_grid(section)
        self._render_topography(section)
        self._render_sea_level(section)
        self._render_section_ends(section)
        self._render_reference_lines(section)
        self._render_polygons(section)
        self._render_surfaces(section)
        self._render_faults(section)
        self._render_horizons(section)
        self._render_wells(section)
        self._render_intersections(section)
        self._render_construct_preview()
        self._render_rubber_band(section)
        self._render_snap_indicator()
        self._render_polygon_in_progress()
        self._render_annotations(section)
        self._render_depth_scale(section)
        self._render_distance_scale(section)
        self._render_seismic_watermark(section)

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
        # Ensure all wells are visible even at high VE (well depth may exceed y_range).
        # Clamp to 0-50 000 m — rules out northing coordinates stored in wrong field.
        for well in self._state.project.wells:
            well_max = float(well.deviation.max_tvd)
            for log_name in well.log_names:
                try:
                    _, hi = well.get_log(log_name).depth_range()
                    hi = float(hi)
                    if 0 < hi < 50_000:
                        well_max = max(well_max, hi)
                except Exception:
                    pass
            if 0 < well_max < 50_000:
                y_range = max(y_range, well_max * 1.05)
        self._ax.set_ylim(y_range, 0.0)   # inverted: 0 at top
        # Prevent imshow / other artists from autoscaling Y away from this range
        self._ax.set_autoscaley_on(False)

        # Labels — check seismic domain to set Y label correctly
        units = section.depth_units
        xlabel = "Distance (m)"
        sds = getattr(section, "seismic_display", None)
        stretch = sds.stretch_mode if sds else "linear"
        _, ex_meta = self._state.get_seismic_for_section(section.name)
        seismic_is_twt = (
            (ex_meta is not None and ex_meta.get("domain") == "twt")
            or any(
                getattr(self._seismic_cache.get(ref.path), "domain", "") == "twt"
                for ref in self._state.project.seismic_refs
            )
        )
        if section.depth_domain == "twt" or (seismic_is_twt and stretch == "native_twt"):
            ylabel = "TWT (ms)"
        elif seismic_is_twt and stretch == "linear":
            ylabel = f"Depth ({units})  [TWT→depth]"
        elif units == "m+ft":
            ylabel = "Depth (m)"
            xlabel = "Distance (m)"
        else:
            ylabel = f"Depth ({units})"

        # Labels and ticks are provided by HUD depth_ruler and scale_bar.
        # Nothing to set on the matplotlib axes here.

        # Dual-unit secondary axes (m + ft)
        if units == "m+ft":
            _m2ft = 3.28084
            try:
                sec_y = self._ax.secondary_yaxis(
                    "right",
                    functions=(lambda m: m * _m2ft, lambda ft: ft / _m2ft),
                )
                sec_y.set_ylabel("Depth (ft)", fontsize=6, color="#888888")
                sec_y.tick_params(labelsize=6, colors="#888888")
                sec_x = self._ax.secondary_xaxis(
                    "top",
                    functions=(lambda m: m * _m2ft, lambda ft: ft / _m2ft),
                )
                sec_x.set_xlabel("Distance (ft)", fontsize=6, color="#888888")
                sec_x.tick_params(labelsize=6, colors="#888888")
            except Exception:
                pass

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
            # Include actual log depth range (LAS-imported wells may have
            # shallower deviation surveys than the actual log extent)
            for log_name in well.log_names:
                try:
                    _, hi = well.get_log(log_name).depth_range()
                    candidates.append(hi)
                except Exception:
                    pass
        sds = getattr(section, "seismic_display", None)
        mode = sds.stretch_mode      if sds else "linear"
        vel  = sds.constant_velocity if sds else 2000.0
        # Extracted seismic (preferred — already projected)
        _ex_data, ex_meta = self._state.get_seismic_for_section(section.name)
        if ex_meta is not None:
            last = float(ex_meta.get("sample_max", 0.0))
            if ex_meta.get("domain") == "twt" and mode == "linear":
                last = last * vel / 2000.0
            candidates.append(last)
        else:
            for ref in self._state.project.seismic_refs:
                ds = self._seismic_cache.get(ref.path)
                if ds is not None:
                    last = float(ds.samples[-1])
                    if ds.domain == "twt" and mode == "linear":
                        last = last * vel / 2000.0
                    candidates.append(last)
        return max(candidates)

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------

    def _render_sea_level(self, section: Section) -> None:
        if not self._show_sea_level:
            return
        yl = self._ax.get_ylim()
        if min(yl) <= 0.0 <= max(yl):
            self._overlay_artists.append(
                self._ax.axhline(0.0, color="#4682B4", linewidth=1.5,
                                 linestyle="-", zorder=2.5, alpha=0.85))
            xl = self._ax.get_xlim()
            self._overlay_artists.append(
                self._ax.text(xl[1], 0.0, "  Sea Level", fontsize=7,
                              color="#4682B4", va="bottom", ha="right",
                              zorder=2.5, alpha=0.85))

    def _render_topography(self, section: Section) -> None:
        topo_data = self._topography.get(section.name)
        if topo_data is None:
            return
        dists, elevs = topo_data
        if len(dists) < 2:
            return
        order = np.argsort(dists)
        dists, elevs = dists[order], elevs[order]
        yl    = self._ax.get_ylim()
        y_top = min(yl)
        self._overlay_artists.append(
            self._ax.fill_between(dists, y_top, elevs,
                                  color="#87CEEB", alpha=0.30, zorder=2.2,
                                  interpolate=True))
        self._overlay_artists.extend(
            self._ax.plot(dists, elevs, color="#222222", linewidth=2.0,
                          zorder=2.3, solid_capstyle="round"))
        self._overlay_artists.append(
            self._ax.text(float(dists[-1]), float(elevs[-1]), "  Surface",
                          fontsize=7, color="#222222", va="bottom", zorder=2.3))

    def _render_grid(self, section: Section) -> None:
        xl  = self._ax.get_xlim()
        yl  = self._ax.get_ylim()
        x_span = abs(xl[1] - xl[0])
        y_span = abs(yl[1] - yl[0])
        x_interval = _nice_interval(x_span / 5)
        y_interval = _nice_interval(y_span / 5)

        xs = np.arange(math.floor(xl[0] / x_interval) * x_interval,
                       xl[1] + x_interval, x_interval)
        ys = np.arange(math.floor(min(yl) / y_interval) * y_interval,
                       max(yl) + y_interval, y_interval)
        segments = []
        for x in xs[:200]:
            segments.append([(x, yl[0]), (x, yl[1])])
        for y in ys[:200]:
            segments.append([(xl[0], y), (xl[1], y)])
        if segments:
            lc = LineCollection(segments, colors="#252832", linewidths=0.6,
                                linestyles="--", zorder=2)
            self._overlay_artists.append(lc)
            self._ax.add_collection(lc)

        self._ax.xaxis.set_major_locator(MultipleLocator(x_interval))
        self._ax.yaxis.set_major_locator(MultipleLocator(y_interval))
        self._ax.ticklabel_format(style="plain", axis="both")

    def _render_seismic_watermark(self, section: Section) -> None:
        sds = getattr(section, "seismic_display", None)
        if sds is None or sds.stretch_mode != "linear":
            return
        has_twt = any(
            (ds := self._seismic_cache.get(ref.path)) is not None
            and getattr(ds, "domain", "") == "twt"
            for ref in self._state.project.seismic_refs
        )
        ex_data, ex_meta = self._state.get_seismic_for_section(section.name)
        if not has_twt and not (ex_meta and ex_meta.get("domain") == "twt"):
            return
        self._overlay_artists.append(
            self._ax.text(
                0.99, 0.01,
                f"Linear stretch  V = {sds.constant_velocity:.0f} m/s",
                fontsize=7, color="#999999", style="italic",
                ha="right", va="bottom", zorder=20,
                transform=self._ax.transAxes,
            ))

    def _render_annotations(self, section: Section) -> None:
        for ann in self._state.project.annotations:
            if ann.section_name and ann.section_name != section.name:
                continue
            r, g, b = ann.color
            color = "#{:02x}{:02x}{:02x}".format(r, g, b)
            px, pz = ann.position
            if ann.anchor_point is not None:
                ax_, az_ = ann.anchor_point
                self._overlay_artists.append(self._ax.annotate(
                    ann.text,
                    xy=(ax_, az_), xytext=(px, pz),
                    fontsize=ann.font_size, color=color,
                    rotation=ann.rotation_degrees,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
                    zorder=15,
                ))
            else:
                self._overlay_artists.append(
                    self._ax.text(
                        px, pz, ann.text,
                        fontsize=ann.font_size, color=color,
                        rotation=ann.rotation_degrees, zorder=15,
                    ))

    def _render_section_ends(self, section: Section) -> None:
        total = section.total_length()
        yl    = self._ax.get_ylim()
        ylo, yhi = min(yl), max(yl)
        kw = dict(color="#666666", linewidth=1.5, alpha=0.7, zorder=2,
                  solid_capstyle="butt")
        self._overlay_artists.extend(self._ax.plot([0, 0],         [ylo, yhi], **kw))
        self._overlay_artists.extend(self._ax.plot([total, total], [ylo, yhi], **kw))

    def _render_reference_lines(self, section: Section) -> None:
        xl = self._ax.get_xlim()
        yl = self._ax.get_ylim()
        ylo, yhi = min(yl), max(yl)
        kw = dict(color="#aaaaaa", linewidth=0.8, linestyle=(0, (6, 4)), zorder=1)
        for rl in self._state.project.reference_lines:
            if not rl.visible:
                continue
            label = rl.name or ""
            if rl.kind == "horizontal":
                self._overlay_artists.append(self._ax.axhline(rl.value, **kw))
                if label:
                    self._overlay_artists.append(
                        self._ax.text(xl[1], rl.value, f" {label}", fontsize=6,
                                      color="#999", va="center", ha="right", zorder=1))
            elif rl.kind == "vertical":
                if rl.map_x is not None and rl.map_y is not None:
                    # Always reproject from map coordinates — stays correct after node moves
                    dist, _ = section.project_point(rl.map_x, rl.map_y)
                else:
                    # Legacy ref line: backfill map coords from current section geometry.
                    # Mutate in place — same object stored in project.reference_lines so
                    # it will be written on the next natural DB save (no signal needed here).
                    dist = rl.value
                    rl.map_x, rl.map_y = section.section_to_map(dist)
                self._overlay_artists.append(self._ax.axvline(dist, **kw))
                if label:
                    self._overlay_artists.append(
                        self._ax.text(dist, ylo, f" {label}", fontsize=6,
                                      color="#999", va="bottom", ha="left",
                                      rotation=90, zorder=1))
            elif rl.kind == "angled":
                ang = math.radians(rl.angle_deg)
                far = max(abs(xl[1] - xl[0]), abs(yhi - ylo)) * 10
                dx  = math.cos(ang) * far
                dy  = -math.sin(ang) * far
                self._overlay_artists.extend(self._ax.plot(
                    [rl.anchor_x - dx, rl.anchor_x + dx],
                    [rl.anchor_y - dy, rl.anchor_y + dy],
                    **kw,
                ))
                if label:
                    self._overlay_artists.append(
                        self._ax.text(rl.anchor_x, rl.anchor_y, f" {label}",
                                      fontsize=6, color="#999", zorder=1))

        # A-Ref rubber band (anchor set, cursor pending)
        if self._ref_line_tool == "a_ref" and self._aref_anchor and self._cursor_data:
            ax_, ay_ = self._aref_anchor
            cx, cy   = self._cursor_data
            dx, dy   = cx - ax_, cy - ay_
            ang_d    = math.degrees(math.atan2(-dy, dx))
            self._overlay_artists.extend(
                self._ax.plot([ax_, cx], [ay_, cy],
                              color="#888", lw=1.0, linestyle="--", zorder=9))
            self._overlay_artists.append(
                self._ax.text(cx, cy, f"  {ang_d:.0f}°", fontsize=7,
                              color="#555", zorder=9))

    def _render_intersections(self, section) -> None:
        topo = self._state.topology
        if topo is None or topo.section_name != section.name:
            return
        ipts = [p for p in topo.intersections if "boundary" not in p.type]
        if not ipts:
            return
        xl, xr = self._ax.get_xlim()
        yl = self._ax.get_ylim()
        y_lo, y_hi = min(yl), max(yl)
        try:
            inv = self._ax.transData.inverted()
            p0 = inv.transform([0, 0])
            p1 = inv.transform([7, 7])
            hw = abs(float(p1[0]) - float(p0[0]))
            hh = abs(float(p1[1]) - float(p0[1]))
        except Exception:
            hw = hh = (xr - xl) * 0.007
        for pt in ipts:
            sx, sy = pt.x, pt.y
            if not (xl <= sx <= xr and y_lo <= sy <= y_hi):
                continue
            self._overlay_artists.extend(
                self._ax.plot([sx - hw, sx + hw], [sy, sy],
                              color="#00CCCC", linewidth=1.8, zorder=10, solid_capstyle="round"))
            self._overlay_artists.extend(
                self._ax.plot([sx, sx], [sy - hh, sy + hh],
                              color="#00CCCC", linewidth=1.8, zorder=10, solid_capstyle="round"))

    def _render_snap_indicator(self) -> None:
        if self._snap_point is None:
            return
        sx, sy = self._snap_point
        s = 8
        try:
            inv = self._ax.transData.inverted()
            p0 = inv.transform([0, 0])
            p1 = inv.transform([s, s])
            dx = abs(float(p1[0]) - float(p0[0]))
            dy = abs(float(p1[1]) - float(p0[1]))
        except Exception:
            return
        kw = dict(color="magenta", lw=1.5, zorder=13)
        self._overlay_artists.extend(self._ax.plot([sx, sx + dx],    [sy, sy + dy],      **kw))
        self._overlay_artists.extend(self._ax.plot([sx + dx, sx],    [sy + dy, sy+2*dy], **kw))
        self._overlay_artists.extend(self._ax.plot([sx, sx - dx],    [sy+2*dy, sy + dy], **kw))
        self._overlay_artists.extend(self._ax.plot([sx - dx, sx],    [sy + dy, sy],      **kw))

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
        d_raw = hp._distances[sec_idxs]
        z_sec = hp._depths[sec_idxs]
        if len(d_raw) == 0:
            return
        # Reproject from map coordinates when available so display stays correct
        # after section geometry changes regardless of whether recompute was called.
        mx = hp._map_x[sec_idxs]
        my = hp._map_y[sec_idxs]
        has_map = ~(np.isnan(mx) | np.isnan(my))
        if np.any(has_map):
            d_sec = d_raw.copy()
            for local_i, full_i in enumerate(np.where(has_map)[0]):
                d_sec[full_i], _ = section.project_point(float(mx[full_i]), float(my[full_i]))
        else:
            d_sec = d_raw

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
        base_z = 8 if category == "Horizons" else 7
        zorder = base_z + 1 if (is_active or is_selected) else base_z

        if is_selected:
            self._overlay_artists.extend(
                self._ax.plot(d_sec, z_sec, color=hp.color,
                              linewidth=render_lw * 3, alpha=0.20,
                              zorder=zorder - 1, solid_capstyle="round"))

        if not decorated:
            self._overlay_artists.extend(
                self._ax.plot(d_sec, z_sec, color=hp.color,
                              linewidth=render_lw, linestyle=ls, zorder=zorder))

        if len(d_sec) >= 2:
            self._render_line_decoration(hp, d_sec, z_sec, category, lw)

        if is_edit:
            for local_i, fi_full in enumerate(sec_idxs):
                d = float(d_sec[local_i])
                z = float(hp._depths[fi_full])
                ms, fc, ec, ew = self._pick_point_style(category, obj_idx, fi_full)
                self._overlay_artists.extend(
                    self._ax.plot(d, z, marker,
                                  markersize=ms, markerfacecolor=fc,
                                  markeredgecolor=ec, markeredgewidth=ew, zorder=11))

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
            self._overlay_artists.extend(
                self._ax.plot(xw, yw, color=hp.color, lw=base_lw, zorder=3))

        elif ct == "disconformity":
            xw, yw = _wavy_coords(self._ax, d_sec, z_sec, 3.0, 20.0)
            self._overlay_artists.extend(
                self._ax.plot(xw, yw, color=hp.color, lw=base_lw, linestyle="--", zorder=3))

        elif ct == "intrusive_contact":
            ticks = _line_ticks(self._ax, d_sec, z_sec, 30.0, 6.0, 1.0)
            for x0, y0, x1, y1 in ticks:
                self._overlay_artists.append(
                    self._ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                                      arrowprops=dict(arrowstyle="x", color=hp.color,
                                                      lw=0.8), zorder=4))

        elif ct == "sequence_boundary":
            self._overlay_artists.extend(
                self._ax.plot(d_sec, z_sec, color=hp.color, lw=2.5, zorder=3))

        elif ct == "maximum_flooding_surface":
            tris = _line_triangles(self._ax, d_sec, z_sec, 40.0, 7.0, -1.0)
            from matplotlib.patches import Polygon as MplPoly
            for verts in tris:
                patch = MplPoly(verts, closed=True,
                                facecolor=hp.color, edgecolor=hp.color,
                                lw=0.5, zorder=4)
                self._overlay_artists.append(patch)
                self._ax.add_patch(patch)

        elif ft == "reverse" or ft == "thrust":
            lw_line = base_lw * (1.3 if ft == "thrust" else 1.0)
            self._overlay_artists.extend(
                self._ax.plot(d_sec, z_sec, color=hp.color, lw=lw_line, zorder=3))
            side = 1.0 if getattr(hp, "dip_direction", "right") == "right" else -1.0
            tris = _line_triangles(self._ax, d_sec, z_sec, 40.0, 8.0, side)
            from matplotlib.patches import Polygon as MplPoly
            for verts in tris:
                patch = MplPoly(verts, closed=True,
                                facecolor=hp.color, edgecolor=hp.color,
                                lw=0.5, zorder=4)
                self._overlay_artists.append(patch)
                self._ax.add_patch(patch)

        elif ft in ("normal", "growth_fault"):
            self._overlay_artists.extend(
                self._ax.plot(d_sec, z_sec, color=hp.color, lw=base_lw, zorder=3))
            side = 1.0 if getattr(hp, "dip_direction", "right") == "right" else -1.0
            ticks = _line_ticks(self._ax, d_sec, z_sec, 40.0, 8.0, side)
            for x0, y0, x1, y1 in ticks:
                self._overlay_artists.extend(
                    self._ax.plot([x0, x1], [y0, y1],
                                  color=hp.color, lw=0.9, zorder=4))

        elif ft == "detachment":
            self._overlay_artists.extend(
                self._ax.plot(d_sec, z_sec, color=hp.color, lw=base_lw * 2, zorder=3))

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

    def _render_image_overlays(self, section: Section) -> None:
        """Render registered raster image overlays at zorder=0."""
        for path, sec_name, (d0, d1), (z0, z1) in self._image_overlays:
            if sec_name != section.name:
                continue
            try:
                import matplotlib.image as mpimg
                img = mpimg.imread(path)
                self._ax.imshow(
                    img,
                    aspect="auto",
                    extent=[d0, d1, z1, z0],
                    origin="upper",
                    zorder=0,
                )
            except Exception:
                pass

    def _display_seismic_data(self, data: np.ndarray) -> np.ndarray:
        """Optionally downsample seismic data to screen resolution for fast display."""
        if not getattr(self, "_fast_display_cb", None) or not self._fast_display_cb.isChecked():
            return data
        try:
            bbox = self._ax.get_window_extent()
            target_w = max(1, int(bbox.width))
            target_h = max(1, int(bbox.height))
            h, w = data.shape[:2]
            step_x = max(1, w // target_w)
            step_y = max(1, h // target_h)
            if step_x > 1 or step_y > 1:
                return data[::step_y, ::step_x]
        except Exception:
            pass
        return data

    def _setup_seismic_artists(self, section: Section) -> None:
        """Create seismic imshow artist(s) — called on every full render."""
        sds      = getattr(section, "seismic_display", None)
        clip_pct = sds.clip_percentile   if sds else 99.0
        gain     = sds.gain              if sds else 1.0
        opacity  = sds.opacity           if sds else 1.0
        cmap_key = sds.colormap          if sds else _DEFAULT_CMAP
        stretch  = sds.stretch_mode      if sds else "linear"
        v_ms     = sds.constant_velocity if sds else 2000.0
        cmap_name = _SEGY_CMAP.get(cmap_key, _DEFAULT_CMAP)

        def _imshow(img_data, dist0, dist1, y_top, y_bot):
            img_data = self._display_seismic_data(img_data)
            # Use a tighter clip (97%) for better trace-to-trace contrast.
            # clip_pct from settings defaults to 99 — cap at 97 for display.
            effective_clip = min(float(clip_pct), 97.0)
            vmax = float(np.percentile(np.abs(img_data), effective_clip) or 1.0) * gain
            # Adaptive interpolation: nearest when many traces are visible (preserves
            # trace character), bilinear when zoomed-out for smoother appearance.
            xl = self._ax.get_xlim()
            visible_dist = abs(xl[1] - xl[0]) if xl[1] != xl[0] else (dist1 - dist0)
            n_traces = img_data.shape[1] if img_data.ndim == 2 else 1
            trace_spacing = (dist1 - dist0) / max(n_traces - 1, 1)
            visible_traces = visible_dist / max(trace_spacing, 1.0)
            interp = "nearest" if visible_traces < 300 else "bilinear"
            art = self._ax.imshow(
                img_data,
                aspect="auto",
                extent=[dist0, dist1, y_bot, y_top],
                origin="upper",
                cmap=cmap_name,
                vmin=-vmax, vmax=vmax,
                interpolation=interp,
                alpha=opacity,
                zorder=1,
            )
            self._seismic_artists.append(art)

        # Extracted seismic (preferred — already projected onto section)
        ex_data, ex_meta = self._state.get_seismic_for_section(section.name)
        if ex_data is not None and ex_meta is not None:
            samples = np.asarray(ex_meta["samples"])
            domain  = ex_meta.get("domain", "twt")
            if stretch == "linear" and domain == "twt":
                scale = v_ms / 2000.0
                y_top, y_bot = float(samples[0]) * scale, float(samples[-1]) * scale
            else:
                y_top, y_bot = float(samples[0]), float(samples[-1])
            if ex_data.shape[1] >= 2:
                # Use the dedicated dist_min/dist_max keys so the imshow extent
                # matches the ACTUAL data coverage, not section.total_length().
                dist0 = float(ex_meta["dist_min"])
                dist1 = float(ex_meta["dist_max"])
                _imshow(ex_data, dist0, dist1, y_top, y_bot)
                self._apply_seismic_boundary_mask(section, dist0, dist1)
            return

        # Fallback: project full SEG-Y on the fly (slow, only when no extraction)
        show_wig = (sds.show_wiggle if sds else False) or (self._display_mode == "wiggle")
        for ref in self._state.project.seismic_refs:
            ds = self._get_or_load_seismic(ref)
            if ds is None or ds.n_traces == 0:
                continue
            proj_key = (section.name, ref.path)
            if proj_key in self._seismic_proj_cache:
                distances, data, perps = self._seismic_proj_cache[proj_key]
            else:
                distances, data, perps = ds.traces_sorted_by_section(section)
                self._seismic_proj_cache[proj_key] = (distances, data, perps)
            mask = np.abs(perps) <= 500.0
            if mask.sum() >= 2:
                distances, data = distances[mask], data[mask]
            if len(distances) < 2:
                continue
            if stretch == "linear" and ds.domain == "twt":
                scale = v_ms / 2000.0
                y_top, y_bot = float(ds.samples[0]) * scale, float(ds.samples[-1]) * scale
            else:
                y_top, y_bot = float(ds.samples[0]), float(ds.samples[-1])
            if show_wig:
                self._render_wiggle(distances, data, ds.samples)
            else:
                dist0f = float(distances[0])
                dist1f = float(distances[-1])
                _imshow(data.T, dist0f, dist1f, y_top, y_bot)
                # Dim seismic outside section bounds (same as extracted path)
                self._apply_seismic_boundary_mask(section, dist0f, dist1f)

    def _apply_seismic_boundary_mask(
        self, section, seis_start: float, seis_end: float,
    ) -> None:
        """Dim seismic outside the section line using Rectangle patches.

        Uses the actual axes limits (not the seismic sample range) so the
        overlay covers the full plot height regardless of seismic extent.
        Rectangle patches are used instead of fill_betweenx because they
        are guaranteed to work with inverted Y axes.
        """
        from matplotlib.patches import Rectangle as _Rect
        sec_end  = section.total_length()
        yl       = self._ax.get_ylim()         # inverted: yl[0] > yl[1]
        y_bottom = max(yl)                      # deeper (larger value = bottom)
        y_top    = min(yl)                      # shallower (smaller = top)
        height   = y_bottom - y_top
        dim_kw   = dict(facecolor="#1E1E1E", alpha=0.55,
                        edgecolor="none", zorder=2)

        if seis_start < 0.0:
            rect = _Rect((seis_start, y_top), -seis_start, height, **dim_kw)
            self._ax.add_patch(rect)
            self._seismic_artists.append(rect)

        if seis_end > sec_end:
            rect = _Rect((sec_end, y_top), seis_end - sec_end, height, **dim_kw)
            self._ax.add_patch(rect)
            self._seismic_artists.append(rect)

        # Thin boundary lines at section endpoints
        for x in (0.0, sec_end):
            ln, = self._ax.plot([x, x], [y_top, y_bottom],
                                color="#AAAAAA", linewidth=0.8,
                                linestyle="-", alpha=0.6, zorder=3)
            self._seismic_artists.append(ln)

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
        aoi = getattr(self._state.project, "aoi", None)
        surfaces = self._state.get_visible_surfaces()
        for surf in surfaces:
            try:
                distances, z_values = surf.intersect_section(section, n_samples=200)
            except Exception:
                continue

            # AOI masking
            if aoi is not None:
                try:
                    map_pts = np.array([section.section_to_map(d) for d in distances])
                    outside = ~aoi.contains_xy(map_pts[:, 0], map_pts[:, 1])
                    z_values[outside] = np.nan
                except Exception:
                    pass

            # Z domain conversion when seismic is in TWT and surface is in depth (or vice versa)
            sds = getattr(section, "seismic_display", None)
            vel = sds.constant_velocity if sds else 2000.0
            if getattr(surf, "z_domain", "depth_m") == "twt_ms" and section.depth_domain != "twt":
                z_values = z_values * vel / 2000.0   # TWT ms → depth m

            valid = ~np.isnan(z_values)
            if not np.any(valid):
                continue

            color = surf.display_color
            kind  = getattr(surf, "kind", "horizon")
            ls    = "--" if kind == "fault" else "-"
            lw    = float(getattr(surf, "line_width", 1.5))

            z_masked = np.ma.masked_invalid(z_values)
            self._overlay_artists.extend(
                self._ax.plot(distances, z_masked, color=color,
                              linewidth=lw, linestyle=ls, alpha=0.9, zorder=6)
            )
            # Name label at the midpoint of valid data
            mid = len(distances) // 2
            if valid[mid]:
                self._overlay_artists.append(
                    self._ax.text(distances[mid], z_values[mid], f" {surf.name}",
                                  fontsize=6, color=color, va="bottom", zorder=6)
                )

    def _render_polygons(self, section: Section) -> None:
        from matplotlib.patches import Polygon as MplPolygon
        for poly in self._state.project.polygons:
            # Only render polygons that belong to this section (or have no section tag)
            poly_sec = getattr(poly, "section_name", "")
            if poly_sec and poly_sec != section.name:
                continue
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
                               hatch=hatch, zorder=4)
            self._overlay_artists.append(patch)
            self._ax.add_patch(patch)
            if hatch:
                hatch_patch = MplPolygon(verts, closed=True,
                                         facecolor="none", alpha=0.35,
                                         edgecolor="black", linewidth=0,
                                         hatch=hatch, zorder=4)
                self._overlay_artists.append(hatch_patch)
                self._ax.add_patch(hatch_patch)

            # Phase 5: formation label inside polygon
            label = getattr(poly, "formation", "") or poly.name
            if label:
                # Representative point (inside polygon, not just centroid)
                cx, cy = float(verts[:, 0].mean()), float(verts[:, 1].mean())
                try:
                    from shapely.geometry import Polygon as _SPoly
                    shp = _SPoly(verts)
                    if shp.is_valid:
                        pt = shp.representative_point()
                        cx, cy = float(pt.x), float(pt.y)
                except Exception:
                    pass

                # Font size based on polygon screen area
                try:
                    pts_s = self._ax.transData.transform(verts)
                    xs_s, ys_s = pts_s[:, 0], pts_s[:, 1]
                    n = len(xs_s)
                    scr_area = abs(sum(
                        xs_s[i] * ys_s[(i+1) % n] - xs_s[(i+1) % n] * ys_s[i]
                        for i in range(n)
                    )) * 0.5
                    fontsize = max(7, min(14, 7 + scr_area / 6000))
                except Exception:
                    fontsize = 8

                # Auto-contrast text color
                try:
                    from matplotlib.colors import to_rgb as _to_rgb
                    r, g, b = _to_rgb(poly.fill_color)
                    lum = r * 0.299 + g * 0.587 + b * 0.114
                    text_color = "black" if lum > 0.55 else "white"
                except Exception:
                    text_color = "black"

                self._overlay_artists.append(
                    self._ax.text(cx, cy, label,
                                  fontsize=fontsize, color=text_color,
                                  ha="center", va="center",
                                  clip_on=True, zorder=5))

    def _render_polygon_in_progress(self) -> None:
        if not self._polygon_drawing or not self._polygon_vertices:
            return
        xs = [v[0] for v in self._polygon_vertices]
        ys = [v[1] for v in self._polygon_vertices]
        self._overlay_artists.extend(
            self._ax.plot(xs, ys, "o-", color="#9467bd", linewidth=1.5,
                          markersize=5, zorder=10))
        if len(xs) >= 2:
            self._overlay_artists.extend(
                self._ax.plot([xs[-1], xs[0]], [ys[-1], ys[0]],
                              "--", color="#9467bd", linewidth=1.0, alpha=0.5, zorder=10))
        if self._cursor_data is not None and len(self._polygon_vertices) >= 1:
            lx, ly = self._polygon_vertices[-1]
            cx, cy = self._cursor_data
            self._overlay_artists.extend(
                self._ax.plot([lx, cx], [ly, cy], "--", color="#9467bd",
                              linewidth=1.0, alpha=0.7, zorder=14))

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
            self._overlay_artists.extend(
                self._ax.plot([d[li], cx], [z[li], cy], **rb_kw))
        if right.any():
            ri = int(np.where(right)[0][0])
            self._overlay_artists.extend(
                self._ax.plot([cx, d[ri]], [cy, z[ri]], **rb_kw))

        if shift_held:
            _, last_d, last_z = self._get_active_pick_last_point()
            if last_d is not None:
                ang = math.atan2(-(cy - last_z), cx - last_d)
                xl = self._ax.get_xlim()
                far = abs(xl[1] - xl[0])
                self._overlay_artists.extend(self._ax.plot(
                    [last_d - far * math.cos(ang), last_d + far * math.cos(ang)],
                    [last_z + far * math.sin(ang), last_z - far * math.sin(ang)],
                    color=color, lw=0.5, linestyle=":", alpha=0.4, zorder=7,
                ))

    def _render_wells(self, section: Section) -> None:
        try:
            inv = self._ax.transData.inverted()
            p0 = inv.transform([0, 0])
            p1 = inv.transform([8, 0])
            tick_w = abs(float(p1[0]) - float(p0[0]))
        except Exception:
            tick_w = section.total_length() * 0.006

        for well in self._state.project.wells:
            collar_dist, perp = well.project_to_section(section)
            if abs(perp) > _WELL_MAX_PERP:
                continue

            # --- Determine stick bottom ---
            # Priority: stored TD (from LAS depth index on import) > deepest log depth index.
            # Reject suspiciously large defaults (e.g. 5000m placeholder on a 3150m well).
            td_stored = float(well.deviation.max_tvd)
            max_log_depth = 0.0
            for log_name in well.log_names:
                try:
                    _, hi = well.get_log(log_name).depth_range()
                    max_log_depth = max(max_log_depth, float(hi))
                except Exception:
                    pass
            # Use the smaller of td_stored and max_log_depth when td looks like
            # the 5000m default and log data says otherwise (sanity: within 20%).
            if (max_log_depth > 0 and td_stored > max_log_depth * 1.2
                    and abs(td_stored - 5000.0) < 1.0):
                well_bottom = max_log_depth
            else:
                well_bottom = max(td_stored, max_log_depth) if max_log_depth > 0 else td_stored
            if well_bottom <= 0:
                continue

            # 1. Well stick — drawn first (lowest zorder) so log curves render on top
            well_color = getattr(well, "color", "#E8E4D0")
            self._overlay_artists.extend(
                self._ax.plot([collar_dist, collar_dist], [0.0, well_bottom],
                              color=well_color, linewidth=2.0,
                              solid_capstyle="butt", zorder=8))

            # 2. Deviated trajectory overlay (only adds value for non-vertical wells)
            distances, tvds = well.section_track(section)
            if len(distances) > 2 or (len(distances) == 2 and
                                       abs(float(distances[0]) - float(distances[-1])) > 1.0):
                self._overlay_artists.extend(
                    self._ax.plot(distances, tvds, color="#AAAAAA", linewidth=1.0,
                                  linestyle="--", zorder=9))

            direction = "E" if perp >= 0 else "W"
            label = f"{well.name}  ({abs(perp):.0f}m {direction})"
            self._overlay_artists.append(
                self._ax.annotate(
                    label,
                    xy=(collar_dist, 0.0),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color=well_color, zorder=10,
                    ha="left", va="bottom",
                ))

            for top_name in well.formation_tops:
                try:
                    td, tz = well.formation_top_in_section(top_name, section)
                except KeyError:
                    continue
                self._overlay_artists.extend(
                    self._ax.plot([td - tick_w, td + tick_w], [tz, tz],
                                  color="#5CB85C", linewidth=1.2, zorder=9))
                self._overlay_artists.append(
                    self._ax.text(td + tick_w * 1.4, tz, top_name,
                                  fontsize=6, color="#5CB85C", va="center", zorder=9))

            # Use explicitly selected log; fall back to auto-detecting a GR curve
            display_log = getattr(well, "display_log", None)
            if display_log and display_log in well.log_names:
                log_name = display_log
            else:
                log_name = next(
                    (n for n in well.log_names
                     if "GR" in n.upper() or n.upper() in ("GAMMA", "GR")),
                    None,
                )
            if log_name:
                self._render_gr_log(well, log_name, collar_dist, section)

    def _render_gr_log(self, well, gr_name: str, collar_dist: float,
                       section: Section) -> None:
        """Render a GR log as a filled wiggle alongside the well stick."""
        try:
            curve = well.get_log(gr_name)
        except KeyError:
            return
        tvd_depths = curve.depths
        values = curve.values
        if len(tvd_depths) < 2:
            return
        # Strip NaN depth samples so the wiggle only draws over actual valid data,
        # not across the NaN-padded tails of the LAS depth index.
        valid = ~np.isnan(values)
        if valid.sum() < 2:
            return
        tvd_depths = tvd_depths[valid]
        values = values[valid]
        # Normalize to [0, 1]
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
        if vmax - vmin < 1.0:
            return
        norm = (values - vmin) / (vmax - vmin)
        # Map to section-space: 50m track width centred on well
        track_w = 50.0
        xs = collar_dist + (norm - 0.5) * track_w
        self._overlay_artists.extend(
            self._ax.plot(xs, tvd_depths, color="#C8A060", linewidth=0.6, zorder=9))
        self._overlay_artists.append(
            self._ax.fill_betweenx(tvd_depths, collar_dist, xs,
                                   where=(norm < 0.5),
                                   color="#FFD060", alpha=0.5, zorder=9))
        self._overlay_artists.append(
            self._ax.fill_betweenx(tvd_depths, collar_dist, xs,
                                   where=(norm >= 0.5),
                                   color="#B0B0B0", alpha=0.35, zorder=9))

    # Formation strip HUD widget replaces the old matplotlib _strat_ax.
    # A thin matplotlib-based chaser column is rendered INSIDE the axes at the left edge.

    def _render_strat_column_chaser(self, section: Section) -> None:
        """Render a thin strat column at the left edge of the section axes."""
        from matplotlib.patches import Rectangle as _Rect
        polygons = [p for p in self._state.project.polygons
                    if not p.section_name or p.section_name == section.name]
        if not polygons:
            return

        # Compute depth range per formation from polygon vertices
        fm_depths: dict[str, tuple[float, float]] = {}
        fm_color:  dict[str, str] = {}
        for poly in polygons:
            name = poly.formation or poly.name or ""
            base = name.rsplit(" (", 1)[0]
            verts = poly._vertices  # (N, 2) of (distance, depth)
            if len(verts) == 0:
                continue
            depths = verts[:, 1]
            d_top, d_bot = float(depths.min()), float(depths.max())
            if base in fm_depths:
                old_top, old_bot = fm_depths[base]
                fm_depths[base] = (min(old_top, d_top), max(old_bot, d_bot))
            else:
                fm_depths[base] = (d_top, d_bot)
                fm_color[base]  = poly.fill_color

        if not fm_depths:
            return

        xl   = self._ax.get_xlim()
        col_w = (xl[1] - xl[0]) * 0.025   # 2.5% of visible width
        col_l = xl[0]                       # flush with left edge

        for name, (d_top, d_bot) in sorted(fm_depths.items(), key=lambda t: t[1][0]):
            hex_col = fm_color.get(name, "#777777")
            rect = _Rect(
                (col_l, d_top), col_w, d_bot - d_top,
                facecolor=hex_col, alpha=0.80,
                edgecolor="#444444", linewidth=0.4,
                zorder=15, clip_on=True,
            )
            self._ax.add_patch(rect)
            self._overlay_artists.append(rect)

            # Label if tall enough relative to visible range
            yl   = self._ax.get_ylim()
            vis  = abs(yl[0] - yl[1])
            if abs(d_bot - d_top) > vis * 0.06:
                short = name[:10]
                lbl = self._ax.text(
                    col_l + col_w / 2, (d_top + d_bot) / 2, short,
                    ha="center", va="center",
                    fontsize=5, color="white", fontweight="bold",
                    rotation=90, zorder=16, clip_on=True,
                )
                self._overlay_artists.append(lbl)

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
        """Return nearest snap target (pick point or topology intersection) within threshold."""
        if not self._snap_active:
            return None
        section = self._state.active_section
        if section is None:
            return None
        sec_name = section.name
        ex, ey = self._to_screen_px_sv(x, y)
        best_dist = float(_SNAP_THRESHOLD)
        best_pt: tuple[float, float] | None = None

        # Pick points
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

        # Topology intersection points (higher priority if closer)
        topo = self._state.topology
        if topo is not None and topo.section_name == sec_name:
            for d, z in topo.get_snap_targets():
                nx, ny = self._to_screen_px_sv(d, z)
                dist = math.hypot(ex - nx, ey - ny)
                if dist < best_dist:
                    best_dist = dist
                    best_pt = (d, z)

        # Section endpoints always snap within 20px
        total = section.total_length()
        for edge_d in (0.0, total):
            # Use the current data-space y as z for the edge snap
            nx, ny = self._to_screen_px_sv(edge_d, y)
            dist = math.hypot(ex - nx, ey - ny)
            if dist < 20.0 and dist < best_dist:
                best_dist = dist
                best_pt = (edge_d, y)

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

    def _undo_last_pick(self) -> None:
        """Middle-click during picking: remove the last placed pick on this section."""
        cat = self._state.active_pick_category
        idx = self._state.active_pick_index
        if cat is None or idx is None:
            return
        section = self._state.active_section
        sec_name = section.name if section else ""
        proj = self._state.project
        picks = proj.horizon_picks if cat == "Horizons" else proj.fault_picks
        if idx >= len(picks):
            return
        hp = picks[idx]
        sec_idxs = hp.section_indices(sec_name)
        if len(sec_idxs) == 0:
            return
        # Remove the last pick on this section
        last_idx = int(sec_idxs[-1])
        hp2 = copy.deepcopy(hp)
        if hp2.n_picks > 1:
            for attr in ("_distances", "_depths", "_section_names",
                         "_confidence", "_quality", "_note"):
                arr = getattr(hp2, attr)
                setattr(hp2, attr, np.delete(arr, last_idx))
        if cat == "Horizons":
            self._state.update_horizon_pick(idx, hp2)
        else:
            self._state.update_fault_pick(idx, hp2)
        remaining = len(hp2.section_indices(sec_name))
        self._flash_hint(f"Removed last pick  ({remaining} remaining)")

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
        # Convert section distance → map coordinates (source of truth for future reprojection)
        map_x, map_y = section.section_to_map(x) if section is not None else (float("nan"), float("nan"))
        hp_after.insert_pick(x, y, sec_name, map_x=map_x, map_y=map_y)

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
        """Return cached SeismicDataset, loading on demand if not yet cached.

        The first call per ref blocks the UI thread while reading the SEG-Y.
        This is acceptable for typical 2D lines (seconds, not minutes).
        Subsequent calls return instantly from the in-memory cache.
        """
        if ref.path not in self._seismic_cache:
            try:
                from PySide6.QtWidgets import QApplication
                from PySide6.QtCore import Qt as _Qt
                QApplication.setOverrideCursor(_Qt.CursorShape.WaitCursor)
                QApplication.processEvents()
                ds = ref.load()
                QApplication.restoreOverrideCursor()
                if ds is not None:
                    self._seismic_cache[ref.path] = ds
            except Exception:
                try:
                    QApplication.restoreOverrideCursor()
                except Exception:
                    pass
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
        from section_tool.core.reference_line import ReferenceLine
        tool = self._ref_line_tool
        if tool == "h_ref":
            rl = ReferenceLine(kind="horizontal", value=y,
                               name=f"H {y:.0f}")
            self._state.add_reference_line(rl)
        elif tool == "v_ref":
            section = self._state.active_section
            map_x, map_y = (section.section_to_map(x)
                            if section is not None else (None, None))
            rl = ReferenceLine(kind="vertical", value=x,
                               name=f"V {x:.0f}", map_x=map_x, map_y=map_y)
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

    def _on_strat_col_click(self, event) -> None:
        """Phase 4: click on strat column → select the horizon nearest clicked depth."""
        clicked_depth = event.ydata
        if clicked_depth is None:
            return
        section = self._state.active_section
        if section is None:
            return
        # Find the horizon whose median depth on this section is nearest
        best_idx, best_dist = None, float("inf")
        for i, hp in enumerate(self._state.project.horizon_picks):
            sec_idxs = hp.section_indices(section.name)
            if len(sec_idxs) == 0:
                continue
            median_d = float(np.median(hp._depths[sec_idxs]))
            d = abs(median_d - clicked_depth)
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_idx is not None:
            self._state.set_active_pick_target("Horizons", best_idx)

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

    def _on_ve_changed(self) -> None:
        value = self._ve_spin.value()
        if value <= 0:
            return
        # VE change only affects ylim, not seismic data — use fast path.
        # Pre-compute new ylim so the fast path can apply it without _setup_axes.
        section = self._state.active_section
        if section is not None:
            max_d = self._compute_max_depth(section)
            y_range = max_d / value
            new_ylim = (y_range, 0.0)
            self._pending_ylim = new_ylim
            self._saved_ylim   = new_ylim   # persist so rubber-band renders keep it
            self._saved_xlim   = None       # let xlim reset to full section width on VE change
            self._pending_xlim = None
            self._user_has_zoomed = False   # full section width on VE change

        if self._ve_lock_btn.isChecked():
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

    def _on_seismic_domain_changed(self, _index: int = 0) -> None:
        from section_tool.core.seismic_settings import SeismicDisplaySettings
        mode = self._seismic_domain_combo.currentData()
        self._seismic_vel_spin.setVisible(mode == "linear")
        section = self._state.active_section
        if section is not None:
            sds = getattr(section, "seismic_display", None) or SeismicDisplaySettings()
            sds.stretch_mode = mode
            section.seismic_display = sds
        self._ax_limits_set = False
        self.request_render()

    def _on_seismic_velocity_changed(self, value: float) -> None:
        from section_tool.core.seismic_settings import SeismicDisplaySettings
        section = self._state.active_section
        if section is not None:
            sds = getattr(section, "seismic_display", None) or SeismicDisplaySettings()
            sds.constant_velocity = value
            section.seismic_display = sds
        self._ax_limits_set = False
        self.request_render()

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

        # ---- Middle-click undoes last placed pick (Phase 4) ----
        if event.button == 2 and (self._picking_active or self._fault_picking):
            self._undo_last_pick()
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
            is_dbl = getattr(event, "dblclick", False)
            self._polygon_vertices.append((x, y))
            self.polygon_vertex_added.emit(x, y)
            if is_dbl and len(self._polygon_vertices) >= 3:
                self.finish_polygon()
            else:
                self.render()
            return
        if event.button == 3 and self._polygon_drawing:
            self.finish_polygon()
            return

        # ---- Phase 2: reference line placement ----
        if event.button == 1 and self._ref_line_tool:
            self._place_reference_line(x, y)
            return

        # ---- Phase 3: construct tools ----
        if event.button == 1 and self._construct_tool:
            self._handle_construct_click(x, y)
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
                        map_x, map_y = (section.section_to_map(x) if section is not None
                                        else (float("nan"), float("nan")))
                        hp.insert_pick(x, y, sec_name, map_x=map_x, map_y=map_y)
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
        # Reset snap at the start of every move — only set it below if eligible
        self._snap_point = None

        if event.xdata is not None and event.ydata is not None:
            cx, cy = float(event.xdata), float(event.ydata)
            self._cursor_data = (cx, cy)

            # Snap only in pick/edit modes — never in select/pan/zoom/measure
            tool = self._state.active_tool
            _snap_active = (
                tool in _SNAP_TOOLS
                or self._picking_active
                or self._fault_picking
                or self._polygon_drawing
                or self._construct_tool is not None
                or (tool == "node_edit" and self._pick_selected is not None)
            )
            if _snap_active:
                self._snap_point = self._compute_snap(cx, cy)

            # Coordinate readout in status bar and game HUD
            self._show_section_coords(cx, cy)
            self.coords_updated.emit(cx, cy)
            # Cross-view cursor tracking: section distance → map coordinates
            section = self._state.active_section
            if section is not None:
                try:
                    mx, my = section.section_to_map(cx)
                    self.cursor_map_pos.emit(mx, my)
                except Exception:
                    pass
            # Snap hint during active picking
            if self._snap_point is not None and (self._picking_active or self._fault_picking):
                sx, sz = self._snap_point
                self._flash_hint(f"Snap: ({sx:.0f} m,  {sz:.0f} m depth)")
        else:
            self._cursor_data = None

        # ---- Pan ----
        if self._sv_pan_anchor is not None:
            try:
                d0 = self._sv_pan_inv.transform(self._sv_pan_anchor)
                d1 = self._sv_pan_inv.transform([event.x, event.y])
                new_xl = (self._sv_pan_xlim0[0] + d0[0] - d1[0],
                          self._sv_pan_xlim0[1] + d0[0] - d1[0])
                new_yl = (self._sv_pan_ylim0[0] + d0[1] - d1[1],
                          self._sv_pan_ylim0[1] + d0[1] - d1[1])
                self._ax.set_xlim(new_xl)
                self._ax.set_ylim(new_yl)
                self._saved_xlim = new_xl   # persist pan position
                self._saved_ylim = new_yl
                self._user_has_zoomed = True
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

        # ---- Rubber band / snap / ref-line / polygon / construct preview ----
        if (self._picking_active or self._fault_picking
                or self._snap_point is not None
                or self._ref_line_tool == "a_ref"
                or self._polygon_drawing
                or self._cst_state != "idle"):
            self._blit_overlays()

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

    def _on_sv_resize(self, _event) -> None:
        self.request_render()

    def _on_scroll_sv(self, event) -> None:
        if event.inaxes is not self._ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        factor = 1.3 if (getattr(event, "step", 0) > 0 or event.button == "up") else 1.0 / 1.3

        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata

        x_range = xlim[1] - xlim[0]
        y_range = ylim[0] - ylim[1]   # positive (y inverted: ylim[0] > ylim[1])

        new_x_range = x_range / factor
        new_y_range = y_range / factor

        x_frac = (xdata - xlim[0]) / x_range if x_range != 0 else 0.5
        y_frac = (ylim[0] - ydata) / y_range if y_range != 0 else 0.5

        new_xlim = (xdata - new_x_range * x_frac,
                    xdata + new_x_range * (1 - x_frac))
        new_ylim = (ydata + new_y_range * y_frac,
                    ydata - new_y_range * (1 - y_frac))
        self._pending_xlim = new_xlim
        self._pending_ylim = new_ylim
        self._saved_xlim   = new_xlim   # persist immediately so rubber-band renders don't reset
        self._saved_ylim   = new_ylim
        self._user_has_zoomed = True
        self.request_render()   # debounced — fires 50 ms after last scroll event

    def _on_sv_key(self, event) -> None:
        if event.key == "escape":
            if self._picking_active or self._fault_picking:
                self._end_pick_sequence()
                return
            if self._polygon_drawing:
                self._polygon_vertices.clear()
                self.set_polygon_drawing(False)
                return
            elif self._cst_state != "idle":
                self._cst_state = "idle"
                self._cst_source = None
                self.render()
                return
            elif self._pick_drag:
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
        self._ax_limits_set = False   # new section gets default limits
        if section is not None and self._ve_lock_btn.isChecked():
            locked_ve = self._ve_spin.value()
            if abs(getattr(section, "vertical_exaggeration", 1.0) - locked_ve) > 0.001:
                idx = self._state.project.sections.index(section)
                sec_copy = copy.deepcopy(section)
                sec_copy.vertical_exaggeration = locked_ve
                self._state.update_section(idx, sec_copy)
                return  # update_section triggers re-render
        self.request_render()

    def _on_data_changed(self, *_args) -> None:
        self.request_render()

    def _on_seismic_refs_changed(self, *_args) -> None:
        self._seismic_cache.clear()
        self._seismic_proj_cache.clear()

        self.request_render()

    def _on_active_section_changed_seismic_invalidate(self, *_args) -> None:
        """Reset zoom state when section changes — new section gets default limits."""
        self._seismic_proj_cache.clear()
        self._user_has_zoomed = False
        self._pending_xlim = None
        self._pending_ylim = None
        self._saved_xlim   = None   # clear persisted zoom — new section, fresh view
        self._saved_ylim   = None
        self._ax_limits_set = False

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

    # ------------------------------------------------------------------
    # Construct tools (extend / trim / parallel) — Phase 4
    # ------------------------------------------------------------------

    def _handle_construct_click(self, x: float, y: float) -> None:
        """2-step state machine for extend/trim/parallel tools."""
        tool = self._construct_tool
        if tool is None:
            return

        if self._cst_state == "idle":
            self._cst_first_click(tool, x, y)
        elif self._cst_state == "source_selected":
            self._cst_second_click(tool, x, y)
            self._cst_state = "idle"
            self._cst_source = None
            self._state.set_active_tool("select")
        self.render()

    def _cst_first_click(self, tool: str, x: float, y: float) -> None:
        """Select the source object for construct operation."""
        section = self._state.active_section
        if section is None:
            return

        if tool == "extend":
            hit = self._find_nearest_endpoint_px(x, y)
            if hit is None:
                self._flash_hint("Click near the endpoint of a line to extend")
                return
            cat, oi, endpoint = hit
            self._cst_source = {"cat": cat, "idx": oi, "endpoint": endpoint}
            self._cst_state = "source_selected"
            self._flash_hint("Now click the target line to extend to")

        elif tool == "trim":
            hit_line = self._find_nearest_pick_line(x, y)
            if hit_line is None:
                self._flash_hint("Click on a line to trim")
                return
            cat, oi = hit_line
            self._cst_source = {"cat": cat, "idx": oi, "click_x": x, "click_y": y}
            self._cst_state = "source_selected"
            self._flash_hint("Now click the cutting line")

        elif tool == "parallel":
            hit_line = self._find_nearest_pick_line(x, y)
            if hit_line is None:
                self._flash_hint("Click on a line to draw parallel to")
                return
            cat, oi = hit_line
            self._cst_source = {"cat": cat, "idx": oi}
            self._cst_state = "source_selected"
            self._flash_hint("Click to place the parallel line")

    def _cst_second_click(self, tool: str, x: float, y: float) -> None:
        """Apply the construct operation."""
        import copy as _cp
        section = self._state.active_section
        if section is None or self._cst_source is None:
            return
        sec_name = section.name

        if tool == "extend":
            src = self._cst_source
            cat, oi = src["cat"], src["idx"]
            endpoint = src["endpoint"]
            picks = (self._state.project.horizon_picks if cat == "Horizons"
                     else self._state.project.fault_picks)
            if oi >= len(picks):
                return
            hp = picks[oi]
            sec_idxs = hp.section_indices(sec_name)
            if len(sec_idxs) < 2:
                return
            d_sec = hp._distances[sec_idxs]
            z_sec = hp._depths[sec_idxs]

            if endpoint == "start":
                slope = (z_sec[1] - z_sec[0]) / max(d_sec[1] - d_sec[0], 1e-9)
                origin_d, origin_z = float(d_sec[0]), float(z_sec[0])
            else:
                slope = (z_sec[-1] - z_sec[-2]) / max(d_sec[-1] - d_sec[-2], 1e-9)
                origin_d, origin_z = float(d_sec[-1]), float(z_sec[-1])

            intersect = self._ray_line_intersection(
                origin_d, origin_z, slope, x, y, sec_name)
            if intersect is None:
                self._flash_hint("Lines do not intersect — extension cancelled")
                return
            ix, iy = intersect
            hp2 = _cp.deepcopy(hp)
            hp2.insert_pick(ix, iy, sec_name)
            if cat == "Horizons":
                self._state.update_horizon_pick(oi, hp2)
            else:
                self._state.update_fault_pick(oi, hp2)

        elif tool == "trim":
            src = self._cst_source
            cat, oi = src["cat"], src["idx"]
            click_x = src["click_x"]
            picks = (self._state.project.horizon_picks if cat == "Horizons"
                     else self._state.project.fault_picks)
            if oi >= len(picks):
                return
            hp = picks[oi]
            sec_idxs = hp.section_indices(sec_name)
            d_sec = hp._distances[sec_idxs]
            z_sec = hp._depths[sec_idxs]

            cut_intersect = self._find_line_intersect_at(x, y, d_sec, z_sec, sec_name)
            if cut_intersect is None:
                self._flash_hint("No intersection found — trim cancelled")
                return
            ix, _ = cut_intersect

            keep_left = click_x <= ix
            keep_mask = (d_sec <= ix) if keep_left else (d_sec >= ix)
            d_keep = d_sec[keep_mask]
            z_keep = z_sec[keep_mask]

            hp3 = _cp.deepcopy(picks[oi])
            idxs_to_remove = sorted(sec_idxs, reverse=True)
            for idx_r in idxs_to_remove:
                if hp3.n_picks > 1:
                    for attr in ("_distances", "_depths", "_section_names",
                                 "_confidence", "_quality", "_note"):
                        arr = getattr(hp3, attr)
                        setattr(hp3, attr, np.delete(arr, idx_r))
            for d, z in zip(d_keep, z_keep):
                hp3.insert_pick(float(d), float(z), sec_name)
            z_intersect_approx = float(np.interp(ix, d_sec, z_sec))
            hp3.insert_pick(ix, z_intersect_approx, sec_name)

            if cat == "Horizons":
                self._state.update_horizon_pick(oi, hp3)
            else:
                self._state.update_fault_pick(oi, hp3)

        elif tool == "parallel":
            src = self._cst_source
            cat, oi = src["cat"], src["idx"]
            picks = (self._state.project.horizon_picks if cat == "Horizons"
                     else self._state.project.fault_picks)
            if oi >= len(picks):
                return
            hp = picks[oi]
            sec_idxs = hp.section_indices(sec_name)
            if len(sec_idxs) < 2:
                return
            d_sec = hp._distances[sec_idxs]
            z_sec = hp._depths[sec_idxs]

            z_at_x = float(np.interp(x, d_sec, z_sec,
                                     left=float(z_sec[0]), right=float(z_sec[-1])))
            offset = y - z_at_x

            new_z = z_sec + offset
            hp_new = HorizonPick(
                distances=d_sec.copy(),
                depths=new_z.copy(),
                name=f"{hp.name} (parallel)",
                color=hp.color,
                section_names=[sec_name] * len(d_sec),
            )
            self._state.add_horizon_pick(hp_new)

    def _find_nearest_endpoint_px(self, x: float, y: float) -> tuple | None:
        """Find the nearest START or END pick point within _PICK_HIT_PX * 1.5 pixels."""
        section = self._state.active_section
        if section is None:
            return None
        sec_name = section.name
        threshold = _PICK_HIT_PX * 1.5
        ex, ey = self._to_screen_px_sv(x, y)
        best, best_dist = None, float("inf")

        for cat, picks in [("Horizons", self._state.project.horizon_picks),
                            ("Faults", self._state.project.fault_picks)]:
            for oi, hp in enumerate(picks):
                sec_idxs = hp.section_indices(sec_name)
                if len(sec_idxs) < 1:
                    continue
                for endpoint, fi in [("start", sec_idxs[0]), ("end", sec_idxs[-1])]:
                    d = float(hp._distances[fi])
                    z = float(hp._depths[fi])
                    nx, ny = self._to_screen_px_sv(d, z)
                    dist = math.hypot(ex - nx, ey - ny)
                    if dist <= threshold and dist < best_dist:
                        best_dist = dist
                        best = (cat, oi, endpoint)
        return best

    def _ray_line_intersection(self, ox: float, oz: float, slope: float,
                               target_x: float, target_y: float,
                               sec_name: str) -> tuple | None:
        """Find where the ray from (ox,oz) with slope intersects any pick line near target."""
        section = self._state.active_section
        if section is None:
            return None
        total = section.total_length()

        hit_line = self._find_nearest_pick_line(target_x, target_y)
        if hit_line is None:
            return None

        cat, oi = hit_line
        picks = (self._state.project.horizon_picks if cat == "Horizons"
                 else self._state.project.fault_picks)
        hp = picks[oi]
        sec_idxs = hp.section_indices(sec_name)
        if len(sec_idxs) < 2:
            return None
        d_sec = hp._distances[sec_idxs]
        z_sec = hp._depths[sec_idxs]

        for i in range(len(d_sec) - 1):
            d0, z0 = float(d_sec[i]), float(z_sec[i])
            d1, z1 = float(d_sec[i+1]), float(z_sec[i+1])
            dd = d1 - d0; dz = z1 - z0
            denom = slope * dd - dz
            if abs(denom) < 1e-9:
                continue
            s = (z0 - oz - slope * d0 + slope * ox) / denom
            if not (0.0 <= s <= 1.0):
                continue
            t = d0 - ox + s * dd
            ix = ox + t
            iz = oz + slope * t
            if 0 <= ix <= total:
                return (ix, iz)
        return None

    def _find_line_intersect_at(self, x: float, y: float,
                                d_sec: np.ndarray, z_sec: np.ndarray,
                                sec_name: str) -> tuple | None:
        """Find intersection of any pick line near (x,y) with the source line."""
        hit = self._find_nearest_pick_line(x, y)
        if hit is None:
            return None
        cat2, oi2 = hit
        picks2 = (self._state.project.horizon_picks if cat2 == "Horizons"
                  else self._state.project.fault_picks)
        hp2 = picks2[oi2]
        sec_idxs2 = hp2.section_indices(sec_name)
        if len(sec_idxs2) < 2:
            return None
        d2 = hp2._distances[sec_idxs2]
        z2 = hp2._depths[sec_idxs2]

        for i in range(len(d_sec) - 1):
            for j in range(len(d2) - 1):
                p = _seg_intersect(
                    d_sec[i], z_sec[i], d_sec[i+1], z_sec[i+1],
                    d2[j], z2[j], d2[j+1], z2[j+1],
                )
                if p is not None:
                    return p
        return None

    def _render_construct_preview(self) -> None:
        if self._cst_state == "idle" or self._cst_source is None:
            return
        tool = self._construct_tool
        section = self._state.active_section
        if section is None or tool not in ("extend", "trim", "parallel"):
            return
        sec_name = section.name
        src = self._cst_source
        cat, oi = src.get("cat"), src.get("idx")
        if cat is None or oi is None:
            return
        picks = (self._state.project.horizon_picks if cat == "Horizons"
                 else self._state.project.fault_picks)
        if oi >= len(picks):
            return
        hp = picks[oi]
        sec_idxs = hp.section_indices(sec_name)
        if len(sec_idxs) < 2:
            return
        d_sec = hp._distances[sec_idxs]
        z_sec = hp._depths[sec_idxs]

        self._overlay_artists.extend(
            self._ax.plot(d_sec, z_sec, color=hp.color, linewidth=3.0, alpha=0.4, zorder=12))

        if tool == "extend" and self._cursor_data is not None:
            endpoint = src.get("endpoint", "end")
            cx, cy = self._cursor_data
            if endpoint == "start":
                ox, oz = float(d_sec[0]), float(z_sec[0])
            else:
                ox, oz = float(d_sec[-1]), float(z_sec[-1])
            self._overlay_artists.extend(
                self._ax.plot([ox, cx], [oz, cy], "--", color=hp.color,
                              lw=1.5, alpha=0.7, zorder=12))

        elif tool == "parallel" and self._cursor_data is not None:
            cx, cy = self._cursor_data
            z_at_cx = float(np.interp(cx, d_sec, z_sec,
                                      left=float(z_sec[0]), right=float(z_sec[-1])))
            offset = cy - z_at_cx
            self._overlay_artists.extend(
                self._ax.plot(d_sec, z_sec + offset, "--", color=hp.color,
                              lw=1.5, alpha=0.6, zorder=12))

    def _flash_hint(self, msg: str) -> None:
        """Show a brief status hint via the parent window's status bar."""
        try:
            w = self.window()
            if hasattr(w, '_status_label'):
                w._status_label.setText(msg)
        except Exception:
            pass

    def _show_section_coords(self, dist: float, depth: float) -> None:
        """Show Dist/Depth + back-calculated geographic coords in the status bar."""
        try:
            w = self.window()
            if not hasattr(w, '_hint_label'):
                return
            section = self._state.active_section
            if section is None:
                return
            # Back-calculate map X,Y from distance along section
            x, y = section.section_to_map(dist)
            units = getattr(section, "depth_units", "m")
            if units == "m+ft":
                depth_str = f"{depth:.0f} m  ({depth * 3.28084:.0f} ft)"
                dist_str  = f"{dist:.0f} m  ({dist * 3.28084:.0f} ft)"
            else:
                depth_str = f"{depth:.0f} {units}"
                dist_str  = f"{dist:.0f} m"
            w._hint_label.setText(
                f"Dist: {dist_str}   Depth: {depth_str}   |   E: {x:,.0f}  N: {y:,.0f}"
            )
        except Exception:
            pass

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

def _seg_intersect(ax, ay, bx, by, cx, cy, dx, dy):
    """Return intersection point of segment AB and CD, or None."""
    dab_x, dab_y = bx - ax, by - ay
    dcd_x, dcd_y = dx - cx, dy - cy
    denom = dab_x * dcd_y - dab_y * dcd_x
    if abs(denom) < 1e-9:
        return None
    t = ((cx - ax) * dcd_y - (cy - ay) * dcd_x) / denom
    u = ((cx - ax) * dab_y - (cy - ay) * dab_x) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return (ax + t * dab_x, ay + t * dab_y)
    return None


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

def _nice_interval(approx: float) -> float:
    """Round *approx* up to a human-friendly interval (1, 2, 5, 10, 20, 50, …)."""
    import math
    if approx <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(approx))
    for mult in (1, 2, 5, 10):
        candidate = mag * mult
        if candidate >= approx:
            return candidate
    return mag * 10


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
