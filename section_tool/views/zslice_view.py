"""ZSliceView — plan view of a horizontal slice (a 'window into 3D' at z0).

A sibling of MapView that REUSES the map's canvas chrome (full-bleed transparent
figure, equal-aspect datalim 1:1, MapHUDLayer's E/N AxisRulers + corner scale bar
+ lat/long readout, world placement) but renders slice-aware content: the AOI,
the section traces in plan, and the piercing dots where each section's picks
cross this slice's elevation (via the already-built slice_crossing), plus the
slice's own horizontal observations.

Deliberately independent of MapView (no fork of map_view._render_impl) — it only
shares the small HUD widgets, so the surface map stays byte-identical. It exposes
the same interface MapHUDLayer needs: ``canvas``, ``axes``, ``cursor_map_pos``,
``_state``.
"""
from __future__ import annotations

import copy
import math

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from section_tool.app_state import AppState
from section_tool.core.geometry import slice_crossing

_SECTION_COLOR = "#94A3B8"
_AOI_COLOR     = "#44AAFF"


class ZSliceView(QWidget):
    """Plan canvas bound to a HorizontalSlice; renders the window at its z0."""

    cursor_map_pos = Signal(float, float)   # world easting/northing on hover
    view_changed   = Signal()
    draw_ended     = Signal()               # right-click/Esc ends the plan trace
    status_message = Signal(str)            # transient hints (e.g. "select a fault")

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._slice = None                  # the active HorizontalSlice
        self._drawing_fault = False         # plan fault-draw mode (z-slice only)
        self._is_rendering = False
        self._redraw = QTimer(self); self._redraw.setSingleShot(True)
        self._redraw.setInterval(50); self._redraw.timeout.connect(self.render)
        self._pan_anchor = None
        self._pan_xlim0 = self._pan_ylim0 = None
        self._setup_ui()

    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        from section_tool.style import BG_CANVAS
        self._fig = Figure(figsize=(8, 6), facecolor=BG_CANVAS)
        self._fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor(BG_CANVAS)
        self._canvas = FigureCanvasQTAgg(self._fig)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._canvas)
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._canvas.mpl_connect("scroll_event", self._on_scroll)
        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.mpl_connect("button_release_event", self._on_release)

    # MapHUDLayer interface ---------------------------------------------
    @property
    def canvas(self) -> FigureCanvasQTAgg:
        return self._canvas

    @property
    def axes(self):
        return self._ax

    # ------------------------------------------------------------------

    def set_slice(self, hslice) -> None:
        self._slice = hslice
        self.render()

    def set_fault_drawing(self, active: bool) -> None:
        """Enable/disable freehand plan fault-draw mode (only meaningful on a z-slice)."""
        self._drawing_fault = bool(active)

    def request_render(self, *_a) -> None:
        if not self._redraw.isActive():
            self._redraw.start()

    @staticmethod
    def _configure_axes(ax) -> None:
        from section_tool.style import BG_CANVAS
        ax.set_facecolor(BG_CANVAS)
        ax.figure.patch.set_facecolor(BG_CANVAS)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.xaxis.set_visible(False); ax.yaxis.set_visible(False)
        ax.set_aspect("equal", adjustable="datalim")

    def render(self, *_a) -> None:
        if self._is_rendering or self._canvas.width() < 4:
            return
        self._is_rendering = True
        try:
            self._render_impl()
        finally:
            self._is_rendering = False

    def _render_impl(self) -> None:
        self._ax.clear()
        self._configure_axes(self._ax)
        if self._slice is None:
            self._canvas.draw_idle(); return

        proj = self._state.project
        picks = list(proj.horizon_picks) + list(proj.fault_picks)
        xs_all: list[float] = []
        ys_all: list[float] = []

        # AOI extent (data to keep — drawn in plan)
        self._render_aoi(xs_all, ys_all)

        # Section traces (where each section crosses this slice) + piercing dots
        for sec in proj.sections:
            sc = slice_crossing(self._slice, sec, picks)
            if sc.locus_kind == "polyline" and sc.locus:
                tx = [p[0] for p in sc.locus]; ty = [p[1] for p in sc.locus]
                self._ax.plot(tx, ty, color=_SECTION_COLOR, lw=1.0, alpha=0.7, zorder=3)
                self._ax.annotate(f" {sec.name}", xy=(tx[0], ty[0]), fontsize=6,
                                  color=_SECTION_COLOR, alpha=0.8, zorder=4)
                xs_all += tx; ys_all += ty
            for pc in sc.piercings:
                col = getattr(pc.pick, "color", "#aaaaaa") or "#aaaaaa"
                self._ax.plot([pc.u], [pc.v], marker="o", markersize=6,
                              markerfacecolor="none", markeredgecolor=col,
                              markeredgewidth=1.4, alpha=0.85, zorder=11)
                xs_all.append(pc.u); ys_all.append(pc.v)

        # The slice's own horizontal observations (none yet — drawing is later)
        for hp in picks:
            idx = hp.indices_for_slice("horizontal", self._slice.name)
            if len(idx):
                mx = hp._map_x[idx]; my = hp._map_y[idx]
                col = getattr(hp, "color", "#aaaaaa") or "#aaaaaa"
                self._ax.plot(mx, my, color=col, lw=1.5, zorder=6)
                xs_all += list(mx); ys_all += list(my)

        self._apply_limits(xs_all, ys_all)
        self._canvas.draw_idle()
        self.view_changed.emit()

    def _render_aoi(self, xs_all, ys_all) -> None:
        aoi = getattr(self._state.project, "aoi", None)
        if aoi is None:
            return
        try:
            coords = list(aoi.polygon.exterior.coords)
            xs = [c[0] for c in coords]; ys = [c[1] for c in coords]
            self._ax.fill(xs, ys, alpha=0.06, color=_AOI_COLOR, zorder=1)
            self._ax.plot(xs, ys, color=_AOI_COLOR, lw=1.2, alpha=0.8, zorder=2)
            xs_all += xs; ys_all += ys
        except Exception:
            pass

    def _apply_limits(self, xs, ys) -> None:
        if not xs or not ys:
            self._ax.set_xlim(-500, 10500); self._ax.set_ylim(-500, 10500)
            return
        xmn, xmx, ymn, ymx = min(xs), max(xs), min(ys), max(ys)
        xpad = max((xmx - xmn) * 0.15, 500.0)
        ypad = max((ymx - ymn) * 0.15, 500.0)
        self._ax.set_xlim(xmn - xpad, xmx + xpad)
        self._ax.set_ylim(ymn - ypad, ymx + ypad)

    # ---- interaction (minimal, mirrors the map) ----------------------
    def _on_motion(self, event) -> None:
        if event.xdata is not None and event.ydata is not None:
            self.cursor_map_pos.emit(float(event.xdata), float(event.ydata))
        if self._pan_anchor is not None and event.x is not None:
            inv = self._ax.transData.inverted()
            d0 = inv.transform(self._pan_anchor); d1 = inv.transform((event.x, event.y))
            dx, dy = d0[0] - d1[0], d0[1] - d1[1]
            self._ax.set_xlim(self._pan_xlim0[0] + dx, self._pan_xlim0[1] + dx)
            self._ax.set_ylim(self._pan_ylim0[0] + dy, self._pan_ylim0[1] + dy)
            self._canvas.draw_idle()

    def _on_press(self, event) -> None:
        # Plan fault-draw mode: left-click extends the trace, right-click ends it.
        if self._drawing_fault and self._slice is not None:
            if event.button == 1 and event.xdata is not None and event.ydata is not None:
                self._add_plan_pick(float(event.xdata), float(event.ydata))
                return
            if event.button == 3:
                self.draw_ended.emit()
                return
        if event.button in (1, 2) and event.x is not None:
            self._pan_anchor = (event.x, event.y)
            self._pan_xlim0 = self._ax.get_xlim(); self._pan_ylim0 = self._ax.get_ylim()

    def _add_plan_pick(self, easting: float, northing: float) -> None:
        """Append a freehand plan point to the active fault on the current slice.

        Mirrors section picking (_add_pick_to_active_target) in plan orientation:
        world (E, N) is the source of truth (map_x/map_y), depth is fixed at
        -z0, slice_kind='horizontal', slice_ref = this slice's name. No
        construction rule is set — plan traces are freehand.
        """
        cat = self._state.active_pick_category
        idx = self._state.active_pick_index
        picks = self._state.project.fault_picks
        if cat != "Faults" or idx is None or idx >= len(picks):
            self.status_message.emit("Select or create a fault first")
            return

        slc = self._slice
        depth = -float(slc.elevation)
        ref = slc.name
        hp_before = copy.deepcopy(picks[idx])
        hp_after = copy.deepcopy(hp_before)

        # distance_along = cumulative distance along the drawn plan trace, so
        # insert_pick (which sorts by distance) keeps points in draw order and
        # the trace renders as a connected polyline, not tangled. Points on THIS
        # slice are the existing horizontal observations referencing it.
        existing = hp_after.indices_for_slice("horizontal", ref)
        if len(existing) > 0:
            last = int(existing[-1])
            seg = math.hypot(easting - float(hp_after._map_x[last]),
                             northing - float(hp_after._map_y[last]))
            dist = float(hp_after._distances[last]) + seg
        else:
            dist = 0.0

        hp_after.insert_pick(dist, depth, section_name=ref,
                             map_x=easting, map_y=northing, slice_kind="horizontal")

        def _do():  self._state.update_fault_pick(idx, copy.deepcopy(hp_after))
        def _undo(): self._state.update_fault_pick(idx, copy.deepcopy(hp_before))
        _do()
        self._state.record_command(
            f"Draw plan trace on {hp_before.name or 'fault'}", undo=_undo, redo=_do)
        self.render()

    def _on_release(self, _event) -> None:
        self._pan_anchor = None

    def _on_scroll(self, event) -> None:
        if event.inaxes is not self._ax:
            return
        f = 1.0 / 1.3 if (getattr(event, "step", 0) > 0 or event.button == "up") else 1.3
        cx = event.xdata; cy = event.ydata
        xl = self._ax.get_xlim(); yl = self._ax.get_ylim()
        relx = (cx - xl[0]) / max(xl[1] - xl[0], 1e-9)
        rely = (cy - yl[0]) / max(yl[1] - yl[0], 1e-9)
        nw = (xl[1] - xl[0]) * f; nh = (yl[1] - yl[0]) * f
        self._ax.set_xlim(cx - nw * relx, cx + nw * (1 - relx))
        self._ax.set_ylim(cy - nh * rely, cy + nh * (1 - rely))
        self._canvas.draw_idle()
