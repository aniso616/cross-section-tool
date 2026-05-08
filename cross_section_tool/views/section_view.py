from __future__ import annotations

from typing import Literal

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from cross_section_tool.app_state import AppState
from cross_section_tool.core.section import Section
from cross_section_tool.io.project import SeismicRef
from cross_section_tool.io.segy import SeismicDataset


class SectionView(QWidget):
    """Matplotlib-based 2D cross-section display panel.

    Renders seismic data, horizon picks, surface intersections, and well
    tracks for the section currently active in :class:`AppState`.  All
    content is redrawn when the relevant :class:`AppState` signals fire.

    Parameters
    ----------
    state:
        Central application state.  The view connects to its signals on
        construction and never holds a reference to individual data
        objects independently.

    Signals
    -------
    horizon_pick_requested(float, float)
        Emitted on a left-click when picking mode is active.
        Arguments are ``(distance_along_section, depth_or_time)``.
    """

    horizon_pick_requested = Signal(float, float)

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._seismic_cache: dict[str, SeismicDataset] = {}
        self._picking_active: bool = False
        self._display_mode: Literal["variable_density", "wiggle"] = "variable_density"
        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self._fig = Figure(figsize=(10, 6), tight_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        self._canvas.mpl_connect("button_press_event", self._on_canvas_click)

    def _connect_signals(self) -> None:
        s = self._state
        s.active_section_changed.connect(self._on_active_section_changed)
        s.project_changed.connect(self.render)
        s.horizon_pick_added.connect(self._on_picks_changed)
        s.horizon_pick_removed.connect(self._on_picks_changed)
        s.horizon_pick_modified.connect(self._on_picks_changed)
        s.well_added.connect(self._on_wells_changed)
        s.well_removed.connect(self._on_wells_changed)
        s.well_modified.connect(self._on_wells_changed)
        s.surface_added.connect(self._on_surfaces_changed)
        s.surface_removed.connect(self._on_surfaces_changed)
        s.surface_modified.connect(self._on_surfaces_changed)
        s.seismic_ref_added.connect(self._on_seismic_refs_changed)
        s.seismic_ref_removed.connect(self._on_seismic_refs_changed)

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

    def set_display_mode(
        self, mode: Literal["variable_density", "wiggle"]
    ) -> None:
        """Switch the seismic display mode and re-render."""
        self._display_mode = mode
        self.render()

    def set_picking_active(self, active: bool) -> None:
        """Enable or disable interactive horizon pick mode."""
        self._picking_active = active

    def clear_seismic_cache(self) -> None:
        """Evict all cached :class:`SeismicDataset` objects."""
        self._seismic_cache.clear()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, *_args) -> None:
        """Full redraw of the active section.

        Clears the axes and redraws seismic, surfaces, picks, and wells.
        Safe to call when no section is active (produces a blank canvas).
        """
        self._ax.clear()
        section = self._state.active_section
        if section is None:
            self._canvas.draw_idle()
            return

        self._setup_axes(section)
        self._render_seismic(section)
        self._render_surfaces(section)
        self._render_horizon_picks(section)
        self._render_wells(section)
        self._canvas.draw_idle()

    def _setup_axes(self, section: Section) -> None:
        total = section.total_length()
        self._ax.set_xlim(0.0, total)
        self._ax.set_xlabel("Distance along section (m)")

        if section.depth_domain == "twt":
            self._ax.set_ylabel("Two-way time (ms)")
        else:
            self._ax.set_ylabel(f"Depth ({section.depth_units})")

        title = section.name if section.name else "Section View"
        self._ax.set_title(title)

    def _render_seismic(self, section: Section) -> None:
        for ref in self._state.project.seismic_refs:
            dataset = self._get_or_load_seismic(ref)
            if dataset is None or dataset.n_traces == 0:
                continue
            distances, data, _ = dataset.traces_sorted_by_section(section)
            if len(distances) < 2:
                continue
            vmax = float(np.percentile(np.abs(data), 95)) or 1.0

            if self._display_mode == "variable_density":
                self._ax.imshow(
                    data.T,
                    aspect="auto",
                    extent=[
                        distances[0], distances[-1],
                        dataset.samples[-1], dataset.samples[0],
                    ],
                    origin="upper",
                    cmap="seismic",
                    vmin=-vmax,
                    vmax=vmax,
                    interpolation="bilinear",
                )
                # Restore depth-down orientation after imshow resets limits
                self._ax.set_ylim(dataset.samples[-1], dataset.samples[0])
            else:
                self._render_seismic_wiggle(distances, data, dataset.samples)

    def _render_seismic_wiggle(
        self,
        distances: np.ndarray,
        data: np.ndarray,
        samples: np.ndarray,
    ) -> None:
        """Draw variable-area wiggle traces."""
        if len(distances) < 2:
            return
        trace_spacing = (distances[-1] - distances[0]) / max(len(distances) - 1, 1)
        scale = trace_spacing * 0.8 / (np.percentile(np.abs(data), 95) or 1.0)
        for i, (dist, trace) in enumerate(zip(distances, data)):
            x = dist + trace * scale
            self._ax.plot(x, samples, "k-", linewidth=0.3)
            pos = np.where(trace > 0, trace, 0)
            self._ax.fill_betweenx(samples, dist, dist + pos * scale, color="k", alpha=0.7)
        self._ax.set_ylim(samples[-1], samples[0])

    def _render_horizon_picks(self, section: Section) -> None:
        total = section.total_length()
        sample_d = np.linspace(0.0, total, 500)
        for pick in self._state.project.horizon_picks:
            depths = pick.sample_many(sample_d)
            valid = ~np.isnan(depths)
            if not np.any(valid):
                continue
            self._ax.plot(
                sample_d[valid],
                depths[valid],
                color=pick.color,
                linewidth=1.5,
                label=pick.name or "_nolegend_",
                zorder=3,
            )

    def _render_surfaces(self, section: Section) -> None:
        for surf in self._state.project.surfaces:
            distances, z_values = surf.profile_along_section(section, n_samples=300)
            valid = ~np.isnan(z_values)
            if not np.any(valid):
                continue
            self._ax.plot(
                distances[valid],
                z_values[valid],
                color="darkorange",
                linewidth=1.5,
                linestyle="--",
                label=surf.name or "_nolegend_",
                alpha=0.85,
                zorder=3,
            )

    def _render_wells(self, section: Section) -> None:
        for well in self._state.project.wells:
            distances, tvds = well.section_track(section)
            collar_dist, _ = well.project_to_section(section)

            # Well track
            self._ax.plot(
                distances, tvds,
                color="#8B4513",
                linewidth=2.0,
                solid_capstyle="round",
                zorder=4,
            )
            # Collar annotation
            if len(tvds) > 0:
                self._ax.annotate(
                    well.name,
                    xy=(collar_dist, tvds[0]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7,
                    color="#8B4513",
                    zorder=5,
                )
            # Formation tops
            for top_name, _md in well.formation_tops.items():
                try:
                    top_dist, top_tvd = well.formation_top_in_section(top_name, section)
                except KeyError:
                    continue
                self._ax.plot(
                    top_dist, top_tvd,
                    marker="<",
                    markersize=6,
                    color="green",
                    zorder=5,
                )
                self._ax.text(
                    top_dist + section.total_length() * 0.005,
                    top_tvd,
                    top_name,
                    fontsize=6,
                    color="green",
                    va="center",
                    zorder=5,
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
    # Slots
    # ------------------------------------------------------------------

    def _on_active_section_changed(self, _section) -> None:
        self.render()

    def _on_picks_changed(self, *_args) -> None:
        self.render()

    def _on_wells_changed(self, *_args) -> None:
        self.render()

    def _on_surfaces_changed(self, *_args) -> None:
        self.render()

    def _on_seismic_refs_changed(self, *_args) -> None:
        self._seismic_cache.clear()
        self.render()

    def _on_canvas_click(self, event) -> None:
        if not self._picking_active:
            return
        if event.inaxes is not self._ax:
            return
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            self.horizon_pick_requested.emit(float(event.xdata), float(event.ydata))
