"""Thermal modeling dialog.

Accessible via Tools → Thermal Modeling.  Runs steady-state, transient,
or inverse thermal models at a chosen position along the active section
and displays the results in a matplotlib canvas.

Modes
-----
Steady-state
    Computes a depth–temperature profile from the current stratigraphic
    column and a constant basal heat flow.
Transient
    Integrates the 1-D heat equation through the burial history derived
    from the restoration sequence.
Inverse
    Monte Carlo search for t-T paths consistent with thermochronometric
    constraints entered by the user.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class ThermalModelingDialog(QDialog):
    """Dialog for running thermal modeling at a section position.

    Parameters
    ----------
    app_state:
        The global :class:`~section_tool.app_state.AppState`.
    section:
        The active :class:`~section_tool.core.section.Section`.
    parent:
        Qt parent widget.
    """

    def __init__(self, app_state, section, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Thermal Modeling")
        self.setMinimumSize(900, 580)

        self._state   = app_state
        self._section = section
        self._manual_burial = None    # user-entered burial fallback (Step 2)

        layout = QHBoxLayout(self)

        # ── Left: controls ────────────────────────────────────────────
        ctrl_panel = QWidget()
        ctrl_panel.setFixedWidth(260)
        ctrl_layout = QVBoxLayout(ctrl_panel)
        ctrl_layout.setContentsMargins(0, 0, 8, 0)

        # Position
        pos_box = QGroupBox("Section position")
        pos_form = QFormLayout(pos_box)
        self._dist_spin = QDoubleSpinBox()
        total = section.total_length() if hasattr(section, "total_length") else 10000.0
        self._dist_spin.setRange(0.0, total)
        self._dist_spin.setValue(total / 2.0)
        self._dist_spin.setSuffix(" m")
        pos_form.addRow("Distance:", self._dist_spin)
        ctrl_layout.addWidget(pos_box)

        # Mode
        mode_box = QGroupBox("Mode")
        mode_form = QFormLayout(mode_box)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Steady-state", "Transient", "Inverse (MC)"])
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_form.addRow("Model:", self._mode_combo)
        ctrl_layout.addWidget(mode_box)

        # Thermal parameters
        param_box = QGroupBox("Parameters")
        param_form = QFormLayout(param_box)

        self._surf_temp = QDoubleSpinBox()
        self._surf_temp.setRange(-20.0, 50.0)
        self._surf_temp.setValue(10.0)
        self._surf_temp.setSuffix(" °C")
        param_form.addRow("Surface T:", self._surf_temp)

        self._heat_flow = QDoubleSpinBox()
        self._heat_flow.setRange(20.0, 300.0)
        self._heat_flow.setValue(60.0)
        self._heat_flow.setSuffix(" mW/m²")
        param_form.addRow("Basal heat flow:", self._heat_flow)

        self._conductivity = QDoubleSpinBox()
        self._conductivity.setRange(0.1, 10.0)
        self._conductivity.setValue(2.5)
        self._conductivity.setDecimals(2)
        self._conductivity.setSuffix(" W/m·K")
        param_form.addRow("Conductivity:", self._conductivity)

        ctrl_layout.addWidget(param_box)

        # Inverse-mode specific
        self._inv_box = QGroupBox("Monte Carlo options")
        inv_form = QFormLayout(self._inv_box)

        self._n_paths = QSpinBox()
        self._n_paths.setRange(100, 100_000)
        self._n_paths.setValue(5000)
        self._n_paths.setSingleStep(1000)
        inv_form.addRow("Paths:", self._n_paths)

        self._n_inflect = QSpinBox()
        self._n_inflect.setRange(2, 10)
        self._n_inflect.setValue(5)
        inv_form.addRow("Inflections:", self._n_inflect)

        self._threshold = QDoubleSpinBox()
        self._threshold.setRange(0.1, 20.0)
        self._threshold.setValue(2.0)
        self._threshold.setDecimals(1)
        inv_form.addRow("χ²/obs threshold:", self._threshold)

        ctrl_layout.addWidget(self._inv_box)
        self._inv_box.setVisible(False)

        # Burial history (restoration↔thermal seam)
        burial_box = QGroupBox("Burial history")
        burial_form = QFormLayout(burial_box)
        self._horizon_combo = QComboBox()
        self._populate_horizon_combo()
        burial_form.addRow("Track horizon:", self._horizon_combo)
        self._burial_btn = QPushButton("Burial history…")
        self._burial_btn.setToolTip("Derive the burial curve at the section position "
                                    "from the restoration sequence (or enter manually).")
        self._burial_btn.clicked.connect(self._open_burial)
        burial_form.addRow(self._burial_btn)
        ctrl_layout.addWidget(burial_box)

        # Observed data
        self._meas_btn = QPushButton("Measurements…")
        self._meas_btn.setToolTip("Enter or import observed thermal / thermochronometric "
                                  "data (Ro, AFT/AHe/ZHe ages, BHT) on a well.")
        self._meas_btn.clicked.connect(self._open_measurements)
        ctrl_layout.addWidget(self._meas_btn)

        # Run button
        self._run_btn = QPushButton("Run Model")
        self._run_btn.clicked.connect(self._run_model)
        ctrl_layout.addWidget(self._run_btn)
        ctrl_layout.addStretch()

        layout.addWidget(ctrl_panel)

        # ── Right: plot canvas ────────────────────────────────────────
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._figure = Figure(figsize=(7, 5), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        layout.addWidget(self._canvas, stretch=1)

        self._ax = self._figure.add_subplot(111)
        self._ax.set_xlabel("Temperature (°C)")
        self._ax.set_ylabel("Depth (m)")
        self._ax.invert_yaxis()
        self._ax.set_title("Thermal model — press Run to compute")
        self._canvas.draw()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_mode_changed(self, idx: int) -> None:
        self._inv_box.setVisible(idx == 2)

    def _open_measurements(self) -> None:
        """Open the per-well measurements editor (observed data)."""
        from section_tool.views.measurements_dialog import MeasurementsDialog
        MeasurementsDialog(self._state, parent=self).exec()

    def _populate_horizon_combo(self) -> None:
        self._horizon_combo.clear()
        sec = self._section.name
        for hp in self._state.project.horizon_picks:
            try:
                if hp.n_picks_for_section(sec) >= 1:
                    self._horizon_combo.addItem(hp.name or "(unnamed)", hp.uuid)
            except Exception:
                pass

    def _current_burial_history(self):
        """Burial curve at the section position: from the restoration sequence when
        available (Step 2 seam), else the user-entered fallback. Never a proxy."""
        from section_tool.core.burial import burial_history_from_restoration
        snap = getattr(self._state, "restoration_snapshot", None)
        seq = self._state.restoration_sequence
        huid = self._horizon_combo.currentData()
        x = self._dist_spin.value()
        if snap is not None and seq.events and huid:
            bh = burial_history_from_restoration(
                seq, huid, x, snapshot=snap, section_name=self._section.name)
            if len(bh.points) >= 2:
                return bh
        return getattr(self, "_manual_burial", None)

    def _open_burial(self) -> None:
        from section_tool.views.burial_history_dialog import BurialHistoryDialog
        dlg = BurialHistoryDialog(self._current_burial_history(), parent=self)
        if dlg.exec() == QDialog.Accepted and dlg.result is not None:
            if dlg.result.source == "user-specified":
                self._manual_burial = dlg.result

    def _run_model(self) -> None:
        mode = self._mode_combo.currentIndex()
        try:
            if mode == 0:
                self._run_steady_state()
            elif mode == 1:
                self._run_transient()
            else:
                self._run_inverse()
        except Exception as exc:
            QMessageBox.warning(self, "Thermal Model Error", str(exc))

    def _run_steady_state(self) -> None:
        """Build a simple 1D column from well formation tops and run steady-state."""
        from section_tool.core.thermal import steady_state_geotherm

        layers = self._build_layers()
        profile = steady_state_geotherm(
            layers,
            surface_temp_C=self._surf_temp.value(),
            basal_heat_flow_mW=self._heat_flow.value(),
        )
        depths = [p[0] for p in profile]
        temps  = [p[1] for p in profile]

        self._ax.clear()
        self._ax.plot(temps, depths, "r-", lw=2, label="Steady-state geotherm")
        self._ax.invert_yaxis()
        self._ax.set_xlabel("Temperature (°C)")
        self._ax.set_ylabel("Depth (m)")
        self._ax.set_title("Steady-state geotherm")
        self._ax.legend()
        self._canvas.draw()

    def _run_transient(self) -> None:
        """Run the transient heat solver over the REAL burial history.

        Burial comes from the restoration sequence (or a user-specified curve) — see
        :meth:`_current_burial_history`. The node grid is scaled at each age by the
        tracked horizon's burial fraction, so the column deepens through time exactly
        as the burial curve says (no fabricated proxy)."""
        from section_tool.core.thermal import transient_1d_heat
        import numpy as np

        burial = self._current_burial_history()
        if burial is None or len(burial.points) < 2:
            QMessageBox.information(
                self, "Burial history",
                "Define a burial history first (Burial history…) — from the "
                "restoration sequence or entered manually.")
            return

        layers = self._build_layers()
        node_depths = np.array([lyr["z_top"] for lyr in layers] +
                               [layers[-1]["z_bottom"]])
        T0 = (self._surf_temp.value() +
              node_depths * self._heat_flow.value() * 1e-3 / self._conductivity.value())

        ages = burial.ages_ma                          # oldest first (Ma)
        depths_curve = burial.depths_m
        present_depth = depths_curve[-1] if depths_curve[-1] > 0 else max(
            float(depths_curve.max()), 1.0)
        scale = depths_curve / present_depth           # burial fraction per age
        burial_grid = np.outer(scale, node_depths)     # (n_times, N)

        T_hist = transient_1d_heat(
            node_depths, T0, burial_grid, ages,
            thermal_conductivity=self._conductivity.value(),
            surface_temp_C=self._surf_temp.value(),
            basal_heat_flow_mW=self._heat_flow.value(),
        )

        self._ax.clear()
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("viridis")
        n = len(ages)
        for j, (T_row, age) in enumerate(zip(T_hist, ages)):
            col = cmap(j / max(n - 1, 1))
            label = "Present" if age == 0 else f"{age:g} Ma"
            self._ax.plot(T_row, node_depths, color=col, lw=2, label=label)
        self._ax.invert_yaxis()
        self._ax.set_xlabel("Temperature (°C)")
        self._ax.set_ylabel("Depth (m)")
        self._ax.set_title(f"Transient thermal history — {burial.source}")
        self._ax.legend(fontsize=7)
        self._canvas.draw()

    def _run_inverse(self) -> None:
        """Run Monte Carlo inverse search."""
        from section_tool.core.thermal_inverse import good_paths_envelope, monte_carlo_search
        from section_tool.core.thermal import aft_age
        import numpy as np

        # Build a simple synthetic observation from the current geotherm
        # (in a real workflow the user would enter measured ages)
        synthetic_aft = aft_age(
            np.array([20.0, 120.0, 20.0]),
            np.array([100.0, 50.0, 0.0]),
        )

        observations = [
            {"type": "AFT", "value": synthetic_aft, "uncertainty": max(synthetic_aft * 0.15, 5.0)},
        ]

        paths = monte_carlo_search(
            observations,
            time_bounds_ma=(100.0, 0.0),
            temp_bounds_C=(10.0, 180.0),
            n_paths=self._n_paths.value(),
            n_inflection_points=self._n_inflect.value(),
            acceptance_threshold=self._threshold.value(),
        )

        self._ax.clear()
        time_grid = np.linspace(100.0, 0.0, 100)

        if paths:
            t_min, t_max, t_mean = good_paths_envelope(paths, time_grid)
            self._ax.fill_between(time_grid, t_min, t_max,
                                  alpha=0.3, color="#4488cc",
                                  label=f"P5–P95 ({len(paths)} paths)")
            self._ax.plot(time_grid, t_mean, "b-", lw=2, label="Mean")
        else:
            self._ax.text(50, 100, "No acceptable paths found.\nTry loosening the threshold.",
                          ha="center", va="center", fontsize=11, color="red")

        self._ax.invert_xaxis()
        self._ax.set_xlabel("Time (Ma)")
        self._ax.set_ylabel("Temperature (°C)")
        self._ax.set_title(f"Inverse model — {len(paths)} accepted paths")
        if paths:
            self._ax.legend()
        self._canvas.draw()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_layers(self) -> list[dict]:
        """Build a simple formation column from the nearest well to *dist_spin*."""
        from section_tool.core.thermal import effective_conductivity_column

        dist = self._dist_spin.value()
        project = self._state.project
        k = self._conductivity.value()

        # Find nearest well
        best_well = None
        best_d = float("inf")
        for well in project.wells:
            dx = well.x - self._section.section_to_map(dist)[0]
            dy = well.y - self._section.section_to_map(dist)[1]
            dd = (dx ** 2 + dy ** 2) ** 0.5
            if dd < best_d:
                best_d = dd
                best_well = well

        layers: list[dict] = []
        if best_well and hasattr(best_well, "formation_tops"):
            tops = sorted(best_well.formation_tops.items(), key=lambda t: t[1])
            for i, (name, md) in enumerate(tops[:-1]):
                layers.append({
                    "z_top":               float(md),
                    "z_bottom":            float(tops[i + 1][1]),
                    "thermal_conductivity": k,
                    "heat_production":      1.0,
                })

        # Fallback: single 5000 m layer
        if not layers:
            layers = [{
                "z_top":               0.0,
                "z_bottom":            5000.0,
                "thermal_conductivity": k,
                "heat_production":      1.0,
            }]
        return layers
