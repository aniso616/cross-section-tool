"""Thermal modeling dialog.

Accessible via Model ▸ Thermal Modeling.  Runs steady-state, transient,
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

from PySide6.QtCore import Qt, Signal
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

    # Emitted (cross-thread, queued) when the off-thread Monte Carlo inverse
    # finishes. Payload: {"paths": [...], "n_obs": int} or {"error": str}.
    inverse_finished = Signal(object)

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
        self._mode_combo.addItems(["Steady-state", "Transient", "Inverse (MC)",
                                   "Forward T–t path"])
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_form.addRow("Model:", self._mode_combo)
        ctrl_layout.addWidget(mode_box)

        # Thermal parameters
        param_box = QGroupBox("Parameters")
        param_form = QFormLayout(param_box)

        self._surf_temp = QDoubleSpinBox()
        self._surf_temp.setRange(-20.0, 50.0)
        # Default from setting: ~4 °C seafloor (marine) vs ~10 °C continental — never
        # hardcoded; the user can override. Detection is best-effort (the marine/
        # onshore setting lives in the depth-stretch knobs).
        self._surf_temp.setValue(4.0 if self._detect_marine() else 10.0)
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
        self._conductivity.setToolTip(
            "Fallback used when no formation picks are available at this position.")
        param_form.addRow("Fallback k:", self._conductivity)

        self._k_source_label = QLabel("—")
        self._k_source_label.setWordWrap(True)
        self._k_source_label.setStyleSheet("font-size:10px; color:#666;")
        param_form.addRow("k source:", self._k_source_label)

        self._k_edit_btn = QPushButton("Edit formation k…")
        self._k_edit_btn.setToolTip(
            "Override the thermal conductivity of individual formations "
            "(session only — not saved to the project).")
        self._k_edit_btn.clicked.connect(self._open_k_editor)
        self._k_edit_btn.setEnabled(False)
        param_form.addRow(self._k_edit_btn)

        self._k_overrides: dict[str, float] = {}

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

        # Predicted-vs-observed fit summary (Forward T–t mode)
        self._fit_label = QLabel("")
        self._fit_label.setWordWrap(True)
        self._fit_label.setStyleSheet("font-size:11px;")
        ctrl_layout.addWidget(self._fit_label)
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

        self._inverse_thread = None
        self.inverse_finished.connect(self._on_inverse_done)
        self._update_k_source_label()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_mode_changed(self, idx: int) -> None:
        self._inv_box.setVisible(idx == 2)

    def _open_measurements(self) -> None:
        """Open the per-well measurements editor (observed data)."""
        from section_tool.views.measurements_dialog import MeasurementsDialog
        MeasurementsDialog(self._state, parent=self).exec()

    def _detect_marine(self) -> bool:
        """Best-effort marine detection for the surface-temperature default."""
        try:
            if str(self._state.get_meta("setting", "")).lower() == "marine":
                return True
        except Exception:
            pass
        vm = getattr(self._state.project, "velocity_model", None)
        cons = getattr(vm, "construction", {}) or {}
        params = cons.get("params", {}) if isinstance(cons, dict) else {}
        return str(params.get("setting", "")).lower() == "marine"

    def _sample_measurements(self) -> list:
        """Measurements on the well nearest the section position (sample point)."""
        layers_well = self._nearest_well()
        return list(getattr(layers_well, "measurements", [])) if layers_well else []

    def _run_forward(self) -> None:
        """Forward T–t path: run the real solver on the real burial history and plot
        the sample point's temperature history (°C) vs time (Ma)."""
        from section_tool.core.thermal import forward_temperature_history
        burial = self._current_burial_history()
        if burial is None or len(burial.points) < 2:
            QMessageBox.information(
                self, "Burial history",
                "Define a burial history first (Burial history…).")
            return
        k_eff = self._column_mean_conductivity(self._build_layers())
        res = forward_temperature_history(
            burial,
            basal_heat_flow_mW=self._heat_flow.value(),
            conductivity=k_eff,
            surface_temp_C=self._surf_temp.value())

        self._ax.clear()
        # Present at LEFT, past at RIGHT (geological convention); temperature
        # increasing DOWNWARD (burial/cooling convention — cooling paths rise).
        self._ax.plot(res.ages_ma, res.temps_C, "-o", color="#cc3333", lw=2,
                      label="T–t path (sample point)")
        for m in self._sample_measurements():
            if m.measurement_type in ("bht", "dst_temp"):
                self._ax.errorbar([0.0], [m.value], yerr=[m.uncertainty or 0.0],
                                  fmt="s", color="#3366cc", capsize=3, zorder=5,
                                  label="BHT / DST (°C)")
        # ── Predicted vs observed (kinetic models) ────────────────────────
        from section_tool.core.thermochron_fit import goodness_of_fit
        t_path = list(zip(res.ages_ma.tolist(), res.temps_C.tolist()))
        fit = goodness_of_fit(t_path, self._sample_measurements())
        # Age-type observed (solid) vs predicted (dashed) on the time axis — style,
        # not colour, distinguishes them; the plot never implies false agreement.
        has_proxy = False
        for mtype in ("aft_age", "ahe_age", "zhe_age"):
            tf = fit.per_type.get(mtype)
            if tf is None:
                continue
            has_proxy = True
            for ov in tf.observed:
                self._ax.axvline(ov, color="#444", linestyle="-", lw=1.0, alpha=0.7)
            self._ax.axvline(tf.predicted, color="#444", linestyle="--", lw=1.4,
                             alpha=0.9)
        if has_proxy:
            self._ax.text(0.02, 0.02, "simplified proxy — see fit panel tooltip",
                          transform=self._ax.transAxes, fontsize=6, color="#888",
                          va="bottom", ha="left")

        self._ax.set_xlabel("Time (Ma) — present at left")
        self._ax.set_ylabel("Temperature (°C)")
        self._ax.invert_yaxis()
        self._ax.set_title(f"Forward T–t — gradient {res.gradient_C_per_km:.0f} °C/km, "
                           f"surface {res.surface_temp_C:.0f} °C")
        handles, labels = self._ax.get_legend_handles_labels()
        if handles:
            uniq = dict(zip(labels, handles))
            self._ax.legend(uniq.values(), uniq.keys(), fontsize=7)
        self._canvas.draw()
        self._update_fit_label(fit)

    def _update_fit_label(self, fit) -> None:
        """Predicted-vs-observed summary + χ² misfit, with the proxy caveat in the
        tooltip (the user sees it in context, not buried)."""
        from section_tool.core.thermal import KINETIC_MODEL_LABELS, KINETIC_MODEL_NOTES
        if not fit.per_type:
            self._fit_label.setText("No measurements at this sample point to compare.")
            self._fit_label.setToolTip("")
            return
        lines = ["<b>Predicted vs observed</b>"]
        notes = []
        for mtype, tf in fit.per_type.items():
            unit = "%Ro" if mtype == "vitrinite_ro" else "Ma"
            label = KINETIC_MODEL_LABELS.get(tf.model_key, mtype)
            lines.append(
                f"{label}: predicted {tf.predicted:.2f}, observed "
                f"{tf.mean_observed:.2f} {unit} (Δ = {tf.discrepancy_pct:+.0f}%)")
            note = KINETIC_MODEL_NOTES.get(tf.model_key)
            if note and note not in notes:
                notes.append(note)
        lines.append(f"<b>χ²/obs = {fit.reduced_chi2:.2f}</b> ({fit.n_obs} obs)")
        self._fit_label.setText("<br>".join(lines))
        self._fit_label.setToolTip("\n".join(notes))

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
        self._update_k_source_label()
        mode = self._mode_combo.currentIndex()
        try:
            if mode == 0:
                self._run_steady_state()
            elif mode == 1:
                self._run_transient()
            elif mode == 3:
                self._run_forward()
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
        k_eff = self._column_mean_conductivity(layers)
        node_depths = np.array([lyr["z_top"] for lyr in layers] +
                               [layers[-1]["z_bottom"]])
        T0 = (self._surf_temp.value() +
              node_depths * self._heat_flow.value() * 1e-3 / k_eff)

        ages = burial.ages_ma                          # oldest first (Ma)
        depths_curve = burial.depths_m
        present_depth = depths_curve[-1] if depths_curve[-1] > 0 else max(
            float(depths_curve.max()), 1.0)
        scale = depths_curve / present_depth           # burial fraction per age
        burial_grid = np.outer(scale, node_depths)     # (n_times, N)

        T_hist = transient_1d_heat(
            node_depths, T0, burial_grid, ages,
            thermal_conductivity=k_eff,
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

    def _inverse_observations(self) -> list[dict]:
        """Map the REAL measurements at the sample point to inverse observations.

        Each thermochronometric type the forward kinetics can predict (Ro, AFT,
        AHe, ZHe) becomes a χ² constraint with its 1-σ uncertainty (a typed
        default when none is recorded). A type with no measurement simply doesn't
        contribute — graceful, no fabrication. BHT/DST temperatures are not t-T
        path constraints in this objective, so they're skipped."""
        from section_tool.core.thermochron_fit import _default_uncertainty
        type_map = {"vitrinite_ro": "Ro", "aft_age": "AFT",
                    "ahe_age": "AHe", "zhe_age": "ZHe"}
        obs: list[dict] = []
        for m in self._sample_measurements():
            obs_type = type_map.get(m.measurement_type)
            if obs_type is None:
                continue
            unc = (getattr(m, "uncertainty", None)
                   or _default_uncertainty(m.measurement_type, m.value))
            obs.append({"type": obs_type, "value": float(m.value),
                        "uncertainty": float(unc)})
        return obs

    def _run_inverse(self) -> None:
        """Monte Carlo inverse on the REAL measurements at the sample point.

        The χ² objective is built from the observed Ro / AFT / AHe / ZHe (each
        with its uncertainty) — no fabricated observation. The search runs OFF
        the UI thread (same pattern as the DEM fetch); the result is delivered
        back to the main thread via :attr:`inverse_finished`."""
        import threading

        observations = self._inverse_observations()
        if not observations:
            QMessageBox.information(
                self, "Inverse model",
                "No thermochronometric measurements at this sample point.\n\n"
                "Enter observed Ro, AFT, AHe or ZHe data (Measurements…) on the "
                "nearest well, then run the inverse.")
            return

        n_paths   = self._n_paths.value()
        n_inflect = self._n_inflect.value()
        threshold = self._threshold.value()

        self._ax.clear()
        self._ax.text(0.5, 0.5, f"Searching {n_paths} t–T paths…",
                      transform=self._ax.transAxes, ha="center", va="center",
                      fontsize=11, color="#666")
        self._ax.set_title("Inverse t–T — running…")
        self._canvas.draw()
        self._run_btn.setEnabled(False)
        self._flash_status(
            f"Inverse: searching {n_paths} t–T paths against "
            f"{len(observations)} observation(s)…")

        def _work():
            from section_tool.core.thermal_inverse import monte_carlo_search
            try:
                paths = monte_carlo_search(
                    observations,
                    time_bounds_ma=(100.0, 0.0),
                    temp_bounds_C=(10.0, 180.0),
                    n_paths=n_paths,
                    n_inflection_points=n_inflect,
                    acceptance_threshold=threshold)
                self.inverse_finished.emit(
                    {"paths": paths, "n_obs": len(observations)})
            except Exception as exc:               # delivered to the UI thread
                self.inverse_finished.emit({"error": str(exc)})

        t = threading.Thread(target=_work, daemon=True)
        self._inverse_thread = t
        t.start()

    def _on_inverse_done(self, result: dict) -> None:
        """Main-thread slot: render the inverse result (queued from the worker)."""
        self._run_btn.setEnabled(True)
        if "error" in result:
            self._flash_status(f"Inverse failed — {result['error']}")
            QMessageBox.warning(self, "Inverse model", result["error"])
            return
        paths = result["paths"]
        self._plot_inverse_envelope(paths, result["n_obs"])
        self._flash_status(
            f"Inverse: {len(paths)} acceptable t–T path(s) found.")

    def _plot_inverse_envelope(self, paths, n_obs: int) -> None:
        """Shaded P5–P95 envelope of acceptable T(t) paths + best-fit (min-χ²) line.

        Present at LEFT (0 Ma), past at right; temperature increases downward."""
        from section_tool.core.thermal_inverse import good_paths_envelope
        import numpy as np

        self._ax.clear()
        time_grid = np.linspace(100.0, 0.0, 100)
        if paths:
            p5, p95, _mean = good_paths_envelope(paths, time_grid)
            # Honest label: this is the spread of ACCEPTED paths, not a formal
            # confidence interval (the proper statement needs full MCMC).
            self._ax.fill_between(
                time_grid, p5, p95, alpha=0.30, color="#4488cc",
                label=f"P5–P95 range of acceptable T(t) paths (N={len(paths)})")
            best = min(paths, key=lambda p: p["chi_squared"])
            self._ax.plot(
                best["ages"], best["temps"], "-", color="#cc3333", lw=2,
                label=f"Best fit (χ²/obs = {best['chi_squared']:.2f})")
        else:
            self._ax.text(
                0.5, 0.5,
                "No acceptable paths found.\nTry loosening the χ²/obs threshold.",
                transform=self._ax.transAxes, ha="center", va="center",
                fontsize=11, color="#cc3333")

        self._ax.set_xlim(0.0, time_grid.max())        # present (0 Ma) at left
        self._ax.set_xlabel("Time (Ma) — present at left")
        self._ax.set_ylabel("Temperature (°C)")
        if not self._ax.yaxis_inverted():
            self._ax.invert_yaxis()                    # temperature increases down
        title = (f"Inverse t–T — {len(paths)} acceptable paths, "
                 f"{n_obs} observation(s)" if paths
                 else "Inverse t–T — no acceptable paths")
        self._ax.set_title(title)
        if paths:
            self._ax.legend(fontsize=8)
        self._canvas.draw()

    def _flash_status(self, msg: str) -> None:
        """Route progress to the main window's status strip, if reachable."""
        fn = getattr(self.parent(), "_flash_status", None)
        if callable(fn):
            try:
                fn(msg)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _nearest_well(self):
        """The well nearest the section position (sample point), or None."""
        dist = self._dist_spin.value()
        try:
            mx, my = self._section.section_to_map(dist)
        except Exception:
            return None
        best_well, best_d = None, float("inf")
        for well in self._state.project.wells:
            dd = ((well.x - mx) ** 2 + (well.y - my) ** 2) ** 0.5
            if dd < best_d:
                best_d, best_well = dd, well
        return best_well

    def _build_layers(self) -> list[dict]:
        """Build the formation column at the sample point with porosity-corrected k.

        For each formation interval (from the nearest well's formation_tops) the
        thermal conductivity, porosity and compaction coefficient come from the
        matching Formation in the project's stratigraphic column (or a session-level
        override via :attr:`_k_overrides`). The effective conductivity is then
        computed with :func:`~section_tool.core.thermal.effective_conductivity_column`
        (Athy porosity + geometric-mean mixing). Falls back to the uniform spinbox
        value when the well has no formation picks."""
        from section_tool.core.thermal import effective_conductivity_column

        fallback_k = self._conductivity.value()
        best_well  = self._nearest_well()

        layers: list[dict] = []
        if best_well and best_well.formation_tops:
            strat = getattr(self._state.project, "strat_column", None)
            tops  = sorted(best_well.formation_tops.items(), key=lambda t: t[1])
            for i, (name, md) in enumerate(tops[:-1]):
                fm    = strat.get_formation(name) if strat else None
                k_ovr = self._k_overrides.get(name)
                k_mat = float(k_ovr if k_ovr is not None else
                              (fm.matrix_thermal_conductivity if fm else fallback_k))
                phi0  = float(fm.porosity_surface       if fm else 0.5)
                c     = float(fm.compaction_coeff        if fm else 0.0005)
                A     = float(fm.radiogenic_heat_production if fm else 1.0)
                layers.append({
                    "name":                name,
                    "z_top":               float(md),
                    "z_bottom":            float(tops[i + 1][1]),
                    "thermal_conductivity": k_mat,
                    "porosity_surface":    phi0,
                    "compaction_coeff":    c,
                    "heat_production":     A,
                })

        if not layers:
            return [{
                "z_top":               0.0,
                "z_bottom":            5000.0,
                "thermal_conductivity": fallback_k,
                "heat_production":      1.0,
            }]

        # Porosity-weighted effective conductivity (Athy + geometric-mean mixing).
        # The effective_conductivity is written back into thermal_conductivity so
        # all three solvers (steady-state, transient, forward T–t) consume the same
        # corrected value without knowing about porosity.
        eff_layers = effective_conductivity_column(layers)
        for lyr, e in zip(layers, eff_layers):
            lyr["thermal_conductivity"] = e["effective_conductivity"]
            lyr["porosity_at_midpoint"] = e["porosity_at_midpoint"]
        return layers

    @staticmethod
    def _column_mean_conductivity(layers: list[dict]) -> float:
        """Harmonic-mean effective conductivity of the column (series resistors).

        Used to produce the scalar k required by the forward T–t and transient
        solvers, which do not yet support per-layer conductivity arrays.
        """
        total_dz = sum(lyr["z_bottom"] - lyr["z_top"] for lyr in layers)
        if total_dz <= 0 or not layers:
            return layers[0]["thermal_conductivity"] if layers else 2.5
        return total_dz / sum(
            (lyr["z_bottom"] - lyr["z_top"])
            / max(lyr["thermal_conductivity"], 1e-9)
            for lyr in layers
        )

    def _update_k_source_label(self) -> None:
        """Refresh the conductivity-source label to reflect the sample position."""
        best_well = self._nearest_well()
        if not best_well or not best_well.formation_tops:
            self._k_source_label.setText("Uniform — no formation picks")
            self._k_edit_btn.setEnabled(False)
            return
        strat = getattr(self._state.project, "strat_column", None)
        tops  = sorted(best_well.formation_tops.items(), key=lambda t: t[1])
        n_int = max(0, len(tops) - 1)
        if n_int == 0:
            self._k_source_label.setText("Uniform — no formation picks")
            self._k_edit_btn.setEnabled(False)
            return
        n_known = sum(1 for nm, _ in tops
                      if strat and strat.get_formation(nm) is not None)
        n_ovr   = sum(1 for nm, _ in tops if nm in self._k_overrides)
        parts   = [f"{n_int} layer(s)", f"{n_known} in strat column"]
        if n_ovr:
            parts.append(f"{n_ovr} overridden")
        self._k_source_label.setText(", ".join(parts))
        self._k_edit_btn.setEnabled(True)

    def _open_k_editor(self) -> None:
        """Open the per-formation conductivity editor (session override)."""
        best_well = self._nearest_well()
        if not best_well:
            return
        strat = getattr(self._state.project, "strat_column", None)
        tops  = sorted(best_well.formation_tops.items(), key=lambda t: t[1])
        rows: list[tuple] = []
        for i, (name, md) in enumerate(tops[:-1]):
            z_bot = tops[i + 1][1]
            fm    = strat.get_formation(name) if strat else None
            dflt  = fm.matrix_thermal_conductivity if fm else self._conductivity.value()
            cur   = self._k_overrides.get(name, dflt)
            rows.append((name, float(md), float(z_bot), float(cur)))
        if not rows:
            return
        dlg = _ConductivityProfileDialog(rows, parent=self)
        if dlg.exec():
            self._k_overrides.update(dlg.overrides())
            self._update_k_source_label()


# ---------------------------------------------------------------------------
# Per-formation conductivity editor (session override only)
# ---------------------------------------------------------------------------

class _ConductivityProfileDialog(QDialog):
    """Table editor for per-formation thermal conductivity.

    Values are session-only overrides and are NOT persisted to the project.
    The dialog pre-fills from Formation objects in the strat column; the user
    can change the k column and confirm with OK.
    """

    def __init__(self, rows: list[tuple], parent=None) -> None:
        # rows: list of (formation_name, z_top, z_bottom, k_current)
        super().__init__(parent)
        self.setWindowTitle("Formation conductivity — session override")
        self.setMinimumWidth(440)

        from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView

        self._rows = rows
        lay = QVBoxLayout(self)

        note = QLabel(
            "Edit the k (W/m·K) column to override per-formation conductivity. "
            "These values are session-only and are not saved to the project.")
        note.setWordWrap(True)
        note.setStyleSheet("font-size:10px; color:#888;")
        lay.addWidget(note)

        self._table = QTableWidget(len(rows), 4)
        self._table.setHorizontalHeaderLabels(
            ["Formation", "Top (m)", "Base (m)", "k  (W/m·K)"])
        for i, (name, z_top, z_bot, k) in enumerate(rows):
            for col, text in enumerate((name, f"{z_top:.1f}", f"{z_bot:.1f}")):
                it = QTableWidgetItem(text)
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self._table.setItem(i, col, it)
            self._table.setItem(i, 3, QTableWidgetItem(f"{k:.3f}"))
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        lay.addWidget(self._table)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def overrides(self) -> dict[str, float]:
        """Return ``{formation_name: k}`` for each row whose k was set."""
        out: dict[str, float] = {}
        for i, (name, *_) in enumerate(self._rows):
            it = self._table.item(i, 3)
            try:
                out[name] = float(it.text())
            except (ValueError, AttributeError):
                pass
        return out
