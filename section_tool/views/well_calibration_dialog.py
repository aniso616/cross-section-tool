"""WellCalibrationDialog — promote assumed velocities to well-tied (M5 UI).

Opt-in: opens only when wells exist.  Pick a well; its formation tops seed the
marker table with depths, and you supply each marker's two-way time (the
checkshot).  Compute runs the robust (Huber/IRLS) fit, promotes the touched
layers to *well-calibrated*, and shows the per-marker depth + TWT residuals (the
consistency diagnostic).  Apply installs the calibrated model.

Time is shown in ms; the controller works in SI seconds.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from section_tool.core.velocity_model import VelocityModel
from section_tool.core.well_calibration import Marker, calibrate_model, marker_residuals


class WellCalibrationDialog(QDialog):
    """Calibrate the project velocity model against a well's T-D control."""

    def __init__(self, state, on_apply=None, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._on_apply = on_apply
        self._calibrated: VelocityModel | None = None
        self.setWindowTitle("Well Calibration — Tie Velocities to a Well")
        self.setMinimumWidth(460)
        self._build_ui()
        self._on_well_changed()

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        self.well = QComboBox()
        for w in getattr(self._state.project, "wells", []):
            self.well.addItem(w.name or "Unnamed", w)
        root.addWidget(QLabel("Well:"))
        root.addWidget(self.well)
        self.well.currentIndexChanged.connect(self._on_well_changed)

        root.addWidget(QLabel("Markers — depth from the well, TWT from the checkshot:"))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Marker", "Depth (m)", "TWT (ms)"])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.table)

        row_btns = QHBoxLayout()
        add_btn = QPushButton("Add row"); add_btn.clicked.connect(lambda: self._add_row())
        del_btn = QPushButton("Remove row"); del_btn.clicked.connect(self._remove_row)
        self._compute_btn = QPushButton("Compute calibration")
        self._compute_btn.clicked.connect(self._compute)
        row_btns.addWidget(add_btn); row_btns.addWidget(del_btn)
        row_btns.addStretch(); row_btns.addWidget(self._compute_btn)
        root.addLayout(row_btns)

        self._report = QLabel("")
        self._report.setStyleSheet("font-family: monospace; font-size: 8pt; color:#333;")
        self._report.setWordWrap(True)
        root.addWidget(QLabel("Result:"))
        root.addWidget(self._report)

        self._buttons = QDialogButtonBox()
        self._apply_btn = self._buttons.addButton("Apply", QDialogButtonBox.ButtonRole.ApplyRole)
        self._buttons.addButton(QDialogButtonBox.StandardButton.Close)
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._apply)
        self._buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.accept)
        root.addWidget(self._buttons)

    # ------------------------------------------------------------------

    def _add_row(self, marker: str = "", depth: float = 0.0, twt_ms: float = 0.0) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(str(marker)))
        self.table.setItem(r, 1, QTableWidgetItem(f"{depth:g}"))
        self.table.setItem(r, 2, QTableWidgetItem(f"{twt_ms:g}"))

    def _remove_row(self) -> None:
        r = self.table.currentRow()
        if r < 0:
            r = self.table.rowCount() - 1
        if r >= 0:
            self.table.removeRow(r)

    def _on_well_changed(self) -> None:
        """Seed the marker table from the selected well's formation tops."""
        self.table.setRowCount(0)
        w = self.well.currentData()
        if w is not None:
            for name, md in sorted(getattr(w, "formation_tops", {}).items(),
                                   key=lambda kv: kv[1]):
                self._add_row(name, float(md), 0.0)
        if self.table.rowCount() == 0:
            self._add_row("", 0.0, 0.0)

    def _markers(self) -> list[Marker]:
        """Rows with a positive TWT become (depth, twt) markers (TWT ms → s)."""
        out: list[Marker] = []
        for r in range(self.table.rowCount()):
            try:
                name = self.table.item(r, 0).text() if self.table.item(r, 0) else ""
                depth = float(self.table.item(r, 1).text())
                twt_ms = float(self.table.item(r, 2).text())
            except (ValueError, AttributeError):
                continue
            if twt_ms > 0.0:
                out.append(Marker(depth_m=depth, twt_s=twt_ms / 1000.0, name=name))
        return out

    def _starting_model(self) -> VelocityModel:
        vm = getattr(self._state.project, "velocity_model", None)
        if vm is not None and not vm.is_empty:
            return vm
        return VelocityModel.average_vz(1800.0, 0.3)   # bootstrap to calibrate

    def _compute(self) -> None:
        markers = self._markers()
        if len(markers) < 2:
            self._report.setText("⚠ Need at least 2 markers with a TWT to calibrate.")
            self._apply_btn.setEnabled(False)
            return
        model = calibrate_model(self._starting_model(), markers)
        self._calibrated = model
        resid = marker_residuals(model, markers)
        lines = [f"{model.method_label}   ({model.provenance})", ""]
        lines.append(f"{'marker':<12}{'depth':>9}{'model':>9}{'Δz(m)':>8}{'Δtwt(ms)':>10}")
        for r in resid:
            lines.append(f"{r['name'][:12]:<12}{r['depth_m']:>9.0f}"
                         f"{r['model_depth_m']:>9.0f}{r['depth_residual_m']:>8.1f}"
                         f"{r['twt_residual_s']*1000:>10.1f}")
        self._report.setText("\n".join(lines))
        self._apply_btn.setEnabled(True)

    def _apply(self) -> None:
        if self._calibrated is None:
            return
        self._state.project.velocity_model = self._calibrated
        if self._on_apply is not None:
            self._on_apply()
