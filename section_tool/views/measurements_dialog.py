"""Measurements editor — enter / import observed thermal & thermochronometric data
on a well (vitrinite Ro, AFT/AHe/ZHe ages, BHT, …). Thermal Step 1.

The interface says what it expects: each type carries its natural units, the CSV
import shows its required format, and out-of-range values are rejected with a
clear message (never silently clamped).
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from section_tool.core.measurements import (
    Measurement, MEASUREMENT_TYPES, MEASUREMENT_TYPE_ORDER, measurement_label,
    default_units, validate_measurement, parse_measurements_csv)


class _MeasurementEditDialog(QDialog):
    """Add / edit a single measurement; validates physical plausibility on accept."""

    def __init__(self, parent=None, measurement: Measurement | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Measurement")
        self.setMinimumWidth(320)
        form = QFormLayout(self)

        self._type = QComboBox()
        for key in MEASUREMENT_TYPE_ORDER:
            self._type.addItem(measurement_label(key), key)
        self._depth = QDoubleSpinBox()
        self._depth.setRange(0.0, 20000.0)
        self._depth.setDecimals(1)
        self._depth.setSuffix(" m")
        self._value = QDoubleSpinBox()
        self._value.setRange(-1e6, 1e6)
        self._value.setDecimals(3)
        self._unc = QDoubleSpinBox()
        self._unc.setRange(0.0, 1e6)
        self._unc.setDecimals(3)
        self._unc.setSpecialValueText("none")
        self._units = QLineEdit()
        self._source = QLineEdit()
        self._hint = QLabel("")
        self._hint.setStyleSheet("color:#888;")

        form.addRow("Type:", self._type)
        form.addRow("Depth:", self._depth)
        form.addRow("Value:", self._value)
        form.addRow("± Uncertainty:", self._unc)
        form.addRow("Units:", self._units)
        form.addRow("Source:", self._source)
        form.addRow("", self._hint)

        self._type.currentIndexChanged.connect(self._on_type_changed)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

        if measurement is not None:
            idx = self._type.findData(measurement.measurement_type)
            if idx >= 0:
                self._type.setCurrentIndex(idx)
            self._depth.setValue(measurement.depth_m)
            self._value.setValue(measurement.value)
            self._unc.setValue(measurement.uncertainty or 0.0)
            self._units.setText(measurement.units)
            self._source.setText(measurement.source)
        self._on_type_changed()
        if measurement is not None:                       # restore edited units
            self._units.setText(measurement.units or default_units(
                self._type.currentData()))

    def _on_type_changed(self) -> None:
        key = self._type.currentData()
        self._units.setText(default_units(key))
        _label, unit, (lo, hi), _excl = MEASUREMENT_TYPES[key]
        self._hint.setText(f"plausible {lo:g}–{hi:g} {unit}")

    def _on_ok(self) -> None:
        key = self._type.currentData()
        reason = validate_measurement(key, self._value.value())
        if reason:
            QMessageBox.warning(self, "Invalid measurement", reason)
            return                                        # reject, don't clamp
        self.accept()

    @property
    def values(self) -> dict:
        unc = self._unc.value()
        return {
            "measurement_type": self._type.currentData(),
            "depth_m": self._depth.value(),
            "value": self._value.value(),
            "uncertainty": unc if unc > 0.0 else None,
            "units": self._units.text().strip() or default_units(self._type.currentData()),
            "source": self._source.text().strip(),
        }


class _ImportCsvDialog(QDialog):
    """Pick a measurement type + units + file for a two-column CSV import."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import measurements (CSV)")
        self.setMinimumWidth(380)
        v = QVBoxLayout(self)
        form = QFormLayout()
        v.addLayout(form)
        self._type = QComboBox()
        for key in MEASUREMENT_TYPE_ORDER:
            self._type.addItem(measurement_label(key), key)
        self._units = QLineEdit()
        form.addRow("Type:", self._type)
        form.addRow("Units:", self._units)
        row = QHBoxLayout()
        self._path = QLineEdit()
        self._path.setReadOnly(True)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(self._path, 1)
        row.addWidget(browse)
        form.addRow("File:", self._wrap(row))
        self._hint = QLabel("")
        self._hint.setStyleSheet("color:#888;")
        v.addWidget(self._hint)
        self._type.currentIndexChanged.connect(self._on_type)
        self._on_type()

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)

    @staticmethod
    def _wrap(layout) -> QWidget:
        w = QWidget()
        w.setLayout(layout)
        return w

    def _on_type(self) -> None:
        key = self._type.currentData()
        self._units.setText(default_units(key))
        self._hint.setText(f"Expected format: depth_m, value   (one row per sample; "
                           f"header optional; value in {default_units(key)})")

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose CSV", "", "CSV / text (*.csv *.txt);;All files (*)")
        if path:
            self._path.setText(path)

    @property
    def selection(self) -> tuple:
        return (self._path.text(), self._type.currentData(),
                self._units.text().strip())


class MeasurementsDialog(QDialog):
    """Per-well table of observed measurements with add / edit / delete + CSV import."""

    _COLS = ["Depth (m)", "Type", "Value", "± Unc", "Units", "Source"]

    def __init__(self, app_state, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Well Measurements")
        self.setMinimumSize(620, 420)
        self._state = app_state

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Well:"))
        self._well_combo = QComboBox()
        for i, w in enumerate(app_state.project.wells):
            self._well_combo.addItem(w.name or f"Well {i + 1}", i)
        self._well_combo.currentIndexChanged.connect(self._rebuild_table)
        top.addWidget(self._well_combo, 1)
        layout.addLayout(top)

        self._table = QTableWidget(0, len(self._COLS))
        self._table.setHorizontalHeaderLabels(self._COLS)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table)

        bar = QHBoxLayout()
        for label, slot in (("Add…", self._add), ("Edit…", self._edit),
                            ("Delete", self._delete), ("Import CSV…", self._import_csv)):
            b = QPushButton(label)
            b.clicked.connect(slot)
            bar.addWidget(b)
        bar.addStretch()
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        bar.addWidget(close)
        layout.addLayout(bar)

        self._rebuild_table()

    # ------------------------------------------------------------------

    def _current_index(self) -> int | None:
        idx = self._well_combo.currentData()
        return idx if idx is not None and idx < len(self._state.project.wells) else None

    def _current_well(self):
        idx = self._current_index()
        return self._state.project.wells[idx] if idx is not None else None

    def _rebuild_table(self) -> None:
        well = self._current_well()
        self._table.setRowCount(0)
        if well is None:
            return
        for m in sorted(well.measurements, key=lambda x: x.depth_m):
            r = self._table.rowCount()
            self._table.insertRow(r)
            unc = "" if m.uncertainty is None else f"{m.uncertainty:g}"
            cells = [f"{m.depth_m:g}", measurement_label(m.measurement_type),
                     f"{m.value:g}", unc, m.units, m.source]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setData(0x0100, m.uuid)              # Qt.UserRole on col 0
                self._table.setItem(r, c, item)

    def _selected_uuid(self) -> str | None:
        rows = self._table.selectedItems()
        if not rows:
            return None
        return self._table.item(self._table.row(rows[0]), 0).data(0x0100)

    def _persist(self, well) -> None:
        idx = self._current_index()
        if idx is not None:
            self._state.update_well(idx, well)            # persists + emits well_modified
        self._rebuild_table()

    def _add(self) -> None:
        well = self._current_well()
        if well is None:
            QMessageBox.information(self, "Measurements", "Add a well first.")
            return
        dlg = _MeasurementEditDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return
        well.add_measurement(Measurement(**dlg.values))
        self._persist(well)

    def _edit(self) -> None:
        well = self._current_well()
        uid = self._selected_uuid()
        if well is None or uid is None:
            return
        target = next((m for m in well._measurements if m.uuid == uid), None)
        if target is None:
            return
        dlg = _MeasurementEditDialog(self, measurement=target)
        if dlg.exec() != QDialog.Accepted:
            return
        v = dlg.values
        target.measurement_type = v["measurement_type"]
        target.depth_m = v["depth_m"]
        target.value = v["value"]
        target.uncertainty = v["uncertainty"]
        target.units = v["units"]
        target.source = v["source"]
        self._persist(well)

    def _delete(self) -> None:
        well = self._current_well()
        uid = self._selected_uuid()
        if well is None or uid is None:
            return
        well.remove_measurement(uid)
        self._persist(well)

    def _import_csv(self) -> None:
        well = self._current_well()
        if well is None:
            QMessageBox.information(self, "Measurements", "Add a well first.")
            return
        dlg = _ImportCsvDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return
        path, mtype, units = dlg.selection
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
                text = fh.read()
            measurements, errors = parse_measurements_csv(text, mtype, units)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Import failed", str(exc))
            return
        for m in measurements:
            well.add_measurement(m)
        self._persist(well)
        msg = f"Imported {len(measurements)} {measurement_label(mtype)} measurement(s)."
        if errors:
            msg += f"\n{len(errors)} row(s) skipped:\n• " + "\n• ".join(errors[:8])
        QMessageBox.information(self, "Import", msg)
