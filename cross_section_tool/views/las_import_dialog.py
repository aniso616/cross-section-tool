"""LAS well import dialog with coordinate verification, CRS handling, and raw header viewer."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QRadioButton, QScrollArea,
    QSizePolicy, QTextEdit, QVBoxLayout, QWidget,
)

if TYPE_CHECKING:
    from lasio import LASFile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _crs_name(epsg: int) -> str:
    try:
        from cross_section_tool.core.crs import get_crs_info
        return get_crs_info(epsg).name
    except Exception:
        return "Unknown CRS"


def _suggest_crs(x: float, y: float) -> str:
    if x == 0.0 and y == 0.0:
        return ""
    if -180.0 <= x <= 180.0 and -90.0 <= y <= 90.0:
        return "Looks like geographic (lon/lat). Consider EPSG:4326."
    if 100_000 <= abs(x) <= 999_999 and 1_000_000 <= abs(y) <= 10_000_000:
        return "Looks like UTM projected coordinates (easting/northing)."
    return ""


# ---------------------------------------------------------------------------
# Raw header viewer
# ---------------------------------------------------------------------------

class _RawHeaderDialog(QDialog):
    """Read-only monospaced view of the complete LAS header sections."""

    def __init__(self, las: "LASFile", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Raw LAS Header")
        self.setMinimumSize(660, 520)

        layout = QVBoxLayout(self)

        te = QTextEdit()
        te.setReadOnly(True)
        font = QFont("Courier New", 9)
        font.setStyleHint(QFont.StyleHint.TypeWriter)
        te.setFont(font)

        lines: list[str] = []

        lines.append("~VERSION ---------------------------------------------------")
        try:
            for key, item in las.version.items():
                lines.append(f"  {key:12s}.{item.unit:8s} {str(item.value):30s} : {item.descr}")
        except Exception:
            pass

        lines.append("")
        lines.append("~WELL -------------------------------------------------------")
        for key, item in las.well.items():
            lines.append(
                f"  {key:12s}.{item.unit:8s} {str(item.value):30s} : {item.descr}"
            )

        lines.append("")
        lines.append("~CURVE ------------------------------------------------------")
        for c in las.curves:
            lines.append(
                f"  {c.mnemonic:12s}.{c.unit:8s}                                : {c.descr}"
            )

        if las.params:
            lines.append("")
            lines.append("~PARAMS -----------------------------------------------------")
            for key, item in las.params.items():
                lines.append(
                    f"  {key:12s}.{item.unit:8s} {str(item.value):30s} : {item.descr}"
                )

        if las.other:
            lines.append("")
            lines.append("~OTHER ------------------------------------------------------")
            lines.append(las.other)

        te.setPlainText("\n".join(lines))
        layout.addWidget(te)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


# ---------------------------------------------------------------------------
# Main import dialog
# ---------------------------------------------------------------------------

class LASImportDialog(QDialog):
    """Review and confirm all settings before importing a LAS well file.

    Parameters
    ----------
    las:
        Already-loaded :class:`lasio.LASFile`.
    path:
        File path (for display only).
    header:
        Dict from :func:`cross_section_tool.io.las.extract_header_full` —
        contains values *and* ``x_source``, ``y_source`` provenance strings.
    project_crs_epsg:
        EPSG code of the current project CRS.
    """

    def __init__(
        self,
        las: "LASFile",
        path: str,
        header: dict,
        project_crs_epsg: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._las = las
        self._path = path
        self._header = header
        self._project_crs = project_crs_epsg

        fname = path.replace("\\", "/").split("/")[-1]
        self.setWindowTitle(f"Import LAS: {fname}")
        self.setMinimumWidth(540)
        self.setMinimumHeight(620)

        root = QVBoxLayout(self)

        # Scrollable form
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        fl = QVBoxLayout(inner)
        fl.setContentsMargins(4, 4, 4, 4)
        self._build_identity(fl)
        self._build_location(fl)
        self._build_crs(fl)
        self._build_curves(fl)
        fl.addStretch()
        scroll.setWidget(inner)
        root.addWidget(scroll, 1)

        # Raw header button
        raw_btn = QPushButton("View Raw Header…")
        raw_btn.clicked.connect(self._show_raw_header)
        root.addWidget(raw_btn)

        # Import / Cancel
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.button(QDialogButtonBox.StandardButton.Ok).setText("Import")
        btns.accepted.connect(self._on_import)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_identity(self, layout: QVBoxLayout) -> None:
        grp = QGroupBox("Well Identity")
        fl = QFormLayout(grp)
        self._name_edit = QLineEdit(self._header.get("well_name") or "")
        fl.addRow("Well name:", self._name_edit)
        self._uwi_edit = QLineEdit(self._header.get("uwi") or "")
        fl.addRow("UWI:", self._uwi_edit)
        layout.addWidget(grp)

    def _build_location(self, layout: QVBoxLayout) -> None:
        grp = QGroupBox("Location")
        fl = QFormLayout(grp)

        x_raw = self._header.get("x")
        y_raw = self._header.get("y")
        x_val = float(x_raw) if x_raw is not None else 0.0
        y_val = float(y_raw) if y_raw is not None else 0.0
        x_src = self._header.get("x_source") or "not found — enter manually"
        y_src = self._header.get("y_source") or "not found — enter manually"

        self._x_spin = self._coord_spin(x_val)
        self._y_spin = self._coord_spin(y_val)

        fl.addRow("X / Easting:", self._src_row(self._x_spin, x_src, x_val == 0.0))
        fl.addRow("Y / Northing:", self._src_row(self._y_spin, y_src, y_val == 0.0))

        kb_val = float(self._header.get("kb") or 0.0)
        self._kb_spin = QDoubleSpinBox()
        self._kb_spin.setRange(-1e5, 1e5)
        self._kb_spin.setDecimals(2)
        self._kb_spin.setValue(kb_val)
        kb_src = self._header.get("kb_source") or ""
        if kb_src:
            fl.addRow("KB elevation (m):", self._src_row(self._kb_spin, kb_src, False))
        else:
            fl.addRow("KB elevation (m):", self._kb_spin)

        gl_val = float(self._header.get("gl") or 0.0)
        self._gl_spin = QDoubleSpinBox()
        self._gl_spin.setRange(-1e5, 1e5)
        self._gl_spin.setDecimals(2)
        self._gl_spin.setValue(gl_val)
        fl.addRow("GL elevation (m):", self._gl_spin)

        suggestion = _suggest_crs(x_val, y_val)
        if suggestion:
            hint = QLabel(f"ℹ  {suggestion}")
            hint.setStyleSheet("color: #666; font-size: 8pt;")
            hint.setWordWrap(True)
            fl.addRow("", hint)

        layout.addWidget(grp)

    def _coord_spin(self, val: float) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(-1e8, 1e8)
        sp.setDecimals(1)
        sp.setValue(val)
        sp.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)
        return sp

    def _src_row(self, widget: QWidget, source: str, warn: bool) -> QWidget:
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(widget, 1)
        lbl = QLabel(source)
        lbl.setStyleSheet("color: grey; font-style: italic; font-size: 8pt;")
        rl.addWidget(lbl)
        if warn:
            widget.setStyleSheet("background: #fffacd;")
        return row

    def _build_crs(self, layout: QVBoxLayout) -> None:
        grp = QGroupBox("Coordinate Reference System")
        vl = QVBoxLayout(grp)

        proj_name = _crs_name(self._project_crs)
        self._crs_same = QRadioButton(
            f"Same as project  (EPSG:{self._project_crs} — {proj_name})"
        )
        self._crs_same.setChecked(True)
        vl.addWidget(self._crs_same)

        spec_row = QWidget()
        sr = QHBoxLayout(spec_row)
        sr.setContentsMargins(0, 0, 0, 0)
        self._crs_specify = QRadioButton("Specify CRS:  EPSG")
        sr.addWidget(self._crs_specify)
        self._epsg_edit = QLineEdit()
        self._epsg_edit.setFixedWidth(72)
        self._epsg_edit.setPlaceholderText("e.g. 32631")
        self._epsg_edit.setEnabled(False)
        sr.addWidget(self._epsg_edit)
        self._epsg_name_lbl = QLabel("")
        self._epsg_name_lbl.setStyleSheet("color: grey; font-size: 8pt;")
        sr.addWidget(self._epsg_name_lbl, 1)
        vl.addWidget(spec_row)

        self._crs_manual = QRadioButton(
            "Unknown — place manually on map after import"
        )
        vl.addWidget(self._crs_manual)

        self._transform_note = QLabel("")
        self._transform_note.setStyleSheet("color: #444; font-size: 8pt;")
        self._transform_note.setWordWrap(True)
        vl.addWidget(self._transform_note)

        self._epsg_edit.textChanged.connect(self._on_epsg_changed)
        self._crs_specify.toggled.connect(
            lambda checked: self._epsg_edit.setEnabled(checked)
        )
        for rb in (self._crs_same, self._crs_specify, self._crs_manual):
            rb.toggled.connect(self._update_transform_note)

        layout.addWidget(grp)

    def _build_curves(self, layout: QVBoxLayout) -> None:
        grp = QGroupBox("Log Curves")
        vl = QVBoxLayout(grp)
        self._curve_checks: dict[str, QCheckBox] = {}

        for curve in self._las.curves[1:]:  # skip depth index
            mn = curve.mnemonic or ""
            skip_default = not mn or mn.upper() == "UNKNOWN"

            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(2, 1, 2, 1)

            cb = QCheckBox(mn or "(unnamed)")
            cb.setChecked(not skip_default)
            rl.addWidget(cb)

            unit_lbl = QLabel(f"[{curve.unit or '—'}]")
            unit_lbl.setStyleSheet("color: grey; font-size: 8pt; min-width: 44px;")
            rl.addWidget(unit_lbl)

            desc_lbl = QLabel(curve.descr or "")
            desc_lbl.setStyleSheet("font-size: 8pt;")
            desc_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            rl.addWidget(desc_lbl, 1)

            try:
                if mn:
                    vals = np.asarray(self._las[mn], dtype=float)
                    valid = vals[~np.isnan(vals)]
                    if len(valid):
                        rng_lbl = QLabel(f"{valid.min():.2g}–{valid.max():.2g}")
                        rng_lbl.setStyleSheet("color: #777; font-size: 8pt;")
                        rl.addWidget(rng_lbl)
            except Exception:
                pass

            vl.addWidget(row)
            if mn:
                self._curve_checks[mn] = cb

        layout.addWidget(grp)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_epsg_changed(self, text: str) -> None:
        text = text.strip()
        if not text:
            self._epsg_name_lbl.setText("")
            self._update_transform_note()
            return
        try:
            epsg = int(text)
            name = _crs_name(epsg)
            ok = "Unknown" not in name
            self._epsg_name_lbl.setStyleSheet(
                f"color: {'green' if ok else 'red'}; font-size: 8pt;"
            )
            self._epsg_name_lbl.setText(name)
        except ValueError:
            self._epsg_name_lbl.setStyleSheet("color: red; font-size: 8pt;")
            self._epsg_name_lbl.setText("invalid")
        self._update_transform_note()

    def _update_transform_note(self) -> None:
        if not self._crs_specify.isChecked():
            self._transform_note.setText("")
            return
        try:
            well_crs = int(self._epsg_edit.text().strip())
            if well_crs != self._project_crs:
                self._transform_note.setText(
                    f"Coordinates will be transformed from EPSG:{well_crs} "
                    f"to EPSG:{self._project_crs} on import."
                )
            else:
                self._transform_note.setText("")
        except ValueError:
            self._transform_note.setText("")

    def _show_raw_header(self) -> None:
        _RawHeaderDialog(self._las, self).exec()

    def _on_import(self) -> None:
        if not self._crs_manual.isChecked():
            x, y = self._x_spin.value(), self._y_spin.value()
            if x == 0.0 and y == 0.0:
                r = QMessageBox.question(
                    self, "Zero Coordinates",
                    "X and Y are both 0.0. Import anyway?\n\n"
                    "(Choose 'No' to go back and enter coordinates or "
                    "select 'Place manually on map'.)",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if r != QMessageBox.StandardButton.Yes:
                    return
        self.accept()

    # ------------------------------------------------------------------
    # Result accessors
    # ------------------------------------------------------------------

    def well_name(self) -> str:
        return self._name_edit.text().strip() or "Unnamed"

    def uwi(self) -> str:
        return self._uwi_edit.text().strip()

    def x(self) -> float:
        return 0.0 if self._crs_manual.isChecked() else self._x_spin.value()

    def y(self) -> float:
        return 0.0 if self._crs_manual.isChecked() else self._y_spin.value()

    def kb(self) -> float:
        return self._kb_spin.value()

    def gl(self) -> float:
        return self._gl_spin.value()

    def well_crs_epsg(self) -> int | None:
        if self._crs_same.isChecked():
            return self._project_crs
        if self._crs_manual.isChecked():
            return None
        try:
            return int(self._epsg_edit.text().strip())
        except ValueError:
            return self._project_crs

    def place_manually(self) -> bool:
        return self._crs_manual.isChecked()

    def selected_curves(self) -> list[str]:
        return [mn for mn, cb in self._curve_checks.items() if cb.isChecked()]
