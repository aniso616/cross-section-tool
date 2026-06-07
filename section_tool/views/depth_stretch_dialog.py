"""DepthStretchDialog — the depth-stretch tool's front door (M3).

A thin Qt shell over the tested StretchSetup controller: declare the setting and
approximate bounding surfaces, pick a method from the ladder, see a live preview
of the resulting velocity model, and Apply (install the model + re-stretch tied
geometry).  Time is shown in ms; the controller works in SI seconds.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel,
    QVBoxLayout)

from section_tool.core.stretch_setup import StretchSetup


class DepthStretchDialog(QDialog):
    """Set up and apply a time→depth stretch through a velocity model."""

    def __init__(self, state, on_apply=None, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._on_apply = on_apply
        self.setWindowTitle("Depth Stretch — Time → Depth Conversion")
        self.setMinimumWidth(420)
        self._build_ui()
        self._refresh_enabled()
        self._update_preview()

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        form = QFormLayout()
        outer.addLayout(form)

        self.setting = QComboBox(); self.setting.addItems(["onshore", "marine"])
        form.addRow("Setting:", self.setting)

        self.method = QComboBox()
        self.method.addItem("Bulk velocity", "bulk")
        self.method.addItem("Average V(z)", "average_vz")
        self.method.addItem("Layered from formations", "layered_from_formations")
        form.addRow("Method:", self.method)

        def _spin(lo, hi, val, step, suffix, dec=0):
            s = QDoubleSpinBox(); s.setRange(lo, hi); s.setValue(val)
            s.setSingleStep(step); s.setSuffix(suffix); s.setDecimals(dec)
            return s

        self.datum_ms    = _spin(0, 10000, 0, 10, " ms")
        self.seafloor_ms = _spin(0, 10000, 400, 10, " ms")
        self.basement_ms = _spin(0, 20000, 3000, 50, " ms")
        form.addRow("Datum / SRD:", self.datum_ms)
        form.addRow("Approx. seafloor:", self.seafloor_ms)
        form.addRow("Approx. basement:", self.basement_ms)

        self.bulk_v  = _spin(500, 6000, 2400, 50, " m/s")
        self.v0      = _spin(500, 6000, 1800, 50, " m/s")
        self.k       = _spin(0.0, 3.0, 0.6, 0.05, " s⁻¹", dec=2)
        self.water_v = _spin(1400, 1600, 1480, 5, " m/s")
        form.addRow("Bulk velocity:", self.bulk_v)
        form.addRow("V₀:", self.v0)
        form.addRow("k:", self.k)
        form.addRow("Water velocity:", self.water_v)

        self._preview = QLabel()
        self._preview.setStyleSheet("color:#333; font-family: monospace; font-size: 8pt;")
        self._preview.setWordWrap(True)
        outer.addWidget(QLabel("Resulting velocity model:"))
        outer.addWidget(self._preview)

        self._buttons = QDialogButtonBox()
        self._apply_btn = self._buttons.addButton("Apply", QDialogButtonBox.ButtonRole.ApplyRole)
        self._buttons.addButton(QDialogButtonBox.StandardButton.Close)
        self._apply_btn.clicked.connect(self._apply)
        self._buttons.rejected.connect(self.reject)
        self._buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.accept)
        outer.addWidget(self._buttons)

        for w in (self.setting, self.method):
            w.currentIndexChanged.connect(self._on_changed)
        for s in (self.datum_ms, self.seafloor_ms, self.basement_ms,
                  self.bulk_v, self.v0, self.k, self.water_v):
            s.valueChanged.connect(self._update_preview)

    # ------------------------------------------------------------------

    def _on_changed(self) -> None:
        self._refresh_enabled()
        self._update_preview()

    def _refresh_enabled(self) -> None:
        marine = self.setting.currentText() == "marine"
        self.seafloor_ms.setEnabled(marine)
        self.water_v.setEnabled(marine)
        method = self.method.currentData()
        self.bulk_v.setEnabled(method == "bulk")
        self.v0.setEnabled(method == "average_vz")
        self.k.setEnabled(method == "average_vz")
        # Interpretation gate: layered needs picked, anchored zone-bounding horizons
        has_tops = bool(self._zone_tops())
        idx = self.method.findData("layered_from_formations")
        item = self.method.model().item(idx)
        item.setEnabled(has_tops)
        if not has_tops and self.method.currentData() == "layered_from_formations":
            self.method.setCurrentIndex(self.method.findData("bulk"))

    def _zone_tops(self):
        """(median TWT anchor, formation) for each seismic-tied horizon — the
        layer tops for the layered method."""
        tops = []
        for hp in getattr(self._state.project, "horizon_picks", []):
            anch = getattr(hp, "_twt_anchor", None)
            if getattr(hp, "seismic_tied", False) and anch is not None and len(anch):
                t = float(np.nanmedian(anch))
                if t == t:
                    tops.append((t, getattr(hp, "formation_below", "") or hp.name))
        return sorted(tops)

    def _read_setup(self) -> StretchSetup:
        return StretchSetup(
            setting=self.setting.currentText(),
            method=self.method.currentData(),
            datum_twt_s=self.datum_ms.value() / 1000.0,
            seafloor_twt_s=self.seafloor_ms.value() / 1000.0,
            basement_twt_s=self.basement_ms.value() / 1000.0,
            bulk_v=self.bulk_v.value(), v0=self.v0.value(), k=self.k.value(),
            water_v=self.water_v.value())

    def _build_model(self):
        setup = self._read_setup()
        return setup.build_model(self._zone_tops(),
                                 getattr(self._state.project, "strat_column", None))

    def _update_preview(self) -> None:
        try:
            model = self._build_model()
        except ValueError as e:
            self._preview.setText(f"⚠ {e}")
            self._apply_btn.setEnabled(False)
            return
        self._apply_btn.setEnabled(True)
        lines = [f"{model.method_label}   ({model.provenance})"]
        for L in model.layers:
            lines.append(f"  {L.top_twt_s*1000:7.0f} ms   {L.method_label}")
        self._preview.setText("\n".join(lines))

    def _apply(self) -> None:
        setup = self._read_setup()
        setup.apply(self._state.project, self._zone_tops())
        if self._on_apply is not None:
            self._on_apply()
