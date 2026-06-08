"""DepthStretchDialog — the time→depth front door (M3, overhauled).

One dialog, the full method ladder: Bulk → Average V(z) → Layered-from-formations
→ Well-calibrated.  Each rung is interpretation-gated (greyed with a reason until
its prerequisites exist).  Fields are disclosed progressively by Method + Setting.
A live layer-cake schematic (right) and a syntax-colored model summary update on
every edit — boundary conversions only, never a trace re-stretch (that is Apply's
job: compute-once-on-Apply / navigate-for-free).

Time is shown in ms; the controller works in SI seconds.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
    QLabel, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget)

from section_tool.core.stretch_setup import StretchSetup, WATER_VELOCITY_MS
from section_tool.core.velocity_model import VelocityModel
from section_tool.core.well_calibration import Marker, calibrate_model, marker_residuals

# Neutral, high-contrast summary text; a few functional accents only.
_TEXT_BASE  = "#E6E6E6"   # legible foreground on the dark theme
_TEXT_MUTED = "#9AA0A6"   # provenance (italic)
_AXIS_TIME  = "#E0A33E"   # schematic TWT-axis ticks (amber)
_AXIS_DEPTH = "#5FB85F"   # schematic depth-axis ticks (green)
_WATER_FILL = "#2B6CB0"   # water band / chip
_SED_FILL   = "#5B6470"   # neutral sediment band / chip (no formation color)


def band_color_hex(layer, strat_column=None) -> str:
    """The schematic band colour for *layer* — reused as the summary row chip so
    summary ↔ schematic share one visual language.  Water → blue; a layered
    formation → its Formation.color (project data); else neutral sediment."""
    name = (getattr(layer, "name", "") or "").lower()
    if "water" in name:
        return _WATER_FILL
    key = getattr(layer, "formation", "") or getattr(layer, "name", "")
    if strat_column is not None and key:
        try:
            f = strat_column.get_formation(key)
            if f is not None and getattr(f, "color", None):
                r, g, b = (int(c) for c in tuple(f.color)[:3])
                return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            pass
    return _SED_FILL

_METHODS = [
    ("bulk", "Bulk velocity"),
    ("average_vz", "Average V(z)"),
    ("layered_from_formations", "Layered from formations"),
    ("well_calibrated", "Well-calibrated"),
]


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without Qt)
# ---------------------------------------------------------------------------

def method_availability(zone_tops, wells) -> dict[str, tuple[bool, str]]:
    """For each ladder rung → (enabled, reason-if-disabled).

    Interpretation-gated: bulk/average need nothing (the bootstrap); layered needs
    picked zone-bounding horizons; well-calibrated needs a well.
    """
    has_tops = bool(zone_tops)
    has_wells = bool(wells)
    return {
        "bulk":                    (True, ""),
        "average_vz":              (True, ""),
        "layered_from_formations": (has_tops,
                                    "" if has_tops else "needs picked zone-bounding horizons"),
        "well_calibrated":         (has_wells,
                                    "" if has_wells else "needs a well with control"),
    }


def format_model_summary_html(model: VelocityModel, strat_column=None) -> str:
    """High-contrast NEUTRAL summary — the contrast is the win, not hues.  The
    only colour is a per-layer chip matching that layer's schematic band (ties
    summary ↔ schematic) plus muted-italic provenance.  Monospace so columns
    align."""
    if model is None or model.is_empty:
        return (f'<div style="font-family:monospace;color:{_TEXT_MUTED};'
                f'font-style:italic;">unconverted</div>')

    def chip(hexc):
        return f'<span style="background-color:{hexc};color:{hexc};">&nbsp;&nbsp;</span>'

    head = (f'<span style="color:{_TEXT_BASE};">{model.method_label}</span>'
            f'&nbsp;<span style="color:{_TEXT_MUTED};font-style:italic;">'
            f'({model.provenance})</span>')
    lines = [head]
    for L in model.layers:
        fn = L.function
        desc = (f"V(z) v0={fn.v0:.0f} k={fn.k:g}" if fn.method == "linear_v0k"
                else f"bulk {fn.v0:.0f} m/s")
        nm = f"  {L.name}" if L.name else ""
        lines.append(
            f'{chip(band_color_hex(L, strat_column))}&nbsp;'
            f'<span style="color:{_TEXT_BASE};">'
            f'{L.top_twt_s * 1000:7.0f} ms&nbsp;&nbsp;{desc}{nm}</span>')
    body = "<br>".join(lines)
    return f'<div style="font-family:monospace;font-size:9pt;color:{_TEXT_BASE};">{body}</div>'


# ---------------------------------------------------------------------------
# Live layer-cake schematic (QPainter; cheap — boundary conversions only)
# ---------------------------------------------------------------------------

class VelocityModelSchematic(QWidget):
    """Layer-cake column, linear in TWT, with TWT (ms) ticks on the left and
    derived Depth (m) ticks on the right so the non-linear stretch is visible."""

    _HEADER_H = 24      # band above the column for the axis headers (no tick collision)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(280, 300)
        self._model: VelocityModel | None = None
        self._max_twt_s = 3.0
        self._strat = None

    def set_model(self, model: VelocityModel, max_twt_s: float, strat_column=None) -> None:
        self._model = model
        self._max_twt_s = max(float(max_twt_s), 1e-3)
        self._strat = strat_column
        self.update()                       # redraw only — never a re-stretch

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#1f1f1f"))
        m = self._model
        if m is None or m.is_empty:
            p.setPen(QColor("#777"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "(no model)")
            return
        # Widened column, pushed right (room for axis numbers on both sides).
        left, right = 52, w - 64
        top, bot = self._HEADER_H + 10, h - 16
        col_h = bot - top

        def y_of(twt_s):
            return top + col_h * min(max(twt_s / self._max_twt_s, 0.0), 1.0)

        # Header band (its own row above the column; clear gap before first tick).
        p.setPen(QColor(_AXIS_TIME))
        p.drawText(QRect(0, 4, left, 16), Qt.AlignmentFlag.AlignLeft, "TWT ms")
        p.setPen(QColor(_AXIS_DEPTH))
        p.drawText(QRect(right, 4, w - right, 16), Qt.AlignmentFlag.AlignRight, "Depth m")

        bounds = [L.top_twt_s for L in m.layers] + [self._max_twt_s]
        for i, L in enumerate(m.layers):
            y0, y1 = y_of(bounds[i]), y_of(bounds[i + 1])
            band = QRect(left, int(y0), right - left, int(y1 - y0) + 1)
            p.fillRect(band, QColor(band_color_hex(L, self._strat)))
            # Provenance → outline (assumed dashed; interpolated purple; calibrated solid green).
            if L.provenance == "well_calibrated":
                pen = QPen(QColor("#5FB85F")); pen.setWidth(2)
            elif L.provenance == "interpolated":
                pen = QPen(QColor("#B07CD6"))
            else:
                pen = QPen(QColor("#9AA0A6")); pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawRect(left, int(y0), right - left, int(y1 - y0))
            # Label the band: layer name + its method/velocity (same as the controls).
            bh = y1 - y0
            if bh >= 14:
                fn = L.function
                desc = (f"V(z) {fn.v0:.0f} m/s" if fn.method == "linear_v0k"
                        else f"Bulk {fn.v0:.0f} m/s")
                p.setPen(QColor("#F5F5F5"))
                if L.name and bh >= 26:
                    p.drawText(QRect(left, int(y0), right - left, int(bh / 2)),
                               Qt.AlignmentFlag.AlignCenter, L.name)
                    p.drawText(QRect(left, int(y0 + bh / 2), right - left, int(bh / 2)),
                               Qt.AlignmentFlag.AlignCenter, desc)
                else:
                    p.drawText(band, Qt.AlignmentFlag.AlignCenter,
                               f"{L.name}  {desc}" if L.name else desc)

        # Axis ticks at every boundary + a mid mark; numbers sit at the boundary,
        # clear of the column edges.
        twt_marks = sorted(set(bounds + [self._max_twt_s * 0.5]))
        for t in twt_marks:
            y = int(y_of(t))
            p.setPen(QColor(_AXIS_TIME))
            p.drawText(QRect(0, y - 7, left - 4, 14),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f"{t * 1000:.0f}")
            try:
                z = m.twt_to_depth(t)
                p.setPen(QColor(_AXIS_DEPTH))
                p.drawText(QRect(right + 4, y - 7, w - right - 4, 14),
                           Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                           f"{z:.0f}")
            except Exception:
                pass
        p.end()


# ---------------------------------------------------------------------------
# The dialog
# ---------------------------------------------------------------------------

class DepthStretchDialog(QDialog):
    def __init__(self, state, on_apply=None, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._on_apply = on_apply
        self.setWindowTitle("Depth Stretch — Time → Depth Conversion")
        self.setMinimumWidth(640)
        self._build_ui()
        self._refresh_method_gating()
        self._on_changed()

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        cols = QHBoxLayout()
        outer.addLayout(cols)

        # Left: controls + summary + how-to
        left = QVBoxLayout()
        cols.addLayout(left, 3)
        self._form = QFormLayout()
        left.addLayout(self._form)

        self.setting = QComboBox()
        # Environment pairing (Marine has the water column).  Land first = the
        # well-free base-case default (no water/replacement layer).
        self.setting.addItem("Land", "onshore")
        self.setting.addItem("Marine", "marine")
        self._form.addRow("Setting:", self.setting)
        self.method = QComboBox()
        for val, label in _METHODS:
            self.method.addItem(label, val)
        self._form.addRow("Method:", self.method)

        def _spin(lo, hi, val, step, suffix, dec=0):
            s = QDoubleSpinBox(); s.setRange(lo, hi); s.setValue(val)
            s.setSingleStep(step); s.setSuffix(suffix); s.setDecimals(dec)
            return s

        self.datum_ms    = _spin(0, 10000, 0, 10, " ms")
        self.seafloor_ms = _spin(0, 10000, 400, 10, " ms")
        self.basement_ms = _spin(0, 20000, 3000, 50, " ms")
        self.bulk_v  = _spin(500, 6000, 2400, 50, " m/s")
        self.v0      = _spin(500, 6000, 1800, 50, " m/s")
        self.k       = _spin(0.0, 3.0, 0.6, 0.05, " s⁻¹", dec=2)
        self.water_v = _spin(1400, 1600, WATER_VELOCITY_MS, 5, " m/s")
        for label, w in [("Datum / SRD:", self.datum_ms),
                         ("Approx. seafloor:", self.seafloor_ms),
                         ("Approx. basement:", self.basement_ms),
                         ("Bulk velocity:", self.bulk_v),
                         ("V₀:", self.v0), ("k:", self.k),
                         ("Water velocity:", self.water_v)]:
            self._form.addRow(label, w)

        # Well-calibrated rung: inline calibration controls (hidden otherwise).
        self.well = QComboBox()
        for wll in getattr(self._state.project, "wells", []):
            self.well.addItem(wll.name or "Unnamed", wll)
        self._form.addRow("Well:", self.well)
        self.markers = QTableWidget(0, 3)
        self.markers.setHorizontalHeaderLabels(["Marker", "Depth (m)", "TWT (ms)"])
        self.markers.setMaximumHeight(150)
        left.addWidget(self.markers)

        self._summary = QTextEdit(); self._summary.setReadOnly(True)
        self._summary.setMaximumHeight(110)
        left.addWidget(QLabel("Velocity model:"))
        left.addWidget(self._summary)

        self._howto = QLabel(
            "set surfaces → pick method → Apply → tune v₀/k and re-apply.  "
            "Layered & well-calibrated unlock with picks / a well.")
        self._howto.setWordWrap(True)
        self._howto.setStyleSheet("color:#9AA0A6; font-size: 8pt;")
        left.addWidget(self._howto)

        # Right: live schematic — pushed right, given the larger share of the width.
        cols.addSpacing(16)
        self._schematic = VelocityModelSchematic()
        cols.addWidget(self._schematic, 4)

        self._buttons = QDialogButtonBox()
        self._apply_btn = self._buttons.addButton("Apply", QDialogButtonBox.ButtonRole.ApplyRole)
        self._buttons.addButton(QDialogButtonBox.StandardButton.Close)
        self._apply_btn.clicked.connect(self._apply)
        self._buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.accept)
        outer.addWidget(self._buttons)

        self.setting.currentIndexChanged.connect(self._on_changed)
        self.method.currentIndexChanged.connect(self._on_changed)
        for s in (self.datum_ms, self.seafloor_ms, self.basement_ms,
                  self.bulk_v, self.v0, self.k, self.water_v):
            s.valueChanged.connect(self._on_changed)
        self.well.currentIndexChanged.connect(self._on_well_changed)
        self.markers.itemChanged.connect(lambda *_: self._on_changed())
        self._on_well_changed()

    # ------------------------------------------------------------------
    # Gating + progressive disclosure
    # ------------------------------------------------------------------

    def _refresh_method_gating(self) -> None:
        avail = method_availability(self._zone_tops(),
                                    getattr(self._state.project, "wells", []))
        for val, _label in _METHODS:
            idx = self.method.findData(val)
            enabled, reason = avail[val]
            item = self.method.model().item(idx)
            item.setEnabled(enabled)
            item.setToolTip(reason)
        cur = self.method.currentData()
        if not avail.get(cur, (True, ""))[0]:
            self.method.setCurrentIndex(self.method.findData("bulk"))

    def _set_row_visible(self, field_widget, visible: bool) -> None:
        try:
            self._form.setRowVisible(field_widget, visible)   # Qt 6.4+
        except Exception:
            field_widget.setVisible(visible)
            lbl = self._form.labelForField(field_widget)
            if lbl is not None:
                lbl.setVisible(visible)

    def _refresh_visibility(self) -> None:
        """Progressive disclosure keyed to Method + Setting."""
        method = self.method.currentData()
        marine = self.setting.currentData() == "marine"
        self._set_row_visible(self.seafloor_ms, marine)
        self._set_row_visible(self.water_v, marine)
        self._set_row_visible(self.bulk_v, method == "bulk")
        self._set_row_visible(self.v0, method == "average_vz")
        self._set_row_visible(self.k, method == "average_vz")
        # Basement scaffolds the layered structure; datum always relevant.
        self._set_row_visible(self.basement_ms, method == "layered_from_formations")
        is_well = method == "well_calibrated"
        self._set_row_visible(self.well, is_well)
        self.markers.setVisible(is_well)

    # ------------------------------------------------------------------

    def _zone_tops(self):
        tops = []
        for hp in getattr(self._state.project, "horizon_picks", []):
            anch = getattr(hp, "_twt_anchor", None)
            if getattr(hp, "seismic_tied", False) and anch is not None and len(anch):
                t = float(np.nanmedian(anch))
                if t == t:
                    tops.append((t, getattr(hp, "formation_below", "") or hp.name))
        return sorted(tops)

    def _on_well_changed(self) -> None:
        self.markers.blockSignals(True)
        self.markers.setRowCount(0)
        w = self.well.currentData()
        if w is not None:
            for name, md in sorted(getattr(w, "formation_tops", {}).items(),
                                   key=lambda kv: kv[1]):
                r = self.markers.rowCount(); self.markers.insertRow(r)
                self.markers.setItem(r, 0, QTableWidgetItem(name))
                self.markers.setItem(r, 1, QTableWidgetItem(f"{md:g}"))
                self.markers.setItem(r, 2, QTableWidgetItem("0"))
        self.markers.blockSignals(False)
        self._on_changed()

    def _well_markers(self) -> list[Marker]:
        out = []
        for r in range(self.markers.rowCount()):
            try:
                name = self.markers.item(r, 0).text() if self.markers.item(r, 0) else ""
                depth = float(self.markers.item(r, 1).text())
                twt_ms = float(self.markers.item(r, 2).text())
            except (ValueError, AttributeError):
                continue
            if twt_ms > 0.0:
                out.append(Marker(depth, twt_ms / 1000.0, name))
        return out

    def _read_setup(self) -> StretchSetup:
        base_method = self.method.currentData()
        if base_method == "well_calibrated":
            base_method = "average_vz"   # calibration promotes the V(z) bootstrap
        return StretchSetup(
            setting=self.setting.currentData(), method=base_method,
            datum_twt_s=self.datum_ms.value() / 1000.0,
            seafloor_twt_s=self.seafloor_ms.value() / 1000.0,
            basement_twt_s=self.basement_ms.value() / 1000.0,
            bulk_v=self.bulk_v.value(), v0=self.v0.value(), k=self.k.value(),
            water_v=self.water_v.value())

    def _build_model(self) -> VelocityModel:
        setup = self._read_setup()
        model = setup.build_model(self._zone_tops(),
                                  getattr(self._state.project, "strat_column", None))
        if self.method.currentData() == "well_calibrated":
            markers = self._well_markers()
            if len(markers) >= 2:
                model = calibrate_model(model, markers)
        return model

    def _on_changed(self) -> None:
        self._refresh_method_gating()
        self._refresh_visibility()
        strat = getattr(self._state.project, "strat_column", None)
        try:
            model = self._build_model()
        except ValueError as e:
            self._summary.setHtml(f'<span style="color:{_TEXT_MUTED};">⚠ {e}</span>')
            self._apply_btn.setEnabled(False)
            self._schematic.set_model(VelocityModel(), self.basement_ms.value() / 1000.0, strat)
            return
        self._apply_btn.setEnabled(True)
        html = format_model_summary_html(model, strat)
        if self.method.currentData() == "well_calibrated":
            html += self._residual_html(model)
        self._summary.setHtml(html)
        max_twt = max(self.basement_ms.value() / 1000.0, 0.5)
        self._schematic.set_model(model, max_twt, strat)

    def _residual_html(self, model) -> str:
        markers = self._well_markers()
        if len(markers) < 2:
            return ('<div style="font-family:monospace;font-size:8pt;color:#9AA0A6;">'
                    'enter ≥2 marker TWTs to calibrate</div>')
        rows = [f'<div style="font-family:monospace;font-size:8pt;color:{_TEXT_BASE};">'
                'residuals (Δz m / Δtwt ms):']
        for r in marker_residuals(model, markers):
            name = r["name"][:10]
            dz = f"{r['depth_residual_m']:+.1f}"
            dt = f"{r['twt_residual_s'] * 1000:+.1f}"
            rows.append(f"&nbsp;{name:<10} {dz} / {dt}")
        return "<br>".join(rows) + "</div>"

    def _apply(self) -> None:
        if self.method.currentData() == "well_calibrated":
            model = self._build_model()
            self._state.project.velocity_model = model
            # Re-derive seismic-tied geometry through the calibrated model so
            # horizons stay glued to their reflectors (setup.apply does this for
            # the other rungs; here we install the model directly).
            from section_tool.core.conversion import restretch_project
            restretch_project(self._state.project, model)
        else:
            setup = self._read_setup()
            setup.apply(self._state.project, self._zone_tops())
        if self._on_apply is not None:
            self._on_apply()
