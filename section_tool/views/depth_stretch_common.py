"""Shared Depth-Stretch view helpers — the neutral model summary, the per-layer
band colours, and the live layer-cake schematic.

Extracted from the old Method×Setting dialog so the recommendation-first panel
(``depth_stretch_panel.py``) reuses the same visual language: summary ↔ schematic
share one colour grammar, provenance shows in line style/weight (not hue), and
formation colours remain project data.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from section_tool.core.velocity_model import VelocityModel, PROVENANCE_LABEL

# Neutral, high-contrast summary text; a few functional accents only.
_TEXT_BASE  = "#E6E6E6"   # legible foreground on the dark theme
_TEXT_MUTED = "#9AA0A6"   # provenance (italic)
_AXIS_TIME  = "#E0A33E"   # schematic TWT-axis ticks (amber)
_AXIS_DEPTH = "#5FB85F"   # schematic depth-axis ticks (green)
_WATER_FILL = "#2B6CB0"   # water band / chip
_SED_FILL   = "#5B6470"   # neutral sediment band / chip (no formation color)

# Ladder method tokens shared with the controller / panel.
_METHODS = [
    ("bulk", "Bulk velocity"),
    ("average_vz", "Average V(z)"),
    ("layered_from_formations", "Layered from formations"),
    ("well_calibrated", "Well-tied"),
]


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


def method_availability(zone_tops, wells) -> dict[str, tuple[bool, str]]:
    """For each base ladder rung → (enabled, reason-if-disabled).

    Interpretation-gated: bulk/average need nothing (the bootstrap); layered needs
    picked zone-bounding horizons; well-tied needs a well.
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

    prov = PROVENANCE_LABEL.get(model.provenance, model.provenance)
    head = (f'<span style="color:{_TEXT_BASE};">{model.method_label}</span>'
            f'&nbsp;<span style="color:{_TEXT_MUTED};font-style:italic;">'
            f'({prov})</span>')
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
        left, right = 52, w - 64
        top, bot = self._HEADER_H + 10, h - 16
        col_h = bot - top

        def y_of(twt_s):
            return top + col_h * min(max(twt_s / self._max_twt_s, 0.0), 1.0)

        p.setPen(QColor(_AXIS_TIME))
        p.drawText(QRect(0, 4, left, 16), Qt.AlignmentFlag.AlignLeft, "TWT ms")
        p.setPen(QColor(_AXIS_DEPTH))
        p.drawText(QRect(right, 4, w - right, 16), Qt.AlignmentFlag.AlignRight, "Depth m")

        bounds = [L.top_twt_s for L in m.layers] + [self._max_twt_s]
        for i, L in enumerate(m.layers):
            y0, y1 = y_of(bounds[i]), y_of(bounds[i + 1])
            band = QRect(left, int(y0), right - left, int(y1 - y0) + 1)
            p.fillRect(band, QColor(band_color_hex(L, self._strat)))
            # Provenance → outline (well-tied solid green; interpolated purple;
            # everything more-grounded gets a heavier weight; assumed dashed grey).
            prov = L.provenance
            if prov in ("well_calibrated", "checkshot_tied"):
                pen = QPen(QColor("#5FB85F")); pen.setWidth(2)
            elif prov in ("sonic_derived", "seismic_velocity", "interpolated"):
                pen = QPen(QColor("#B07CD6"))
            else:
                pen = QPen(QColor("#9AA0A6")); pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawRect(left, int(y0), right - left, int(y1 - y0))
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
