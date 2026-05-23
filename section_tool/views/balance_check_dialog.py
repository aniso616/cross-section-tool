"""Balance check dialog.

Accessible via Tools → Check Section Balance. Reports line lengths of
horizon picks and areas of section polygons for the active section.
"""
from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


def _pick_line_length(distances: np.ndarray, depths: np.ndarray) -> float:
    """Arc length of a horizon pick polyline (section-space units)."""
    if len(distances) < 2:
        return 0.0
    return float(np.hypot(np.diff(distances), np.diff(depths)).sum())


class BalanceCheckDialog(QDialog):
    """Balance check report for the active section.

    Shows horizon pick line lengths and polygon areas (both in km / km²).
    """

    def __init__(self, app_state, section, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Balance Check — {section.name}")
        self.setMinimumWidth(560)

        proj = app_state._project
        sec_length_m = section.total_length()
        sec_length_km = sec_length_m / 1000.0

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ── Section summary ───────────────────────────────────────────────
        hdr = QLabel(f"<b>Section:</b> {section.name}   "
                     f"<b>Length:</b> {sec_length_km:.3f} km")
        hdr.setTextFormat(Qt.RichText)
        layout.addWidget(hdr)

        # ── Horizon line lengths ──────────────────────────────────────────
        hz_box = QGroupBox("Horizon Line Lengths")
        hz_layout = QVBoxLayout(hz_box)

        picks = [hp for hp in proj.horizon_picks if hp.n_picks_for_section(section.name) >= 2]
        if picks:
            hz_table = QTableWidget(len(picks), 4)
            hz_table.setHorizontalHeaderLabels(["Name", "Picks", "Length (km)", "Extent (km)"])
            hz_table.horizontalHeader().setStretchLastSection(True)
            hz_table.verticalHeader().setVisible(False)
            hz_table.setEditTriggers(QTableWidget.NoEditTriggers)
            hz_table.setSelectionMode(QTableWidget.SingleSelection)

            for row, hp in enumerate(picks):
                dist, depth = hp.picks_for_section(section.name)
                length_km = _pick_line_length(dist, depth) / 1000.0
                d_min_km = float(dist.min()) / 1000.0
                d_max_km = float(dist.max()) / 1000.0
                hz_table.setItem(row, 0, _cell(hp.name or "(unnamed)"))
                hz_table.setItem(row, 1, _cell(str(len(dist)), align=Qt.AlignRight | Qt.AlignVCenter))
                hz_table.setItem(row, 2, _cell(f"{length_km:.3f}", align=Qt.AlignRight | Qt.AlignVCenter))
                hz_table.setItem(row, 3, _cell(f"{d_min_km:.2f} – {d_max_km:.2f}", align=Qt.AlignRight | Qt.AlignVCenter))

            hz_table.resizeColumnsToContents()
            hz_layout.addWidget(hz_table)
        else:
            hz_layout.addWidget(QLabel("No horizon picks on this section."))

        layout.addWidget(hz_box)

        # ── Polygon areas ─────────────────────────────────────────────────
        poly_box = QGroupBox("Polygon Areas")
        poly_layout = QVBoxLayout(poly_box)

        section_polys = [p for p in proj.polygons if p.section_name == section.name]
        if section_polys:
            poly_table = QTableWidget(len(section_polys), 4)
            poly_table.setHorizontalHeaderLabels(["Name", "Formation", "Area (km²)", "Vertices"])
            poly_table.horizontalHeader().setStretchLastSection(True)
            poly_table.verticalHeader().setVisible(False)
            poly_table.setEditTriggers(QTableWidget.NoEditTriggers)
            poly_table.setSelectionMode(QTableWidget.SingleSelection)

            total_area_m2 = 0.0
            for row, poly in enumerate(section_polys):
                area_km2 = poly.area / 1e6
                total_area_m2 += poly.area
                poly_table.setItem(row, 0, _cell(poly.name or "(unnamed)"))
                poly_table.setItem(row, 1, _cell(poly.formation or ""))
                poly_table.setItem(row, 2, _cell(f"{area_km2:.4f}", align=Qt.AlignRight | Qt.AlignVCenter))
                poly_table.setItem(row, 3, _cell(str(poly.n_vertices), align=Qt.AlignRight | Qt.AlignVCenter))

            poly_table.resizeColumnsToContents()
            poly_layout.addWidget(poly_table)

            total_area_km2 = total_area_m2 / 1e6
            summary_row = QHBoxLayout()
            summary_row.addWidget(QLabel(f"<b>Total area:</b> {total_area_km2:.4f} km²"))
            if sec_length_km > 0:
                depth_km = total_area_km2 / sec_length_km
                summary_row.addWidget(QLabel(
                    f"   <b>Depth to detachment:</b> {depth_km:.3f} km"
                    f"<span style='color:#777; font-size:small;'> (area ÷ length)</span>"
                ))
            summary_row.addStretch()
            poly_layout.addLayout(summary_row)
        else:
            poly_layout.addWidget(QLabel("No polygons on this section."))

        layout.addWidget(poly_box)

        # ── Close button ─────────────────────────────────────────────────
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def _cell(text: str, align: Qt.Alignment = Qt.AlignLeft | Qt.AlignVCenter) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(align)
    item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
    return item
