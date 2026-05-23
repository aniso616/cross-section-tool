"""Attribute table dialog.

Accessible via Tools → Attribute Table. Shows all geological elements
(horizons, faults, polygons, wells) in sortable per-category tables.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class AttributeTableDialog(QDialog):
    """Tabbed read-only view of all geological element attributes."""

    def __init__(self, app_state, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Attribute Table")
        self.setMinimumSize(740, 480)

        proj = app_state._project
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        tabs = QTabWidget()
        tabs.addTab(_build_horizon_tab(proj), f"Horizons ({len(proj.horizon_picks)})")
        tabs.addTab(_build_fault_tab(proj),   f"Faults ({len(proj.fault_picks)})")
        tabs.addTab(_build_polygon_tab(proj), f"Polygons ({len(proj.polygons)})")
        tabs.addTab(_build_well_tab(proj),    f"Wells ({len(proj.wells)})")
        layout.addWidget(tabs)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def _build_horizon_tab(proj) -> QTableWidget:
    headers = ["Name", "Contact Type", "Formation Above", "Formation Below",
               "Age (Ma)", "Confidence", "Picks", "Color"]
    rows = proj.horizon_picks
    table = _make_table(len(rows), headers)
    for r, hp in enumerate(rows):
        age = f"{hp.age_ma:.2f}" if hp.age_ma is not None else ""
        table.setItem(r, 0, _cell(hp.name or ""))
        table.setItem(r, 1, _cell(getattr(hp, "contact_type", "") or ""))
        table.setItem(r, 2, _cell(getattr(hp, "formation_above", "") or ""))
        table.setItem(r, 3, _cell(getattr(hp, "formation_below", "") or ""))
        table.setItem(r, 4, _cell(age, Qt.AlignRight | Qt.AlignVCenter))
        conf = getattr(hp, "confidence", 1.0)
        table.setItem(r, 5, _cell(f"{conf:.2f}", Qt.AlignRight | Qt.AlignVCenter))
        table.setItem(r, 6, _cell(str(hp.n_picks), Qt.AlignRight | Qt.AlignVCenter))
        table.setItem(r, 7, _cell(hp.color))
    table.resizeColumnsToContents()
    table.setSortingEnabled(True)
    return table


def _build_fault_tab(proj) -> QTableWidget:
    headers = ["Name", "Fault Type", "Dip Direction", "Sense of Slip",
               "Displacement (m)", "Picks", "Color"]
    rows = proj.fault_picks
    table = _make_table(len(rows), headers)
    for r, fp in enumerate(rows):
        disp = f"{fp.displacement:.1f}" if getattr(fp, "displacement", None) is not None else ""
        table.setItem(r, 0, _cell(fp.name or ""))
        table.setItem(r, 1, _cell(getattr(fp, "fault_type", "") or ""))
        table.setItem(r, 2, _cell(getattr(fp, "dip_direction", "") or ""))
        table.setItem(r, 3, _cell(getattr(fp, "sense_of_slip", "") or ""))
        table.setItem(r, 4, _cell(disp, Qt.AlignRight | Qt.AlignVCenter))
        table.setItem(r, 5, _cell(str(fp.n_picks), Qt.AlignRight | Qt.AlignVCenter))
        table.setItem(r, 6, _cell(fp.color))
    table.resizeColumnsToContents()
    table.setSortingEnabled(True)
    return table


def _build_polygon_tab(proj) -> QTableWidget:
    headers = ["Name", "Formation", "Section", "Area (km²)", "Vertices", "Fill Color"]
    rows = proj.polygons
    table = _make_table(len(rows), headers)
    for r, poly in enumerate(rows):
        area_km2 = poly.area / 1e6
        table.setItem(r, 0, _cell(poly.name or ""))
        table.setItem(r, 1, _cell(getattr(poly, "formation", "") or ""))
        table.setItem(r, 2, _cell(getattr(poly, "section_name", "") or ""))
        table.setItem(r, 3, _cell(f"{area_km2:.4f}", Qt.AlignRight | Qt.AlignVCenter))
        table.setItem(r, 4, _cell(str(poly.n_vertices), Qt.AlignRight | Qt.AlignVCenter))
        table.setItem(r, 5, _cell(poly.fill_color))
    table.resizeColumnsToContents()
    table.setSortingEnabled(True)
    return table


def _build_well_tab(proj) -> QTableWidget:
    headers = ["Name", "UWI", "X (m)", "Y (m)", "KB (m)", "TD (m)", "Color"]
    rows = proj.wells
    table = _make_table(len(rows), headers)
    for r, well in enumerate(rows):
        kb = f"{well.kb:.1f}" if hasattr(well, "kb") and well.kb is not None else ""
        td = (f"{well.deviation.max_tvd:.1f}"
              if hasattr(well, "deviation") and well.deviation else "")
        table.setItem(r, 0, _cell(well.name or ""))
        table.setItem(r, 1, _cell(getattr(well, "uwi", "") or ""))
        table.setItem(r, 2, _cell(f"{well.x:.1f}", Qt.AlignRight | Qt.AlignVCenter))
        table.setItem(r, 3, _cell(f"{well.y:.1f}", Qt.AlignRight | Qt.AlignVCenter))
        table.setItem(r, 4, _cell(kb, Qt.AlignRight | Qt.AlignVCenter))
        table.setItem(r, 5, _cell(td, Qt.AlignRight | Qt.AlignVCenter))
        table.setItem(r, 6, _cell(getattr(well, "color", "") or ""))
    table.resizeColumnsToContents()
    table.setSortingEnabled(True)
    return table


def _make_table(n_rows: int, headers: list[str]) -> QTableWidget:
    table = QTableWidget(n_rows, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.horizontalHeader().setStretchLastSection(True)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    table.setSelectionMode(QTableWidget.SingleSelection)
    table.setAlternatingRowColors(True)
    return table


def _cell(text: str, align: Qt.Alignment = Qt.AlignLeft | Qt.AlignVCenter) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(align)
    item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
    return item
