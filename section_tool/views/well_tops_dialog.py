"""Well tops CSV import dialog with column mapping and preview."""
from __future__ import annotations

import csv
import io

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout,
    QGroupBox, QLabel, QTableWidget, QTableWidgetItem,
    QVBoxLayout,
)

# Required columns with friendly names and defaults
_REQUIRED_COLS = [
    ("well_name",       "Well name",        "well_name"),
    ("x",               "X / Easting",      "x"),
    ("y",               "Y / Northing",      "y"),
    ("kb_elevation",    "KB elevation",      "kb"),
    ("formation_name",  "Formation name",    "formation"),
    ("md",              "Measured depth",    "md"),
    ("tvd",             "True vertical depth","tvd"),
]

_PREVIEW_ROWS = 5


class WellTopsDialog(QDialog):
    """Show first N rows of a well-tops CSV and let user map columns."""

    def __init__(self, csv_path: str, crs_epsg: int = 32632, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Well Tops CSV")
        self.setMinimumWidth(600)
        self._csv_path = csv_path
        self._crs_epsg = crs_epsg
        self._headers: list[str] = []
        self._rows:    list[list[str]] = []
        self._combos:  dict[str, QComboBox] = {}

        self._load_csv()
        self._build_ui()

    # ------------------------------------------------------------------

    def _load_csv(self) -> None:
        with open(self._csv_path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.reader(fh)
            self._headers = next(reader, [])
            for row in reader:
                self._rows.append(row)
                if len(self._rows) >= 500:
                    break

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Preview table
        preview_grp = QGroupBox(f"Preview  ({len(self._rows)} rows detected)")
        pv = QVBoxLayout(preview_grp)
        tbl = QTableWidget(min(_PREVIEW_ROWS, len(self._rows)), len(self._headers))
        tbl.setHorizontalHeaderLabels(self._headers)
        tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        tbl.setMaximumHeight(130)
        for r, row in enumerate(self._rows[:_PREVIEW_ROWS]):
            for c, val in enumerate(row[:len(self._headers)]):
                tbl.setItem(r, c, QTableWidgetItem(val))
        tbl.resizeColumnsToContents()
        pv.addWidget(tbl)
        layout.addWidget(preview_grp)

        # Column mapping
        map_grp = QGroupBox("Column mapping")
        mf = QFormLayout(map_grp)
        cols_with_blank = ["(not present)"] + list(self._headers)
        for key, friendly, default_guess in _REQUIRED_COLS:
            combo = QComboBox()
            combo.addItems(cols_with_blank)
            # Auto-select best match
            best = self._guess_column(default_guess)
            if best is not None:
                combo.setCurrentIndex(best + 1)  # +1 for "(not present)" entry
            mf.addRow(f"{friendly}:", combo)
            self._combos[key] = combo
        layout.addWidget(map_grp)

        # CRS
        crs_grp = QGroupBox("CRS")
        cf = QFormLayout(crs_grp)
        self._crs_label = QLabel(f"EPSG:{self._crs_epsg}  (project CRS)")
        cf.addRow("Coordinate system:", self._crs_label)
        layout.addWidget(crs_grp)

        bbox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        layout.addWidget(bbox)

    def _guess_column(self, hint: str) -> int | None:
        """Return 0-based index into self._headers for the best matching column."""
        hint_lower = hint.lower()
        # Exact match first
        for i, h in enumerate(self._headers):
            if h.lower() == hint_lower:
                return i
        # Substring match
        for i, h in enumerate(self._headers):
            if hint_lower in h.lower() or h.lower() in hint_lower:
                return i
        return None

    def _col_index(self, key: str) -> int | None:
        """Return the CSV column index selected for *key*, or None."""
        combo = self._combos[key]
        idx = combo.currentIndex()
        if idx == 0:
            return None
        return idx - 1   # subtract 1 for "(not present)" entry

    # ------------------------------------------------------------------
    # Result properties — consumed by the import handler
    # ------------------------------------------------------------------

    def load_wells(self):
        """Parse the CSV with the selected column mapping.

        Returns a list of ``Well`` objects.
        Raises ``ValueError`` with a message if required columns are missing.
        """
        from section_tool.core.wells import Well

        ci_wn  = self._col_index("well_name")
        ci_x   = self._col_index("x")
        ci_y   = self._col_index("y")
        ci_kb  = self._col_index("kb_elevation")
        ci_fm  = self._col_index("formation_name")
        ci_md  = self._col_index("md")
        ci_tvd = self._col_index("tvd")

        if ci_wn is None or ci_x is None or ci_y is None:
            raise ValueError("well_name, X, and Y columns are required.")

        wells: dict[str, Well] = {}

        for row in self._rows:
            def _get(ci):
                if ci is None or ci >= len(row):
                    return None
                v = row[ci].strip()
                return v if v else None

            well_name = _get(ci_wn)
            if not well_name:
                continue
            try:
                x = float(_get(ci_x) or "nan")
                y = float(_get(ci_y) or "nan")
            except ValueError:
                continue

            kb = float(_get(ci_kb) or "0") if ci_kb is not None else 0.0
            if well_name not in wells:
                wells[well_name] = Well(name=well_name, x=x, y=y, kb=kb)

            fm_name = _get(ci_fm)
            md_str  = _get(ci_md)
            tvd_str = _get(ci_tvd)

            if fm_name and md_str:
                try:
                    md_val = float(md_str)
                    wells[well_name].add_formation_top(fm_name, md_val)
                except ValueError:
                    pass

        return list(wells.values())
