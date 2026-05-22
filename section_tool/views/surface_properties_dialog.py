"""Surface properties dialog — data info and display options."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QLabel, QWidget,
)

from .properties_dialog import PropertiesDialog


class SurfacePropertiesDialog(PropertiesDialog):

    def _build_tabs(self) -> None:
        self.tabs.addTab(self._make_general_tab(), "General")
        self.tabs.addTab(self._make_data_tab(), "Data")
        self.tabs.addTab(self._make_display_tab(), "Display")

    def _make_data_tab(self) -> QWidget:
        w = QWidget(); form = QFormLayout(w)
        form.setVerticalSpacing(6); form.setHorizontalSpacing(10)

        src = self.obj_data.get("source_file", "")
        if src and len(src) > 50:
            src = "…" + src[-47:]
        form.addRow("Source:", QLabel(src))
        form.addRow("Format:", QLabel(self.obj_data.get("source_format", "")))
        form.addRow("Points:", QLabel(f"{int(self.obj_data.get('n_points', 0)):,}"))
        form.addRow("Z Domain:", QLabel(self.obj_data.get("z_domain", "")))
        form.addRow("Z Units:", QLabel(self.obj_data.get("z_units", "")))
        zr = self.obj_data.get("z_range", (0, 0))
        form.addRow("Z Range:", QLabel(f"{zr[0]:.1f} — {zr[1]:.1f}"))
        form.addRow("CRS (EPSG):", QLabel(str(self.obj_data.get("crs_epsg", ""))))
        b = self.obj_data.get("bounds", (0, 0, 0, 0))
        form.addRow("Bounds X:", QLabel(f"{b[0]:.0f} — {b[2]:.0f}"))
        form.addRow("Bounds Y:", QLabel(f"{b[1]:.0f} — {b[3]:.0f}"))

        return w

    def _make_display_tab(self) -> QWidget:
        w = QWidget(); form = QFormLayout(w)
        form.setVerticalSpacing(6); form.setHorizontalSpacing(10)

        self._widgets["interpolation"] = QComboBox()
        self._widgets["interpolation"].addItems(["linear", "nearest"])
        cur = self.obj_data.get("interpolation", "linear")
        idx = self._widgets["interpolation"].findText(cur)
        if idx >= 0:
            self._widgets["interpolation"].setCurrentIndex(idx)
        form.addRow("Interpolation:", self._widgets["interpolation"])

        map_disp_vals = ["contours", "bbox_only", "points", "none"]
        self._widgets["map_display"] = QComboBox()
        self._widgets["map_display"].addItems(
            ["Contour Map", "Bounding Box Only", "Data Points", "None"])
        cur_md = self.obj_data.get("map_display", "contours")
        if cur_md in map_disp_vals:
            self._widgets["map_display"].setCurrentIndex(
                map_disp_vals.index(cur_md))
        form.addRow("Map Display:", self._widgets["map_display"])

        self._widgets["contour_interval"] = QDoubleSpinBox()
        self._widgets["contour_interval"].setRange(0.1, 10000)
        self._widgets["contour_interval"].setDecimals(1)
        self._widgets["contour_interval"].setValue(
            float(self.obj_data.get("contour_interval", 50)))
        form.addRow("Contour Interval:", self._widgets["contour_interval"])

        return w

    def _collect_values(self) -> dict:
        vals = super()._collect_values()
        map_disp_vals = ["contours", "bbox_only", "points", "none"]
        vals["map_display"] = map_disp_vals[
            self._widgets["map_display"].currentIndex()]
        return vals
