"""Well properties dialog — geometry, display, logs."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QLabel, QListWidget, QSpinBox, QWidget,
)

from .properties_dialog import PropertiesDialog


class WellPropertiesDialog(PropertiesDialog):

    def _build_tabs(self) -> None:
        self.tabs.addTab(self._make_general_tab(), "General")
        self.tabs.addTab(self._make_geometry_tab(), "Geometry")
        self.tabs.addTab(self._make_display_tab(), "Display")
        self.tabs.addTab(self._make_logs_tab(), "Logs")
        self.tabs.addTab(self._make_tops_tab(), "Tops")

    def _make_geometry_tab(self) -> QWidget:
        w = QWidget(); form = QFormLayout(w)
        form.setVerticalSpacing(6); form.setHorizontalSpacing(10)

        self._widgets["x"] = QDoubleSpinBox()
        self._widgets["x"].setRange(-1e9, 1e9); self._widgets["x"].setDecimals(2)
        self._widgets["x"].setValue(float(self.obj_data.get("x", 0)))
        form.addRow("Easting (X):", self._widgets["x"])

        self._widgets["y"] = QDoubleSpinBox()
        self._widgets["y"].setRange(-1e9, 1e9); self._widgets["y"].setDecimals(2)
        self._widgets["y"].setValue(float(self.obj_data.get("y", 0)))
        form.addRow("Northing (Y):", self._widgets["y"])

        self._widgets["kb_elevation"] = QDoubleSpinBox()
        self._widgets["kb_elevation"].setRange(-10000, 10000)
        self._widgets["kb_elevation"].setDecimals(2)
        self._widgets["kb_elevation"].setSuffix(" m")
        self._widgets["kb_elevation"].setValue(
            float(self.obj_data.get("kb_elevation", 0)))
        form.addRow("KB Elevation:", self._widgets["kb_elevation"])

        self._widgets["td"] = QDoubleSpinBox()
        self._widgets["td"].setRange(0, 20000)
        self._widgets["td"].setDecimals(1)
        self._widgets["td"].setSuffix(" m")
        self._widgets["td"].setValue(float(self.obj_data.get("td", 0)))
        form.addRow("Total Depth:", self._widgets["td"])

        return w

    def _make_display_tab(self) -> QWidget:
        w = QWidget(); form = QFormLayout(w)
        form.setVerticalSpacing(6); form.setHorizontalSpacing(10)

        self._widgets["show_deviation"] = QCheckBox()
        self._widgets["show_deviation"].setChecked(
            bool(self.obj_data.get("show_deviation", True)))
        form.addRow("Show Deviation Path:", self._widgets["show_deviation"])

        self._widgets["show_tops"] = QCheckBox()
        self._widgets["show_tops"].setChecked(
            bool(self.obj_data.get("show_tops", True)))
        form.addRow("Show Formation Tops:", self._widgets["show_tops"])

        lbl_fmt_vals = ["name_only", "name_offset", "name_td", "custom"]
        self._widgets["label_format"] = QComboBox()
        self._widgets["label_format"].addItems(
            ["Name only", "Name + offset", "Name + TD", "Custom"])
        cur = self.obj_data.get("label_format", "name_offset")
        if cur in lbl_fmt_vals:
            self._widgets["label_format"].setCurrentIndex(
                lbl_fmt_vals.index(cur))
        form.addRow("Label Format:", self._widgets["label_format"])

        return w

    def _make_logs_tab(self) -> QWidget:
        w = QWidget(); form = QFormLayout(w)
        form.setVerticalSpacing(6); form.setHorizontalSpacing(10)

        available = self.obj_data.get("available_logs", [])
        self._widgets["display_log"] = QComboBox()
        self._widgets["display_log"].addItem("None")
        for log_name in available:
            self._widgets["display_log"].addItem(log_name)
        cur = self.obj_data.get("display_log", "None")
        idx = self._widgets["display_log"].findText(cur)
        if idx >= 0:
            self._widgets["display_log"].setCurrentIndex(idx)
        form.addRow("Display Log:", self._widgets["display_log"])

        self._widgets["log_width_pct"] = QSpinBox()
        self._widgets["log_width_pct"].setRange(1, 20)
        self._widgets["log_width_pct"].setValue(
            int(self.obj_data.get("log_width_pct", 2)))
        self._widgets["log_width_pct"].setSuffix(" % of section")
        form.addRow("Log Track Width:", self._widgets["log_width_pct"])

        return w

    def _make_tops_tab(self) -> QWidget:
        w = QWidget(); form = QFormLayout(w)
        tops = QListWidget()
        for top in self.obj_data.get("formation_tops", []):
            tops.addItem(f"{top.get('name','?')}  @  {top.get('depth', 0):.1f} m")
        tops.setMaximumHeight(200)
        form.addRow(tops)
        return w

    def _collect_values(self) -> dict:
        vals = super()._collect_values()
        lbl_fmt_vals = ["name_only", "name_offset", "name_td", "custom"]
        vals["label_format"] = lbl_fmt_vals[
            self._widgets["label_format"].currentIndex()]
        return vals
