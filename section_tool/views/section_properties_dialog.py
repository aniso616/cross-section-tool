"""Section properties dialog — geometry (read-only) and display settings."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QLabel, QWidget,
)

from .properties_dialog import PropertiesDialog


class SectionPropertiesDialog(PropertiesDialog):

    def _build_tabs(self) -> None:
        self.tabs.addTab(self._make_general_tab(), "General")
        self.tabs.addTab(self._make_geometry_tab(), "Geometry")
        self.tabs.addTab(self._make_display_tab(), "Display")

    def _make_geometry_tab(self) -> QWidget:
        w = QWidget(); form = QFormLayout(w)
        form.setVerticalSpacing(6); form.setHorizontalSpacing(10)

        # Read-only computed fields
        form.addRow("Length:", QLabel(
            f"{float(self.obj_data.get('total_length', 0)):.1f} m"))
        form.addRow("Azimuth:", QLabel(
            f"{float(self.obj_data.get('azimuth', 0)):.1f}°"))
        form.addRow("Nodes:", QLabel(str(self.obj_data.get("n_nodes", 2))))
        form.addRow("CRS (EPSG):", QLabel(
            str(self.obj_data.get("crs_epsg", ""))))

        return w

    def _make_display_tab(self) -> QWidget:
        w = QWidget(); form = QFormLayout(w)
        form.setVerticalSpacing(6); form.setHorizontalSpacing(10)

        self._widgets["display_domain"] = QComboBox()
        self._widgets["display_domain"].addItems(["depth", "twt"])
        dd = self.obj_data.get("display_domain", "depth")
        self._widgets["display_domain"].setCurrentIndex(
            0 if dd == "depth" else 1)
        form.addRow("Display Domain:", self._widgets["display_domain"])

        self._widgets["vertical_exaggeration"] = QDoubleSpinBox()
        self._widgets["vertical_exaggeration"].setRange(0.1, 100.0)
        self._widgets["vertical_exaggeration"].setDecimals(2)
        self._widgets["vertical_exaggeration"].setValue(
            float(self.obj_data.get("vertical_exaggeration", 1.0)))
        form.addRow("Vertical Exaggeration:", self._widgets["vertical_exaggeration"])

        self._widgets["depth_units"] = QComboBox()
        self._widgets["depth_units"].addItems(["m", "ft", "km", "ms"])
        du = self.obj_data.get("depth_units", "m")
        idx = self._widgets["depth_units"].findText(du)
        if idx >= 0:
            self._widgets["depth_units"].setCurrentIndex(idx)
        form.addRow("Depth Units:", self._widgets["depth_units"])

        return w
