"""Polygon properties dialog — geology, fill, label."""
from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox,
    QFormLayout, QHBoxLayout, QLineEdit, QPushButton, QSpinBox, QWidget,
)

from .properties_dialog import PropertiesDialog


_LITHOLOGIES = [
    "unknown", "sandstone", "shale", "limestone", "dolomite",
    "conglomerate", "siltstone", "mudstone", "coal", "salt",
    "anhydrite", "gypsum", "chert", "marl", "volcanic",
    "igneous", "metamorphic", "basement",
]
_FILL_PATTERNS = ["solid", "none", "hatched", "crosshatch", "dotted"]


class PolygonPropertiesDialog(PropertiesDialog):

    def _build_tabs(self) -> None:
        self.tabs.addTab(self._make_general_tab(), "General")
        self.tabs.addTab(self._make_geology_tab(), "Geology")
        self.tabs.addTab(self._make_fill_tab(), "Fill")
        self.tabs.addTab(self._make_label_tab(), "Label")

    def _make_geology_tab(self) -> QWidget:
        w = QWidget(); form = QFormLayout(w)
        form.setVerticalSpacing(6); form.setHorizontalSpacing(10)

        self._widgets["formation_name"] = QLineEdit(
            self.obj_data.get("formation_name", ""))
        form.addRow("Formation:", self._widgets["formation_name"])

        self._widgets["lithology"] = QComboBox()
        self._widgets["lithology"].addItems(
            [l.replace("_", " ").title() for l in _LITHOLOGIES])
        cur = self.obj_data.get("lithology", "unknown")
        if cur in _LITHOLOGIES:
            self._widgets["lithology"].setCurrentIndex(_LITHOLOGIES.index(cur))
        form.addRow("Lithology:", self._widgets["lithology"])

        self._widgets["age_ma"] = QDoubleSpinBox()
        self._widgets["age_ma"].setRange(0, 4600)
        self._widgets["age_ma"].setDecimals(2)
        self._widgets["age_ma"].setSuffix(" Ma")
        self._widgets["age_ma"].setValue(float(self.obj_data.get("age_ma") or 0))
        form.addRow("Age:", self._widgets["age_ma"])

        return w

    def _make_fill_tab(self) -> QWidget:
        w = QWidget(); form = QFormLayout(w)
        form.setVerticalSpacing(6); form.setHorizontalSpacing(10)

        # Separate fill colour button
        self._fill_color = self.obj_data.get("fill_color", self._color_value)
        if isinstance(self._fill_color, tuple):
            r, g, b = self._fill_color
            self._fill_color = f"#{r:02x}{g:02x}{b:02x}"
        self._fill_btn = QPushButton()
        self._fill_btn.setFixedSize(40, 22)
        self._fill_btn.setStyleSheet(
            f"background-color:{self._fill_color}; border:1px solid #555;")
        self._fill_btn.clicked.connect(self._pick_fill_color)
        form.addRow("Fill Color:", self._fill_btn)

        self._widgets["fill_opacity"] = QDoubleSpinBox()
        self._widgets["fill_opacity"].setRange(0.0, 1.0)
        self._widgets["fill_opacity"].setSingleStep(0.05)
        self._widgets["fill_opacity"].setDecimals(2)
        self._widgets["fill_opacity"].setValue(
            float(self.obj_data.get("fill_opacity", 0.6)))
        form.addRow("Fill Opacity:", self._widgets["fill_opacity"])

        self._widgets["fill_pattern"] = QComboBox()
        self._widgets["fill_pattern"].addItems(
            [p.replace("_", " ").title() for p in _FILL_PATTERNS])
        cur = self.obj_data.get("fill_pattern", "solid")
        if cur in _FILL_PATTERNS:
            self._widgets["fill_pattern"].setCurrentIndex(
                _FILL_PATTERNS.index(cur))
        form.addRow("Fill Pattern:", self._widgets["fill_pattern"])

        self._widgets["hatch_angle"] = QSpinBox()
        self._widgets["hatch_angle"].setRange(0, 180)
        self._widgets["hatch_angle"].setValue(
            int(self.obj_data.get("hatch_angle", 45)))
        self._widgets["hatch_angle"].setSuffix("°")
        form.addRow("Hatch Angle:", self._widgets["hatch_angle"])

        return w

    def _pick_fill_color(self) -> None:
        qc = QColorDialog.getColor(QColor(self._fill_color), self, "Fill Colour")
        if qc.isValid():
            self._fill_color = qc.name()
            self._fill_btn.setStyleSheet(
                f"background-color:{self._fill_color}; border:1px solid #555;")

    def _make_label_tab(self) -> QWidget:
        w = QWidget(); form = QFormLayout(w)
        form.setVerticalSpacing(6); form.setHorizontalSpacing(10)

        self._widgets["show_label"] = QCheckBox()
        self._widgets["show_label"].setChecked(
            bool(self.obj_data.get("show_label", True)))
        form.addRow("Show Label:", self._widgets["show_label"])

        self._widgets["label_text_source"] = QComboBox()
        self._widgets["label_text_source"].addItems(
            ["Formation Name", "Lithology", "Age", "Custom"])
        ls = self.obj_data.get("label_text_source", "Formation Name")
        idx = self._widgets["label_text_source"].findText(ls)
        if idx >= 0:
            self._widgets["label_text_source"].setCurrentIndex(idx)
        form.addRow("Label Text:", self._widgets["label_text_source"])

        self._widgets["custom_label"] = QLineEdit(
            self.obj_data.get("custom_label", ""))
        form.addRow("Custom Label:", self._widgets["custom_label"])

        return w

    def _collect_values(self) -> dict:
        vals = super()._collect_values()
        vals["fill_color"] = self._fill_color
        vals["lithology"] = _LITHOLOGIES[
            self._widgets["lithology"].currentIndex()]
        vals["fill_pattern"] = _FILL_PATTERNS[
            self._widgets["fill_pattern"].currentIndex()]
        return vals
