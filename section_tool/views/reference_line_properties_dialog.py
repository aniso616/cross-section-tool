"""Reference line properties dialog."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QLabel, QLineEdit, QWidget,
)

from .properties_dialog import PropertiesDialog


class ReferenceLinePropertiesDialog(PropertiesDialog):

    def _build_tabs(self) -> None:
        self.tabs.addTab(self._make_general_tab(), "General")
        self.tabs.addTab(self._make_value_tab(), "Value")
        self.tabs.addTab(self._make_label_tab(), "Label")

    def _make_value_tab(self) -> QWidget:
        w = QWidget(); form = QFormLayout(w)
        form.setVerticalSpacing(6); form.setHorizontalSpacing(10)

        form.addRow("Line Type:",
                    QLabel(self.obj_data.get("kind", "horizontal").title()))

        self._widgets["value"] = QDoubleSpinBox()
        self._widgets["value"].setRange(-1e9, 1e9)
        self._widgets["value"].setDecimals(2)
        kind = self.obj_data.get("kind", "horizontal")
        self._widgets["value"].setSuffix(" m" if kind == "horizontal" else "")
        self._widgets["value"].setValue(float(self.obj_data.get("value", 0)))
        form.addRow("Value:", self._widgets["value"])

        return w

    def _make_label_tab(self) -> QWidget:
        w = QWidget(); form = QFormLayout(w)
        form.setVerticalSpacing(6); form.setHorizontalSpacing(10)

        self._widgets["custom_label"] = QLineEdit(
            self.obj_data.get("custom_label", ""))
        form.addRow("Label Text:", self._widgets["custom_label"])

        lp_vals = ["start", "middle", "end"]
        self._widgets["label_position"] = QComboBox()
        self._widgets["label_position"].addItems(["Start", "Middle", "End"])
        cur = self.obj_data.get("label_position", "end")
        if cur in lp_vals:
            self._widgets["label_position"].setCurrentIndex(
                lp_vals.index(cur))
        form.addRow("Label Position:", self._widgets["label_position"])

        return w

    def _collect_values(self) -> dict:
        vals = super()._collect_values()
        lp_vals = ["start", "middle", "end"]
        vals["label_position"] = lp_vals[
            self._widgets["label_position"].currentIndex()]
        return vals
