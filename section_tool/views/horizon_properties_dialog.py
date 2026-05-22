"""Horizon properties dialog — stratigraphy, pick display, snapping."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QLineEdit, QSpinBox, QWidget,
)

from .properties_dialog import PropertiesDialog


_CONTACT_TYPES = [
    "conformable", "unconformity", "disconformity",
    "angular_unconformity", "nonconformity",
    "fault_contact", "intrusive_contact", "marker_bed",
]
_CONTACT_LABELS = [ct.replace("_", " ").title() for ct in _CONTACT_TYPES]


class HorizonPropertiesDialog(PropertiesDialog):

    def _build_tabs(self) -> None:
        self.tabs.addTab(self._make_general_tab(), "General")
        self.tabs.addTab(self._make_stratigraphy_tab(), "Stratigraphy")
        self.tabs.addTab(self._make_picks_tab(), "Picks")

    def _make_stratigraphy_tab(self) -> QWidget:
        w = QWidget(); form = QFormLayout(w)
        form.setVerticalSpacing(6); form.setHorizontalSpacing(10)

        self._widgets["contact_type"] = QComboBox()
        for lbl in _CONTACT_LABELS:
            self._widgets["contact_type"].addItem(lbl)
        cur = self.obj_data.get("contact_type", "conformable")
        for i, ct in enumerate(_CONTACT_TYPES):
            if ct == cur:
                self._widgets["contact_type"].setCurrentIndex(i); break
        form.addRow("Contact Type:", self._widgets["contact_type"])

        self._widgets["formation_above"] = QLineEdit(
            self.obj_data.get("formation_above", ""))
        form.addRow("Formation Above:", self._widgets["formation_above"])

        self._widgets["formation_below"] = QLineEdit(
            self.obj_data.get("formation_below", ""))
        form.addRow("Formation Below:", self._widgets["formation_below"])

        self._widgets["age_ma"] = QDoubleSpinBox()
        self._widgets["age_ma"].setRange(0, 4600)
        self._widgets["age_ma"].setDecimals(2)
        self._widgets["age_ma"].setSuffix(" Ma")
        self._widgets["age_ma"].setValue(float(self.obj_data.get("age_ma") or 0))
        form.addRow("Age:", self._widgets["age_ma"])

        conf_vals = ["high", "medium", "low", "speculative"]
        self._widgets["confidence_level"] = QComboBox()
        self._widgets["confidence_level"].addItems([v.title() for v in conf_vals])
        cur_conf = str(self.obj_data.get("confidence_level", "high")).lower()
        if cur_conf in conf_vals:
            self._widgets["confidence_level"].setCurrentIndex(conf_vals.index(cur_conf))
        form.addRow("Confidence:", self._widgets["confidence_level"])

        return w

    def _make_picks_tab(self) -> QWidget:
        w = QWidget(); form = QFormLayout(w)
        form.setVerticalSpacing(6); form.setHorizontalSpacing(10)

        self._widgets["show_picks"] = QCheckBox()
        self._widgets["show_picks"].setChecked(
            bool(self.obj_data.get("show_picks", True)))
        form.addRow("Show Pick Points:", self._widgets["show_picks"])

        self._widgets["pick_size"] = QSpinBox()
        self._widgets["pick_size"].setRange(2, 30)
        self._widgets["pick_size"].setValue(int(self.obj_data.get("pick_size", 7)))
        self._widgets["pick_size"].setSuffix(" px")
        form.addRow("Pick Symbol Size:", self._widgets["pick_size"])

        self._widgets["show_label"] = QCheckBox()
        self._widgets["show_label"].setChecked(
            bool(self.obj_data.get("show_label", True)))
        form.addRow("Show Name Label:", self._widgets["show_label"])

        return w

    def _collect_values(self) -> dict:
        vals = super()._collect_values()
        # Map combo display text back to stored value
        ct_idx = self._widgets["contact_type"].currentIndex()
        vals["contact_type"] = _CONTACT_TYPES[ct_idx]
        conf_idx = self._widgets["confidence_level"].currentIndex()
        vals["confidence_level"] = ["high", "medium", "low", "speculative"][conf_idx]
        return vals
