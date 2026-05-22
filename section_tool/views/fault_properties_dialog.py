"""Fault properties dialog — kinematics and display options."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QLineEdit, QSpinBox, QWidget,
)

from .properties_dialog import PropertiesDialog


_FAULT_TYPES = [
    "normal", "reverse", "thrust", "strike_slip",
    "oblique_normal", "oblique_reverse", "detachment",
    "listric", "growth_fault", "unknown",
]


class FaultPropertiesDialog(PropertiesDialog):

    def _build_tabs(self) -> None:
        self.tabs.addTab(self._make_general_tab(), "General")
        self.tabs.addTab(self._make_kinematics_tab(), "Kinematics")
        self.tabs.addTab(self._make_display_tab(), "Display")

    def _make_kinematics_tab(self) -> QWidget:
        w = QWidget(); form = QFormLayout(w)
        form.setVerticalSpacing(6); form.setHorizontalSpacing(10)

        self._widgets["fault_type"] = QComboBox()
        self._widgets["fault_type"].addItems(
            [ft.replace("_", " ").title() for ft in _FAULT_TYPES])
        cur = self.obj_data.get("fault_type", "normal")
        if cur in _FAULT_TYPES:
            self._widgets["fault_type"].setCurrentIndex(_FAULT_TYPES.index(cur))
        form.addRow("Fault Type:", self._widgets["fault_type"])

        self._widgets["dip_direction"] = QComboBox()
        self._widgets["dip_direction"].addItems(["Right", "Left"])
        dd = self.obj_data.get("dip_direction", "right")
        self._widgets["dip_direction"].setCurrentIndex(0 if dd == "right" else 1)
        form.addRow("Dip Direction:", self._widgets["dip_direction"])

        self._widgets["displacement"] = QDoubleSpinBox()
        self._widgets["displacement"].setRange(0, 100_000)
        self._widgets["displacement"].setDecimals(1)
        self._widgets["displacement"].setSuffix(" m")
        self._widgets["displacement"].setValue(
            float(self.obj_data.get("displacement") or 0))
        form.addRow("Displacement:", self._widgets["displacement"])

        conf_vals = ["high", "medium", "low", "speculative"]
        self._widgets["confidence_level"] = QComboBox()
        self._widgets["confidence_level"].addItems([v.title() for v in conf_vals])
        cur_conf = str(self.obj_data.get("confidence_level", "high")).lower()
        if cur_conf in conf_vals:
            self._widgets["confidence_level"].setCurrentIndex(
                conf_vals.index(cur_conf))
        form.addRow("Confidence:", self._widgets["confidence_level"])

        return w

    def _make_display_tab(self) -> QWidget:
        w = QWidget(); form = QFormLayout(w)
        form.setVerticalSpacing(6); form.setHorizontalSpacing(10)

        self._widgets["show_ticks"] = QCheckBox()
        self._widgets["show_ticks"].setChecked(
            bool(self.obj_data.get("show_ticks", True)))
        form.addRow("Show Hanging-Wall Ticks:", self._widgets["show_ticks"])

        self._widgets["tick_spacing"] = QSpinBox()
        self._widgets["tick_spacing"].setRange(20, 500)
        self._widgets["tick_spacing"].setValue(
            int(self.obj_data.get("tick_spacing", 80)))
        self._widgets["tick_spacing"].setSuffix(" px")
        form.addRow("Tick Spacing:", self._widgets["tick_spacing"])

        self._widgets["tick_length"] = QSpinBox()
        self._widgets["tick_length"].setRange(5, 50)
        self._widgets["tick_length"].setValue(
            int(self.obj_data.get("tick_length", 15)))
        self._widgets["tick_length"].setSuffix(" px")
        form.addRow("Tick Length:", self._widgets["tick_length"])

        self._widgets["show_label"] = QCheckBox()
        self._widgets["show_label"].setChecked(
            bool(self.obj_data.get("show_label", False)))
        form.addRow("Show Name Label:", self._widgets["show_label"])

        return w

    def _collect_values(self) -> dict:
        vals = super()._collect_values()
        ft_idx = self._widgets["fault_type"].currentIndex()
        vals["fault_type"] = _FAULT_TYPES[ft_idx]
        vals["dip_direction"] = (
            "right" if self._widgets["dip_direction"].currentIndex() == 0 else "left")
        conf_idx = self._widgets["confidence_level"].currentIndex()
        vals["confidence_level"] = ["high", "medium", "low", "speculative"][conf_idx]
        return vals
