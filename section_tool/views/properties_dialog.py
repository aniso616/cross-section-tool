"""Properties dialog base class — tabbed form for any project object."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSpinBox, QComboBox, QTabWidget,
    QTextEdit, QVBoxLayout, QWidget,
)


_STYLE_OPTIONS = [("Solid", "solid"), ("Dashed", "dashed"),
                  ("Dotted", "dotted"), ("Dash-Dot", "dashdot")]


class PropertiesDialog(QDialog):
    """Generic properties dialog — subclasses add type-specific tabs."""

    properties_changed = Signal(dict)

    def __init__(self, obj_type: str, obj_data: dict, parent=None) -> None:
        super().__init__(parent)
        self.obj_type = obj_type
        self.obj_data = dict(obj_data)
        self._widgets: dict[str, object] = {}
        self._color_value: str = obj_data.get("color", "#888888")
        if isinstance(self._color_value, tuple):
            r, g, b = self._color_value
            self._color_value = f"#{r:02x}{g:02x}{b:02x}"

        self.setWindowTitle(f"{obj_type.replace('_', ' ').title()} Properties")
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self._build_tabs()

        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply
        )
        bb.accepted.connect(self._on_ok)
        bb.rejected.connect(self.reject)
        bb.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._on_apply)
        layout.addWidget(bb)

    def _build_tabs(self) -> None:
        self.tabs.addTab(self._make_general_tab(), "General")

    # ------------------------------------------------------------------
    # General tab
    # ------------------------------------------------------------------

    def _make_general_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setVerticalSpacing(6)
        form.setHorizontalSpacing(10)

        # Name
        self._widgets["name"] = QLineEdit(self.obj_data.get("name", ""))
        form.addRow("Name:", self._widgets["name"])

        # Visible
        self._widgets["visible"] = QCheckBox()
        self._widgets["visible"].setChecked(bool(self.obj_data.get("visible", True)))
        form.addRow("Visible:", self._widgets["visible"])

        # Color
        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(40, 22)
        self._refresh_color_btn()
        self._color_btn.clicked.connect(self._pick_color)
        form.addRow("Color:", self._color_btn)

        # Opacity
        self._widgets["opacity"] = QDoubleSpinBox()
        self._widgets["opacity"].setRange(0.0, 1.0)
        self._widgets["opacity"].setSingleStep(0.05)
        self._widgets["opacity"].setDecimals(2)
        self._widgets["opacity"].setValue(float(self.obj_data.get("opacity", 1.0)))
        form.addRow("Opacity:", self._widgets["opacity"])

        # Line style
        self._widgets["line_style"] = QComboBox()
        for label, val in _STYLE_OPTIONS:
            self._widgets["line_style"].addItem(label, val)
        cur_style = self.obj_data.get("line_style", "solid")
        for i, (_, val) in enumerate(_STYLE_OPTIONS):
            if val == cur_style:
                self._widgets["line_style"].setCurrentIndex(i)
                break
        form.addRow("Line Style:", self._widgets["line_style"])

        # Line width
        self._widgets["line_width"] = QDoubleSpinBox()
        self._widgets["line_width"].setRange(0.1, 10.0)
        self._widgets["line_width"].setSingleStep(0.5)
        self._widgets["line_width"].setDecimals(1)
        self._widgets["line_width"].setValue(float(self.obj_data.get("line_width", 1.5)))
        form.addRow("Line Width:", self._widgets["line_width"])

        # Notes
        self._widgets["notes"] = QTextEdit()
        self._widgets["notes"].setPlainText(self.obj_data.get("notes", ""))
        self._widgets["notes"].setMaximumHeight(70)
        form.addRow("Notes:", self._widgets["notes"])

        return w

    # ------------------------------------------------------------------
    # Color helpers
    # ------------------------------------------------------------------

    def _refresh_color_btn(self) -> None:
        self._color_btn.setStyleSheet(
            f"background-color:{self._color_value}; border:1px solid #555;"
        )

    def _pick_color(self) -> None:
        qc = QColorDialog.getColor(QColor(self._color_value), self, "Choose Colour")
        if qc.isValid():
            self._color_value = qc.name()
            self._refresh_color_btn()

    # ------------------------------------------------------------------
    # Value collection
    # ------------------------------------------------------------------

    def _collect_values(self) -> dict:
        result = {"color": self._color_value}
        for key, widget in self._widgets.items():
            if isinstance(widget, QLineEdit):
                result[key] = widget.text().strip()
            elif isinstance(widget, QCheckBox):
                result[key] = widget.isChecked()
            elif isinstance(widget, QSpinBox):
                result[key] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                result[key] = widget.value()
            elif isinstance(widget, QComboBox):
                if key == "line_style":
                    result[key] = widget.currentData()
                else:
                    result[key] = widget.currentText()
            elif isinstance(widget, QTextEdit):
                result[key] = widget.toPlainText()
        return result

    def _on_apply(self) -> None:
        self.properties_changed.emit(self._collect_values())

    def _on_ok(self) -> None:
        self._on_apply()
        self.accept()
