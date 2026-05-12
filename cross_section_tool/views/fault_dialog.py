"""Dialog for creating / editing fault attributes (Phase B)."""
from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup, QColorDialog, QComboBox, QDialog, QDialogButtonBox,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QRadioButton, QVBoxLayout, QWidget,
)

FAULT_TYPES = [
    "normal", "reverse", "thrust", "strike_slip", "detachment", "growth_fault",
]


class FaultDialog(QDialog):
    """Create or edit a fault line."""

    def __init__(self, parent=None, *,
                 name: str = "",
                 fault_type: str = "normal",
                 color: str = "#d62728",
                 dip_direction: str = "right") -> None:
        super().__init__(parent)
        self.setWindowTitle("Fault Properties")
        self.setMinimumWidth(320)
        self._color = color
        self._setup_ui(name, fault_type, color, dip_direction)

    def _setup_ui(self, name, ft, color, dd) -> None:
        form = QFormLayout()

        self._name_edit = QLineEdit(name)
        form.addRow("Name:", self._name_edit)

        self._type_combo = QComboBox()
        for ft_val in FAULT_TYPES:
            self._type_combo.addItem(ft_val.replace("_", " ").title(), ft_val)
        idx = FAULT_TYPES.index(ft) if ft in FAULT_TYPES else 0
        self._type_combo.setCurrentIndex(idx)
        form.addRow("Fault type:", self._type_combo)

        # Dip direction radio buttons
        dip_widget = QWidget()
        dip_hb = QHBoxLayout(dip_widget)
        dip_hb.setContentsMargins(0, 0, 0, 0)
        self._dd_left  = QRadioButton("Left  (hanging wall left)")
        self._dd_right = QRadioButton("Right (hanging wall right)")
        self._dd_right.setChecked(dd == "right")
        self._dd_left.setChecked(dd == "left")
        bg = QButtonGroup(self)
        bg.addButton(self._dd_left)
        bg.addButton(self._dd_right)
        dip_hb.addWidget(self._dd_left)
        dip_hb.addWidget(self._dd_right)
        form.addRow("Dip direction:", dip_widget)

        color_widget = self._make_color_row(color)
        form.addRow("Color:", color_widget)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)

        vbox = QVBoxLayout(self)
        vbox.addLayout(form)
        vbox.addWidget(bb)

    def _make_color_row(self, color: str):
        w = QWidget()
        hb = QHBoxLayout(w)
        hb.setContentsMargins(0, 0, 0, 0)
        self._color_swatch = QLabel()
        self._color_swatch.setFixedSize(24, 16)
        self._apply_swatch(color)
        hb.addWidget(self._color_swatch)
        btn = QPushButton("Choose…")
        btn.clicked.connect(self._pick_color)
        hb.addWidget(btn)
        hb.addStretch()
        return w

    def _apply_swatch(self, color: str) -> None:
        self._color_swatch.setStyleSheet(
            f"background:{color}; border:1px solid #888;")

    def _pick_color(self) -> None:
        c = QColorDialog.getColor(QColor(self._color), self)
        if c.isValid():
            self._color = c.name()
            self._apply_swatch(self._color)

    @property
    def name(self) -> str:
        return self._name_edit.text().strip()

    @property
    def fault_type(self) -> str:
        return self._type_combo.currentData()

    @property
    def color(self) -> str:
        return self._color

    @property
    def dip_direction(self) -> str:
        return "right" if self._dd_right.isChecked() else "left"
