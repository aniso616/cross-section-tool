"""Dialog for creating / editing horizon / contact attributes (Phase A)."""
from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog, QComboBox, QDialog, QDialogButtonBox,
    QFormLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout,
)

CONTACT_TYPES = [
    "conformable", "unconformity", "angular_unconformity", "disconformity",
    "intrusive_contact", "marker_bed", "sequence_boundary",
    "maximum_flooding_surface",
]


class HorizonDialog(QDialog):
    """Create or edit a horizon / contact line."""

    def __init__(self, parent=None, *,
                 name: str = "",
                 contact_type: str = "conformable",
                 color: str = "#2ca02c",
                 formation_above: str = "",
                 formation_below: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Horizon / Contact Properties")
        self.setMinimumWidth(340)
        self._color = color
        self._setup_ui(name, contact_type, color, formation_above, formation_below)

    def _setup_ui(self, name, ct, color, fa, fb) -> None:
        form = QFormLayout()

        self._name_edit = QLineEdit(name)
        form.addRow("Name:", self._name_edit)

        self._type_combo = QComboBox()
        for ct_val in CONTACT_TYPES:
            self._type_combo.addItem(ct_val.replace("_", " ").title(), ct_val)
        idx = CONTACT_TYPES.index(ct) if ct in CONTACT_TYPES else 0
        self._type_combo.setCurrentIndex(idx)
        form.addRow("Contact type:", self._type_combo)

        color_row_widget = self._make_color_row(color)
        form.addRow("Color:", color_row_widget)

        self._fa_edit = QLineEdit(fa)
        form.addRow("Formation above:", self._fa_edit)

        self._fb_edit = QLineEdit(fb)
        form.addRow("Formation below:", self._fb_edit)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)

        vbox = QVBoxLayout(self)
        vbox.addLayout(form)
        vbox.addWidget(bb)

    def _make_color_row(self, color: str):
        from PySide6.QtWidgets import QWidget, QHBoxLayout
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

    # ------------------------------------------------------------------
    # Result accessors
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name_edit.text().strip()

    @property
    def contact_type(self) -> str:
        return self._type_combo.currentData()

    @property
    def color(self) -> str:
        return self._color

    @property
    def formation_above(self) -> str:
        return self._fa_edit.text().strip()

    @property
    def formation_below(self) -> str:
        return self._fb_edit.text().strip()
