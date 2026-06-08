"""SEG-Y import dialog — geometry, vertical domain, scalar.

The import is the input-domain *boundary*: because the project (and every
section) is always depth, the only place a volume can be declared time is here.
The dialog separates the **geometry** block (target section, X/Y headers,
coordinate scalar, corridor width) from the **vertical-domain** declaration
(two-way time vs depth, with the sample units that match).
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QGroupBox, QLabel, QVBoxLayout,
)

# segyio TraceField byte positions for common coordinate headers
_COORD_FIELDS = [
    ("CDP_X  (181)", 181),
    ("CDP_Y  (185)", 185),
    ("SourceX (73)", 73),
    ("SourceY (77)", 77),
    ("GroupX  (81)", 81),
    ("GroupY  (85)", 85),
    ("ReceiverGroupElevation (41)", 41),
]


class SeismicImportDialog(QDialog):
    """Dialog shown after the user selects a SEG-Y file.

    Declares the volume's geometry (how its traces relate to a section) and its
    vertical domain (time vs depth).  A time volume cannot display directly in a
    depth section — it is converted through the Depth Stretch tool.
    """

    def __init__(self, sections: list, filename: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Import SEG-Y: {filename}")
        self.setMinimumWidth(400)
        self._sections = sections

        layout = QVBoxLayout(self)

        # ── Geometry: how the traces relate to a section ───────────────
        geo_grp = QGroupBox("Geometry")
        gf = QFormLayout(geo_grp)

        self._section_combo = QComboBox()
        self._section_combo.addItem("All sections (auto-detect nearest)", None)
        for sec in sections:
            self._section_combo.addItem(sec.name or "Unnamed", sec)
        gf.addRow("Target section:", self._section_combo)

        self._x_combo = QComboBox()
        self._y_combo = QComboBox()
        for label, byte in _COORD_FIELDS:
            self._x_combo.addItem(label, byte)
            self._y_combo.addItem(label, byte)
        self._x_combo.setCurrentIndex(0)   # CDP_X
        self._y_combo.setCurrentIndex(1)   # CDP_Y
        gf.addRow("X header:", self._x_combo)
        gf.addRow("Y header:", self._y_combo)

        self._scalar_check = QCheckBox("Apply coordinate scalar (auto-detect from header)")
        self._scalar_check.setChecked(True)
        gf.addRow("", self._scalar_check)

        self._max_offset_spin = QDoubleSpinBox()
        self._max_offset_spin.setRange(0.0, 100000.0)
        self._max_offset_spin.setDecimals(0)
        self._max_offset_spin.setSingleStep(50.0)
        self._max_offset_spin.setValue(500.0)
        self._max_offset_spin.setSuffix(" m")
        gf.addRow("Max distance from section:", self._max_offset_spin)

        layout.addWidget(geo_grp)

        # ── Vertical domain: the input-domain boundary ─────────────────
        dom_grp = QGroupBox("Vertical domain")
        df = QFormLayout(dom_grp)
        self._domain_combo = QComboBox()
        self._domain_combo.addItem("Two-way time (TWT)", "twt")
        self._domain_combo.addItem("Depth", "depth")
        df.addRow("Domain:", self._domain_combo)

        self._units_combo = QComboBox()
        df.addRow("Sample units:", self._units_combo)

        self._domain_note = QLabel()
        self._domain_note.setWordWrap(True)
        self._domain_note.setStyleSheet("color: #888; font-size: 8pt;")
        df.addRow("", self._domain_note)
        layout.addWidget(dom_grp)

        self._domain_combo.currentIndexChanged.connect(self._on_domain_changed)
        self._on_domain_changed()   # populate units + note for the default (TWT)

        # Buttons
        bbox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        layout.addWidget(bbox)

    # ------------------------------------------------------------------

    def _on_domain_changed(self) -> None:
        """Sample units (and the explanatory note) follow the chosen domain:
        ms/s for two-way time, m/ft for depth."""
        domain = self._domain_combo.currentData()
        self._units_combo.clear()
        if domain == "twt":
            self._units_combo.addItem("ms", "ms")
            self._units_combo.addItem("s", "s")
            self._domain_note.setText(
                "Time volume — it cannot display directly in a depth section. "
                "Convert it to depth with the Depth Stretch tool after extraction."
            )
        else:
            self._units_combo.addItem("m", "m")
            self._units_combo.addItem("ft", "ft")
            self._domain_note.setText(
                "Depth volume — displays directly in the depth section."
            )

    # ------------------------------------------------------------------
    # Result properties
    # ------------------------------------------------------------------

    @property
    def x_field(self) -> int:
        return int(self._x_combo.currentData())

    @property
    def y_field(self) -> int:
        return int(self._y_combo.currentData())

    @property
    def apply_scalar(self) -> bool:
        return self._scalar_check.isChecked()

    @property
    def domain(self) -> str:
        return str(self._domain_combo.currentData())

    @property
    def depth_units(self) -> str:
        return str(self._units_combo.currentData())

    @property
    def max_offset(self) -> float:
        return float(self._max_offset_spin.value())

    @property
    def target_section(self):
        """The chosen section, or None for auto-detect."""
        return self._section_combo.currentData()
