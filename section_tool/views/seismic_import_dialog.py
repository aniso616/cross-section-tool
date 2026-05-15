"""SEG-Y import dialog — section, header fields, domain, scalar."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox,
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

_DOMAINS = [("TWT (ms)", "twt"), ("Depth (m)", "depth"), ("Depth (ft)", "depth")]
_DEPTH_UNITS = [("ms", "ms"), ("m", "m"), ("ft", "ft")]


class SeismicImportDialog(QDialog):
    """Dialog shown after the user selects a SEG-Y file.

    Lets the user choose which section to drape on, the X/Y header
    bytes, depth domain, and whether to apply the coordinate scalar.
    """

    def __init__(self, sections: list, filename: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Import SEG-Y: {filename}")
        self.setMinimumWidth(380)
        self._sections = sections

        layout = QVBoxLayout(self)

        # Section selection
        sec_grp = QGroupBox("Drape on section")
        sf = QFormLayout(sec_grp)
        self._section_combo = QComboBox()
        self._section_combo.addItem("(all / auto-detect)", None)
        for sec in sections:
            self._section_combo.addItem(sec.name or "Unnamed", sec)
        sf.addRow("Section:", self._section_combo)
        layout.addWidget(sec_grp)

        # Coordinate headers
        coord_grp = QGroupBox("Coordinate trace headers")
        cf = QFormLayout(coord_grp)
        self._x_combo = QComboBox()
        self._y_combo = QComboBox()
        for label, byte in _COORD_FIELDS:
            self._x_combo.addItem(label, byte)
            self._y_combo.addItem(label, byte)
        self._x_combo.setCurrentIndex(0)   # CDP_X
        self._y_combo.setCurrentIndex(1)   # CDP_Y
        cf.addRow("X field:", self._x_combo)
        cf.addRow("Y field:", self._y_combo)
        self._scalar_check = QCheckBox("Apply coordinate scalar (auto-detect from header)")
        self._scalar_check.setChecked(True)
        cf.addRow("", self._scalar_check)
        layout.addWidget(coord_grp)

        # Domain
        dom_grp = QGroupBox("Depth domain")
        df = QFormLayout(dom_grp)
        self._domain_combo = QComboBox()
        for label, val in _DOMAINS:
            self._domain_combo.addItem(label, val)
        df.addRow("Domain:", self._domain_combo)
        self._units_combo = QComboBox()
        for label, val in _DEPTH_UNITS:
            self._units_combo.addItem(label, val)
        df.addRow("Sample units:", self._units_combo)
        layout.addWidget(dom_grp)

        # Note
        note = QLabel(
            "<i>Traces farther than 500 m perpendicular to the section will be excluded.</i>"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #888; font-size: 8pt;")
        layout.addWidget(note)

        # Buttons
        bbox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        layout.addWidget(bbox)

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
