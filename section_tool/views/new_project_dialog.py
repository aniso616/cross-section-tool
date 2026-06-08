"""New Project dialog — name, CRS, units, depth range."""
from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QRadioButton, QVBoxLayout, QWidget,
)


# ---------------------------------------------------------------------------
# Common projected CRS presets  (label, epsg)
# ---------------------------------------------------------------------------

_COMMON_CRS: list[tuple[str, int | None]] = [
    ("Enter EPSG code manually", None),
    # North Sea / Europe
    ("WGS 84 / UTM zone 28N  (EPSG:32628)", 32628),
    ("WGS 84 / UTM zone 29N  (EPSG:32629)", 32629),
    ("WGS 84 / UTM zone 30N  (EPSG:32630)", 32630),
    ("WGS 84 / UTM zone 31N  (EPSG:32631)", 32631),
    ("WGS 84 / UTM zone 32N  (EPSG:32632)", 32632),
    ("WGS 84 / UTM zone 33N  (EPSG:32633)", 32633),
    ("WGS 84 / UTM zone 34N  (EPSG:32634)", 32634),
    ("WGS 84 / UTM zone 35N  (EPSG:32635)", 32635),
    ("WGS 84 / UTM zone 36N  (EPSG:32636)", 32636),
    ("WGS 84 / UTM zone 37N  (EPSG:32637)", 32637),
    ("British National Grid  (EPSG:27700)", 27700),
    ("ED50 / UTM zone 31N  (EPSG:23031)", 23031),
    # North America
    ("NAD83 / UTM zone 10N  (EPSG:26910)", 26910),
    ("NAD83 / UTM zone 11N  (EPSG:26911)", 26911),
    ("NAD83 / UTM zone 12N  (EPSG:26912)", 26912),
    ("NAD83 / UTM zone 13N  (EPSG:26913)", 26913),
    ("NAD83 / UTM zone 14N  (EPSG:26914)", 26914),
    ("NAD83 / UTM zone 15N  (EPSG:26915)", 26915),
    ("NAD83 / UTM zone 16N  (EPSG:26916)", 26916),
    ("NAD83 / UTM zone 17N  (EPSG:26917)", 26917),
    ("NAD83 / UTM zone 18N  (EPSG:26918)", 26918),
    ("NAD83 / UTM zone 19N  (EPSG:26919)", 26919),
    # Australia
    ("GDA2020 / MGA zone 50  (EPSG:7850)", 7850),
    ("GDA2020 / MGA zone 51  (EPSG:7851)", 7851),
    ("GDA2020 / MGA zone 52  (EPSG:7852)", 7852),
    ("GDA2020 / MGA zone 53  (EPSG:7853)", 7853),
    ("GDA2020 / MGA zone 54  (EPSG:7854)", 7854),
    ("GDA2020 / MGA zone 55  (EPSG:7855)", 7855),
    # Geographic — will show warning
    ("WGS 84 geographic  (EPSG:4326)  ⚠", 4326),
]


def _crs_info(epsg: int):
    try:
        from section_tool.core.crs import get_crs_info
        return get_crs_info(epsg)
    except Exception:
        return None


class NewProjectDialog(QDialog):
    """Dialog for creating a new project: name, CRS, units, and depth range."""

    def __init__(self, current_crs: int = 32632, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.setMinimumWidth(460)

        root = QVBoxLayout(self)

        # ── Project identity ───────────────────────────────────────────
        id_grp = QGroupBox("Project")
        id_fl = QFormLayout(id_grp)
        self._name_edit = QLineEdit("Untitled")
        self._name_edit.selectAll()
        id_fl.addRow("Project name:", self._name_edit)

        # Location row
        loc_row = QWidget()
        lr = QHBoxLayout(loc_row)
        lr.setContentsMargins(0, 0, 0, 0)
        default_loc = os.path.join(os.path.expanduser("~"), "Projects", "Untitled")
        self._location_edit = QLineEdit(default_loc)
        lr.addWidget(self._location_edit, 1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_location)
        lr.addWidget(browse_btn)
        id_fl.addRow("Project folder:", loc_row)

        root.addWidget(id_grp)

        # Auto-update location when name changes
        self._name_edit.textChanged.connect(self._sync_location)

        # ── CRS ────────────────────────────────────────────────────────
        crs_grp = QGroupBox("Coordinate Reference System")
        crs_vl = QVBoxLayout(crs_grp)

        # Dropdown of presets
        self._preset_combo = QComboBox()
        for label, _epsg in _COMMON_CRS:
            self._preset_combo.addItem(label)
        crs_vl.addWidget(self._preset_combo)

        # Manual EPSG entry row
        epsg_row = QHBoxLayout()
        epsg_row.addWidget(QLabel("EPSG code:"))
        self._epsg_edit = QLineEdit(str(current_crs))
        self._epsg_edit.setFixedWidth(80)
        epsg_row.addWidget(self._epsg_edit)
        self._crs_name_lbl = QLabel("")
        self._crs_name_lbl.setStyleSheet("font-size: 9pt;")
        epsg_row.addWidget(self._crs_name_lbl, 1)
        crs_vl.addLayout(epsg_row)

        self._crs_warn_lbl = QLabel("")
        self._crs_warn_lbl.setStyleSheet("color: #cc6600; font-size: 8pt;")
        self._crs_warn_lbl.setWordWrap(True)
        crs_vl.addWidget(self._crs_warn_lbl)

        root.addWidget(crs_grp)

        # ── Depth settings ─────────────────────────────────────────────
        depth_grp = QGroupBox("Project Units & Depth")
        depth_fl = QFormLayout(depth_grp)

        # Horizontal units — auto from CRS, read-only
        self._horiz_lbl = QLabel("metres (from CRS)")
        self._horiz_lbl.setStyleSheet("color: grey;")
        depth_fl.addRow("Horizontal units:", self._horiz_lbl)

        # Depth units
        du_row = QWidget()
        du_rl = QHBoxLayout(du_row)
        du_rl.setContentsMargins(0, 0, 0, 0)
        self._depth_units_group = QButtonGroup(self)
        self._depth_units_group.setExclusive(True)
        self._depth_m_rb = QRadioButton("Metres")
        self._depth_ft_rb = QRadioButton("Feet")
        self._depth_m_rb.setChecked(True)
        self._depth_units_group.addButton(self._depth_m_rb, 0)
        self._depth_units_group.addButton(self._depth_ft_rb, 1)
        du_rl.addWidget(self._depth_m_rb)
        du_rl.addWidget(self._depth_ft_rb)
        du_rl.addStretch()
        depth_fl.addRow("Depth units:", du_row)

        # No project-level domain choice: the section is always depth/TVD.
        # Time (TWT) seismic is an *input*, declared at SEG-Y import and converted
        # to depth through the Depth Stretch tool — never a project domain.

        # Default depth range
        self._depth_min_spin = QDoubleSpinBox()
        self._depth_min_spin.setRange(-10000, 0)
        self._depth_min_spin.setDecimals(0)
        self._depth_min_spin.setValue(0)
        depth_fl.addRow("Default depth min (m):", self._depth_min_spin)

        self._depth_max_spin = QDoubleSpinBox()
        self._depth_max_spin.setRange(100, 50000)
        self._depth_max_spin.setDecimals(0)
        self._depth_max_spin.setValue(5000)
        depth_fl.addRow("Default depth max (m):", self._depth_max_spin)

        root.addWidget(depth_grp)

        # ── Buttons ────────────────────────────────────────────────────
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.button(QDialogButtonBox.StandardButton.Ok).setText("Create")
        btns.accepted.connect(self._on_create)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        # Wire signals
        self._epsg_edit.textChanged.connect(self._on_epsg_changed)
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)

        # Initialise CRS display
        self._on_epsg_changed(str(current_crs))

        # Pre-select matching preset if it exists
        self._sync_preset_to_epsg(current_crs)

    # ------------------------------------------------------------------

    def _on_epsg_changed(self, text: str) -> None:
        text = text.strip()
        try:
            epsg = int(text)
            info = _crs_info(epsg)
            if info is None:
                raise ValueError
            self._crs_name_lbl.setStyleSheet("color: green; font-size: 9pt;")
            self._crs_name_lbl.setText(info.name)
            if not info.is_projected:
                self._crs_warn_lbl.setText(
                    "⚠  Geographic CRS — section geometry requires a projected CRS "
                    "(e.g. UTM). Distances will be in degrees, not metres."
                )
                self._horiz_lbl.setText("degrees (geographic)")
            else:
                self._crs_warn_lbl.setText("")
                units = info.linear_units or "metres"
                self._horiz_lbl.setText(units)
        except (ValueError, TypeError):
            self._crs_name_lbl.setStyleSheet("color: red; font-size: 9pt;")
            self._crs_name_lbl.setText("Unknown CRS" if text else "")
            self._crs_warn_lbl.setText("")
            self._horiz_lbl.setText("—")

    def _on_preset_changed(self, index: int) -> None:
        _label, epsg = _COMMON_CRS[index]
        if epsg is None:
            return  # "Enter manually" — leave EPSG field as-is
        self._epsg_edit.setText(str(epsg))

    def _sync_preset_to_epsg(self, epsg: int) -> None:
        for i, (_label, code) in enumerate(_COMMON_CRS):
            if code == epsg:
                self._preset_combo.setCurrentIndex(i)
                return

    def _browse_location(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Choose Project Location",
            os.path.dirname(self._location_edit.text()),
        )
        if folder:
            name = self._name_edit.text().strip() or "Untitled"
            self._location_edit.setText(os.path.join(folder, name))

    def _sync_location(self, name: str) -> None:
        """Keep the folder name in sync with the project name."""
        cur = self._location_edit.text()
        parent = os.path.dirname(cur)
        self._location_edit.setText(os.path.join(parent, name or "Untitled"))

    def _on_create(self) -> None:
        if not self._name_edit.text().strip():
            self._name_edit.setPlaceholderText("Name required")
            self._name_edit.setStyleSheet("border: 1px solid red;")
            return
        if not self._location_edit.text().strip():
            self._location_edit.setStyleSheet("border: 1px solid red;")
            return
        try:
            int(self._epsg_edit.text().strip())
        except ValueError:
            self._epsg_edit.setStyleSheet("border: 1px solid red;")
            return
        self.accept()

    # ------------------------------------------------------------------
    # Result accessors
    # ------------------------------------------------------------------

    def project_name(self) -> str:
        return self._name_edit.text().strip() or "Untitled"

    def crs_epsg(self) -> int:
        try:
            return int(self._epsg_edit.text().strip())
        except ValueError:
            return 32632

    def depth_units(self) -> str:
        return "ft" if self._depth_ft_rb.isChecked() else "m"

    def depth_domain(self) -> str:
        # The project is always depth/TVD (architectural rule: the section is
        # always depth).  Retained so callers need no change.
        return "depth"

    def default_depth_min(self) -> float:
        return self._depth_min_spin.value()

    def default_depth_max(self) -> float:
        return self._depth_max_spin.value()

    def folder_path(self) -> str:
        return self._location_edit.text().strip()
