"""Project Properties dialog — edit name, units, domain and depth range.

CRS is set at project creation and is shown read-only.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)


class ProjectPropertiesDialog(QDialog):
    """Dialog for viewing/editing project-level properties.

    Parameters
    ----------
    state:
        The application :class:`~section_tool.app_state.AppState` instance.
    parent:
        Optional Qt parent widget.
    """

    def __init__(self, state, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        proj = state.project

        self.setWindowTitle("Project Properties")
        self.setMinimumWidth(420)

        root = QVBoxLayout(self)

        # ── Identity ──────────────────────────────────────────────────────
        id_grp = QGroupBox("Project")
        id_fl = QFormLayout(id_grp)

        self._name_edit = QLineEdit(proj.name)
        id_fl.addRow("Project name:", self._name_edit)

        crs_text = f"EPSG:{proj.crs_epsg}" if proj.crs_epsg else "Not set"
        self._crs_lbl = QLabel(crs_text)
        self._crs_lbl.setStyleSheet("color: grey;")
        id_fl.addRow("CRS:", self._crs_lbl)

        crs_note = QLabel("CRS is set at project creation and cannot be changed.")
        crs_note.setStyleSheet("font-size: 8pt; color: grey;")
        crs_note.setWordWrap(True)
        id_fl.addRow("", crs_note)

        root.addWidget(id_grp)

        # ── Units & Depth ─────────────────────────────────────────────────
        depth_grp = QGroupBox("Units & Depth")
        depth_fl = QFormLayout(depth_grp)

        self._depth_units_combo = QComboBox()
        self._depth_units_combo.addItems(["m", "ft"])
        idx = self._depth_units_combo.findText(proj.depth_units)
        if idx >= 0:
            self._depth_units_combo.setCurrentIndex(idx)
        depth_fl.addRow("Depth units:", self._depth_units_combo)

        self._domain_combo = QComboBox()
        self._domain_combo.addItems(["md", "twt"])
        dom_idx = self._domain_combo.findText(proj.depth_domain)
        if dom_idx >= 0:
            self._domain_combo.setCurrentIndex(dom_idx)
        depth_fl.addRow("Depth domain:", self._domain_combo)

        self._depth_min_spin = QDoubleSpinBox()
        self._depth_min_spin.setRange(-10000, 50000)
        self._depth_min_spin.setDecimals(0)
        self._depth_min_spin.setValue(proj.default_depth_min)
        depth_fl.addRow("Default depth min:", self._depth_min_spin)

        self._depth_max_spin = QDoubleSpinBox()
        self._depth_max_spin.setRange(-10000, 50000)
        self._depth_max_spin.setDecimals(0)
        self._depth_max_spin.setValue(proj.default_depth_max)
        depth_fl.addRow("Default depth max:", self._depth_max_spin)

        root.addWidget(depth_grp)

        # ── Buttons ───────────────────────────────────────────────────────
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._apply)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def _apply(self) -> None:
        self._state.set_project_properties(
            name=self._name_edit.text().strip() or self._state.project.name,
            depth_units=self._depth_units_combo.currentText(),
            depth_domain=self._domain_combo.currentText(),
            default_depth_min=self._depth_min_spin.value(),
            default_depth_max=self._depth_max_spin.value(),
        )
        self.accept()
