"""Dialog for generating a gridded Surface from horizon picks."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFormLayout, QLabel, QMessageBox, QSpinBox, QVBoxLayout,
)


class GenerateSurfaceDialog(QDialog):
    """Let the user configure and run surface generation from horizon picks.

    Parameters
    ----------
    state : AppState
        Application state (used to read picks and call add_surface).
    parent : QWidget | None
    """

    def __init__(self, state, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Generate Surface from Horizon")
        self.setMinimumWidth(380)
        self._state = state
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        proj = self._state.project

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        # Horizon combo ---------------------------------------------------
        self._horizon_combo = QComboBox()
        names = sorted({hp.name for hp in proj.horizon_picks if hp.name})
        for n in names:
            self._horizon_combo.addItem(n)
        form.addRow("Horizon:", self._horizon_combo)

        # Info label (updated when combo changes) -------------------------
        self._info_label = QLabel()
        self._info_label.setWordWrap(True)
        form.addRow("", self._info_label)

        # Grid resolution -------------------------------------------------
        self._res_spin = QSpinBox()
        self._res_spin.setRange(10, 10_000)
        self._res_spin.setValue(100)
        self._res_spin.setSuffix(" m")
        self._res_spin.setToolTip("Approximate grid cell size in CRS units")
        form.addRow("Grid resolution:", self._res_spin)

        # Method ----------------------------------------------------------
        self._method_combo = QComboBox()
        self._method_combo.addItem("Linear (Delaunay)", "linear")
        self._method_combo.addItem("Nearest neighbour", "nearest")
        form.addRow("Interpolation method:", self._method_combo)

        # Clip to AOI -----------------------------------------------------
        self._aoi_check = QCheckBox("Clip to AOI")
        has_aoi = proj.aoi is not None
        self._aoi_check.setChecked(has_aoi)
        self._aoi_check.setEnabled(has_aoi)
        if not has_aoi:
            self._aoi_check.setToolTip("No AOI is defined for this project")
        form.addRow("", self._aoi_check)

        # Buttons ---------------------------------------------------------
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        bb.accepted.connect(self._on_accept)
        bb.rejected.connect(self.reject)

        vbox = QVBoxLayout(self)
        vbox.addLayout(form)
        vbox.addWidget(bb)

        # Connect combo change after layout is complete
        self._horizon_combo.currentTextChanged.connect(self._update_info)
        self._update_info(self._horizon_combo.currentText())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_info(self, horizon_name: str) -> None:
        """Update the info label to show pick count and section count."""
        proj = self._state.project
        n_pts      = 0
        section_names: set[str] = set()

        for hp in proj.horizon_picks:
            if hp.name != horizon_name:
                continue
            n = hp.n_picks
            n_pts += n
            for sn in hp._section_names:
                sn = str(sn)
                if sn:
                    section_names.add(sn)

        n_secs = len(section_names)
        if n_pts == 0:
            msg = "No picks found for this horizon."
        elif n_pts < 3:
            msg = (
                f"{n_pts} pick point(s) across {n_secs} section(s) — "
                "need at least 3 to build a surface."
            )
        else:
            msg = f"{n_pts} pick point(s) across {n_secs} section(s)."
        self._info_label.setText(msg)

    # ------------------------------------------------------------------
    # Accept handler
    # ------------------------------------------------------------------

    def _on_accept(self) -> None:
        from section_tool.core.surface_builder import build_surface_from_picks

        horizon_name = self._horizon_combo.currentText()
        resolution   = float(self._res_spin.value())
        method       = self._method_combo.currentData()
        proj         = self._state.project
        aoi          = proj.aoi if self._aoi_check.isChecked() else None

        try:
            surface = build_surface_from_picks(
                proj,
                horizon_name,
                grid_resolution=resolution,
                method=method,
                aoi=aoi,
            )
        except ValueError as exc:
            QMessageBox.warning(
                self,
                "Generate Surface",
                str(exc),
            )
            return

        self._state.add_surface(surface)
        self.accept()
