"""Formation Properties dialog — multi-tab (Phase E)."""
from __future__ import annotations

import math

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSlider, QTabWidget,
    QVBoxLayout, QWidget,
)

from cross_section_tool.core.formation import Formation, LITHOLOGY_DEFAULTS

_LITHOLOGIES = list(LITHOLOGY_DEFAULTS.keys()) + ["conglomerate", "coal", "volcanic"]
_PATTERNS    = ["none", "sandstone", "shale", "siltstone", "limestone",
                "dolomite", "conglomerate", "coal", "salt", "basement", "volcanic"]


def _spin(lo, hi, step, value, decimals=2) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    s.setValue(value)
    return s


class FormationDialog(QDialog):
    """Edit all attributes of a Formation."""

    def __init__(self, formation: Formation, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Formation Properties — {formation.name}")
        self.setMinimumSize(480, 560)
        self._fm = formation
        self._color = "#{:02x}{:02x}{:02x}".format(*[int(c) for c in formation.color])
        self._setup_ui()

    def _setup_ui(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self._tab_general(),  "General")
        tabs.addTab(self._tab_lithology(), "Lithology")
        tabs.addTab(self._tab_physical(),  "Physical")
        tabs.addTab(self._tab_thermal(),   "Thermal")
        tabs.addTab(self._tab_computed(),  "Computed")

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._on_accept)
        bb.rejected.connect(self.reject)

        vbox = QVBoxLayout(self)
        vbox.addWidget(tabs)
        vbox.addWidget(bb)

    # ------------------------------------------------------------------
    # Tab: General
    # ------------------------------------------------------------------

    def _tab_general(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self._name_edit = QLineEdit(self._fm.name)
        form.addRow("Name:", self._name_edit)

        self._age_top  = _spin(-1000, 5000, 1, self._fm.age_top_ma or 0, 1)
        self._age_base = _spin(-1000, 5000, 1, self._fm.age_base_ma or 0, 1)
        form.addRow("Age top (Ma):", self._age_top)
        form.addRow("Age base (Ma):", self._age_base)

        # Color picker
        color_w = QWidget()
        color_h = QHBoxLayout(color_w)
        color_h.setContentsMargins(0, 0, 0, 0)
        self._color_swatch = QLabel()
        self._color_swatch.setFixedSize(32, 16)
        self._apply_swatch()
        color_h.addWidget(self._color_swatch)
        cb = QPushButton("Choose…")
        cb.clicked.connect(self._pick_color)
        color_h.addWidget(cb)
        color_h.addStretch()
        form.addRow("Color:", color_w)

        # Opacity slider
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(int(self._fm.opacity * 100))
        form.addRow("Opacity (%):", self._opacity_slider)

        # Lithology pattern
        self._pattern_combo = QComboBox()
        for p in _PATTERNS:
            self._pattern_combo.addItem(p)
        idx = _PATTERNS.index(self._fm.lithology_pattern) \
              if self._fm.lithology_pattern in _PATTERNS else 0
        self._pattern_combo.setCurrentIndex(idx)
        form.addRow("Pattern:", self._pattern_combo)

        self._pattern_scale = _spin(0.1, 4.0, 0.1, self._fm.pattern_scale)
        form.addRow("Pattern scale:", self._pattern_scale)

        return w

    # ------------------------------------------------------------------
    # Tab: Lithology
    # ------------------------------------------------------------------

    def _tab_lithology(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self._litho_combo = QComboBox()
        for lit in _LITHOLOGIES:
            self._litho_combo.addItem(lit.title(), lit)
        cur = self._fm.primary_lithology
        idx = _LITHOLOGIES.index(cur) if cur in _LITHOLOGIES else 0
        self._litho_combo.setCurrentIndex(idx)
        self._litho_combo.currentIndexChanged.connect(self._on_lithology_changed)
        form.addRow("Primary lithology:", self._litho_combo)

        reset_btn = QPushButton("Reset to defaults")
        reset_btn.clicked.connect(self._on_lithology_changed)
        form.addRow("", reset_btn)

        self._sand_spin   = _spin(0, 1, 0.05, self._fm.sand_fraction)
        self._shale_spin  = _spin(0, 1, 0.05, self._fm.shale_fraction)
        self._carb_spin   = _spin(0, 1, 0.05, self._fm.carbonate_fraction)
        form.addRow("Sand fraction:", self._sand_spin)
        form.addRow("Shale fraction:", self._shale_spin)
        form.addRow("Carbonate fraction:", self._carb_spin)

        return w

    # ------------------------------------------------------------------
    # Tab: Physical
    # ------------------------------------------------------------------

    def _tab_physical(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self._phi0 = _spin(0, 1, 0.01, self._fm.porosity_surface)
        self._c    = _spin(0, 0.01, 0.0001, self._fm.compaction_coeff, 6)
        self._rho  = _spin(1000, 5000, 10, self._fm.grain_density, 0)
        form.addRow("Surface porosity φ₀:", self._phi0)
        form.addRow("Compaction coeff c (1/m):", self._c)
        form.addRow("Grain density (kg/m³):", self._rho)

        # Athy's law preview plot
        fig = Figure(figsize=(3.5, 2.5), tight_layout=True)
        self._phy_ax  = fig.add_subplot(111)
        self._phy_canvas = FigureCanvasQTAgg(fig)
        self._phy_canvas.setFixedHeight(160)
        self._phi0.valueChanged.connect(self._update_athy_plot)
        self._c.valueChanged.connect(self._update_athy_plot)
        self._update_athy_plot()
        form.addRow("Porosity–depth:", self._phy_canvas)

        return w

    def _update_athy_plot(self) -> None:
        phi0 = self._phi0.value()
        c    = self._c.value()
        z = np.linspace(0, 5000, 200)
        phi = phi0 * np.exp(-c * z)
        ax = self._phy_ax
        ax.clear()
        ax.plot(phi, z, "b-", lw=1.5)
        ax.set_xlabel("Porosity", fontsize=7)
        ax.set_ylabel("Depth (m)", fontsize=7)
        ax.invert_yaxis()
        ax.tick_params(labelsize=6)
        self._phy_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Tab: Thermal
    # ------------------------------------------------------------------

    def _tab_thermal(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self._k_cond  = _spin(0, 20, 0.1, self._fm.matrix_thermal_conductivity)
        self._heat_p  = _spin(0, 20, 0.1, self._fm.radiogenic_heat_production)
        self._cp      = _spin(0, 5000, 10, self._fm.specific_heat_capacity, 0)
        form.addRow("Thermal conductivity (W/m·K):", self._k_cond)
        form.addRow("Heat production (µW/m³):", self._heat_p)
        form.addRow("Specific heat (J/kg·K):", self._cp)
        return w

    # ------------------------------------------------------------------
    # Tab: Computed (read-only)
    # ------------------------------------------------------------------

    def _tab_computed(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        phi0 = self._fm.porosity_surface
        c    = self._fm.compaction_coeff
        rho  = self._fm.grain_density

        # Porosity at 1 km, 3 km
        phi1k = phi0 * math.exp(-c * 1000) if c > 0 else phi0
        phi3k = phi0 * math.exp(-c * 3000) if c > 0 else phi0
        form.addRow("Porosity at 1 km:", QLabel(f"{phi1k:.3f}"))
        form.addRow("Porosity at 3 km:", QLabel(f"{phi3k:.3f}"))

        bd1k = rho * (1 - phi1k) + 1000.0 * phi1k
        form.addRow("Bulk density at 1 km (kg/m³):", QLabel(f"{bd1k:.0f}"))

        note = QLabel(
            "<small>Decompacted thickness and area are available once "
            "the polygon is linked to a formation via the Project panel.</small>"
        )
        note.setWordWrap(True)
        form.addRow(note)
        return w

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_swatch(self) -> None:
        self._color_swatch.setStyleSheet(
            f"background:{self._color}; border:1px solid #888;")

    def _pick_color(self) -> None:
        c = QColorDialog.getColor(QColor(self._color), self)
        if c.isValid():
            self._color = c.name()
            self._apply_swatch()

    def _on_lithology_changed(self) -> None:
        lit = self._litho_combo.currentData()
        defaults = LITHOLOGY_DEFAULTS.get(lit, {})
        if "porosity_surface" in defaults:
            self._phi0.setValue(defaults["porosity_surface"])
        if "compaction_coeff" in defaults:
            self._c.setValue(defaults["compaction_coeff"])
        if "grain_density" in defaults:
            self._rho.setValue(defaults["grain_density"])
        if "matrix_thermal_conductivity" in defaults:
            self._k_cond.setValue(defaults["matrix_thermal_conductivity"])
        if "radiogenic_heat_production" in defaults:
            self._heat_p.setValue(defaults["radiogenic_heat_production"])
        if "specific_heat_capacity" in defaults:
            self._cp.setValue(defaults["specific_heat_capacity"])

    def _on_accept(self) -> None:
        self._fm.name                      = self._name_edit.text().strip()
        self._fm.age_top_ma                = self._age_top.value() or None
        self._fm.age_base_ma               = self._age_base.value() or None
        rgb = QColor(self._color)
        self._fm.color                     = (rgb.red(), rgb.green(), rgb.blue())
        self._fm.opacity                   = self._opacity_slider.value() / 100.0
        self._fm.lithology_pattern         = self._pattern_combo.currentText()
        self._fm.pattern_scale             = self._pattern_scale.value()
        self._fm.primary_lithology         = self._litho_combo.currentData()
        self._fm.sand_fraction             = self._sand_spin.value()
        self._fm.shale_fraction            = self._shale_spin.value()
        self._fm.carbonate_fraction        = self._carb_spin.value()
        self._fm.porosity_surface          = self._phi0.value()
        self._fm.compaction_coeff          = self._c.value()
        self._fm.grain_density             = self._rho.value()
        self._fm.matrix_thermal_conductivity = self._k_cond.value()
        self._fm.radiogenic_heat_production  = self._heat_p.value()
        self._fm.specific_heat_capacity    = self._cp.value()
        self.accept()
