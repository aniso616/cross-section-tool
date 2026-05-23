"""Phase 4 — Context toolbar: changes based on active tool."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout,
    QLabel, QPushButton, QSizePolicy, QToolButton, QWidget,
)

from section_tool.app_state import AppState


def _qlbl(text: str) -> QLabel:
    """Return a QLabel with explicit dark style for the light options bar."""
    lbl = QLabel(text)
    lbl.setStyleSheet("font-size: 8pt; color: #333333;")
    return lbl


class ContextToolbar(QWidget):
    """Slim 32px bar directly below the section header.

    Content switches based on the active tool so the most-needed controls
    are always one click away.
    """

    # Emitted for end-pick, close-polygon, etc.
    action_requested = Signal(str)

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self.setFixedHeight(32)
        # Minimal own stylesheet — inherits global QSS for most widgets
        self.setStyleSheet(
            "QWidget#ContextToolbar { border-bottom: 1px solid #aaa; }"
            "QLabel { font-size: 8pt; }"
            "QPushButton { font-size: 8pt; padding: 2px 10px; }"
            "QComboBox { font-size: 8pt; min-width: 90px; }"
        )
        self.setObjectName("ContextToolbar")

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(8, 2, 8, 2)
        self._layout.setSpacing(4)

        self._rebuilding = False
        self._state.tool_changed.connect(self._on_tool_changed)
        self._state.active_pick_target_changed.connect(
            lambda *_: self._on_tool_changed(self._state.active_tool))
        self._on_tool_changed(state.active_tool)

    # ------------------------------------------------------------------

    @staticmethod
    def _flat_btn(label: str, tooltip: str, sig: str,
                  target, *, danger: bool = False) -> "QPushButton":
        """Return a compact flat-style QPushButton wired to *target*."""
        btn = QPushButton(label)
        btn.setFlat(True)
        btn.setFixedHeight(22)
        btn.setToolTip(tooltip)
        btn.setStyleSheet(
            "QPushButton { border: none; border-radius: 3px; padding: 2px 8px; font-size: 8pt; }"
            "QPushButton:hover { background: rgba(59,130,246,0.18); }"
            "QPushButton:pressed { background: rgba(59,130,246,0.35); }"
            + ("QPushButton { color: #c44; }" if danger else "")
        )
        btn.clicked.connect(lambda _, s=sig: target.emit(s))
        return btn

    @staticmethod
    def _icon_tool_btn(label: str, tooltip: str, sig: str,
                       target, *, danger: bool = False) -> "QToolButton":
        """Return a 20×20 flat QToolButton (icon-only, no border, hover only)."""
        btn = QToolButton()
        btn.setText(label)
        btn.setFixedSize(20, 20)
        btn.setToolTip(tooltip)
        danger_style = "color: #c44;" if danger else ""
        btn.setStyleSheet(
            f"QToolButton {{ border: none; border-radius: 3px; font-size: 8pt; {danger_style} }}"
            "QToolButton:hover { background: rgba(0,0,0,0.10); }"
            "QToolButton:pressed { background: rgba(0,0,0,0.20); }"
        )
        btn.clicked.connect(lambda _, s=sig: target.emit(s))
        return btn

    def _clear(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _on_tool_changed(self, tool_id: str) -> None:
        if self._rebuilding:
            return
        self._rebuilding = True
        self._clear()
        builders = {
            "select":           self._build_select,
            "node_edit":        self._build_node_edit,
            "horizon_pick":     self._build_horizon_pick,
            "fault_pick":       self._build_fault_pick,
            "polygon":          self._build_polygon,
            "h_ref":            self._build_ref_line,
            "v_ref":            self._build_ref_line,
            "a_ref":            self._build_ref_line,
            "extend":           self._build_construct,
            "trim":             self._build_construct,
            "parallel":         self._build_construct,
            "dip_constrained":  self._build_dip_constrained,
            "kink_band":        self._build_kink_band,
            "measure":          self._build_measure,
        }
        fn = builders.get(tool_id, self._build_default)
        fn()
        self._layout.addStretch()
        self._rebuilding = False

    # ------------------------------------------------------------------
    # Per-tool content
    # ------------------------------------------------------------------

    def _build_default(self) -> None:
        proj = self._state.project
        sec  = self._state.active_section
        parts = [f"Project: {proj.name or 'Untitled'}",
                 f"CRS: EPSG:{proj.crs_epsg}"]
        if sec is not None:
            parts.append(f"Section: {sec.name or 'Unnamed'}")
        lbl = QLabel("  |  ".join(parts))
        lbl.setStyleSheet("font-size: 8pt; color: #555555;")
        self._layout.addWidget(lbl)

    def _build_select(self) -> None:
        lbl = QLabel("Select Object (V)")
        lbl.setStyleSheet("font-size: 8pt; padding-right: 6px; color: #333333;")
        self._layout.addWidget(lbl)
        sep = QLabel("|"); sep.setStyleSheet("color: #888;")
        self._layout.addWidget(sep)
        self._layout.addWidget(self._icon_tool_btn(
            "✕", "Delete selected object  (Del)", "delete_object",
            self.action_requested, danger=True))
        self._layout.addWidget(self._icon_tool_btn(
            "⎘", "Copy selected object  (Ctrl+C)", "copy_object", self.action_requested))
        self._layout.addWidget(self._icon_tool_btn(
            "⎙", "Paste object  (Ctrl+V)", "paste_object", self.action_requested))

    def _build_node_edit(self) -> None:
        lbl = QLabel("Node Edit (A)")
        lbl.setStyleSheet("font-size: 8pt; padding-right: 6px; color: #333333;")
        self._layout.addWidget(lbl)
        sep = QLabel("|"); sep.setStyleSheet("color: #888;")
        self._layout.addWidget(sep)
        self._layout.addWidget(self._flat_btn(
            "Insert",  "Insert node at cursor", "insert_node", self.action_requested))
        self._layout.addWidget(self._flat_btn(
            "Delete",  "Delete selected node",  "delete_node", self.action_requested, danger=True))
        self._layout.addWidget(self._flat_btn(
            "Smooth",  "Smooth picks",          "smooth_picks", self.action_requested))

        self._layout.addWidget(_qlbl("  Snap:"))
        snap_cb = QCheckBox()
        snap_cb.setChecked(True)
        snap_cb.stateChanged.connect(
            lambda s: self.action_requested.emit("snap_on" if s else "snap_off"))
        self._layout.addWidget(snap_cb)
        self._layout.addWidget(_qlbl("Tol:"))
        tol = QDoubleSpinBox()
        tol.setRange(1, 50); tol.setValue(15); tol.setSuffix(" px"); tol.setFixedWidth(72)
        self._layout.addWidget(tol)

    def _build_horizon_pick(self) -> None:
        self._layout.addWidget(_qlbl("Target:"))
        combo = self._pick_target_combo("Horizons")
        self._layout.addWidget(combo)

        # Contact type quick-set
        self._layout.addWidget(_qlbl("  Type:"))
        from section_tool.views.horizon_dialog import CONTACT_TYPES
        ct_combo = QComboBox(); ct_combo.setFixedWidth(130)
        for ct in CONTACT_TYPES:
            ct_combo.addItem(ct.replace("_", " ").title(), ct)
        self._layout.addWidget(ct_combo)

        self._layout.addWidget(self._flat_btn(
            "⏹ End Pick", "Finish picking  (Right-click or Escape)",
            "end_pick", self.action_requested, danger=True))

    def _build_fault_pick(self) -> None:
        self._layout.addWidget(_qlbl("Target:"))
        combo = self._pick_target_combo("Faults")
        self._layout.addWidget(combo)

        from section_tool.views.fault_dialog import FAULT_TYPES
        ft_combo = QComboBox(); ft_combo.setFixedWidth(100)
        for ft in FAULT_TYPES:
            ft_combo.addItem(ft.replace("_", " ").title(), ft)
        self._layout.addWidget(ft_combo)

        self._layout.addWidget(self._flat_btn(
            "⏹ End Pick", "Finish picking  (Right-click or Escape)",
            "end_pick", self.action_requested, danger=True))

    def _build_polygon(self) -> None:
        self._layout.addWidget(_qlbl("Target:"))
        combo = QComboBox(); combo.setMinimumWidth(120)
        proj = self._state.project
        for i, p in enumerate(proj.polygons):
            combo.addItem(p.name or f"Polygon {i+1}", i)
        combo.addItem("+ New Polygon…", -1)
        combo.currentIndexChanged.connect(lambda i: (
            self.action_requested.emit("new_polygon")
            if combo.itemData(i) == -1 else None
        ))
        self._layout.addWidget(combo)
        sep = QLabel("|"); sep.setStyleSheet("color: #888;")
        self._layout.addWidget(sep)
        self._layout.addWidget(self._flat_btn(
            "Close", "Close polygon  (Right-click)", "close_polygon", self.action_requested))
        self._layout.addWidget(self._flat_btn(
            "Cancel", "Cancel polygon  (Escape)", "cancel_polygon",
            self.action_requested, danger=True))

    def _build_ref_line(self) -> None:
        tool = self._state.active_tool
        labels = {"h_ref": "Horizontal Ref (H)", "v_ref": "Vertical Ref (V)",
                  "a_ref": "Angled Ref (A)"}
        self._layout.addWidget(_qlbl(labels.get(tool, "Reference Line") + "  |"))
        self._layout.addWidget(_qlbl("Click on section to place guide."))
        if tool == "a_ref":
            self._layout.addWidget(_qlbl("  1st click: anchor  2nd click: direction"))

    def _build_measure(self) -> None:
        self._layout.addWidget(_qlbl("Measure  |"))
        self._dist_lbl  = _qlbl("Distance: —")
        self._depth_lbl = _qlbl("Depth Δ: —")
        self._angle_lbl = _qlbl("Angle: —°")
        for lbl in (self._dist_lbl, self._depth_lbl, self._angle_lbl):
            self._layout.addWidget(lbl)
        clr = QPushButton("Clear")
        clr.clicked.connect(lambda: self.action_requested.emit("measure_clear"))
        self._layout.addWidget(clr)

    # ------------------------------------------------------------------

    def _build_construct(self) -> None:
        labels = {
            "extend":  "Extend  |  Click an endpoint, then the target line.",
            "trim":    "Trim  |  Click the keep-side of a line, then the cutting line.",
            "parallel": "Parallel  |  Click reference horizon, then click to place copy.",
        }
        tool = self._state.active_tool
        lbl = QLabel(labels.get(tool, "Construct"))
        lbl.setStyleSheet("font-size: 8pt; color: #333333;")
        self._layout.addWidget(lbl)
        self._layout.addWidget(self._flat_btn(
            "Cancel", "Cancel  (Escape)", "cancel_construct",
            self.action_requested, danger=True))

    def _build_dip_constrained(self) -> None:
        self._layout.addWidget(_qlbl("Dip-constrained  |"))
        self._layout.addWidget(_qlbl("Dip:"))
        dip_spin = QDoubleSpinBox()
        dip_spin.setRange(-89.0, 89.0)
        dip_spin.setValue(0.0)
        dip_spin.setSuffix("°")
        dip_spin.setFixedWidth(72)
        dip_spin.valueChanged.connect(
            lambda v: self.action_requested.emit(f"cst_param:dip_deg:{v}"))
        self._layout.addWidget(dip_spin)
        self._layout.addWidget(self._flat_btn(
            "Cancel", "Cancel  (Escape)", "cancel_construct",
            self.action_requested, danger=True))

    def _build_kink_band(self) -> None:
        self._layout.addWidget(_qlbl("Kink Band  |"))
        for label, param, default in [
            ("Axial:", "axial_dip", 45.0),
            ("Fore:",  "fore_dip",  30.0),
            ("Back:",  "back_dip",   0.0),
        ]:
            self._layout.addWidget(_qlbl(label))
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 89.0)
            spin.setValue(default)
            spin.setSuffix("°")
            spin.setFixedWidth(68)
            spin.valueChanged.connect(
                lambda v, p=param: self.action_requested.emit(f"cst_param:{p}:{v}"))
            self._layout.addWidget(spin)
        self._layout.addWidget(self._flat_btn(
            "Cancel", "Cancel  (Escape)", "cancel_construct",
            self.action_requested, danger=True))

    # ------------------------------------------------------------------

    def _pick_target_combo(self, category: str) -> QComboBox:
        """Build a combobox listing existing pick objects + '+ New'."""
        combo = QComboBox(); combo.setMinimumWidth(120)
        proj = self._state.project
        picks = proj.horizon_picks if category == "Horizons" else proj.fault_picks
        for i, hp in enumerate(picks):
            combo.addItem(hp.name or f"{category[:-1]} {i+1}", i)

        combo.addItem(f"+ New {category[:-1]}…", -1)

        cur_idx = self._state.active_pick_index
        if cur_idx is not None and 0 <= cur_idx < len(picks):
            combo.blockSignals(True)
            combo.setCurrentIndex(cur_idx)
            combo.blockSignals(False)

        def _on_change(i):
            val = combo.itemData(i)
            if val == -1:
                self.action_requested.emit(
                    "new_horizon" if category == "Horizons" else "new_fault")
            else:
                self._state.set_active_pick_target(category, val)

        combo.currentIndexChanged.connect(_on_change)
        return combo
