"""Phase 4 — Context toolbar: changes based on active tool."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout,
    QLabel, QPushButton, QSizePolicy, QWidget,
)

from cross_section_tool.app_state import AppState


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
            "select":       self._build_select,
            "node_edit":    self._build_node_edit,
            "horizon_pick": self._build_horizon_pick,
            "fault_pick":   self._build_fault_pick,
            "polygon":      self._build_polygon,
            "h_ref":        self._build_ref_line,
            "v_ref":        self._build_ref_line,
            "a_ref":        self._build_ref_line,
            "measure":      self._build_measure,
        }
        fn = builders.get(tool_id, self._build_default)
        fn()
        self._layout.addStretch()
        self._rebuilding = False

    # ------------------------------------------------------------------
    # Per-tool content
    # ------------------------------------------------------------------

    def _build_default(self) -> None:
        self._layout.addWidget(QLabel(
            self._state.active_tool.replace("_", " ").title()
            + "  |  Space to pan temporarily"))

    def _build_select(self) -> None:
        lbl = QLabel("Select Object (V)")
        lbl.setStyleSheet("font-size: 8pt; padding-right: 6px;")
        self._layout.addWidget(lbl)
        sep = QLabel("|"); sep.setStyleSheet("color: #888;")
        self._layout.addWidget(sep)
        self._layout.addWidget(self._flat_btn(
            "Delete", "Delete selected object", "delete_object",
            self.action_requested, danger=True))
        self._layout.addWidget(self._flat_btn(
            "Copy",   "Copy selected object",   "copy_object",   self.action_requested))
        self._layout.addWidget(self._flat_btn(
            "Paste",  "Paste object",            "paste_object",  self.action_requested))

    def _build_node_edit(self) -> None:
        lbl = QLabel("Node Edit (A)")
        lbl.setStyleSheet("font-size: 8pt; padding-right: 6px;")
        self._layout.addWidget(lbl)
        sep = QLabel("|"); sep.setStyleSheet("color: #888;")
        self._layout.addWidget(sep)
        self._layout.addWidget(self._flat_btn(
            "Insert",  "Insert node at cursor", "insert_node", self.action_requested))
        self._layout.addWidget(self._flat_btn(
            "Delete",  "Delete selected node",  "delete_node", self.action_requested, danger=True))
        self._layout.addWidget(self._flat_btn(
            "Smooth",  "Smooth picks",          "smooth_picks", self.action_requested))

        self._layout.addWidget(QLabel("  Snap:"))
        snap_cb = QCheckBox()
        snap_cb.setChecked(True)
        snap_cb.stateChanged.connect(
            lambda s: self.action_requested.emit("snap_on" if s else "snap_off"))
        self._layout.addWidget(snap_cb)
        self._layout.addWidget(QLabel("Tol:"))
        tol = QDoubleSpinBox()
        tol.setRange(1, 50); tol.setValue(15); tol.setSuffix(" px"); tol.setFixedWidth(72)
        self._layout.addWidget(tol)

    def _build_horizon_pick(self) -> None:
        self._layout.addWidget(QLabel("Target:"))
        combo = self._pick_target_combo("Horizons")
        self._layout.addWidget(combo)

        # Contact type quick-set
        self._layout.addWidget(QLabel("  Type:"))
        from cross_section_tool.views.horizon_dialog import CONTACT_TYPES
        ct_combo = QComboBox(); ct_combo.setFixedWidth(130)
        for ct in CONTACT_TYPES:
            ct_combo.addItem(ct.replace("_", " ").title(), ct)
        self._layout.addWidget(ct_combo)

        self._layout.addWidget(self._flat_btn(
            "⏹ End Pick", "Finish picking  (Right-click or Escape)",
            "end_pick", self.action_requested, danger=True))

    def _build_fault_pick(self) -> None:
        self._layout.addWidget(QLabel("Target:"))
        combo = self._pick_target_combo("Faults")
        self._layout.addWidget(combo)

        from cross_section_tool.views.fault_dialog import FAULT_TYPES
        ft_combo = QComboBox(); ft_combo.setFixedWidth(100)
        for ft in FAULT_TYPES:
            ft_combo.addItem(ft.replace("_", " ").title(), ft)
        self._layout.addWidget(ft_combo)

        self._layout.addWidget(self._flat_btn(
            "⏹ End Pick", "Finish picking  (Right-click or Escape)",
            "end_pick", self.action_requested, danger=True))

    def _build_polygon(self) -> None:
        self._layout.addWidget(QLabel("Polygon"))
        sep = QLabel("|"); sep.setStyleSheet("color: #888;")
        self._layout.addWidget(sep)
        self._layout.addWidget(self._flat_btn(
            "Close", "Close polygon", "close_polygon", self.action_requested))
        self._layout.addWidget(self._flat_btn(
            "Cancel", "Cancel polygon", "cancel_polygon",
            self.action_requested, danger=True))

    def _build_ref_line(self) -> None:
        tool = self._state.active_tool
        labels = {"h_ref": "Horizontal Ref (H)", "v_ref": "Vertical Ref (V)",
                  "a_ref": "Angled Ref (A)"}
        self._layout.addWidget(QLabel(labels.get(tool, "Reference Line") + "  |"))
        self._layout.addWidget(QLabel("Click on section to place guide."))
        if tool == "a_ref":
            self._layout.addWidget(QLabel("  1st click: anchor  2nd click: direction"))

    def _build_measure(self) -> None:
        self._layout.addWidget(QLabel("Measure  |"))
        self._dist_lbl  = QLabel("Distance: —")
        self._depth_lbl = QLabel("Depth Δ: —")
        self._angle_lbl = QLabel("Angle: —°")
        for lbl in (self._dist_lbl, self._depth_lbl, self._angle_lbl):
            self._layout.addWidget(lbl)
        clr = QPushButton("Clear")
        clr.clicked.connect(lambda: self.action_requested.emit("measure_clear"))
        self._layout.addWidget(clr)

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
