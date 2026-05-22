from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from section_tool.app_state import AppState


# ---------------------------------------------------------------------------
# Category labels and per-object type colours
# ---------------------------------------------------------------------------

_CATEGORIES = ["Sections", "Horizons", "Faults", "Reference Lines",
               "Polygons", "Wells", "Surfaces"]
_DEFAULT_COLORS = {
    "Sections":        "#1f77b4",
    "Horizons":        "#2ca02c",
    "Faults":          "#d62728",
    "Reference Lines": "#999999",
    "Wells":           "#8B4513",
    "Polygons":        "#9467bd",
    "Surfaces":        "#E87722",
}
_ICONS = {
    "Sections":        "⟋",
    "Horizons":        "─",
    "Faults":          "╲",
    "Reference Lines": "·",
    "Polygons":        "■",
    "Wells":           "▼",
    "Surfaces":        "◇",
}


class _ColorSwatch(QLabel):
    """Tiny clickable colour rectangle that opens a QColorDialog on click."""

    color_changed = Signal(str)   # hex colour string

    def __init__(self, color: str, parent=None) -> None:
        super().__init__(parent)
        self._color = color
        self.setFixedSize(14, 14)
        self._apply()

    def _apply(self) -> None:
        self.setStyleSheet(
            f"background:{self._color}; border:1px solid #888; border-radius:2px;"
        )

    @property
    def color(self) -> str:
        return self._color

    def set_color(self, color: str) -> None:
        self._color = color
        self._apply()

    def mousePressEvent(self, event) -> None:
        chosen = QColorDialog.getColor(QColor(self._color), self, "Choose Colour")
        if chosen.isValid():
            self._color = chosen.name()
            self._apply()
            self.color_changed.emit(self._color)


_STYLE_LABELS  = ["─────", "- - -", "· · ·", "-·-·-"]
_STYLE_VALUES  = ["solid", "dashed", "dotted", "dashdot"]


class _ObjectRow(QWidget):
    """A single row widget: [checkbox] [swatch] [name] [width] [style]."""

    visibility_changed = Signal(bool)
    color_changed      = Signal(str)
    rename_requested   = Signal(str)
    line_width_changed = Signal(float)
    line_style_changed = Signal(str)

    def __init__(self, name: str, color: str, visible: bool = True,
                 line_width: float = 1.5, line_style: str = "solid",
                 show_stroke: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._name = name

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(4)

        from PySide6.QtWidgets import QCheckBox
        self._check = QCheckBox()
        self._check.setChecked(visible)
        self._check.toggled.connect(self.visibility_changed.emit)
        layout.addWidget(self._check)

        self._swatch = _ColorSwatch(color)
        self._swatch.color_changed.connect(self.color_changed.emit)
        layout.addWidget(self._swatch)

        self._label = QLabel(name)
        self._label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        font = QFont()
        font.setPointSize(9)
        self._label.setFont(font)
        layout.addWidget(self._label)

        self._width_spin: QDoubleSpinBox | None = None
        self._style_combo: QComboBox | None = None

        if show_stroke:
            self._width_spin = QDoubleSpinBox()
            self._width_spin.setRange(0.5, 6.0)
            self._width_spin.setSingleStep(0.5)
            self._width_spin.setValue(float(line_width))
            self._width_spin.setFixedWidth(44)
            self._width_spin.setDecimals(1)
            self._width_spin.setToolTip("Line width (pt)")
            self._width_spin.valueChanged.connect(self.line_width_changed.emit)
            layout.addWidget(self._width_spin)

            self._style_combo = QComboBox()
            for label in _STYLE_LABELS:
                self._style_combo.addItem(label)
            idx = _STYLE_VALUES.index(line_style) if line_style in _STYLE_VALUES else 0
            self._style_combo.setCurrentIndex(idx)
            self._style_combo.setFixedWidth(54)
            self._style_combo.setToolTip("Line style")
            self._style_combo.currentIndexChanged.connect(
                lambda i: self.line_style_changed.emit(_STYLE_VALUES[i])
            )
            layout.addWidget(self._style_combo)

    @property
    def is_visible(self) -> bool:
        return self._check.isChecked()

    @property
    def color(self) -> str:
        return self._swatch.color

    @property
    def line_width(self) -> float:
        return self._width_spin.value() if self._width_spin else 1.5

    @property
    def line_style(self) -> str:
        if self._style_combo is None:
            return "solid"
        return _STYLE_VALUES[self._style_combo.currentIndex()]

    @property
    def name(self) -> str:
        return self._name

    def set_name(self, name: str) -> None:
        self._name = name
        self._label.setText(name)

    def set_color(self, color: str) -> None:
        self._swatch.set_color(color)


class ProjectPanel(QDockWidget):
    """Dockable 'Project' panel with a QTreeWidget listing all project objects.

    Structure::

        ▼ Sections
            ☑ ⟋ Section 1
            ☑ ⟋ Section 2
        ▼ Horizons
            ☑ ─ Top Mancos
        ...

    Signals
    -------
    visibility_changed(category, index, visible)
        Emitted when the user toggles the visibility checkbox of an object.
    object_color_changed(category, index, color)
        Emitted when the user picks a new colour for an object.
    object_renamed(category, index, new_name)
        Emitted when the user renames an object inline.
    object_deleted(category, index)
        Emitted when the user selects Delete from the context menu.
    object_moved(category, from_index, to_index)
        Emitted when Move Up / Move Down is used.
    add_requested(category)
        Emitted when the + button is clicked.
    """

    properties_requested      = Signal(str, int)   # Phase A/B/E
    visibility_changed        = Signal(str, int, bool)
    object_color_changed      = Signal(str, int, str)
    object_line_width_changed = Signal(str, int, float)
    object_line_style_changed = Signal(str, int, str)
    object_renamed            = Signal(str, int, str)
    object_deleted            = Signal(str, int)
    object_moved              = Signal(str, int, int)
    add_requested             = Signal(str)
    # Emitted when a Horizon/Fault is clicked — signals the active pick target
    pick_target_selected      = Signal(str, int)
    # Well-specific actions
    create_ew_section_through_well = Signal(int)   # well index
    create_ns_section_through_well = Signal(int)   # well index

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__("Project", parent)
        self._state = state
        self._setup_ui()
        self._connect_state_signals()
        self._rebuild()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )

        inner = QWidget()
        vbox = QVBoxLayout(inner)
        vbox.setContentsMargins(2, 2, 2, 2)
        vbox.setSpacing(2)

        # Mode indicator pill at top
        self._mode_label = QLabel("● Select")
        self._mode_label.setStyleSheet("""
            QLabel {
                background-color: #1a3050;
                color: #8ab4d8;
                padding: 3px 8px;
                font-size: 9pt;
                font-weight: bold;
            }
        """)
        vbox.addWidget(self._mode_label)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setColumnCount(1)
        self._tree.setIndentation(12)
        self._tree.setUniformRowHeights(True)
        self._tree.setStyleSheet(
            "QTreeWidget::item { min-height: 22px; padding: 1px 2px; }")
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        self._tree.itemDoubleClicked.connect(self._on_double_click)
        vbox.addWidget(self._tree)

        # + button at the bottom
        add_row = QHBoxLayout()
        add_row.setContentsMargins(0, 0, 0, 0)
        self._add_btn = QPushButton("+ Add")
        self._add_btn.setFlat(True)
        self._add_btn.setFixedHeight(22)
        font = QFont()
        font.setPointSize(8)
        self._add_btn.setFont(font)
        self._add_btn.clicked.connect(self._on_add_clicked)
        add_row.addWidget(self._add_btn)
        add_row.addStretch()
        vbox.addLayout(add_row)

        self.setWidget(inner)
        self.setMinimumWidth(180)

    _TOOL_LABELS = {
        "select":       "● Select",
        "node_edit":    "● Nodes",
        "pan":          "● Pan",
        "zoom":         "● Zoom",
        "horizon_pick": "● Horizon Pick",
        "fault_pick":   "● Fault Pick",
        "polygon":      "● Polygon",
        "measure":      "● Measure",
        "new_section":  "● Draw Section",
        "h_ref":        "● H Reference",
        "v_ref":        "● V Reference",
    }

    def set_mode(self, tool_id: str) -> None:
        label = self._TOOL_LABELS.get(tool_id, f"● {tool_id.replace('_', ' ').title()}")
        active = tool_id not in ("select", "pan", "zoom", "")
        color  = "#2563EB" if active else "#1a3050"
        text_c = "#ffffff" if active else "#8ab4d8"
        self._mode_label.setStyleSheet(f"""
            QLabel {{
                background-color: {color};
                color: {text_c};
                padding: 3px 8px;
                font-size: 9pt;
                font-weight: bold;
            }}
        """)
        self._mode_label.setText(label)

    def _connect_state_signals(self) -> None:
        s = self._state
        s.tool_changed.connect(self.set_mode)
        s.project_changed.connect(self._rebuild)
        s.section_added.connect(lambda _: self._rebuild())
        s.section_removed.connect(lambda _: self._rebuild())
        s.section_modified.connect(lambda *_: self._rebuild())
        s.horizon_pick_added.connect(lambda _: self._rebuild())
        s.horizon_pick_removed.connect(lambda _: self._rebuild())
        s.horizon_pick_modified.connect(lambda *_: self._rebuild())
        s.fault_pick_added.connect(lambda _: self._rebuild())
        s.fault_pick_removed.connect(lambda _: self._rebuild())
        s.fault_pick_modified.connect(lambda *_: self._rebuild())
        s.well_added.connect(lambda _: self._rebuild())
        s.well_removed.connect(lambda _: self._rebuild())
        s.well_modified.connect(lambda *_: self._rebuild())
        s.reference_line_added.connect(lambda _: self._rebuild())
        s.reference_line_removed.connect(lambda _: self._rebuild())
        s.reference_line_modified.connect(lambda *_: self._rebuild())
        # Update Add button label when active tool changes
        s.tool_changed.connect(self._update_add_btn_label)
        # Emit pick-target when user clicks a tree item
        self._tree.itemClicked.connect(self._on_item_clicked)

    # ------------------------------------------------------------------
    # Tree population
    # ------------------------------------------------------------------

    def _rebuild(self, *_args) -> None:
        """Repopulate the tree from current project state."""
        # Phase 6: suppress per-item signals during bulk rebuild
        self._tree.blockSignals(True)
        self._tree.setUpdatesEnabled(False)
        try:
            self._tree.clear()
            proj = self._state.project

            self._category_items: dict[str, QTreeWidgetItem] = {}
            self._row_widgets: dict[tuple[str, int], _ObjectRow] = {}

            for cat in _CATEGORIES:
                cat_item = QTreeWidgetItem([cat])
                cat_item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                )
                font = QFont()
                font.setBold(True)
                font.setPointSize(9)
                cat_item.setFont(0, font)
                self._tree.addTopLevelItem(cat_item)
                self._category_items[cat] = cat_item

                objects = self._objects_for_category(cat)
                show_stroke = cat in ("Horizons", "Faults")
                for idx, (name, color, lw, ls) in enumerate(objects):
                    child = QTreeWidgetItem()
                    child.setFlags(
                        Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                    )
                    cat_item.addChild(child)

                    row = _ObjectRow(name, color, line_width=lw, line_style=ls,
                                     show_stroke=show_stroke)
                    row.visibility_changed.connect(
                        lambda v, c=cat, i=idx: self.visibility_changed.emit(c, i, v)
                    )
                    row.color_changed.connect(
                        lambda col, c=cat, i=idx: self.object_color_changed.emit(c, i, col)
                    )
                    row.line_width_changed.connect(
                        lambda w, c=cat, i=idx: self.object_line_width_changed.emit(c, i, w)
                    )
                    row.line_style_changed.connect(
                        lambda s, c=cat, i=idx: self.object_line_style_changed.emit(c, i, s)
                    )
                    self._tree.setItemWidget(child, 0, row)
                    self._row_widgets[(cat, idx)] = row

                cat_item.setExpanded(True)

            # Vector layers section
            vector_layers = self._state.get_vector_layers() if hasattr(self._state, "get_vector_layers") else []
            if vector_layers:
                layers_item = QTreeWidgetItem(["Layers"])
                layers_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                font = QFont(); font.setBold(True); font.setPointSize(9)
                layers_item.setFont(0, font)
                self._tree.addTopLevelItem(layers_item)
                for lyr in vector_layers:
                    child = QTreeWidgetItem([lyr.get("name", "Layer")])
                    child.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                                   | Qt.ItemFlag.ItemIsUserCheckable)
                    child.setCheckState(0, Qt.CheckState.Checked)
                    layers_item.addChild(child)
                layers_item.setExpanded(True)
        finally:
            self._tree.blockSignals(False)
            self._tree.setUpdatesEnabled(True)

    def _objects_for_category(
        self, category: str
    ) -> list[tuple[str, str, float, str]]:
        """Return (name, color, line_width, line_style) tuples for a category."""
        proj = self._state.project
        _dw, _ds = 1.5, "solid"
        if category == "Sections":
            return [(s.name or f"Section {i+1}", _DEFAULT_COLORS["Sections"], _dw, _ds)
                    for i, s in enumerate(proj.sections)]
        if category == "Horizons":
            return [(h.name or f"Horizon {i+1}", h.color,
                     getattr(h, "line_width", _dw), getattr(h, "line_style", _ds))
                    for i, h in enumerate(proj.horizon_picks)]
        if category == "Faults":
            return [(f.name or f"Fault {i+1}", f.color,
                     getattr(f, "line_width", _dw), getattr(f, "line_style", _ds))
                    for i, f in enumerate(proj.fault_picks)]
        if category == "Reference Lines":
            return [(rl.name or f"{'H' if rl.kind == 'horizontal' else 'V'} {rl.value}",
                     rl.color, _dw, _ds)
                    for rl in proj.reference_lines]
        if category == "Polygons":
            return [(p.name or f"Polygon {i+1}", p.fill_color, 1.5, "solid")
                    for i, p in enumerate(proj.polygons)]
        if category == "Wells":
            return [(w.name or f"Well {i+1}",
                     getattr(w, "color", _DEFAULT_COLORS["Wells"]), _dw, _ds)
                    for i, w in enumerate(proj.wells)]
        if category == "Surfaces":
            surfs = proj.surfaces if hasattr(proj, "surfaces") else []
            return [(s.name or f"Surface {i+1}",
                     s.display_color if hasattr(s, "display_color") else _DEFAULT_COLORS["Surfaces"],
                     float(getattr(s, "line_width", _dw)), _ds)
                    for i, s in enumerate(surfs)]
        return []

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _selected_category_and_index(
        self,
    ) -> tuple[str, int] | None:
        """Return (category, index) for the currently selected object row."""
        items = self._tree.selectedItems()
        if not items:
            return None
        item = items[0]
        parent = item.parent()
        if parent is None:
            return None  # category header selected, not an object
        cat_text = parent.text(0)
        idx = parent.indexOfChild(item)
        return cat_text, idx

    def _selected_category(self) -> str | None:
        """Return the category name of the selected item (header or object)."""
        sel = self._tree.currentItem()
        if sel is None:
            return None
        if sel.parent() is None:
            return sel.text(0)
        return sel.parent().text(0)

    def _on_item_clicked(self, item: QTreeWidgetItem, _col: int) -> None:
        """Single click on any object row activates it."""
        parent = item.parent()
        if parent is None:
            return  # category header
        cat = parent.text(0)
        idx = parent.indexOfChild(item)
        if cat in ("Horizons", "Faults"):
            self._state.set_active_pick_target(cat, idx)
            self.pick_target_selected.emit(cat, idx)
        elif cat == "Sections":
            proj = self._state.project
            if idx < len(proj.sections):
                self._state.set_active_section(proj.sections[idx])

    def _on_context_menu(self, pos) -> None:
        item_at = self._tree.itemAt(pos)
        if item_at is None:
            return

        # Category header: "Add …" only
        if item_at.parent() is None:
            cat = item_at.text(0)
            menu = QMenu(self)
            lbl = "Add Horizon…" if cat == "Horizons" else \
                  "Add Fault…"   if cat == "Faults"   else f"Add {cat}"
            add_act = menu.addAction(lbl)
            if menu.exec(self._tree.viewport().mapToGlobal(pos)) is add_act:
                self.add_requested.emit(cat)
            return

        result = self._selected_category_and_index()
        if result is None:
            return
        cat, idx = result

        menu = QMenu(self)

        # --- Properties (all types) ---
        menu.addAction("Properties…",
                       lambda: self._show_properties_dialog(cat, idx))
        menu.addSeparator()

        # --- Visibility ---
        row = self._row_widgets.get((cat, idx))
        if row:
            vis_act = menu.addAction("Visible")
            vis_act.setCheckable(True)
            vis_act.setChecked(row.is_visible)
            vis_act.triggered.connect(
                lambda v, c=cat, i=idx: self.visibility_changed.emit(c, i, v))

        # --- Color ---
        menu.addAction("Color…",
                       lambda: self._quick_color_change(cat, idx))

        # --- Line style submenu ---
        style_menu = menu.addMenu("Line Style")
        for label, val in [("Solid", "solid"), ("Dashed", "dashed"),
                            ("Dotted", "dotted"), ("Dash-Dot", "dashdot")]:
            style_menu.addAction(
                label,
                lambda v=val, c=cat, i=idx: self._apply_prop(c, i, {"line_style": v}),
            )

        # --- Line width submenu ---
        width_menu = menu.addMenu("Line Width")
        for w in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
            width_menu.addAction(
                str(w),
                lambda wv=w, c=cat, i=idx: self._apply_prop(c, i, {"line_width": wv}),
            )

        menu.addSeparator()

        # --- Well-specific quick actions ---
        if cat == "Wells":
            # Display Log submenu
            try:
                well = self._state.project.wells[idx]
                log_names = list(well.log_names)
                if log_names:
                    log_menu = menu.addMenu("Display Log")
                    current_log = getattr(well, "display_log", None)
                    none_act = log_menu.addAction("Auto (GR)")
                    none_act.setCheckable(True)
                    none_act.setChecked(current_log is None)
                    none_act.triggered.connect(
                        lambda *_, i=idx: self._set_well_display_log(i, None))
                    log_menu.addSeparator()
                    for ln in log_names:
                        a = log_menu.addAction(ln)
                        a.setCheckable(True)
                        a.setChecked(ln == current_log)
                        a.triggered.connect(
                            lambda *_, i=idx, l=ln: self._set_well_display_log(i, l))
                    menu.addSeparator()
            except (IndexError, AttributeError):
                pass
            menu.addAction("Create E–W Section Through Well",
                           lambda: self.create_ew_section_through_well.emit(idx))
            menu.addAction("Create N–S Section Through Well",
                           lambda: self.create_ns_section_through_well.emit(idx))
            menu.addSeparator()

        # --- Smart actions (type-specific) ---
        self._add_smart_actions(menu, cat, idx)

        menu.addSeparator()

        # --- Standard CRUD ---
        menu.addAction("Rename…", lambda: self._rename_item(cat, idx))
        menu.addSeparator()
        up_act   = menu.addAction("Move Up")
        down_act = menu.addAction("Move Down")
        menu.addSeparator()
        del_act  = menu.addAction("Delete")

        chosen = menu.exec(self._tree.viewport().mapToGlobal(pos))
        if chosen is del_act:
            self._confirm_delete(cat, idx)
        elif chosen is up_act and idx > 0:
            self.object_moved.emit(cat, idx, idx - 1)
        elif chosen is down_act:
            n = self._category_items.get(cat, item_at.parent()).childCount()
            if idx < n - 1:
                self.object_moved.emit(cat, idx, idx + 1)

    def _on_double_click(self, item: QTreeWidgetItem, _col: int) -> None:
        """Double-click on an object row → inline rename."""
        if item.parent() is None:
            return  # header row
        cat = item.parent().text(0)
        idx = item.parent().indexOfChild(item)
        self._rename_item(cat, idx)

    def _rename_item(self, cat: str, idx: int) -> None:
        row = self._row_widgets.get((cat, idx))
        if row is None:
            return
        from PySide6.QtWidgets import QLineEdit
        new_name, ok = QInputDialog.getText(
            self, "Rename", f"New name for {row.name}:",
            QLineEdit.EchoMode.Normal, row.name,
        )
        if ok and new_name.strip():
            row.set_name(new_name.strip())
            self.object_renamed.emit(cat, idx, new_name.strip())

    def _update_add_btn_label(self, tool: str) -> None:
        """Update the + Add button label to reflect what it will create."""
        labels = {
            "horizon_pick": "+ Horizon",
            "fault_pick":   "+ Fault",
            "polygon":      "+ Polygon",
        }
        self._add_btn.setText(labels.get(tool, "+ Add"))

    def _on_add_clicked(self) -> None:
        """Context-sensitive add: active pick tool > tree selection > popup menu."""
        tool = self._state.active_tool
        if tool == "horizon_pick":
            self.add_requested.emit("Horizons")
            return
        if tool == "fault_pick":
            self.add_requested.emit("Faults")
            return
        if tool == "polygon":
            self.add_requested.emit("Polygons")
            return

        cat = self._selected_category()
        if cat in ("Sections", "Horizons", "Faults", "Polygons", "Reference Lines"):
            self.add_requested.emit(cat)
            return

        # Fallback: popup menu — never silently default to Sections
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        menu.addAction("Add Section",       lambda: self.add_requested.emit("Sections"))
        menu.addAction("Add Horizon",       lambda: self.add_requested.emit("Horizons"))
        menu.addAction("Add Fault",         lambda: self.add_requested.emit("Faults"))
        menu.addAction("Add Polygon",       lambda: self.add_requested.emit("Polygons"))
        menu.addAction("Add Reference Line",lambda: self.add_requested.emit("Reference Lines"))
        menu.popup(self._add_btn.mapToGlobal(self._add_btn.rect().bottomLeft()))

    # ------------------------------------------------------------------
    # Properties dialog
    # ------------------------------------------------------------------

    def _show_properties_dialog(self, cat: str, idx: int) -> None:
        obj_data = self._get_object_data(cat, idx)
        dialog_cls = self._get_dialog_class(cat)
        dlg = dialog_cls(cat, obj_data, parent=self)
        dlg.properties_changed.connect(
            lambda vals, c=cat, i=idx: self._apply_prop(c, i, vals))
        dlg.exec()

    def _get_dialog_class(self, cat: str):
        from .properties_dialog import PropertiesDialog
        from .horizon_properties_dialog import HorizonPropertiesDialog
        from .fault_properties_dialog import FaultPropertiesDialog
        from .polygon_properties_dialog import PolygonPropertiesDialog
        from .well_properties_dialog import WellPropertiesDialog
        from .section_properties_dialog import SectionPropertiesDialog
        from .surface_properties_dialog import SurfacePropertiesDialog
        from .reference_line_properties_dialog import ReferenceLinePropertiesDialog
        return {
            "Horizons":       HorizonPropertiesDialog,
            "Faults":         FaultPropertiesDialog,
            "Polygons":       PolygonPropertiesDialog,
            "Wells":          WellPropertiesDialog,
            "Sections":       SectionPropertiesDialog,
            "Surfaces":       SurfacePropertiesDialog,
            "Reference Lines":ReferenceLinePropertiesDialog,
        }.get(cat, PropertiesDialog)

    def _get_object_data(self, cat: str, idx: int) -> dict:
        """Extract properties dict from the project object at (cat, idx)."""
        import math
        proj = self._state.project
        try:
            if cat == "Horizons":
                hp = proj.horizon_picks[idx]
                return {
                    "name": hp.name, "color": hp.color,
                    "line_width": hp.line_width, "line_style": hp.line_style,
                    "contact_type": getattr(hp, "contact_type", "conformable"),
                    "formation_above": getattr(hp, "formation_above", ""),
                    "formation_below": getattr(hp, "formation_below", ""),
                    "age_ma": getattr(hp, "age_ma", None),
                    "confidence": getattr(hp, "confidence", 1.0),
                }
            if cat == "Faults":
                fp = proj.fault_picks[idx]
                return {
                    "name": fp.name, "color": fp.color,
                    "line_width": fp.line_width, "line_style": fp.line_style,
                    "fault_type": getattr(fp, "fault_type", "normal"),
                    "dip_direction": getattr(fp, "dip_direction", "right"),
                    "displacement": getattr(fp, "displacement", None),
                }
            if cat == "Sections":
                sec = proj.sections[idx]
                try:
                    azs = sec.segment_azimuths()
                    az = float(azs[0]) if len(azs) >= 1 else 0.0
                except Exception:
                    az = 0.0
                return {
                    "name": sec.name,
                    "total_length": sec.total_length(),
                    "azimuth": az,
                    "n_nodes": sec.n_nodes,
                    "crs_epsg": sec.crs_epsg,
                    "display_domain": getattr(sec, "display_domain", "depth"),
                    "depth_units": sec.depth_units,
                    "vertical_exaggeration": sec.vertical_exaggeration,
                }
            if cat == "Polygons":
                poly = proj.polygons[idx]
                return {
                    "name": getattr(poly, "name", ""),
                    "color": getattr(poly, "fill_color", "#9467bd"),
                    "fill_color": getattr(poly, "fill_color", "#9467bd"),
                    "fill_opacity": getattr(poly, "fill_alpha", 0.6),
                    "formation_name": getattr(poly, "formation", ""),
                }
            if cat == "Wells":
                w = proj.wells[idx]
                return {
                    "name": w.name,
                    "color": getattr(w, "color", "#E8C46A"),
                    "x": w.x, "y": w.y,
                    "kb_elevation": w.kb,
                    "td": w.deviation.max_tvd if w.deviation else 0,
                    "available_logs": list(w.log_names),
                }
            if cat == "Surfaces":
                surf = proj.surfaces[idx]
                b = surf.bounds()
                zr = surf.z_range()
                return {
                    "name": surf.name,
                    "color": surf.display_color,
                    "n_points": surf.n_points,
                    "z_domain": surf.z_domain,
                    "z_units": surf.z_units,
                    "z_range": zr,
                    "bounds": b,
                    "crs_epsg": surf.crs_epsg,
                    "source_file": surf.source_file,
                    "source_format": surf.source_format,
                    "interpolation": surf.interpolation,
                    "line_width": surf.line_width,
                    "visible": surf.visible,
                }
            if cat == "Reference Lines":
                rl = proj.reference_lines[idx]
                return {
                    "name": rl.name, "color": rl.color,
                    "kind": rl.kind, "value": rl.value,
                    "visible": rl.visible,
                }
        except (IndexError, AttributeError):
            pass
        return {"name": ""}

    def _apply_prop(self, cat: str, idx: int, values: dict) -> None:
        """Apply a values dict to the object at (cat, idx) and emit signals."""
        import copy
        proj = self._state.project
        try:
            if cat == "Horizons" and idx < len(proj.horizon_picks):
                hp = copy.deepcopy(proj.horizon_picks[idx])
                for k, v in values.items():
                    if hasattr(hp, k):
                        setattr(hp, k, v)
                self._state.update_horizon_pick(idx, hp)
                # Sync the row widget color/width/style immediately
                row = self._row_widgets.get((cat, idx))
                if row and "color" in values:
                    row.set_color(values["color"])

            elif cat == "Faults" and idx < len(proj.fault_picks):
                fp = copy.deepcopy(proj.fault_picks[idx])
                for k, v in values.items():
                    if hasattr(fp, k):
                        setattr(fp, k, v)
                self._state.update_fault_pick(idx, fp)
                row = self._row_widgets.get((cat, idx))
                if row and "color" in values:
                    row.set_color(values["color"])

            elif cat == "Polygons" and idx < len(proj.polygons):
                poly = copy.deepcopy(proj.polygons[idx])
                # "color" from quick-change maps to fill_color
                if "color" in values:
                    poly.fill_color = values["color"]
                if "fill_color" in values:
                    poly.fill_color = values["fill_color"]
                if "fill_opacity" in values:
                    poly.fill_alpha = float(values["fill_opacity"])
                if "formation_name" in values:
                    poly.formation = values["formation_name"]
                if "name" in values:
                    poly.name = values["name"]
                self._state.update_polygon(idx, poly)

            elif cat == "Wells" and idx < len(proj.wells):
                import copy as _copy
                well = _copy.deepcopy(proj.wells[idx])
                if "color" in values:
                    well.color = values["color"]
                if "x" in values:
                    well.x = float(values["x"])
                if "y" in values:
                    well.y = float(values["y"])
                if "kb_elevation" in values:
                    well.kb = float(values["kb_elevation"])
                if "name" in values and values["name"]:
                    well.name = values["name"]
                self._state.update_well(idx, well)
                row = self._row_widgets.get((cat, idx))
                if row and "color" in values:
                    row.set_color(values["color"])

            elif cat == "Sections" and idx < len(proj.sections):
                sec = copy.deepcopy(proj.sections[idx])
                if "name" in values:
                    sec.name = values["name"]
                if "vertical_exaggeration" in values:
                    sec.vertical_exaggeration = float(
                        values["vertical_exaggeration"])
                if "display_domain" in values:
                    sec.display_domain = values["display_domain"]
                if "depth_units" in values:
                    sec.depth_units = values["depth_units"]
                self._state.update_section(idx, sec)

            elif cat == "Surfaces" and idx < len(proj.surfaces):
                surf = proj.surfaces[idx]
                if "name" in values:
                    surf.name = values["name"]
                if "color" in values:
                    c = values["color"]
                    if isinstance(c, str) and c.startswith("#"):
                        r = int(c[1:3], 16); g = int(c[3:5], 16); b = int(c[5:7], 16)
                        surf.color = (r, g, b)
                if "line_width" in values:
                    surf.line_width = float(values["line_width"])
                if "visible" in values:
                    surf.visible = bool(values["visible"])
                if "interpolation" in values:
                    surf.interpolation = values["interpolation"]
                    surf.invalidate_cache()
                self._state.update_surface(idx, surf)

            elif cat == "Reference Lines" and idx < len(proj.reference_lines):
                rl = proj.reference_lines[idx]
                if "name" in values:
                    rl.name = values["name"]
                if "color" in values:
                    rl.color = values["color"]
                if "visible" in values:
                    rl.visible = bool(values["visible"])
                if "value" in values:
                    rl.value = float(values["value"])
                self._state.update_reference_line(idx, rl)

        except Exception:
            pass

    # ------------------------------------------------------------------
    # Quick actions
    # ------------------------------------------------------------------

    def _quick_color_change(self, cat: str, idx: int) -> None:
        from PySide6.QtWidgets import QColorDialog
        from PySide6.QtGui import QColor
        data = self._get_object_data(cat, idx)
        cur = data.get("color", "#888888")
        if isinstance(cur, tuple):
            r, g, b = cur; cur = f"#{r:02x}{g:02x}{b:02x}"
        qc = QColorDialog.getColor(QColor(cur), self, "Choose Colour")
        if qc.isValid():
            self._apply_prop(cat, idx, {"color": qc.name()})
            row = self._row_widgets.get((cat, idx))
            if row:
                row.set_color(qc.name())

    def _confirm_delete(self, cat: str, idx: int) -> None:
        from PySide6.QtWidgets import QMessageBox
        data = self._get_object_data(cat, idx)
        name = data.get("name", f"{cat} {idx}")
        reply = QMessageBox.question(
            self, "Delete",
            f"Delete {cat.rstrip('s').lower()} '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.object_deleted.emit(cat, idx)

    # ------------------------------------------------------------------
    # Smart actions
    # ------------------------------------------------------------------

    def _add_smart_actions(self, menu, cat: str, idx: int) -> None:
        """Append type-specific smart actions to *menu*."""
        from PySide6.QtWidgets import QMessageBox

        def stub(title):
            QMessageBox.information(self, title, "Not yet implemented.")

        if cat == "Horizons":
            menu.addSeparator()
            menu.addAction("Clear Picks on Active Section",
                           lambda: self._clear_picks_active_section(idx))
            menu.addAction("Clear All Picks",
                           lambda: self._clear_picks_all(idx))
            menu.addAction("Convert to 3D Surface",
                           lambda: stub("Convert to 3D Surface"))

        elif cat == "Faults":
            menu.addSeparator()
            menu.addAction("Flip Dip Direction",
                           lambda: self._flip_fault_dip(idx))

        elif cat == "Polygons":
            menu.addSeparator()
            menu.addAction("Merge with Adjacent (same formation)",
                           lambda: stub("Merge Polygons"))

        elif cat == "Surfaces":
            menu.addSeparator()
            menu.addAction("Sample at All Wells",
                           lambda: stub("Sample Surface at Wells"))
            menu.addAction("Convert to Section Picks",
                           lambda: stub("Convert to Picks"))

        elif cat == "Sections":
            menu.addSeparator()
            menu.addAction("Re-extract Seismic",
                           lambda: stub("Re-extract Seismic"))

    def _clear_picks_active_section(self, horizon_idx: int) -> None:
        import copy
        from PySide6.QtWidgets import QMessageBox
        sec = self._state.active_section
        if sec is None:
            QMessageBox.information(self, "No Active Section",
                                    "Select a section first.")
            return
        proj = self._state.project
        if horizon_idx >= len(proj.horizon_picks):
            return
        hp = proj.horizon_picks[horizon_idx]
        reply = QMessageBox.question(
            self, "Clear Picks",
            f"Clear all picks for '{hp.name}' on '{sec.name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        new_hp = copy.deepcopy(hp)
        # Remove only picks belonging to this section
        keep = new_hp._section_names != sec.name
        import numpy as np
        for attr in ("_distances", "_depths", "_section_names",
                     "_confidence", "_quality", "_note", "_map_x", "_map_y"):
            setattr(new_hp, attr, getattr(new_hp, attr)[keep])
        self._state.update_horizon_pick(horizon_idx, new_hp)

    def _clear_picks_all(self, horizon_idx: int) -> None:
        from PySide6.QtWidgets import QMessageBox
        from section_tool.core.surfaces import HorizonPick
        proj = self._state.project
        if horizon_idx >= len(proj.horizon_picks):
            return
        hp = proj.horizon_picks[horizon_idx]
        reply = QMessageBox.question(
            self, "Clear All Picks",
            f"Clear ALL picks for '{hp.name}' on every section?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        empty = HorizonPick.empty(name=hp.name, color=hp.color,
                                  line_width=hp.line_width,
                                  line_style=hp.line_style)
        self._state.update_horizon_pick(horizon_idx, empty)

    def _set_well_display_log(self, well_idx: int, log_name: str | None) -> None:
        """Set the displayed log curve for a well and re-render."""
        import copy
        proj = self._state.project
        if well_idx >= len(proj.wells):
            return
        well = copy.deepcopy(proj.wells[well_idx])
        well.display_log = log_name
        self._state.update_well(well_idx, well)

    def _flip_fault_dip(self, fault_idx: int) -> None:
        import copy
        proj = self._state.project
        if fault_idx >= len(proj.fault_picks):
            return
        fp = copy.deepcopy(proj.fault_picks[fault_idx])
        cur = getattr(fp, "dip_direction", "right")
        fp.dip_direction = "left" if cur == "right" else "right"
        self._state.update_fault_pick(fault_idx, fp)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def is_visible(self, category: str, index: int) -> bool:
        """Return the visibility state of an object row."""
        row = self._row_widgets.get((category, index))
        return row.is_visible if row else True

    def color_of(self, category: str, index: int) -> str:
        """Return the colour hex string of an object row."""
        row = self._row_widgets.get((category, index))
        return row.color if row else "#888888"
