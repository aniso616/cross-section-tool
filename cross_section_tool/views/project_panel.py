from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QColorDialog,
    QDockWidget,
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

from cross_section_tool.app_state import AppState


# ---------------------------------------------------------------------------
# Category labels and per-object type colours
# ---------------------------------------------------------------------------

_CATEGORIES = ["Sections", "Horizons", "Faults", "Polygons"]
_DEFAULT_COLORS = {
    "Sections": "#1f77b4",
    "Horizons": "#2ca02c",
    "Faults":   "#d62728",
    "Polygons": "#9467bd",
}
_ICONS = {
    "Sections": "⟋",
    "Horizons": "─",
    "Faults":   "╲",
    "Polygons": "■",
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


class _ObjectRow(QWidget):
    """A single row widget: [checkbox] [swatch] [name label]."""

    visibility_changed = Signal(bool)
    color_changed = Signal(str)
    rename_requested = Signal(str)

    def __init__(self, name: str, color: str, visible: bool = True,
                 parent=None) -> None:
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

    @property
    def is_visible(self) -> bool:
        return self._check.isChecked()

    @property
    def color(self) -> str:
        return self._swatch.color

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

    visibility_changed = Signal(str, int, bool)
    object_color_changed = Signal(str, int, str)
    object_renamed = Signal(str, int, str)
    object_deleted = Signal(str, int)
    object_moved = Signal(str, int, int)
    add_requested = Signal(str)

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

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setColumnCount(1)
        self._tree.setIndentation(12)
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

    def _connect_state_signals(self) -> None:
        s = self._state
        s.project_changed.connect(self._rebuild)
        s.section_added.connect(lambda _: self._rebuild())
        s.section_removed.connect(lambda _: self._rebuild())
        s.section_modified.connect(lambda *_: self._rebuild())
        s.horizon_pick_added.connect(lambda _: self._rebuild())
        s.horizon_pick_removed.connect(lambda _: self._rebuild())
        s.horizon_pick_modified.connect(lambda *_: self._rebuild())
        s.well_added.connect(lambda _: self._rebuild())
        s.well_removed.connect(lambda _: self._rebuild())

    # ------------------------------------------------------------------
    # Tree population
    # ------------------------------------------------------------------

    def _rebuild(self, *_args) -> None:
        """Repopulate the tree from current project state."""
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
            for idx, (name, color) in enumerate(objects):
                child = QTreeWidgetItem()
                child.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                )
                cat_item.addChild(child)

                row = _ObjectRow(name, color)
                row.visibility_changed.connect(
                    lambda v, c=cat, i=idx: self.visibility_changed.emit(c, i, v)
                )
                row.color_changed.connect(
                    lambda col, c=cat, i=idx: self.object_color_changed.emit(c, i, col)
                )
                self._tree.setItemWidget(child, 0, row)
                self._row_widgets[(cat, idx)] = row

            cat_item.setExpanded(True)

    def _objects_for_category(
        self, category: str
    ) -> list[tuple[str, str]]:
        """Return (name, color) pairs for a category from the project."""
        proj = self._state.project
        if category == "Sections":
            return [(s.name or f"Section {i+1}", _DEFAULT_COLORS["Sections"])
                    for i, s in enumerate(proj.sections)]
        if category == "Horizons":
            return [(h.name or f"Horizon {i+1}", h.color)
                    for i, h in enumerate(proj.horizon_picks)]
        if category == "Faults":
            return []   # Faults stored in future work
        if category == "Polygons":
            return []   # Polygons stored in future work
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

    def _on_context_menu(self, pos) -> None:
        result = self._selected_category_and_index()
        if result is None:
            return
        cat, idx = result

        menu = QMenu(self)
        rename_act = menu.addAction("Rename")
        menu.addSeparator()
        up_act = menu.addAction("Move Up")
        down_act = menu.addAction("Move Down")
        menu.addSeparator()
        del_act = menu.addAction("Delete")

        chosen = menu.exec(self._tree.viewport().mapToGlobal(pos))
        if chosen is rename_act:
            self._rename_item(cat, idx)
        elif chosen is del_act:
            self.object_deleted.emit(cat, idx)
        elif chosen is up_act and idx > 0:
            self.object_moved.emit(cat, idx, idx - 1)
        elif chosen is down_act:
            n = self._category_items[cat].childCount()
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
        new_name, ok = QInputDialog.getText(
            self, "Rename", f"New name for {row.name}:", text=row.name
        )
        if ok and new_name.strip():
            row.set_name(new_name.strip())
            self.object_renamed.emit(cat, idx, new_name.strip())

    def _on_add_clicked(self) -> None:
        cat = self._selected_category() or "Sections"
        self.add_requested.emit(cat)

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
