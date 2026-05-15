"""Simple dialog for viewing and editing the stratigraphic column."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QInputDialog, QLabel,
    QListWidget, QListWidgetItem, QPushButton, QVBoxLayout,
)

from section_tool.app_state import AppState
from section_tool.core.formation import Formation


class StratColumnDialog(QDialog):
    """Minimal editor for the project's stratigraphic column.

    Youngest formation at top, oldest at bottom.
    """

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self.setWindowTitle("Stratigraphic Column")
        self.resize(320, 480)
        self._setup_ui()
        self._populate()

    def _setup_ui(self) -> None:
        vbox = QVBoxLayout(self)
        vbox.addWidget(QLabel("Youngest  ▲  (drag to reorder)"))

        self._list = QListWidget()
        self._list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        vbox.addWidget(self._list, stretch=1)

        vbox.addWidget(QLabel("Oldest  ▼"))

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add…")
        add_btn.clicked.connect(self._on_add)
        btn_row.addWidget(add_btn)
        del_btn = QPushButton("Remove")
        del_btn.clicked.connect(self._on_remove)
        btn_row.addWidget(del_btn)
        vbox.addLayout(btn_row)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._on_accept)
        bb.rejected.connect(self.reject)
        vbox.addWidget(bb)

    def _populate(self) -> None:
        self._list.clear()
        for f in self._state.project.strat_column.formations:
            item = QListWidgetItem(
                f"{f.name}" + (f"  [{f.lithology}]" if f.lithology else "")
            )
            item.setData(Qt.ItemDataRole.UserRole, f.name)
            self._list.addItem(item)

    def _on_add(self) -> None:
        name, ok = QInputDialog.getText(self, "Add Formation", "Formation name:")
        if not ok or not name.strip():
            return
        self._state.project.strat_column.add_formation(Formation(name=name.strip()))
        self._populate()

    def _on_remove(self) -> None:
        item = self._list.currentItem()
        if item is None:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        self._state.project.strat_column.remove_formation(name)
        self._populate()

    def _on_accept(self) -> None:
        """Commit the current list order back to the column."""
        col = self._state.project.strat_column
        ordered = [self._list.item(i).data(Qt.ItemDataRole.UserRole)
                   for i in range(self._list.count())]
        for pos, name in enumerate(ordered):
            col.reorder(name, pos)
        self.accept()
