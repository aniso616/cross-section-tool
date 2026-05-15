"""Keyboard Shortcuts reference dialog.

Generated from ``MainWindow.SHORTCUT_REGISTRY`` so it is always in sync with
the actual registered shortcuts — no manual maintenance required.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class KeyboardShortcutsDialog(QDialog):
    """Read-only table showing every registered keyboard shortcut.

    Parameters
    ----------
    registry:
        The ``MainWindow.SHORTCUT_REGISTRY`` list — each entry is a
        ``(key_sequence, description, category)`` tuple.
    parent:
        Parent widget (typically the main window).
    """

    def __init__(
        self,
        registry: list[tuple[str, str, str]],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumSize(480, 500)
        self._build_ui(registry)

    def _build_ui(self, registry: list[tuple[str, str, str]]) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        title = QLabel("<b>Keyboard Shortcuts</b>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        note = QLabel(
            "All shortcuts are application-level — they work regardless of "
            "which panel has focus.  Space (temporary pan) requires holding "
            "the key and is handled separately."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        # Group entries by category, preserving registry order
        categories: dict[str, list[tuple[str, str]]] = {}
        for key, desc, cat in registry:
            categories.setdefault(cat, []).append((key, desc))

        table = QTableWidget(0, 3)
        table.setHorizontalHeaderLabels(["Category", "Shortcut", "Description"])
        table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)

        row = 0
        for cat, entries in categories.items():
            for key, desc in entries:
                table.insertRow(row)
                table.setItem(row, 0, QTableWidgetItem(cat))
                key_item = QTableWidgetItem(key)
                key_item.setFont(_monospace_font())
                table.setItem(row, 1, key_item)
                table.setItem(row, 2, QTableWidgetItem(desc))
                row += 1

        layout.addWidget(table)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.accept)
        layout.addWidget(buttons)


def _monospace_font():
    from PySide6.QtGui import QFont
    f = QFont("Courier New")
    f.setStyleHint(QFont.StyleHint.Monospace)
    f.setPointSize(9)
    return f
