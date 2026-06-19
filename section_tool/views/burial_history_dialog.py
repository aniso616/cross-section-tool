"""Burial-history viewer / editor (Thermal Step 2).

Shows the ``(age, depth)`` burial curve for the tracked horizon at the section
position — labelled with its provenance ("from restoration sequence …" or
"user-specified") — as a table plus a depth-vs-time preview (the standard burial
curve: time on x, depth increasing downward). When there is no restoration-derived
curve, the table is editable so the user can enter one (never a hardcoded proxy).
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from section_tool.core.burial import BurialHistory, manual_burial_history


class BurialHistoryDialog(QDialog):
    def __init__(self, burial: BurialHistory | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Burial history")
        self.setMinimumSize(560, 420)
        # editable when there's nothing restoration-derived to show
        self._editable = burial is None or burial.source == "user-specified"
        self.result: BurialHistory | None = burial

        layout = QHBoxLayout(self)

        left = QVBoxLayout()
        src = (burial.source if burial is not None
               else "user-specified (no restoration sequence)")
        self._source_label = QLabel(f"<b>Source:</b> {src}")
        self._source_label.setTextFormat(Qt.RichText)
        self._source_label.setWordWrap(True)
        left.addWidget(self._source_label)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Age (Ma)", "Depth (m)"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        if not self._editable:
            self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        left.addWidget(self._table)

        if self._editable:
            row = QHBoxLayout()
            add = QPushButton("Add row")
            add.clicked.connect(self._add_row)
            rem = QPushButton("Remove row")
            rem.clicked.connect(self._remove_row)
            row.addWidget(add)
            row.addWidget(rem)
            row.addStretch()
            left.addLayout(row)

        layout.addLayout(left, 1)

        # depth-vs-time preview
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self._figure = Figure(figsize=(4, 3), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._canvas, 1)
        self._ax = self._figure.add_subplot(111)

        for age, depth in (burial.points if burial is not None else []):
            self._append_row(age, depth)
        if self._editable and self._table.rowCount() == 0:
            self._append_row(0.0, 0.0)
        if self._editable:
            self._table.itemChanged.connect(lambda *_: self._refresh_preview())

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)
        left.addWidget(btns)

        self._refresh_preview()

    # ------------------------------------------------------------------

    def _append_row(self, age, depth) -> None:
        r = self._table.rowCount()
        self._table.insertRow(r)
        self._table.setItem(r, 0, QTableWidgetItem(f"{float(age):g}"))
        self._table.setItem(r, 1, QTableWidgetItem(f"{float(depth):g}"))

    def _add_row(self) -> None:
        self._append_row(0.0, 0.0)

    def _remove_row(self) -> None:
        rows = {i.row() for i in self._table.selectedItems()}
        for r in sorted(rows, reverse=True):
            self._table.removeRow(r)
        self._refresh_preview()

    def _read_rows(self) -> list:
        pairs = []
        for r in range(self._table.rowCount()):
            try:
                a = float(self._table.item(r, 0).text())
                d = float(self._table.item(r, 1).text())
            except (ValueError, AttributeError):
                continue
            pairs.append((a, d))
        return pairs

    def _refresh_preview(self) -> None:
        pairs = self._read_rows()
        self._ax.clear()
        if pairs:
            pts = sorted(pairs, key=lambda p: p[0])
            ages = [p[0] for p in pts]
            depths = [p[1] for p in pts]
            self._ax.plot(ages, depths, "o-", color="#cc6633", lw=1.5)
        self._ax.set_xlabel("Age (Ma)")
        self._ax.set_ylabel("Depth (m)")
        self._ax.invert_xaxis()                  # oldest on the left
        self._ax.invert_yaxis()                  # depth increases downward
        self._ax.set_title("Burial curve")
        self._canvas.draw()

    def _on_ok(self) -> None:
        if self._editable:
            self.result = manual_burial_history(self._read_rows())
        self.accept()
