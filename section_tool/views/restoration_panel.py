"""Kinematic restoration panel.

Displays the project's :class:`~section_tool.core.restoration.RestorationSequence`
as an ordered list and provides controls to add, remove, reorder, and
step through restoration events.

Layout
------
  ┌─────────────────────────────────────────┐
  │ ← Present │  Step 2/4  │  Step →        │ ← navigation bar
  ├─────────────────────────────────────────┤
  │ [+] [−] [↑] [↓]                         │ ← event toolbar
  ├─────────────────────────────────────────┤
  │  # │ Name            │ Age (Ma)          │
  │  1 │ Remove Oligocene│ 34.0              │
  │  2 │ Remove Eocene   │ 56.0              │
  │  …                                       │
  └─────────────────────────────────────────┘
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class _EventEditDialog(QDialog):
    """Simple dialog for adding / editing a RestorationEvent."""

    def __init__(self, parent=None, event=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Restoration Event")
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self._name = QLineEdit()
        self._age = QDoubleSpinBox()
        self._age.setRange(0.0, 4600.0)
        self._age.setDecimals(2)
        self._age.setSuffix(" Ma")
        self._age.setSpecialValueText("Unknown")
        self._desc = QLineEdit()

        form.addRow("Name:", self._name)
        form.addRow("Age:", self._age)
        form.addRow("Description:", self._desc)

        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self,
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        if event is not None:
            self._name.setText(event.name)
            self._age.setValue(event.age_ma if event.age_ma is not None else 0.0)
            self._desc.setText(event.description)

    @property
    def values(self) -> dict:
        age = self._age.value()
        return {
            "name": self._name.text().strip() or "Event",
            "age_ma": age if age > 0.0 else None,
            "description": self._desc.text().strip(),
        }


class RestorationPanel(QWidget):
    """Sidebar panel showing the restoration sequence.

    Emits :attr:`step_changed` whenever the user navigates to a different
    restoration step; the section view can connect to this to show only
    the elements that existed at that time.

    Parameters
    ----------
    app_state:
        The global :class:`~section_tool.app_state.AppState`.
    parent:
        Qt parent widget.
    """

    step_changed = Signal(int)  # emitted with the new step index

    def __init__(self, app_state, parent=None) -> None:
        super().__init__(parent)
        self._state = app_state
        self._building = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Navigation bar
        nav = QHBoxLayout()
        self._btn_back = QPushButton("◀ Present")
        self._btn_back.setFixedHeight(24)
        self._step_label = QLabel("Step 0")
        self._step_label.setAlignment(Qt.AlignCenter)
        self._btn_fwd = QPushButton("Step ▶")
        self._btn_fwd.setFixedHeight(24)
        nav.addWidget(self._btn_back)
        nav.addWidget(self._step_label, stretch=1)
        nav.addWidget(self._btn_fwd)
        layout.addLayout(nav)

        # Event toolbar
        toolbar = QHBoxLayout()
        self._btn_add    = QPushButton("+")
        self._btn_remove = QPushButton("−")
        self._btn_up     = QPushButton("↑")
        self._btn_down   = QPushButton("↓")
        for btn in (self._btn_add, self._btn_remove, self._btn_up, self._btn_down):
            btn.setFixedSize(28, 24)
            toolbar.addWidget(btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Event table
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["#", "Name", "Age (Ma)"])
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table)

        # Wire buttons
        self._btn_back.clicked.connect(self._step_back)
        self._btn_fwd.clicked.connect(self._step_forward)
        self._btn_add.clicked.connect(self._add_event)
        self._btn_remove.clicked.connect(self._remove_event)
        self._btn_up.clicked.connect(self._move_up)
        self._btn_down.clicked.connect(self._move_down)
        self._table.doubleClicked.connect(self._edit_event)

        self.rebuild()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def rebuild(self) -> None:
        """Re-populate the table from the current restoration sequence."""
        self._building = True
        seq = self._sequence
        self._table.setRowCount(0)
        for i, ev in enumerate(seq.events):
            self._table.insertRow(i)
            self._table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self._table.setItem(i, 1, QTableWidgetItem(ev.name))
            age_str = f"{ev.age_ma:.1f}" if ev.age_ma is not None else "—"
            self._table.setItem(i, 2, QTableWidgetItem(age_str))
        self._update_nav()
        self._building = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _sequence(self):
        return self._state.restoration_sequence

    def _update_nav(self) -> None:
        seq = self._sequence
        n = len(seq.events)
        step = seq.current_step
        self._step_label.setText(f"Step {step} / {n}")
        self._btn_back.setEnabled(step > 0)
        self._btn_fwd.setEnabled(step < n)

    def _set_step(self, step: int) -> None:
        seq = self._sequence
        seq.current_step = max(0, min(step, len(seq.events)))
        self._state.set_restoration_sequence(seq)
        self._update_nav()
        self.step_changed.emit(seq.current_step)

    def _step_back(self) -> None:
        self._set_step(self._sequence.current_step - 1)

    def _step_forward(self) -> None:
        self._set_step(self._sequence.current_step + 1)

    def _selected_index(self) -> int | None:
        rows = self._table.selectedItems()
        if not rows:
            return None
        return self._table.row(rows[0])

    def _add_event(self) -> None:
        from section_tool.core.restoration import RestorationEvent
        dlg = _EventEditDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return
        vals = dlg.values
        seq = self._sequence
        next_id = max((e.event_id for e in seq.events), default=0) + 1
        ev = RestorationEvent(
            event_id=next_id,
            name=vals["name"],
            age_ma=vals["age_ma"],
            description=vals["description"],
        )
        seq.add_event(ev)
        self._state.set_restoration_sequence(seq)
        self.rebuild()

    def _remove_event(self) -> None:
        idx = self._selected_index()
        if idx is None:
            return
        seq = self._sequence
        ev = seq.events[idx]
        answer = QMessageBox.question(
            self, "Remove event",
            f"Remove restoration event "{ev.name}"?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        seq.remove_event(ev.event_id)
        self._state.set_restoration_sequence(seq)
        self.rebuild()

    def _move_up(self) -> None:
        idx = self._selected_index()
        if idx is None:
            return
        seq = self._sequence
        ev = seq.events[idx]
        if seq.move_event_up(ev.event_id):
            self._state.set_restoration_sequence(seq)
            self.rebuild()
            self._table.selectRow(idx - 1)

    def _move_down(self) -> None:
        idx = self._selected_index()
        if idx is None:
            return
        seq = self._sequence
        ev = seq.events[idx]
        if seq.move_event_down(ev.event_id):
            self._state.set_restoration_sequence(seq)
            self.rebuild()
            self._table.selectRow(idx + 1)

    def _edit_event(self) -> None:
        idx = self._selected_index()
        if idx is None:
            return
        seq = self._sequence
        ev = seq.events[idx]
        dlg = _EventEditDialog(self, event=ev)
        if dlg.exec() != QDialog.Accepted:
            return
        vals = dlg.values
        ev.name = vals["name"]
        ev.age_ma = vals["age_ma"]
        ev.description = vals["description"]
        self._state.set_restoration_sequence(seq)
        self.rebuild()
        self._table.selectRow(idx)
