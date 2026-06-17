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
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class _EventEditDialog(QDialog):
    """Dialog for adding / editing a RestorationEvent.

    Beyond name / age / description, it lets the user pick **which elements this
    event removes**. The element list is resolved live from the section (passed
    in as ``removable``) and writes UUIDs — never names — into the event.

    Parameters
    ----------
    event:
        The RestorationEvent being edited (``None`` when adding).
    removable:
        ``[(uuid, name, type_label), …]`` — the section's removable elements.
    already_removed:
        ``{uuid: step_number}`` for elements removed by an EARLIER event, so the
        picker can flag (not forbid) a redundant re-removal.
    """

    def __init__(self, parent=None, event=None, *, removable=None,
                 already_removed=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Restoration Event")
        self.setMinimumWidth(360)

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

        # ── Element picker ────────────────────────────────────────────────
        grp = QGroupBox("Elements removed by this event")
        gl = QVBoxLayout(grp)
        self._elem_list = QListWidget()
        self._elem_list.setMinimumHeight(150)
        gl.addWidget(self._elem_list)
        layout.addWidget(grp)

        checked = set(event.remove_element_ids) if event is not None else set()
        already = already_removed or {}
        for uid, name, type_label in (removable or []):
            item = QListWidgetItem(f"{name or '(unnamed)'}  ·  {type_label}")
            item.setData(Qt.UserRole, uid)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            if uid in already:
                # Redundant — already removed upstream. Flag it, don't forbid it.
                item.setText(item.text() + f"   — removed at step {already[uid]}")
                item.setForeground(QColor("#888888"))
            item.setCheckState(Qt.Checked if uid in checked else Qt.Unchecked)
            self._elem_list.addItem(item)

        # Unresolved legacy names (renamed/deleted) — display-only, not selectable.
        for nm in (getattr(event, "remove_element_names", []) if event else []):
            item = QListWidgetItem(f"{nm}  ·  (unresolved)")
            item.setFlags(Qt.ItemIsEnabled)
            item.setForeground(QColor("#cc6666"))
            self._elem_list.addItem(item)

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
        ids = []
        for i in range(self._elem_list.count()):
            item = self._elem_list.item(i)
            uid = item.data(Qt.UserRole)
            if uid and item.checkState() == Qt.Checked:
                ids.append(uid)
        return {
            "name": self._name.text().strip() or "Event",
            "age_ma": age if age > 0.0 else None,
            "description": self._desc.text().strip(),
            "remove_element_ids": ids,
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
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["#", "Name", "Age (Ma)", "Removes"])
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
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
            n_removed = len(ev.remove_element_ids) + len(ev.remove_element_names)
            self._table.setItem(i, 3, QTableWidgetItem(str(n_removed) if n_removed else "—"))
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

    def _removable_elements(self) -> "list[tuple[str, str, str]]":
        """``(uuid, name, type)`` for the active section's removable elements.

        Resolved live so it always reflects reality (rename-safe — keyed by uuid).
        Horizons / faults are listed when they appear on the active section;
        polygons when tagged to it (or untagged). With no active section,
        everything removable is offered.
        """
        proj = self._state.project
        sec = self._state.active_section
        sec_name = sec.name if sec is not None else None

        def on_section(pick) -> bool:
            if sec_name is None:
                return True
            try:
                return pick.n_picks_for_section(sec_name) >= 1
            except Exception:
                return True

        out: list[tuple[str, str, str]] = []
        for hp in proj.horizon_picks:
            if on_section(hp):
                out.append((hp.uuid, hp.name, "Horizon"))
        for fp in proj.fault_picks:
            if on_section(fp):
                out.append((fp.uuid, fp.name, "Fault"))
        for poly in proj.polygons:
            psec = getattr(poly, "section_name", "")
            if sec_name is None or not psec or psec == sec_name:
                out.append((poly.uuid, poly.name, "Polygon"))
        return out

    def _already_removed_before(self, event_index: int) -> "dict[str, int]":
        """``{uuid: step_number}`` for elements removed by an earlier event.

        Restoration is sequential — an element removed by an earlier step is
        already gone, so re-removing it is redundant (flagged, not forbidden).
        """
        out: dict[str, int] = {}
        for step_i, ev in enumerate(self._sequence.events[:event_index], start=1):
            for uid in ev.remove_element_ids:
                out.setdefault(uid, step_i)
        return out

    def _add_event(self) -> None:
        from section_tool.core.restoration import RestorationEvent
        seq = self._sequence
        # A new event is appended last, so every existing event is "earlier".
        dlg = _EventEditDialog(self, removable=self._removable_elements(),
                               already_removed=self._already_removed_before(len(seq.events)))
        if dlg.exec() != QDialog.Accepted:
            return
        vals = dlg.values
        next_id = max((e.event_id for e in seq.events), default=0) + 1
        ev = RestorationEvent(
            event_id=next_id,
            name=vals["name"],
            age_ma=vals["age_ma"],
            description=vals["description"],
            remove_element_ids=vals["remove_element_ids"],
        )
        seq.add_event(ev)
        self._state.set_restoration_sequence(seq)
        self.rebuild()
        self.step_changed.emit(seq.current_step)   # refresh the section via the existing path

    def _remove_event(self) -> None:
        idx = self._selected_index()
        if idx is None:
            return
        seq = self._sequence
        ev = seq.events[idx]
        answer = QMessageBox.question(
            self, "Remove event",
            f"Remove restoration event \"{ev.name}\"?",
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
        dlg = _EventEditDialog(self, event=ev, removable=self._removable_elements(),
                               already_removed=self._already_removed_before(idx))
        if dlg.exec() != QDialog.Accepted:
            return
        vals = dlg.values
        ev.name = vals["name"]
        ev.age_ma = vals["age_ma"]
        ev.description = vals["description"]
        ev.remove_element_ids = vals["remove_element_ids"]
        self._state.set_restoration_sequence(seq)
        self.rebuild()
        self._table.selectRow(idx)
        self.step_changed.emit(seq.current_step)   # live: section reflects the new set
