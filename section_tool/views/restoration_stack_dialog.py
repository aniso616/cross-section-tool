"""Restoration stack dialog.

Accessible via Model ▸ Restoration Stack. Shows all restoration steps
in a timeline table: what is removed at each step and what is still
present, giving a full-sequence audit view.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class RestorationStackDialog(QDialog):
    """Timeline view of the full restoration sequence.

    Each row is one restoration event. Columns show:
    - Step number
    - Event name and age
    - Elements removed at that step
    - Cumulative removed count
    """

    def __init__(self, app_state, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Restoration Stack")
        self.setMinimumWidth(620)
        self.setMinimumHeight(400)

        seq = app_state.restoration_sequence
        proj = app_state._project

        all_names: set[str] = set()
        id_to_name: dict[str, str] = {}        # UUID → display name, for removals
        for coll in (proj.horizon_picks, proj.fault_picks, proj.polygons):
            for obj in coll:
                nm = getattr(obj, "name", "")
                if nm:
                    all_names.add(nm)
                uid = getattr(obj, "uuid", None)
                if uid:
                    id_to_name[uid] = nm or uid
        line_names = {rl.uuid: (rl.name or "(unnamed)")
                      for rl in getattr(proj, "reference_lines", [])
                      if getattr(rl, "uuid", None)}

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # ── Summary header ────────────────────────────────────────────────
        n_events = len(seq.events)
        current = seq.current_step
        hdr = QLabel(
            f"<b>Events:</b> {n_events}   "
            f"<b>Current step:</b> {current}   "
            f"<b>Named elements:</b> {len(all_names)}"
        )
        hdr.setTextFormat(Qt.RichText)
        layout.addWidget(hdr)

        # ── Step table ────────────────────────────────────────────────────
        box = QGroupBox("Restoration Steps  (step 0 = present day)")
        box_layout = QVBoxLayout(box)

        if not seq.events:
            box_layout.addWidget(QLabel("No restoration events defined.\n"
                                        "Use the Restoration Panel (Ctrl+6) to add events."))
        else:
            table = QTableWidget(n_events + 1, 6)
            table.setHorizontalHeaderLabels(
                ["Step", "Event Name", "Age (Ma)", "Removed at this step",
                 "Cumulative removed", "Algorithm / assumptions"])
            table.horizontalHeader().setStretchLastSection(True)
            table.verticalHeader().setVisible(False)
            table.setEditTriggers(QTableWidget.NoEditTriggers)
            table.setSelectionMode(QTableWidget.SingleSelection)
            table.setAlternatingRowColors(True)

            cumulative: set[str] = set()

            # Row 0: present day
            table.setItem(0, 0, _cell("0", Qt.AlignCenter | Qt.AlignVCenter))
            table.setItem(0, 1, _cell("Present day"))
            table.setItem(0, 2, _cell("—", Qt.AlignCenter | Qt.AlignVCenter))
            table.setItem(0, 3, _cell("—"))
            table.setItem(0, 4, _cell("0 / " + str(len(all_names))))
            table.setItem(0, 5, _cell("— (present day)"))
            if current == 0:
                _highlight_row(table, 0)

            for row_i, ev in enumerate(seq.events, start=1):
                # Resolved UUIDs → names, plus any unresolved legacy names (shown
                # with a marker so a broken reference is visible, not hidden).
                removed_here = [id_to_name.get(uid, uid) for uid in ev.remove_element_ids]
                removed_here += [f"{nm} (?)" for nm in ev.remove_element_names]
                cumulative.update(removed_here)
                age_str = f"{ev.age_ma:.1f}" if ev.age_ma is not None else "—"
                removed_str = ", ".join(removed_here) if removed_here else "(none)"
                cum_str = f"{len(cumulative)} / {len(all_names)}"

                table.setItem(row_i, 0, _cell(str(row_i), Qt.AlignCenter | Qt.AlignVCenter))
                table.setItem(row_i, 1, _cell(ev.name))
                table.setItem(row_i, 2, _cell(age_str, Qt.AlignCenter | Qt.AlignVCenter))
                table.setItem(row_i, 3, _cell(removed_str))
                table.setItem(row_i, 4, _cell(cum_str, Qt.AlignCenter | Qt.AlignVCenter))
                table.setItem(row_i, 5, _cell(_algo_summary(ev, line_names)))
                if current == row_i:
                    _highlight_row(table, row_i)

            table.resizeColumnsToContents()
            table.horizontalHeader().setStretchLastSection(True)
            box_layout.addWidget(table)

        layout.addWidget(box)

        # ── Close ─────────────────────────────────────────────────────────
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def _algo_summary(ev, line_names: dict) -> str:
    """Compact, auditable summary of an event's restoration assumptions:
    algorithm + pin/datum (line name or numeric) + key params."""
    from section_tool.core.kinematics import ALGORITHM_LABELS
    algo = getattr(ev, "algorithm", "none")
    if algo in ("none", None):
        return "— (remove only)"
    parts = [ALGORITHM_LABELS.get(algo, algo)]
    p = getattr(ev, "params", {}) or {}
    if getattr(ev, "pin_line_id", None):
        parts.append(f"pin: {line_names.get(ev.pin_line_id, 'line')}")
    elif "pin_x" in p:
        parts.append(f"pin x={p['pin_x']:g}")
    if getattr(ev, "datum_line_id", None):
        parts.append(f"datum: {line_names.get(ev.datum_line_id, 'line')}")
    elif "datum_y" in p:
        parts.append(f"datum={p['datum_y']:g}")
    for k in ("dx", "dy", "shear_angle", "slip"):
        if k in p:
            parts.append(f"{k}={p[k]:g}")
    return ", ".join(parts)


def _cell(text: str, align: Qt.Alignment = Qt.AlignLeft | Qt.AlignVCenter) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(align)
    item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
    return item


def _highlight_row(table: QTableWidget, row: int) -> None:
    """Mark the current restoration step row with bold text."""
    for col in range(table.columnCount()):
        item = table.item(row, col)
        if item is not None:
            font = item.font()
            font.setBold(True)
            item.setFont(font)
