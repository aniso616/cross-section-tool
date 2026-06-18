"""RestorationPanel Step 2: the event editor defines which elements an event
removes (UUID-keyed), flags upstream removals, and persists through the sequence."""
from __future__ import annotations

import sys

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog

from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick
from section_tool.core.restoration import RestorationEvent
from section_tool.views.restoration_panel import RestorationPanel, _EventEditDialog


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


def _state_with_horizon(name="Top Chalk"):
    state = AppState()
    state.add_section(Section([(0.0, 0.0), (1000.0, 0.0)], name="L1", crs_epsg=32631))
    state.set_active_section(state.project.sections[0])
    hp = HorizonPick([0.0, 1000.0], [100.0, 200.0], name=name,
                     section_names=["L1", "L1"])
    state.project.horizon_picks.append(hp)
    return state, hp


# ── dialog ────────────────────────────────────────────────────────────────

def test_dialog_values_returns_checked_uuids(qapp):
    removable = [("uuid-a", "Top", "Horizon"), ("uuid-b", "Base", "Horizon")]
    dlg = _EventEditDialog(removable=removable)
    dlg._elem_list.item(0).setCheckState(Qt.Checked)
    assert dlg.values["remove_element_ids"] == ["uuid-a"]
    dlg._elem_list.item(1).setCheckState(Qt.Checked)
    assert dlg.values["remove_element_ids"] == ["uuid-a", "uuid-b"]


def test_dialog_prechecks_existing_event_ids(qapp):
    removable = [("uuid-a", "Top", "Horizon"), ("uuid-b", "Base", "Horizon")]
    ev = RestorationEvent(1, "e", remove_element_ids=["uuid-b"])
    dlg = _EventEditDialog(event=ev, removable=removable)
    assert dlg._elem_list.item(0).checkState() == Qt.Unchecked
    assert dlg._elem_list.item(1).checkState() == Qt.Checked


def test_dialog_flags_already_removed_upstream(qapp):
    dlg = _EventEditDialog(removable=[("uuid-a", "Top", "Horizon")],
                           already_removed={"uuid-a": 2})
    item = dlg._elem_list.item(0)
    assert "removed at step 2" in item.text()
    # flagged, not forbidden — still checkable
    assert bool(item.flags() & Qt.ItemIsUserCheckable)


def test_dialog_unresolved_names_are_display_only(qapp):
    ev = RestorationEvent(1, "e", remove_element_names=["Ghost Bed"])
    dlg = _EventEditDialog(event=ev, removable=[])
    item = dlg._elem_list.item(0)
    assert "Ghost Bed" in item.text() and "unresolved" in item.text()
    assert not (item.flags() & Qt.ItemIsUserCheckable)
    assert dlg.values["remove_element_ids"] == []           # never selectable


# ── panel ─────────────────────────────────────────────────────────────────

def test_panel_add_event_writes_uuid_and_persists(qapp, monkeypatch):
    state, hp = _state_with_horizon()
    panel = RestorationPanel(state)

    def fake_exec(self):
        self._name.setText("Remove Chalk")
        self._elem_list.item(0).setCheckState(Qt.Checked)    # the horizon
        return QDialog.Accepted
    monkeypatch.setattr(_EventEditDialog, "exec", fake_exec)

    panel._add_event()
    seq = state.restoration_sequence
    assert len(seq.events) == 1
    assert seq.events[0].remove_element_ids == [hp.uuid]     # UUID, not name
    assert seq.events[0].name == "Remove Chalk"
    # reflected in the table's "Removes" count
    assert panel._table.item(0, 3).text() == "1"


def test_panel_edit_round_trips_checked_set(qapp, monkeypatch):
    state, hp = _state_with_horizon()
    seq = state.restoration_sequence
    seq.add_event(RestorationEvent(1, "e"))                  # no removals yet
    state.set_restoration_sequence(seq)
    panel = RestorationPanel(state)
    panel._table.selectRow(0)

    def fake_exec(self):
        self._elem_list.item(0).setCheckState(Qt.Checked)
        return QDialog.Accepted
    monkeypatch.setattr(_EventEditDialog, "exec", fake_exec)
    panel._edit_event()
    assert state.restoration_sequence.events[0].remove_element_ids == [hp.uuid]

    # Re-opening the editor shows it checked (persisted round-trip).
    dlg = _EventEditDialog(event=state.restoration_sequence.events[0],
                           removable=panel._removable_elements())
    assert dlg._elem_list.item(0).checkState() == Qt.Checked


def test_panel_picker_resolves_by_uuid_after_rename(qapp):
    state, hp = _state_with_horizon()
    seq = state.restoration_sequence
    seq.add_event(RestorationEvent(1, "e", remove_element_ids=[hp.uuid]))
    state.set_restoration_sequence(seq)
    hp.name = "Renamed Chalk"                                # rename after the event
    panel = RestorationPanel(state)
    dlg = _EventEditDialog(event=seq.events[0], removable=panel._removable_elements())
    item = dlg._elem_list.item(0)
    assert item.data(Qt.UserRole) == hp.uuid                # keyed by uuid
    assert "Renamed Chalk" in item.text()                   # shows the live name
    assert item.checkState() == Qt.Checked                  # still checked


def test_dialog_param_visibility_follows_algorithm(qapp):
    dlg = _EventEditDialog(removable=[])
    dlg._algo.setCurrentIndex(dlg._algo.findData("flexural_slip"))
    dlg._update_param_visibility()
    assert dlg._p_pin_x.isVisibleTo(dlg) and dlg._p_datum_y.isVisibleTo(dlg)
    assert not dlg._p_dx.isVisibleTo(dlg) and not dlg._p_slip.isVisibleTo(dlg)


def test_panel_add_event_writes_algorithm_and_params(qapp, monkeypatch):
    state, hp = _state_with_horizon()
    panel = RestorationPanel(state)

    def fake_exec(self):
        self._name.setText("Unfold")
        self._algo.setCurrentIndex(self._algo.findData("rigid_translation"))
        self._p_dx.setValue(250.0)
        self._p_dy.setValue(-10.0)
        return QDialog.Accepted
    monkeypatch.setattr(_EventEditDialog, "exec", fake_exec)

    panel._add_event()
    ev = state.restoration_sequence.events[0]
    assert ev.algorithm == "rigid_translation"
    assert ev.params == {"dx": 250.0, "dy": -10.0}


def test_dialog_prefills_algorithm_and_params_on_edit(qapp):
    ev = RestorationEvent(1, "e", algorithm="simple_shear",
                          params={"shear_angle": 30.0, "datum_y": 0.0})
    dlg = _EventEditDialog(event=ev, removable=[])
    assert dlg._algo.currentData() == "simple_shear"
    assert dlg._p_shear.value() == pytest.approx(30.0)
    assert dlg.values["params"] == {"shear_angle": 30.0, "datum_y": 0.0}


def test_panel_already_removed_reflects_earlier_event(qapp):
    state, hp = _state_with_horizon()
    seq = state.restoration_sequence
    seq.add_event(RestorationEvent(1, "first", remove_element_ids=[hp.uuid]))
    seq.add_event(RestorationEvent(2, "second"))
    state.set_restoration_sequence(seq)
    panel = RestorationPanel(state)
    already = panel._already_removed_before(1)               # before the 2nd event
    assert already.get(hp.uuid) == 1                         # removed at step 1
