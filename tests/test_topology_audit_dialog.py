"""Phase 2 scenario 11 — TopologyAuditDialog integration (headless).

audit_section() and the fix actions are unit-tested in test_topology_audit.py.
This harness covers the dialog glue that nothing else exercises: running the
audit on the active section on open, populating the table, the summary line,
and the Fix / Fix-All buttons cascading back through AppState.
"""
from __future__ import annotations

import sys

import pytest
from PySide6.QtWidgets import QApplication

from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick
from section_tool.views.topology_audit_dialog import TopologyAuditDialog


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


def _state_with_duplicate():
    """AppState whose active section has a horizon with an exact duplicate node
    (an auto-fixable topology issue)."""
    state = AppState()
    sec = Section([(0.0, 0.0), (1000.0, 0.0)], name="L1")
    state.add_section(sec)
    state.set_active_section(sec)
    hp = HorizonPick.empty(name="Dup")
    hp.insert_pick(0.0, 100.0, "L1")
    hp.insert_pick(500.0, 200.0, "L1")
    hp.insert_pick(500.0, 200.0, "L1")   # exact duplicate -> auto-fixable
    hp.insert_pick(1000.0, 300.0, "L1")
    state.add_horizon_pick(hp)
    return state


def _state_clean():
    state = AppState()
    sec = Section([(0.0, 0.0), (1000.0, 0.0)], name="L1")
    state.add_section(sec)
    state.set_active_section(sec)
    hp = HorizonPick([0.0, 500.0, 1000.0], [100.0, 200.0, 300.0],
                     name="Clean", section_names=["L1", "L1", "L1"])
    state.add_horizon_pick(hp)
    return state


# ---------------------------------------------------------------------------
# Audit runs on open and populates the table
# ---------------------------------------------------------------------------

class TestDialogPopulates:

    def test_audit_runs_on_construction(self, qapp):
        dlg = TopologyAuditDialog(_state_with_duplicate())
        assert len(dlg._issues) >= 1
        assert any("duplicate" in i.description.lower() for i in dlg._issues)

    def test_table_row_count_matches_issues(self, qapp):
        dlg = TopologyAuditDialog(_state_with_duplicate())
        assert dlg._table.rowCount() == len(dlg._issues)

    def test_summary_reports_fixable(self, qapp):
        dlg = TopologyAuditDialog(_state_with_duplicate())
        assert "auto-fixable" in dlg._summary_label.text()
        assert dlg._fix_all_btn.isEnabled()


# ---------------------------------------------------------------------------
# Fix-all and single-fix cascade through state and re-run the audit
# ---------------------------------------------------------------------------

class TestDialogFixes:

    def test_fix_all_removes_duplicate(self, qapp):
        state = _state_with_duplicate()
        dlg = TopologyAuditDialog(state)
        assert any(i.auto_fixable for i in dlg._issues)

        dlg._on_fix_all()   # applies fixes, then re-runs the audit

        # the duplicate node is gone from the underlying pick...
        idxs = state.project.horizon_picks[0].section_indices("L1")
        assert len(idxs) == 3
        # ...and the re-run audit no longer reports a duplicate
        assert not any("duplicate" in i.description.lower() for i in dlg._issues)

    def test_single_fix_button_action(self, qapp):
        state = _state_with_duplicate()
        dlg = TopologyAuditDialog(state)
        dupe = next(i for i in dlg._issues if i.auto_fixable)
        dlg._apply_fix(dupe)
        assert len(state.project.horizon_picks[0].section_indices("L1")) == 3


# ---------------------------------------------------------------------------
# Edge cases: clean section and no active section
# ---------------------------------------------------------------------------

class TestDialogEdgeCases:

    def test_clean_section_reports_no_issues(self, qapp):
        dlg = TopologyAuditDialog(_state_clean())
        assert dlg._issues == []
        assert dlg._table.rowCount() == 0
        assert dlg._summary_label.text() == "No issues found."
        assert not dlg._fix_all_btn.isEnabled()

    def test_no_active_section(self, qapp):
        state = AppState()   # no section added/activated
        dlg = TopologyAuditDialog(state)
        assert dlg._issues == []
        assert dlg._table.rowCount() == 0
        assert dlg._summary_label.text() == "No active section."
