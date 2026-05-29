"""Phase 2 scenario 6 — construction-tool STATE-MACHINE wiring in SectionView.

The tool classes (Dip/Parallel/Kink) and the trim/extend geometry primitives
are already unit-tested (test_construction_tools.py). This harness covers the
SectionView glue that orchestrates them — the part the driving checklist cares
about and nothing else exercises:

  * tool activation / reset via set_ref_line_tool
  * Dip-constrained and Kink-band click sequences create picks
  * continuous mode: extend/trim stay armed after a commit; parallel/dip don't
  * free-extend end-to-end adds an endpoint and re-arms
  * right-click cancels an in-progress construction
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication

from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick
from section_tool.core.construction import DipConstrainedRule, KinkBandRule
from section_tool.views.section_view import SectionView


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


def _view(qapp, picks=()):
    state = AppState()
    sec = Section([(0.0, 0.0), (10000.0, 0.0)], name="L1")
    state.add_section(sec)
    state.set_active_section(sec)
    for hp in picks:
        state.add_horizon_pick(hp)
    view = SectionView(state)
    view.render()   # initialise transData for any screen-space hit tests
    return view, state


def _line(name="BL"):
    return HorizonPick([0.0, 1000.0], [100.0, 200.0], name=name, section_names=["L1", "L1"])


# ---------------------------------------------------------------------------
# Tool activation / reset
# ---------------------------------------------------------------------------

class TestToolActivation:

    def test_activate_sets_construct_tool_and_idle(self, qapp):
        view, _ = _view(qapp)
        view.set_ref_line_tool("extend")
        assert view._construct_tool == "extend"
        assert view._cst_state == "idle"
        assert view._cst_source is None

    def test_switch_away_resets_tool_objects(self, qapp):
        view, _ = _view(qapp)
        view.set_ref_line_tool("dip_constrained")
        view._cst_dip_tool._anchor = (1.0, 2.0)         # simulate mid-sequence
        view.set_ref_line_tool("select")               # non-construct tool
        assert view._construct_tool is None
        assert view._cst_dip_tool.state == "idle"       # reset


# ---------------------------------------------------------------------------
# Dip-constrained: two clicks -> a new horizon pick with the rule attached
# ---------------------------------------------------------------------------

class TestDipConstrained:

    def test_two_clicks_create_pick(self, qapp):
        view, state = _view(qapp)
        view.set_ref_line_tool("dip_constrained")
        view._cst_dip_tool.dip_deg = 0.0
        n0 = len(state.project.horizon_picks)

        view._handle_construct_click(0.0, 100.0)        # anchor
        assert view._cst_dip_tool.state == "anchor_set"
        assert len(state.project.horizon_picks) == n0

        view._handle_construct_click(1000.0, 100.0)     # extent -> commit
        assert len(state.project.horizon_picks) == n0 + 1
        new = state.project.horizon_picks[-1]
        assert isinstance(new.construction_rule, DipConstrainedRule)
        assert state.active_tool == "select"            # tool auto-deactivates


# ---------------------------------------------------------------------------
# Kink-band: select backlimb -> axial click -> forelimb pick
# ---------------------------------------------------------------------------

class TestKinkBand:

    def test_reference_then_axial_creates_forelimb(self, qapp):
        view, state = _view(qapp, picks=[_line("Backlimb")])
        view.set_ref_line_tool("kink_band")
        view._selected_object = ("Horizons", 0)
        n0 = len(state.project.horizon_picks)

        view._handle_construct_click(300.0, 130.0)      # sets reference
        assert view._cst_kink_tool.state == "ref_selected"
        assert len(state.project.horizon_picks) == n0

        view._handle_construct_click(600.0, 0.0)        # axial -> forelimb
        assert len(state.project.horizon_picks) == n0 + 1
        assert isinstance(state.project.horizon_picks[-1].construction_rule, KinkBandRule)


# ---------------------------------------------------------------------------
# Continuous mode branching (isolates the wiring; geometry stubbed True)
# ---------------------------------------------------------------------------

class TestContinuousMode:

    def _arm(self, view, tool):
        view.set_ref_line_tool(tool)
        view._cst_source = {"cat": "Horizons", "idx": 0, "endpoint": "end"}
        view._cst_state = "source_selected"

    def test_extend_stays_armed_after_commit(self, qapp, monkeypatch):
        view, state = _view(qapp, picks=[_line()])
        self._arm(view, "extend")
        monkeypatch.setattr(view, "_cst_second_click", lambda *a, **k: True)
        view._handle_construct_click(2000.0, 50.0)
        assert view._cst_state == "source_selected"     # re-armed for next click

    def test_trim_stays_armed_after_commit(self, qapp, monkeypatch):
        view, state = _view(qapp, picks=[_line()])
        self._arm(view, "trim")
        monkeypatch.setattr(view, "_cst_second_click", lambda *a, **k: True)
        view._handle_construct_click(500.0, 150.0)
        assert view._cst_state == "source_selected"

    def test_parallel_returns_to_idle_after_commit(self, qapp, monkeypatch):
        view, state = _view(qapp, picks=[_line()])
        self._arm(view, "parallel")
        monkeypatch.setattr(view, "_cst_second_click", lambda *a, **k: True)
        view._handle_construct_click(500.0, 250.0)
        assert view._cst_state == "idle"
        assert view._cst_source is None
        assert state.active_tool == "select"


# ---------------------------------------------------------------------------
# Free-extend end-to-end: adds a new endpoint, re-arms (continuous)
# ---------------------------------------------------------------------------

class TestFreeExtend:

    def test_extend_adds_endpoint_and_rearms(self, qapp):
        view, state = _view(qapp, picks=[_line("H")])
        view._selected_object = ("Horizons", 0)
        view.set_ref_line_tool("extend")

        view._handle_construct_click(500.0, 150.0)       # sel preset -> arm source_selected
        assert view._cst_state == "source_selected"

        n_before = state.project.horizon_picks[0].n_picks
        view._handle_construct_click(3000.0, 250.0)      # second click -> free extend
        assert state.project.horizon_picks[0].n_picks == n_before + 1
        assert view._cst_state == "source_selected"      # continuous: still armed


# ---------------------------------------------------------------------------
# Right-click cancels an in-progress construction
# ---------------------------------------------------------------------------

class TestRightClickCancel:

    def test_right_click_resets_state(self, qapp):
        view, state = _view(qapp, picks=[_line()])
        view.set_ref_line_tool("extend")
        view._cst_source = {"cat": "Horizons", "idx": 0, "endpoint": "end"}
        view._cst_state = "source_selected"

        evt = SimpleNamespace(inaxes=view._ax, x=None, y=None,
                              xdata=500.0, ydata=150.0, button=3, dblclick=False)
        view._on_sv_press(evt)

        assert view._cst_state == "idle"
        assert view._cst_source is None
        assert state.active_tool == "select"
