"""Build B: drawing freehand plan fault traces on the active z-slice.

Mirrors section fault picking in plan orientation: select fault → click along its
trace in world E/N at fixed depth -z0, slice_kind='horizontal', freehand.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.core.slices import HorizontalSlice
from section_tool.core.surfaces import HorizonPick
from section_tool.views.zslice_view import ZSliceView

# A deliberately non-monotonic click order (zigzag) — tests that cumulative
# distance preserves DRAW order rather than sorting by easting.
TRACE = [(600_000.0, 6_080_000.0), (601_000.0, 6_080_000.0),
         (600_500.0, 6_081_000.0), (602_000.0, 6_080_500.0)]


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


def _setup(qapp, folder=None):
    st = AppState()
    if folder:
        st.new_project(name="B", crs_epsg=32631, folder_path=folder)
    st.add_section(Section([(600_000, 6_080_000), (612_000, 6_092_000)], name="L1"))
    st.add_fault_pick(HorizonPick.empty(name="F1", color="#d62728"))
    hs = HorizontalSlice("Z-1500", -1500.0, 32631)
    st.project.horizontal_slices.append(hs)
    st.set_active_slice(hs)
    st.set_active_pick_target("Faults", 0)
    view = ZSliceView(st)
    view.set_slice(hs)
    view.set_fault_drawing(True)
    return st, view, hs


def _draw(view, pts):
    for e, n in pts:
        view._add_plan_pick(e, n)


# ---------------------------------------------------------------------------
# Drawing mechanics
# ---------------------------------------------------------------------------

class TestPlanDraw:
    def test_points_stored_as_horizontal_with_world_and_depth(self, qapp):
        st, view, hs = _setup(qapp)
        _draw(view, TRACE)
        f = st.project.fault_picks[0]
        idx = f.indices_for_slice("horizontal", "Z-1500")
        assert len(idx) == len(TRACE)
        assert all(f._slice_kinds[i] == "horizontal" for i in idx)
        assert np.allclose(f._depths[idx], 1500.0)            # depth = -z0
        assert all(f._section_names[i] == "Z-1500" for i in idx)

    def test_world_coords_are_source_of_truth(self, qapp):
        st, view, _ = _setup(qapp)
        _draw(view, TRACE)
        f = st.project.fault_picks[0]
        idx = f.indices_for_slice("horizontal", "Z-1500")
        assert list(zip(f._map_x[idx], f._map_y[idx])) == \
            [pytest.approx(p) for p in TRACE]

    def test_cumulative_distance_preserves_draw_order(self, qapp):
        # The subtle detail: a zigzag must render in DRAW order, not sorted by E.
        st, view, _ = _setup(qapp)
        _draw(view, TRACE)
        f = st.project.fault_picks[0]
        idx = f.indices_for_slice("horizontal", "Z-1500")
        dists = [float(f._distances[i]) for i in idx]
        assert dists == sorted(dists)                          # monotonic
        assert dists[0] == 0.0 and dists == sorted(set(dists)) # strictly increasing
        # the polyline (map_x[idx], map_y[idx]) is in draw order
        assert [float(f._map_x[i]) for i in idx] == [p[0] for p in TRACE]

    def test_freehand_no_construction_rule(self, qapp):
        st, view, _ = _setup(qapp)
        _draw(view, TRACE)
        assert st.project.fault_picks[0].construction_rule is None

    def test_no_active_fault_target_is_noop(self, qapp):
        st, view, _ = _setup(qapp)
        st.set_active_pick_target("Faults", 99)   # invalid index
        warned = []
        view.status_message.connect(lambda m: warned.append(m))
        view._add_plan_pick(600_000.0, 6_080_000.0)
        assert warned and "fault" in warned[0].lower()
        assert st.project.fault_picks[0].n_picks == 0


# ---------------------------------------------------------------------------
# Press routing (left draws, right ends) only in draw mode
# ---------------------------------------------------------------------------

class TestPressRouting:
    def test_left_click_draws_in_mode(self, qapp):
        st, view, _ = _setup(qapp)
        view._on_press(SimpleNamespace(button=1, xdata=600_000.0, ydata=6_080_000.0,
                                       x=10.0, y=10.0))
        assert st.project.fault_picks[0].n_picks == 1

    def test_right_click_ends_trace(self, qapp):
        st, view, _ = _setup(qapp)
        ended = []
        view.draw_ended.connect(lambda: ended.append(True))
        view._on_press(SimpleNamespace(button=3, xdata=1.0, ydata=2.0, x=1.0, y=2.0))
        assert ended == [True]

    def test_left_click_pans_when_not_drawing(self, qapp):
        st, view, _ = _setup(qapp)
        view.set_fault_drawing(False)
        view._on_press(SimpleNamespace(button=1, xdata=1.0, ydata=2.0, x=5.0, y=6.0))
        assert view._pan_anchor == (5.0, 6.0)        # pan armed, no pick added
        assert st.project.fault_picks[0].n_picks == 0


# ---------------------------------------------------------------------------
# Round-trip: a fault carries a section trace AND a plan trace (the payoff)
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_plan_trace_round_trips_freehand(self, qapp, tmp_path):
        folder = str(tmp_path / "proj")
        st, view, _ = _setup(qapp, folder=folder)
        # give the fault a section observation too, so it spans both slice kinds
        st.project.fault_picks[0].insert_pick(0.0, 700.0, section_name="L1",
                                              map_x=600_000.0, map_y=6_080_000.0)
        _draw(view, TRACE)
        uuid = st.project.fault_picks[0].uuid
        st.save_project()

        dst = AppState(); dst.open_project(folder)
        f = dst.project.fault_picks[0]
        assert f.uuid == uuid
        assert set(f.section_names()) == {"L1"}                # section observation
        idx = f.indices_for_slice("horizontal", "Z-1500")      # plan observation
        assert len(idx) == len(TRACE)
        assert [float(f._map_x[i]) for i in idx] == [p[0] for p in TRACE]   # order kept
        assert np.allclose(f._depths[idx], 1500.0)
        assert f.construction_rule is None                     # freehand survived
