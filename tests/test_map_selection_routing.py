"""Map-click section selection/creation routes through the active-slice router.

Build B prerequisite: picking or drawing a section on the map must switch the
active *workspace* away from a z-slice (route via set_active_slice), not just
set active_section directly.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication

from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.core.slices import HorizontalSlice
from section_tool.views.map_view import MapView


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


def _state_with_zslice_active(qapp):
    st = AppState()
    st.set_active_tool("select")
    sec = Section([(0.0, 0.0), (1000.0, 0.0)], name="L1")
    st.add_section(sec)
    hs = HorizontalSlice("Z-1500", -1500.0)
    st.project.horizontal_slices.append(hs)
    st.set_active_slice(hs)                 # z-slice is the active workspace
    return st, sec, hs


class TestSelectionRouting:
    def test_section_selection_switches_workspace_off_zslice(self, qapp):
        st, sec, hs = _state_with_zslice_active(qapp)
        assert st.active_slice is hs        # precondition: z-slice active
        view = MapView(st)
        view._find_nearest_section = lambda x, y: 0   # click hits the section
        evt = SimpleNamespace(button=1, xdata=500.0, ydata=0.0, x=10.0, y=10.0,
                              dblclick=False, inaxes=view._ax)
        view._on_canvas_press(evt)
        # routed through set_active_slice → workspace is now the section
        assert st.active_slice is sec
        assert st.active_section is sec     # delegation to set_active_section intact

    def test_quick_section_creation_routes(self, qapp):
        st, _sec, hs = _state_with_zslice_active(qapp)
        view = MapView(st)
        before = len(st.project.sections)
        view._create_section_at(500.0, 500.0, "ew")   # quick E-W section
        assert len(st.project.sections) == before + 1
        new_sec = st.project.sections[-1]
        assert st.active_slice is new_sec   # created section is the active workspace
        assert st.active_section is new_sec
