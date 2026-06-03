"""ZSliceView: the plan view renders the window-at-z0 content (headless)."""
from __future__ import annotations

import sys

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.core.slices import HorizontalSlice
from section_tool.core.surfaces import HorizonPick
from section_tool.views.zslice_view import ZSliceView


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


@pytest.fixture
def state(qapp):
    st = AppState()
    st.add_section(Section([(0.0, 0.0), (2000.0, 0.0)], name="L1"))   # E-W trace
    # fault on L1 dipping 1000 -> 2000 over 0 -> 2000; crosses z0=-1500 at d=1000
    st.add_fault_pick(HorizonPick(np.array([0.0, 2000.0]), np.array([1000.0, 2000.0]),
                                  name="F1", color="#d62728", section_names=["L1", "L1"]))
    st.project.horizontal_slices.append(HorizontalSlice("Z-1500", -1500.0))
    return st


class TestZSliceRender:
    def test_exposes_map_hud_interface(self, state):
        zv = ZSliceView(state)
        assert hasattr(zv, "canvas") and hasattr(zv, "axes")
        assert hasattr(zv, "cursor_map_pos")          # signal MapHUDLayer needs

    def test_renders_section_trace_and_piercing(self, state):
        zv = ZSliceView(state)
        zv.resize(400, 400)
        zv.set_slice(state.project.horizontal_slices[0])
        ax = zv.axes
        # the section trace is drawn in plan (an E-W line at y=0)
        traces = [l for l in ax.lines if list(l.get_ydata()) == [0.0, 0.0]]
        assert traces, "section trace not drawn in plan"
        # piercing dot: open-circle marker, single point, at world (1000, 0)
        dots = [l for l in ax.lines if l.get_marker() == "o" and len(l.get_xdata()) == 1]
        assert len(dots) == 1
        assert dots[0].get_xdata()[0] == pytest.approx(1000.0)
        assert dots[0].get_ydata()[0] == pytest.approx(0.0)

    def test_equal_aspect_datalim(self, state):
        zv = ZSliceView(state)
        zv.set_slice(state.project.horizontal_slices[0])
        assert zv.axes.get_aspect() == 1.0
        assert zv.axes.get_adjustable() == "datalim"

    def test_no_slice_renders_empty(self, state):
        zv = ZSliceView(state)
        zv.render()                                    # no slice set
        assert len(zv.axes.lines) == 0

    def test_cursor_signal_emits(self, state):
        zv = ZSliceView(state)
        seen = []
        zv.cursor_map_pos.connect(lambda x, y: seen.append((x, y)))
        from types import SimpleNamespace
        zv._on_motion(SimpleNamespace(xdata=500.0, ydata=250.0, x=None, y=None))
        assert seen == [(500.0, 250.0)]
