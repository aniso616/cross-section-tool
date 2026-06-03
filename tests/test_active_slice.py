"""Model A: active_slice alongside active_section — additive, behaviour-identical.

The critical guarantee: selecting a Section via set_active_slice is byte-identical
to set_active_section today; a HorizontalSlice leaves active_section untouched.
"""
from __future__ import annotations

import sys

import pytest
from PySide6.QtWidgets import QApplication

from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.core.slices import HorizontalSlice


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


@pytest.fixture
def state(qapp):
    st = AppState()
    st.add_section(Section([(0.0, 0.0), (1000.0, 0.0)], name="L1"))
    st.add_section(Section([(0.0, 500.0), (1000.0, 500.0)], name="L2"))
    return st


# ---------------------------------------------------------------------------
# Section via set_active_slice ≡ set_active_section (the regression gate)
# ---------------------------------------------------------------------------

class TestSectionDelegation:
    def test_section_sets_active_section(self, state):
        sec = state.project.sections[0]
        state.set_active_slice(sec)
        assert state.active_section is sec
        assert state.active_slice is sec

    def test_section_fires_both_signals(self, state):
        sec = state.project.sections[0]
        seen = {"section": [], "slice": []}
        state.active_section_changed.connect(lambda s: seen["section"].append(s))
        state.active_slice_changed.connect(lambda s: seen["slice"].append(s))
        state.set_active_slice(sec)
        assert seen["section"] == [sec]   # delegated → existing signal fires as today
        assert seen["slice"] == [sec]

    def test_identical_to_set_active_section(self, state):
        # set_active_slice(section) leaves the world in the same state as
        # set_active_section(section) would.
        sec = state.project.sections[1]
        state.set_active_slice(sec)
        via_slice = state.active_section
        state.set_active_section(None)
        state.set_active_section(sec)
        via_section = state.active_section
        assert via_slice is via_section is sec

    def test_none_clears_active_section(self, state):
        state.set_active_slice(state.project.sections[0])
        state.set_active_slice(None)
        assert state.active_section is None
        assert state.active_slice is None


# ---------------------------------------------------------------------------
# HorizontalSlice does NOT disturb active_section
# ---------------------------------------------------------------------------

class TestHorizontalRouting:
    def test_horizontal_leaves_active_section_untouched(self, state):
        sec = state.project.sections[0]
        state.set_active_slice(sec)                     # a section is active
        hs = HorizontalSlice("Z-1500", -1500.0)
        state.set_active_slice(hs)
        assert state.active_slice is hs                 # slice selection updated
        assert state.active_section is sec              # active_section UNCHANGED

    def test_horizontal_does_not_fire_active_section_changed(self, state):
        hs = HorizontalSlice("Z-1500", -1500.0)
        seen_section, seen_slice = [], []
        state.active_section_changed.connect(lambda s: seen_section.append(s))
        state.active_slice_changed.connect(lambda s: seen_slice.append(s))
        state.set_active_slice(hs)
        assert seen_section == []                       # section surface undisturbed
        assert seen_slice == [hs]

    def test_kind_predicate_distinguishes(self, state):
        state.set_active_slice(HorizontalSlice("Z", -1000.0))
        assert state.active_slice.kind == "horizontal"
        state.set_active_slice(state.project.sections[0])
        assert state.active_slice.kind == "section"

    def test_emits_only_on_change(self, state):
        hs = HorizontalSlice("Z", -1000.0)
        seen = []
        state.active_slice_changed.connect(lambda s: seen.append(s))
        state.set_active_slice(hs)
        state.set_active_slice(hs)        # same → no second emit
        assert seen == [hs]
