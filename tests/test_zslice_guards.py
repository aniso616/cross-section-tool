"""Section-only tools are disabled when a horizontal z-slice is the workspace."""
from __future__ import annotations

import sys

import pytest
from PySide6.QtWidgets import QApplication

from section_tool.views.tool_palette import ToolPalette


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


SECTION_PLANE = ["horizon_pick", "fault_pick", "polygon", "h_ref", "v_ref",
                 "a_ref", "extend", "trim", "parallel", "dip_constrained",
                 "kink_band", "measure", "node_edit"]
ALWAYS_ON = ["select", "pan", "zoom", "new_section"]


class TestSectionWorkspaceGuard:
    def test_section_plane_tools_disabled_on_zslice(self, qapp):
        p = ToolPalette()
        # a section + picks exist, but a z-slice is the active workspace
        p.update_tool_availability(has_section=True, has_picks=True,
                                   section_workspace=False)
        for tid in SECTION_PLANE:
            assert not p._buttons[tid]._btn.isEnabled(), f"{tid} should be disabled"

    def test_navigation_and_draw_stay_enabled_on_zslice(self, qapp):
        p = ToolPalette()
        p.update_tool_availability(has_section=True, has_picks=True,
                                   section_workspace=False)
        for tid in ALWAYS_ON:
            assert p._buttons[tid]._btn.isEnabled(), f"{tid} should stay enabled"

    def test_default_section_workspace_unchanged(self, qapp):
        # Backward compat: default section_workspace=True behaves as before.
        p = ToolPalette()
        p.update_tool_availability(has_section=True, has_picks=True)
        for tid in SECTION_PLANE:
            assert p._buttons[tid]._btn.isEnabled()

    def test_construction_tools_in_section_workspace_set(self):
        for tid in ("dip_constrained", "kink_band", "extend", "trim", "parallel"):
            assert tid in ToolPalette._SECTION_WORKSPACE


class TestZSliceWorkspaceGuard:
    """The plan fault-draw tool is the inverse: enabled ONLY on a z-slice."""

    def test_plan_fault_disabled_in_section_workspace(self, qapp):
        p = ToolPalette()
        p.update_tool_availability(has_section=True, has_picks=True,
                                   section_workspace=True)
        assert not p._buttons["plan_fault"]._btn.isEnabled()

    def test_plan_fault_enabled_on_zslice(self, qapp):
        p = ToolPalette()
        p.update_tool_availability(has_section=True, has_picks=True,
                                   section_workspace=False)
        assert p._buttons["plan_fault"]._btn.isEnabled()

    def test_plan_fault_enabled_on_zslice_without_section_or_picks(self, qapp):
        # Independent of has_section/has_picks — only the workspace matters.
        p = ToolPalette()
        p.update_tool_availability(has_section=False, has_picks=False,
                                   section_workspace=False)
        assert p._buttons["plan_fault"]._btn.isEnabled()

    def test_plan_fault_in_zslice_workspace_set_only(self):
        assert "plan_fault" in ToolPalette._ZSLICE_WORKSPACE
        assert "plan_fault" not in ToolPalette._SECTION_WORKSPACE
        assert ToolPalette._ZSLICE_WORKSPACE.isdisjoint(ToolPalette._SECTION_WORKSPACE)

    def test_plan_fault_registered_as_tool(self):
        from section_tool.views.tool_palette import _TOOL_IDS
        assert "plan_fault" in _TOOL_IDS


class TestPredicate:
    def test_zslice_active_predicate(self, qapp):
        from section_tool.app_state import AppState
        from section_tool.core.section import Section
        from section_tool.core.slices import HorizontalSlice
        st = AppState()
        st.add_section(Section([(0.0, 0.0), (1000.0, 0.0)], name="L1"))
        # mimic the app helper
        def zslice_active():
            return getattr(st.active_slice, "kind", None) == "horizontal"
        st.set_active_slice(st.project.sections[0])
        assert zslice_active() is False
        st.set_active_slice(HorizontalSlice("Z", -1000.0))
        assert zslice_active() is True
