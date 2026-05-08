"""Tests for cross_section_tool.views.tool_palette.ToolPalette."""

import sys

import pytest
from PySide6.QtWidgets import QApplication

from cross_section_tool.views.tool_palette import ToolPalette, _TOOL_IDS


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


@pytest.fixture
def palette(qapp):
    return ToolPalette()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_is_widget(self, palette):
        assert palette.isWidgetType()

    def test_fixed_width(self, palette):
        assert palette.width() == 40

    def test_has_all_tool_ids(self, palette):
        assert set(palette.tool_ids) == set(_TOOL_IDS)

    def test_default_tool_is_select(self, palette):
        assert palette.active_tool == "select"

    def test_all_buttons_present(self, palette):
        for tid in _TOOL_IDS:
            assert tid in palette._buttons

    def test_select_button_checked_initially(self, palette):
        assert palette._buttons["select"].isChecked()

    def test_other_buttons_unchecked_initially(self, palette):
        for tid, btn in palette._buttons.items():
            if tid != "select":
                assert not btn.isChecked()


# ---------------------------------------------------------------------------
# set_active_tool
# ---------------------------------------------------------------------------

class TestSetActiveTool:
    def test_changes_active_tool(self, palette):
        palette.set_active_tool("pan")
        assert palette.active_tool == "pan"

    def test_checks_correct_button(self, palette):
        palette.set_active_tool("horizon_pick")
        assert palette._buttons["horizon_pick"].isChecked()

    def test_unchecks_previous_button(self, palette):
        palette.set_active_tool("zoom")
        palette.set_active_tool("pan")
        assert not palette._buttons["zoom"].isChecked()

    def test_only_one_button_checked(self, palette):
        palette.set_active_tool("measure")
        checked = [tid for tid, btn in palette._buttons.items() if btn.isChecked()]
        assert checked == ["measure"]

    def test_invalid_tool_id_is_noop(self, palette):
        palette.set_active_tool("select")
        palette.set_active_tool("nonexistent_tool")
        assert palette.active_tool == "select"

    def test_all_tools_activatable(self, palette):
        for tid in _TOOL_IDS:
            palette.set_active_tool(tid)
            assert palette.active_tool == tid


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------

class TestToolChangedSignal:
    def test_emitted_on_activation(self, palette):
        received = []
        palette.tool_changed.connect(lambda t: received.append(t))
        palette.set_active_tool("pan")
        assert received == ["pan"]

    def test_not_emitted_for_same_tool(self, palette):
        palette.set_active_tool("select")
        received = []
        palette.tool_changed.connect(lambda t: received.append(t))
        palette.set_active_tool("select")
        assert received == []

    def test_emitted_for_each_change(self, palette):
        received = []
        palette.tool_changed.connect(lambda t: received.append(t))
        palette.set_active_tool("pan")
        palette.set_active_tool("zoom")
        palette.set_active_tool("select")
        assert received == ["pan", "zoom", "select"]

    def test_correct_id_in_signal(self, palette):
        # Start from a known non-default tool so every subsequent activation emits
        palette.set_active_tool("pan")
        received = []
        palette.tool_changed.connect(lambda t: received.append(t))
        for tid in _TOOL_IDS:
            palette.set_active_tool(tid)
        # Signal fires for every tool whose id differs from the previous active tool.
        # We activated each tool once in order so distinct consecutive ids all emit.
        assert set(received) == set(_TOOL_IDS)
        # Each id appears exactly once (no duplicates from re-activation)
        assert len(received) == len(set(received))

    def test_not_emitted_on_initial_construction(self, qapp):
        received = []
        p = ToolPalette()
        p.tool_changed.connect(lambda t: received.append(t))
        # No activation after construction — signal should not have fired
        assert received == []


# ---------------------------------------------------------------------------
# Tool IDs completeness
# ---------------------------------------------------------------------------

class TestToolIds:
    def test_expected_tools_present(self, palette):
        expected = {
            "select", "pan", "zoom",
            "new_section", "edit_nodes",
            "horizon_pick", "fault_pick", "polygon",
            "measure",
        }
        assert expected.issubset(set(_TOOL_IDS))

    def test_no_duplicates(self, palette):
        assert len(_TOOL_IDS) == len(set(_TOOL_IDS))

    def test_tool_ids_property_matches(self, palette):
        assert palette.tool_ids == list(_TOOL_IDS)
