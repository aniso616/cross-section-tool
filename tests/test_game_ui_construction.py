"""Construction-tool surfacing in the game UI (presentation wiring).

These exercise the access/visibility wiring added when the construction tools
were re-surfaced in SectionMainWindow: palette membership, the T->Trim /
W->well-top key fix, the AppState-keyed indicator labels, command-palette
discoverability, and the ContextToolbar parameter edit reaching the live tool.

They are deliberately window-free (no SectionMainWindow) to stay fast and to
avoid the VTK/pyvista 3D-viewer teardown segfault; the wiring under test does
not depend on the full window.
"""
from __future__ import annotations

import sys

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QWidget

from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.hud.command_palette import CommandPalette
from section_tool.interaction.tool_manager import (
    APPSTATE_TOOL_LABELS,
    TOOL_KEYS,
    ToolManager,
)
from section_tool.views.context_toolbar import ContextToolbar
from section_tool.views.section_view import SectionView
from section_tool.views.tool_palette import ToolPalette

CONSTRUCTION = ("extend", "trim", "parallel", "dip_constrained", "kink_band")


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


# ---------------------------------------------------------------------------
# Palette surfaces the construction tools as first-class buttons
# ---------------------------------------------------------------------------

class TestPaletteSurfacesConstruction:
    def test_all_construction_tools_have_buttons(self, qapp):
        pal = ToolPalette()
        for tid in CONSTRUCTION:
            assert tid in pal.tool_ids
            assert tid in pal._buttons

    def test_construction_tool_activatable_and_checked(self, qapp):
        pal = ToolPalette()
        for tid in CONSTRUCTION:
            pal.set_active_tool(tid)
            assert pal.active_tool == tid
            assert pal._buttons[tid].isChecked()


# ---------------------------------------------------------------------------
# Indicator labels exist for construction tools (status strip + corner)
# ---------------------------------------------------------------------------

class TestIndicatorLabels:
    def test_construction_tools_labelled(self):
        for tid in CONSTRUCTION:
            assert APPSTATE_TOOL_LABELS.get(tid)   # present and non-empty


# ---------------------------------------------------------------------------
# Command palette discoverability
# ---------------------------------------------------------------------------

class TestCommandPaletteDiscoverability:
    def test_construction_commands_registered(self, qapp):
        cp = CommandPalette(QWidget())
        ids = {c["id"] for c in cp._commands}
        for cid in ("tool_extend", "tool_trim", "tool_parallel",
                    "tool_dip", "tool_kink"):
            assert cid in ids

    def test_well_top_pick_moved_off_T(self, qapp):
        cp = CommandPalette(QWidget())
        pick = next(c for c in cp._commands if c["id"] == "tool_pick")
        assert pick["shortcut"] == "W"


# ---------------------------------------------------------------------------
# T-key collision resolved: T = trim, W = well-top pick
# ---------------------------------------------------------------------------

class TestKeyCollision:
    def test_T_not_a_tool_key(self):
        assert Qt.Key.Key_T not in TOOL_KEYS

    def test_W_is_well_top_pick(self):
        assert TOOL_KEYS.get(Qt.Key.Key_W) == "pick"

    def test_handle_T_not_consumed(self):
        # T is no longer a ToolManager key, so it falls through to the
        # application 'T -> Trim' shortcut.
        assert ToolManager().handle_key(Qt.Key.Key_T) is False

    def test_handle_W_activates_pick(self):
        mgr = ToolManager()
        assert mgr.handle_key(Qt.Key.Key_W) is True
        assert mgr.active == "pick"


# ---------------------------------------------------------------------------
# ContextToolbar parameter edits reach the live construction tool
# ---------------------------------------------------------------------------

class TestContextToolbarParamReachesTool:
    def _view(self, qapp):
        state = AppState()
        sec = Section([(0.0, 0.0), (10000.0, 0.0)], name="L1")
        state.add_section(sec)
        state.set_active_section(sec)
        return state, SectionView(state)

    def test_dip_spinner_sets_tool_dip(self, qapp):
        state, view = self._view(qapp)
        ctx = ContextToolbar(state)
        ctx.action_requested.connect(view._on_context_action)
        state.set_active_tool("dip_constrained")
        spins = ctx.findChildren(QDoubleSpinBox)
        assert spins, "dip-constrained context bar should expose a Dip spinner"
        spins[0].setValue(33.0)
        assert view._cst_dip_tool.dip_deg == 33.0

    def test_kink_spinners_set_tool_dips(self, qapp):
        state, view = self._view(qapp)
        ctx = ContextToolbar(state)
        ctx.action_requested.connect(view._on_context_action)
        state.set_active_tool("kink_band")
        spins = ctx.findChildren(QDoubleSpinBox)
        assert len(spins) >= 3            # axial, fore, back
        spins[0].setValue(50.0)
        spins[1].setValue(35.0)
        spins[2].setValue(5.0)
        assert view._cst_kink_tool.axial_surface_dip_deg == 50.0
        assert view._cst_kink_tool.fore_dip_deg == 35.0
        assert view._cst_kink_tool.back_dip_deg == 5.0

    def test_dark_theme_keeps_params_wired(self, qapp):
        state, view = self._view(qapp)
        ctx = ContextToolbar(state)
        ctx.action_requested.connect(view._on_context_action)
        ctx.set_dark_theme()
        assert ctx._dark
        state.set_active_tool("dip_constrained")
        spins = ctx.findChildren(QDoubleSpinBox)
        assert spins
        spins[0].setValue(12.0)
        assert view._cst_dip_tool.dip_deg == 12.0
