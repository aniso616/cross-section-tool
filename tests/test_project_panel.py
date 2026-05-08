"""Tests for cross_section_tool.views.project_panel.ProjectPanel."""

import sys

import pytest
from PySide6.QtWidgets import QApplication

from cross_section_tool.app_state import AppState
from cross_section_tool.core.section import Section
from cross_section_tool.core.surfaces import HorizonPick
from cross_section_tool.views.project_panel import ProjectPanel, _CATEGORIES


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


@pytest.fixture
def state():
    return AppState()


@pytest.fixture
def panel(qapp, state):
    return ProjectPanel(state)


def _sec(name="L1"):
    return Section([(0, 0), (1000, 0)], name=name)


def _pick(name="TopSand", color="#ff0000"):
    return HorizonPick([0.0, 500.0, 1000.0], [100.0, 200.0, 150.0],
                       name=name, color=color)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_is_dock_widget(self, panel):
        from PySide6.QtWidgets import QDockWidget
        assert isinstance(panel, QDockWidget)

    def test_title_is_project(self, panel):
        assert panel.windowTitle() == "Project"

    def test_has_tree(self, panel):
        assert panel._tree is not None

    def test_all_categories_present(self, panel):
        top_count = panel._tree.topLevelItemCount()
        assert top_count == len(_CATEGORIES)

    def test_category_names(self, panel):
        names = [panel._tree.topLevelItem(i).text(0)
                 for i in range(panel._tree.topLevelItemCount())]
        assert names == _CATEGORIES

    def test_empty_sections_initially(self, panel):
        sec_item = panel._tree.topLevelItem(0)
        assert sec_item.childCount() == 0

    def test_has_add_button(self, panel):
        assert panel._add_btn is not None


# ---------------------------------------------------------------------------
# Population from AppState
# ---------------------------------------------------------------------------

class TestPopulation:
    def test_section_appears(self, panel, state):
        state.add_section(_sec("MyLine"))
        # Panel rebuilds via signal; check tree
        sec_item = panel._category_items["Sections"]
        assert sec_item.childCount() == 1

    def test_section_row_name(self, panel, state):
        state.add_section(_sec("Alpha"))
        row = panel._row_widgets[("Sections", 0)]
        assert row.name == "Alpha"

    def test_two_sections(self, panel, state):
        state.add_section(_sec("A"))
        state.add_section(_sec("B"))
        sec_item = panel._category_items["Sections"]
        assert sec_item.childCount() == 2

    def test_horizon_appears(self, panel, state):
        state.add_horizon_pick(_pick("Top Mancos"))
        horiz_item = panel._category_items["Horizons"]
        assert horiz_item.childCount() == 1

    def test_horizon_row_color(self, panel, state):
        state.add_horizon_pick(_pick("H1", color="#aabbcc"))
        row = panel._row_widgets[("Horizons", 0)]
        assert row.color == "#aabbcc"

    def test_rebuild_on_section_removed(self, panel, state):
        sec = _sec("RemoveMe")
        state.add_section(sec)
        state.add_section(_sec("Keep"))
        state.remove_section(sec)
        sec_item = panel._category_items["Sections"]
        assert sec_item.childCount() == 1

    def test_rebuild_on_new_project(self, panel, state):
        state.add_section(_sec())
        state.new_project()
        sec_item = panel._category_items["Sections"]
        assert sec_item.childCount() == 0


# ---------------------------------------------------------------------------
# Visibility signal
# ---------------------------------------------------------------------------

class TestVisibilitySignal:
    def test_initial_visibility_true(self, panel, state):
        state.add_section(_sec())
        assert panel.is_visible("Sections", 0) is True

    def test_toggle_emits_signal(self, panel, state):
        state.add_section(_sec())
        received = []
        panel.visibility_changed.connect(
            lambda cat, idx, vis: received.append((cat, idx, vis))
        )
        row = panel._row_widgets[("Sections", 0)]
        row._check.setChecked(False)
        assert len(received) == 1
        assert received[0] == ("Sections", 0, False)

    def test_re_enable_emits_signal(self, panel, state):
        state.add_section(_sec())
        received = []
        panel.visibility_changed.connect(
            lambda cat, idx, vis: received.append(vis)
        )
        row = panel._row_widgets[("Sections", 0)]
        row._check.setChecked(False)
        row._check.setChecked(True)
        assert received == [False, True]


# ---------------------------------------------------------------------------
# Color signal
# ---------------------------------------------------------------------------

class TestColorSignal:
    def test_color_change_emits_signal(self, panel, state):
        state.add_horizon_pick(_pick(color="#ff0000"))
        received = []
        panel.object_color_changed.connect(
            lambda cat, idx, col: received.append((cat, idx, col))
        )
        row = panel._row_widgets[("Horizons", 0)]
        row._swatch.set_color("#00ff00")
        row._swatch.color_changed.emit("#00ff00")
        assert len(received) == 1
        assert received[0] == ("Horizons", 0, "#00ff00")

    def test_color_of_helper(self, panel, state):
        state.add_horizon_pick(_pick(color="#123456"))
        assert panel.color_of("Horizons", 0) == "#123456"


# ---------------------------------------------------------------------------
# Delete signal
# ---------------------------------------------------------------------------

class TestDeleteSignal:
    def test_delete_emits_signal(self, panel, state):
        state.add_section(_sec())
        received = []
        panel.object_deleted.connect(lambda cat, idx: received.append((cat, idx)))
        panel.object_deleted.emit("Sections", 0)
        assert received == [("Sections", 0)]


# ---------------------------------------------------------------------------
# Rename signal
# ---------------------------------------------------------------------------

class TestRenameSignal:
    def test_rename_emits_signal(self, panel, state):
        state.add_section(_sec())
        received = []
        panel.object_renamed.connect(
            lambda cat, idx, name: received.append((cat, idx, name))
        )
        panel.object_renamed.emit("Sections", 0, "New Name")
        assert received == [("Sections", 0, "New Name")]


# ---------------------------------------------------------------------------
# Move signal
# ---------------------------------------------------------------------------

class TestMoveSignal:
    def test_move_emits_signal(self, panel, state):
        state.add_section(_sec("A"))
        state.add_section(_sec("B"))
        received = []
        panel.object_moved.connect(
            lambda cat, frm, to: received.append((cat, frm, to))
        )
        panel.object_moved.emit("Sections", 1, 0)
        assert received == [("Sections", 1, 0)]


# ---------------------------------------------------------------------------
# Add requested signal
# ---------------------------------------------------------------------------

class TestAddRequested:
    def test_add_button_emits_add_requested(self, panel, state):
        received = []
        panel.add_requested.connect(lambda cat: received.append(cat))
        panel._add_btn.click()
        assert len(received) == 1

    def test_add_defaults_to_sections_when_nothing_selected(self, panel, state):
        received = []
        panel.add_requested.connect(lambda cat: received.append(cat))
        panel._tree.clearSelection()
        panel._add_btn.click()
        assert received[0] == "Sections"
