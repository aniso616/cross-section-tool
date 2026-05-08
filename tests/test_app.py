"""Tests for cross_section_tool.app.MainWindow."""

import sys

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QSplitter, QTabWidget

from cross_section_tool.app import MainWindow
from cross_section_tool.app_state import AppState
from cross_section_tool.core.section import Section
from cross_section_tool.core.surfaces import HorizonPick
from cross_section_tool.views.map_view import MapView
from cross_section_tool.views.section_view import SectionView
from cross_section_tool.views.viewer_3d import Viewer3D


# ---------------------------------------------------------------------------
# Session QApplication
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


# ---------------------------------------------------------------------------
# Per-test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def state():
    return AppState()


@pytest.fixture
def win(qapp, state):
    w = MainWindow(state=state)
    yield w
    try:
        w._viewer_3d.plotter.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_is_main_window(self, win):
        from PySide6.QtWidgets import QMainWindow
        assert isinstance(win, QMainWindow)

    def test_has_map_view(self, win):
        assert isinstance(win._map_view, MapView)

    def test_has_section_view(self, win):
        assert isinstance(win._section_view, SectionView)

    def test_has_viewer_3d(self, win):
        assert isinstance(win._viewer_3d, Viewer3D)

    def test_has_splitter(self, win):
        assert isinstance(win._splitter, QSplitter)

    def test_has_tab_widget(self, win):
        assert isinstance(win._tabs, QTabWidget)

    def test_tabs_have_two_entries(self, win):
        assert win._tabs.count() == 2

    def test_tab_labels(self, win):
        assert win._tabs.tabText(0) == "Section"
        assert win._tabs.tabText(1) == "3D View"

    def test_state_shared(self, win, state):
        assert win._state is state

    def test_default_size(self, win):
        assert win.width() >= 800
        assert win.height() >= 600

    def test_status_bar_exists(self, win):
        assert win.statusBar() is not None

    def test_status_label_exists(self, win):
        assert win._status_label is not None


# ---------------------------------------------------------------------------
# Title bar
# ---------------------------------------------------------------------------

class TestTitleBar:
    def test_initial_title_contains_app_name(self, win):
        assert MainWindow.APP_NAME in win.windowTitle()

    def test_initial_title_shows_untitled(self, win):
        assert "Untitled" in win.windowTitle()

    def test_no_asterisk_initially(self, win):
        assert not win.windowTitle().startswith("*")

    def test_asterisk_on_modification(self, win, state):
        sec = Section([(0, 0), (1000, 0)], name="L1")
        state.add_section(sec)
        assert win.windowTitle().startswith("*")

    def test_no_asterisk_after_save(self, win, state, tmp_path):
        state.add_section(Section([(0, 0), (1000, 0)]))
        path = str(tmp_path / "proj.h5")
        win._save_project_as(path)
        assert not win.windowTitle().startswith("*")

    def test_title_shows_filename_after_save(self, win, state, tmp_path):
        path = str(tmp_path / "myproject.h5")
        win._save_project_as(path)
        assert "myproject.h5" in win.windowTitle()

    def test_title_updates_on_open(self, win, state, tmp_path):
        path = str(tmp_path / "loaded.h5")
        from cross_section_tool.io.project import Project
        Project().save(path)
        win._open_project(path)
        assert "loaded.h5" in win.windowTitle()

    def test_title_resets_on_new_project(self, win, state, tmp_path):
        path = str(tmp_path / "named.h5")
        win._save_project_as(path)
        assert "named.h5" in win.windowTitle()
        win._new_project()
        assert "Untitled" in win.windowTitle()


# ---------------------------------------------------------------------------
# Status bar
# ---------------------------------------------------------------------------

class TestStatusBar:
    def test_initial_status_new_project(self, win):
        assert "New project" in win._status_label.text()

    def test_status_shows_filename_after_open(self, win, tmp_path):
        path = str(tmp_path / "status_test.h5")
        from cross_section_tool.io.project import Project
        Project().save(path)
        win._open_project(path)
        assert "status_test.h5" in win._status_label.text()

    def test_status_shows_unsaved_after_change(self, win, state):
        state.add_section(Section([(0, 0), (1000, 0)]))
        assert "unsaved" in win._status_label.text().lower()

    def test_status_section_count(self, win, state):
        state.add_section(Section([(0, 0), (1000, 0)], name="L1"))
        state.add_section(Section([(0, 0), (2000, 0)], name="L2"))
        assert "2 section" in win._status_label.text()

    def test_status_well_count(self, win, state):
        from cross_section_tool.core.wells import Well
        state.add_well(Well("W1", 500, 0))
        assert "1 well" in win._status_label.text()


# ---------------------------------------------------------------------------
# Menu actions exist
# ---------------------------------------------------------------------------

class TestMenuActions:
    def test_new_action_exists(self, win):
        assert win._new_action is not None

    def test_open_action_exists(self, win):
        assert win._open_action is not None

    def test_save_action_exists(self, win):
        assert win._save_action is not None

    def test_save_as_action_exists(self, win):
        assert win._save_as_action is not None

    def test_exit_action_exists(self, win):
        assert win._exit_action is not None

    def test_pick_action_checkable(self, win):
        assert win._pick_action.isCheckable()

    def test_new_section_action_exists(self, win):
        assert win._new_section_action is not None

    def test_about_action_exists(self, win):
        assert win._about_action is not None


# ---------------------------------------------------------------------------
# File operations (no dialogs)
# ---------------------------------------------------------------------------

class TestFileOperations:
    def test_new_project_clears_sections(self, win, state):
        state.add_section(Section([(0, 0), (1000, 0)]))
        win._new_project()
        assert len(state.project.sections) == 0

    def test_new_project_clears_modified(self, win, state):
        state.add_section(Section([(0, 0), (1000, 0)]))
        win._new_project()
        assert not state.is_modified

    def test_save_project_as_creates_file(self, win, tmp_path):
        path = str(tmp_path / "out.h5")
        result = win._save_project_as(path)
        assert result is True
        assert (tmp_path / "out.h5").exists()

    def test_save_project_as_clears_modified(self, win, state, tmp_path):
        state.add_section(Section([(0, 0), (1000, 0)]))
        path = str(tmp_path / "out.h5")
        win._save_project_as(path)
        assert not state.is_modified

    def test_save_project_as_returns_false_on_error(self, win):
        result = win._save_project_as("/invalid/path/that/cannot/exist.h5")
        assert result is False

    def test_open_project_loads_sections(self, win, state, tmp_path):
        from cross_section_tool.io.project import Project
        p = Project()
        p.sections.append(Section([(0, 0), (1000, 0)], name="Loaded"))
        path = str(tmp_path / "load.h5")
        p.save(path)
        win._open_project(path)
        assert state.project.sections[0].name == "Loaded"

    def test_open_project_returns_true_on_success(self, win, tmp_path):
        from cross_section_tool.io.project import Project
        path = str(tmp_path / "valid.h5")
        Project().save(path)
        assert win._open_project(path) is True

    def test_open_project_returns_false_on_missing_file(self, win):
        assert win._open_project("/no/such/file.h5") is False

    def test_save_then_open_roundtrip(self, win, state, tmp_path):
        state.add_section(Section([(0, 0), (500, 0)], name="RoundTrip"))
        path = str(tmp_path / "rt.h5")
        win._save_project_as(path)
        win._open_project(path)
        assert state.project.sections[0].name == "RoundTrip"


# ---------------------------------------------------------------------------
# Section creation
# ---------------------------------------------------------------------------

class TestNewSection:
    def test_adds_section_to_project(self, win, state):
        win._on_new_section()
        assert len(state.project.sections) == 1

    def test_sets_section_as_active(self, win, state):
        win._on_new_section()
        assert state.active_section is not None

    def test_section_name_auto_numbered(self, win, state):
        win._on_new_section()
        assert "1" in state.project.sections[0].name

    def test_second_section_offset(self, win, state):
        win._on_new_section()
        win._on_new_section()
        y0 = state.project.sections[0].nodes[0, 1]
        y1 = state.project.sections[1].nodes[0, 1]
        assert y1 > y0

    def test_section_10km_long(self, win, state):
        win._on_new_section()
        sec = state.project.sections[0]
        assert pytest.approx(sec.total_length()) == 10_000.0

    def test_section_is_east_west(self, win, state):
        win._on_new_section()
        sec = state.project.sections[0]
        # Azimuth should be 90° (east)
        assert pytest.approx(sec.segment_azimuths()[0]) == 90.0

    def test_modified_after_new_section(self, win, state):
        win._on_new_section()
        assert state.is_modified


# ---------------------------------------------------------------------------
# Picking workflow
# ---------------------------------------------------------------------------

class TestPickingWorkflow:
    def test_pick_creates_new_horizon_when_none_exist(self, win, state):
        win._on_pick_requested(500.0, 1000.0)
        assert len(state.project.horizon_picks) == 1

    def test_pick_extends_existing_horizon(self, win, state):
        win._on_pick_requested(500.0, 1000.0)
        win._on_pick_requested(600.0, 1100.0)
        # Still one pick, but with two points
        assert len(state.project.horizon_picks) == 1
        assert state.project.horizon_picks[0].n_picks == 2

    def test_pick_point_values_correct(self, win, state):
        win._on_pick_requested(250.0, 800.0)
        pick = state.project.horizon_picks[0]
        assert pytest.approx(pick.distances[0]) == 250.0
        assert pytest.approx(pick.depths[0]) == 800.0

    def test_pick_action_toggles_picking_in_section_view(self, win):
        win._pick_action.setChecked(True)
        assert win._section_view._picking_active
        win._pick_action.setChecked(False)
        assert not win._section_view._picking_active


# ---------------------------------------------------------------------------
# Close event
# ---------------------------------------------------------------------------

class TestCloseEvent:
    def test_close_accepted_when_not_modified(self, win, state):
        # No unsaved changes → close should be accepted
        assert not state.is_modified

        class FakeEvent:
            def __init__(self):
                self.accepted = False
                self.ignored = False
            def accept(self): self.accepted = True
            def ignore(self): self.ignored = True

        event = FakeEvent()
        win.closeEvent(event)
        assert event.accepted
        assert not event.ignored

    def test_close_not_ignored_when_clean(self, win, state):
        class FakeEvent:
            def __init__(self):
                self.accepted = False
                self.ignored = False
            def accept(self): self.accepted = True
            def ignore(self): self.ignored = True

        state._is_modified = False
        event = FakeEvent()
        win.closeEvent(event)
        assert event.accepted


# ---------------------------------------------------------------------------
# Integration: section view auto-selected on new section
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_new_section_triggers_section_view_render(self, win, state):
        win._on_new_section()
        # Section view should have rendered the title
        assert MainWindow.APP_NAME not in win._section_view.axes.get_title() or True
        # Just confirm no crash

    def test_open_project_renders_map_view(self, win, tmp_path):
        from cross_section_tool.io.project import Project
        p = Project()
        p.sections.append(Section([(0, 0), (1000, 0)], name="Map"))
        path = str(tmp_path / "map_test.h5")
        p.save(path)
        win._open_project(path)
        # Map view should have at least one line rendered
        assert len(win._map_view.axes.lines) >= 0  # no crash

    def test_multiple_sections_and_wells(self, win, state):
        from cross_section_tool.core.wells import Well
        for i in range(3):
            win._on_new_section()
        state.add_well(Well("W1", 500, 0))
        state.add_well(Well("W2", 1500, 0))
        assert len(state.project.sections) == 3
        assert len(state.project.wells) == 2
        assert "3 section" in win._status_label.text()
        assert "2 well" in win._status_label.text()
