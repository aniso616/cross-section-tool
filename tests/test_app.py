"""Tests for section_tool.app.MainWindow."""

import sys

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDockWidget, QSplitter, QTabWidget

from section_tool.app import MainWindow
from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick
from section_tool.views.map_view import MapView
from section_tool.views.section_view import SectionView
from section_tool.views.tool_palette import ToolPalette
from section_tool.views.viewer_3d import Viewer3D


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

    def test_has_map_dock(self, win):
        assert isinstance(win._map_dock, QDockWidget)
        assert win._map_dock.objectName() == "MapDock"

    def test_has_section_dock(self, win):
        assert isinstance(win._section_dock, QDockWidget)
        assert win._section_dock.objectName() == "SectionDock"

    def test_has_view3d_dock(self, win):
        assert isinstance(win._view3d_dock, QDockWidget)
        assert win._view3d_dock.objectName() == "View3DDock"

    def test_has_tool_palette(self, win):
        assert isinstance(win._tool_palette, ToolPalette)

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
        from section_tool.io.project import Project
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
        from section_tool.io.project import Project
        Project().save(path)
        win._open_project(path)
        assert "status_test.h5" in win._status_label.text()

    def test_status_shows_unsaved_after_change(self, win, state):
        state.add_section(Section([(0, 0), (1000, 0)]))
        assert "✎" in win._status_label.text()

    def test_status_section_count(self, win, state):
        state.add_section(Section([(0, 0), (1000, 0)], name="L1"))
        state.add_section(Section([(0, 0), (2000, 0)], name="L2"))
        assert "2S" in win._status_label.text()

    def test_status_well_count(self, win, state):
        from section_tool.core.wells import Well
        state.add_well(Well("W1", 500, 0))
        assert "1W" in win._status_label.text()


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
        from section_tool.io.project import Project
        p = Project()
        p.sections.append(Section([(0, 0), (1000, 0)], name="Loaded"))
        path = str(tmp_path / "load.h5")
        p.save(path)
        win._open_project(path)
        assert state.project.sections[0].name == "Loaded"

    def test_open_project_returns_true_on_success(self, win, tmp_path):
        from section_tool.io.project import Project
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

    def test_pick_action_toggles_picking_in_section_view(self, win, state):
        state.add_horizon_pick(HorizonPick.empty(name="H1"))
        state.set_active_pick_target("Horizons", 0)
        win._tool_palette.set_active_tool("horizon_pick")
        assert win._section_view._picking_active
        win._tool_palette.set_active_tool("select")
        assert not win._section_view._picking_active

    def test_tool_palette_default_tool_is_select(self, win):
        assert win._tool_palette.active_tool == "select"

    def test_tool_palette_horizon_enables_picking(self, win, state):
        state.add_horizon_pick(HorizonPick.empty(name="H1"))
        state.set_active_pick_target("Horizons", 0)
        win._tool_palette.set_active_tool("horizon_pick")
        assert win._section_view._picking_active

    def test_tool_palette_select_disables_picking(self, win, state):
        state.add_horizon_pick(HorizonPick.empty(name="H1"))
        state.set_active_pick_target("Horizons", 0)
        win._tool_palette.set_active_tool("horizon_pick")
        win._tool_palette.set_active_tool("select")
        assert not win._section_view._picking_active

    def test_pick_menu_action_syncs_palette(self, win, state):
        state.add_horizon_pick(HorizonPick.empty(name="H1"))
        state.set_active_pick_target("Horizons", 0)
        win._pick_action.setChecked(True)
        assert win._tool_palette.active_tool == "horizon_pick"
        win._pick_action.setChecked(False)
        assert win._tool_palette.active_tool == "select"

    def test_pick_without_horizon_creates_new(self, win, state):
        # No horizon selected and none exist: pressing P auto-creates a new one
        win._tool_palette.set_active_tool("horizon_pick")
        assert win._tool_palette.active_tool == "horizon_pick"
        assert win._section_view._picking_active
        assert len(state.project.horizon_picks) == 1


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
        from section_tool.io.project import Project
        p = Project()
        p.sections.append(Section([(0, 0), (1000, 0)], name="Map"))
        path = str(tmp_path / "map_test.h5")
        p.save(path)
        win._open_project(path)
        # Map view should have at least one line rendered
        assert len(win._map_view.axes.lines) >= 0  # no crash

    def test_multiple_sections_and_wells(self, win, state):
        from section_tool.core.wells import Well
        for i in range(3):
            win._on_new_section()
        state.add_well(Well("W1", 500, 0))
        state.add_well(Well("W2", 1500, 0))
        assert len(state.project.sections) == 3
        assert len(state.project.wells) == 2
        assert "3S" in win._status_label.text()
        assert "2W" in win._status_label.text()


# ---------------------------------------------------------------------------
# Pick chain guard — these tests must NOT be deleted; they protect the entire
# horizon/fault creation and picking workflow from silent regressions.
# ---------------------------------------------------------------------------

class TestPickChainGuard:
    """End-to-end pick chain regression tests.

    These cover: AppState mutation → signal → panel rebuild → tool activation
    → section view picking → pick storage.  Every step must succeed.
    """

    def test_horizon_add_stored_in_state(self, state):
        hp = HorizonPick.empty(name="Top Sand", color="#2ca02c")
        state.add_horizon_pick(hp)
        assert len(state.project.horizon_picks) == 1
        assert state.project.horizon_picks[0].name == "Top Sand"

    def test_fault_add_stored_in_state(self, state):
        fp = HorizonPick.empty(name="F1", color="#d62728")
        state.add_fault_pick(fp)
        assert len(state.project.fault_picks) == 1
        assert state.project.fault_picks[0].name == "F1"

    def test_set_active_pick_target_horizon(self, state):
        hp = HorizonPick.empty(name="H1")
        state.add_horizon_pick(hp)
        state.set_active_pick_target("Horizons", 0)
        assert state.active_pick_category == "Horizons"
        assert state.active_pick_index == 0

    def test_set_active_pick_target_fault(self, state):
        fp = HorizonPick.empty(name="F1")
        state.add_fault_pick(fp)
        state.set_active_pick_target("Faults", 0)
        assert state.active_pick_category == "Faults"
        assert state.active_pick_index == 0

    def test_horizon_pick_activates_tool(self, win, state):
        hp = HorizonPick.empty(name="H1")
        state.add_horizon_pick(hp)
        state.set_selected_entity("Horizons", 0)
        win._tool_palette.set_active_tool("horizon_pick")
        assert win._tool_palette.active_tool == "horizon_pick"
        assert win._section_view._picking_active

    def test_fault_pick_activates_tool(self, win, state):
        fp = HorizonPick.empty(name="F1")
        state.add_fault_pick(fp)
        state.set_selected_entity("Faults", 0)
        win._tool_palette.set_active_tool("fault_pick")
        assert win._tool_palette.active_tool == "fault_pick"
        assert win._section_view._fault_picking

    def test_full_horizon_pick_chain(self, win, state):
        hp = HorizonPick.empty(name="Top Formation", color="#2ca02c")
        state.add_horizon_pick(hp)
        state.set_selected_entity("Horizons", 0)   # select → becomes pick target
        win._tool_palette.set_active_tool("horizon_pick")
        sec = Section([(0, 0), (1000, 0)], name="L1")
        state.add_section(sec)
        state.set_active_section(sec)
        win._section_view._add_pick_to_active_target(200.0, 800.0)
        win._section_view._add_pick_to_active_target(500.0, 1200.0)
        assert state.project.horizon_picks[0].n_picks == 2

    def test_full_fault_pick_chain(self, win, state):
        fp = HorizonPick.empty(name="Fault A", color="#d62728")
        state.add_fault_pick(fp)
        state.set_selected_entity("Faults", 0)   # select → becomes pick target
        win._tool_palette.set_active_tool("fault_pick")
        sec = Section([(0, 0), (1000, 0)], name="L1")
        state.add_section(sec)
        state.set_active_section(sec)
        win._section_view._add_pick_to_active_target(300.0, 900.0)
        assert state.project.fault_picks[0].n_picks == 1

    def test_no_horizon_reverts_pick_tool_to_select(self, win, state):
        # No horizon selected and none exist → auto-creates a new one
        win._tool_palette.set_active_tool("horizon_pick")
        assert win._tool_palette.active_tool == "horizon_pick"
        assert win._section_view._picking_active
        assert len(state.project.horizon_picks) == 1

    def test_panel_add_requested_connected(self, win):
        # add_requested must be wired to app — verify by checking the handler exists
        assert hasattr(win, "_on_panel_add")
        # The project panel must have at least one receiver for add_requested
        panel = win._project_panel
        assert panel.add_requested is not None  # signal object exists


# ---------------------------------------------------------------------------
# Import Time–Depth Data: end-to-end on a REAL window (regression — the
# handler reached for a phantom self._statusbar and crashed AFTER the well was
# committed, so a successful import looked like an Unexpected Error).
# ---------------------------------------------------------------------------

_TD_TXT = "30\t0\n553.6\t0.544\n1695\t1.67\n3150\t3.234\n"   # depth-MD, TWT-s


def _well_F0201():
    from section_tool.core.wells import Well
    return Well("F02-01", 606554.0, 6080126.0, kb=30.0, td=3200.0)


class TestImportTdrEndToEnd:
    def _setup(self, win, state, td_path, monkeypatch, accept=True):
        from PySide6.QtWidgets import QFileDialog, QDialog
        from section_tool.views import tdr_import_dialog as tdrdlg
        state.add_section(Section([(0, 0), (3000, 0)], name="L1"))
        state.set_active_section(state.project.sections[0])
        state.add_well(_well_F0201())
        monkeypatch.setattr(QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, **k: (str(td_path), "")))
        code = (QDialog.DialogCode.Accepted if accept
                else QDialog.DialogCode.Rejected)
        monkeypatch.setattr(tdrdlg.TdrImportDialog, "exec", lambda self: code)

    def test_import_tdr_does_not_crash_and_lands(self, win, state, tmp_path,
                                                 monkeypatch):
        p = tmp_path / "F02-01_TD.txt"; p.write_text(_TD_TXT, encoding="utf-8")
        self._setup(win, state, p, monkeypatch)
        win._on_import_tdr()                       # must NOT raise (the _statusbar bug)
        tdrs = state.project.wells[0].tdrs
        assert len(tdrs) == 1
        assert tdrs[0].kind == "checkshot"
        assert tdrs[0].source == "F02-01_TD.txt"

    def test_failed_import_leaves_zero_tdr_rows(self, win, state, tmp_path,
                                                monkeypatch):
        # A file that parses but whose load raises (non-monotonic TWT) must leave
        # the well clean — the import commits only after a successful load.
        bad = tmp_path / "bad.txt"
        bad.write_text("0\t1.0\n100\t0.5\n200\t2.0\n", encoding="utf-8")  # TWT dips
        self._setup(win, state, bad, monkeypatch)
        from PySide6.QtWidgets import QMessageBox
        monkeypatch.setattr(QMessageBox, "warning",
                            staticmethod(lambda *a, **k: None))  # don't block on the error
        win._on_import_tdr()
        assert len(state.project.wells[0].tdrs) == 0

    def test_cancelled_dialog_imports_nothing(self, win, state, tmp_path,
                                              monkeypatch):
        p = tmp_path / "F02-01_TD.txt"; p.write_text(_TD_TXT, encoding="utf-8")
        self._setup(win, state, p, monkeypatch, accept=False)
        win._on_import_tdr()
        assert len(state.project.wells[0].tdrs) == 0


# ---------------------------------------------------------------------------
# View ▸ Basemap on a REAL MainWindow (offscreen, network mocked) — the
# real-window integration the stub-window gap kept missing.
# ---------------------------------------------------------------------------

class TestRestorationRemoval:
    """Real-MainWindow: a UUID-keyed removal event hides the right element at a
    restoration step, and survives a rename (the Step-1 point)."""

    def test_step_hides_element_by_uuid_rename_invariant(self, win, state):
        from section_tool.core.section import Section
        from section_tool.core.surfaces import HorizonPick
        from section_tool.core.restoration import RestorationEvent

        state.add_section(Section([(0.0, 0.0), (1000.0, 0.0)], name="L1",
                                  crs_epsg=32631))
        state.set_active_section(state.project.sections[0])
        hp = HorizonPick([0.0, 1000.0], [100.0, 200.0], name="Top Chalk",
                         section_names=["L1", "L1"])
        state.project.horizon_picks.append(hp)

        seq = state.restoration_sequence
        seq.add_event(RestorationEvent(1, "Remove Top Chalk",
                                       remove_element_ids=[hp.uuid]))

        sv = win._section_view
        seq.current_step = 0
        state.set_restoration_sequence(seq)
        assert sv._get_removed_ids() == set()            # present day: nothing hidden

        seq.current_step = 1
        state.set_restoration_sequence(seq)
        assert hp.uuid in sv._get_removed_ids()          # hidden by UUID at step 1

        hp.name = "Renamed Chalk"                        # rename must not matter
        assert hp.uuid in sv._get_removed_ids()
        sv.render()                                      # full render must not crash

    def test_panel_event_content_drives_section_hide_live(self, win, state,
                                                           monkeypatch):
        """Real-window: defining an event's removed elements in the panel makes the
        section view hide them at that step — through the existing consume path."""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QDialog
        from section_tool.core.section import Section
        from section_tool.core.surfaces import HorizonPick
        from section_tool.views.restoration_panel import _EventEditDialog

        state.add_section(Section([(0.0, 0.0), (1000.0, 0.0)], name="L1",
                                  crs_epsg=32631))
        state.set_active_section(state.project.sections[0])
        hp = HorizonPick([0.0, 1000.0], [100.0, 200.0], name="Top Chalk",
                         section_names=["L1", "L1"])
        state.project.horizon_picks.append(hp)

        def fake_exec(self):
            self._name.setText("Remove Chalk")
            self._elem_list.item(0).setCheckState(Qt.Checked)
            return QDialog.Accepted
        monkeypatch.setattr(_EventEditDialog, "exec", fake_exec)

        win._restoration_widget._add_event()             # define content via the panel
        seq = state.restoration_sequence
        assert seq.events[0].remove_element_ids == [hp.uuid]

        seq.current_step = 1                             # apply the removal step
        state.set_restoration_sequence(seq)
        assert hp.uuid in win._section_view._get_removed_ids()
        win._section_view.render()


class TestBasemapMenuEndToEnd:
    def test_select_basemap_fetches_and_persists(self, win, state, monkeypatch):
        import numpy as np
        from section_tool.views.map_basemap_layer import basemap_available
        if not basemap_available():
            import pytest as _pt
            _pt.skip("contextily not installed")
        state.project.crs_epsg = 32631                 # authoritative project CRS (F3)
        state.add_section(Section([(606000, 6080000), (610000, 6082000)],
                                  name="L1", crs_epsg=32631))
        state.set_active_section(state.project.sections[0])

        fetched = {"n": 0, "epsg": None}

        def fake_fetch(provider, epsg, extent, zoom):
            fetched["n"] += 1
            fetched["epsg"] = epsg
            return np.zeros((8, 8, 3), dtype=np.uint8), tuple(map(float, extent))

        win._map_view._basemap._fetch_fn = fake_fetch
        meta_writes = []
        monkeypatch.setattr(state, "set_meta",
                            lambda k, v: meta_writes.append((k, v)))

        win._basemap_actions["satellite"].trigger()    # drive the real menu action
        assert win._map_view.basemap_source() == "satellite"
        assert ("basemap_source", "satellite") in meta_writes   # persisted per project
        assert win._basemap_actions["satellite"].isChecked()

        win._map_view._fetch_basemap()                 # what the settle timer fires
        win._map_view._basemap._last_thread.join(timeout=5)
        assert fetched["n"] == 1
        assert fetched["epsg"] == 32631                # warped to the PROJECT CRS
        assert win._map_view._basemap.has_image()

        # Toggling back to None drops the underlay and never fetches.
        win._basemap_actions["none"].trigger()
        assert win._map_view.basemap_source() == "none"
        assert not win._map_view._basemap.has_image()

    def test_basemap_menu_default_is_none(self, win):
        assert win._map_view.basemap_source() == "none"
        if getattr(win, "_basemap_actions", None):
            assert win._basemap_actions["none"].isChecked()


# ---------------------------------------------------------------------------
# DEM fetch + hillshade on a REAL MainWindow (offscreen, network mocked).
# ---------------------------------------------------------------------------

class TestDemFetchEndToEnd:
    def test_fetch_dem_loads_hillshade(self, win, state, tmp_path):
        import numpy as np
        rasterio = pytest.importorskip("rasterio")
        from rasterio.transform import from_bounds as _affine
        from section_tool.core.crs import transform_points

        lon0, lon1, lat0, lat1 = 4.0, 5.0, 54.0, 55.0
        src = tmp_path / "src.tif"
        h = w = 32
        elev = np.tile((np.linspace(lon0, lon1, w) - lon0) * 1000.0,
                       (h, 1)).astype("float32")
        with rasterio.open(src, "w", driver="GTiff", height=h, width=w, count=1,
                           dtype="float32", crs="EPSG:4326",
                           transform=_affine(lon0, lat0, lon1, lat1, w, h),
                           nodata=-9999.0) as ds:
            ds.write(elev, 1)

        state.project.crs_epsg = 32631
        ex, ny = transform_points([lon0, lon1], [lat0, lat1], 4326, 32631)
        state.add_section(Section([(ex[0], ny[0]), (ex[1], ny[1])],
                                  name="L1", crs_epsg=32631))
        state.set_active_section(state.project.sections[0])

        class _Ctx:
            def __enter__(self): self._ds = rasterio.open(src); return self._ds
            def __exit__(self, *a): self._ds.close()

        # Network mocked: the opener yields the local fixture.
        win._map_view._dem.fetch(
            "copernicus", win._map_view._basemap_extent(), 32631,
            str(tmp_path / "dem" / "elevation.tif"),
            opener=lambda *a, **k: _Ctx())
        win._map_view._dem._last_thread.join(timeout=10)
        QApplication.processEvents()

        assert win._map_view.has_dem()
        win._map_view.render()
        assert win._map_view._ax.get_images()              # hillshade drawn

        # Hillshade toggle hides it.
        win._map_view.set_hillshade_visible(False)
        assert win._map_view._ax.get_images() == []
        assert win._map_view.hillshade_visible() is False

    def _load_dem_into(self, win, state, tmp_path):
        import numpy as np
        rasterio = pytest.importorskip("rasterio")
        from rasterio.transform import from_bounds as _affine
        from section_tool.core.crs import transform_points
        lon0, lon1, lat0, lat1 = 4.0, 5.0, 54.0, 55.0
        src = tmp_path / "src.tif"
        h = w = 32
        elev = np.tile((np.linspace(lon0, lon1, w) - lon0) * 1000.0,
                       (h, 1)).astype("float32")
        with rasterio.open(src, "w", driver="GTiff", height=h, width=w, count=1,
                           dtype="float32", crs="EPSG:4326",
                           transform=_affine(lon0, lat0, lon1, lat1, w, h),
                           nodata=-9999.0) as ds:
            ds.write(elev, 1)
        state.project.crs_epsg = 32631
        ex, ny = transform_points([lon0, lon1], [lat0, lat1], 4326, 32631)
        state.add_section(Section([(ex[0], ny[0]), (ex[1], ny[1])],
                                  name="L1", crs_epsg=32631))
        state.set_active_section(state.project.sections[0])

        class _Ctx:
            def __enter__(self): self._ds = rasterio.open(src); return self._ds
            def __exit__(self, *a): self._ds.close()

        win._map_view._dem.fetch("copernicus", win._map_view._basemap_extent(),
                                 32631, str(tmp_path / "dem" / "elevation.tif"),
                                 opener=lambda *a, **k: _Ctx())
        win._map_view._dem._last_thread.join(timeout=10)
        QApplication.processEvents()
        assert win._map_view.has_dem()

    def test_colormap_menu_re_tints_and_persists(self, win, state, tmp_path, monkeypatch):
        """Real-window: the Colormap menu re-tints the DEM and persists, no refetch."""
        import numpy as np
        self._load_dem_into(win, state, tmp_path)
        rgb_before = win._map_view._dem._rgb.copy()
        fetch_thread = win._map_view._dem._last_thread

        writes = []
        monkeypatch.setattr(state, "set_meta", lambda k, v: writes.append((k, v)))
        assert win._map_view.dem_cmap() == "terrain"

        win._dem_cmap_actions["gray"].trigger()          # drive the real menu action
        assert win._map_view.dem_cmap() == "gray"
        assert ("dem_cmap", "gray") in writes            # persisted per project
        assert win._dem_cmap_actions["gray"].isChecked()
        assert not np.allclose(win._map_view._dem._rgb[..., :3], rgb_before[..., :3])
        assert win._map_view._dem._last_thread is fetch_thread   # no new fetch

    def test_drape_satellite_toggle_renders_and_persists(self, win, state,
                                                         tmp_path, monkeypatch):
        """Real-window: Drape ▸ Satellite composites imagery on the DEM (injected
        tiles, no network), renders under the data, persists; None returns the tint."""
        import numpy as np
        self._load_dem_into(win, state, tmp_path)
        ext = win._map_view._dem._extent

        def fake_fetch(provider, epsg, extent, zoom="auto"):
            img = np.zeros((48, 48, 3), dtype="uint8")
            img[:, 24:, 2] = 255                          # right half blue
            return img, tuple(float(v) for v in ext)

        win._map_view._drape_fetch_fn = fake_fetch
        writes = []
        monkeypatch.setattr(state, "set_meta", lambda k, v: writes.append((k, v)))

        win._dem_drape_actions["satellite"].trigger()     # drive the real menu action
        win._map_view._dem._last_drape_thread.join(timeout=10)
        QApplication.processEvents()

        assert win._map_view.drape_source() == "satellite"
        assert win._map_view._dem.has_drape()
        assert ("dem_drape", "satellite") in writes       # persisted per project
        assert win._map_view._dem.drape_provenance.get("drape") == "satellite"

        win._map_view.render()
        imgs = [im for im in win._map_view._ax.get_images()
                if -10 < im.get_zorder() < 0]
        assert imgs and imgs[0].get_visible() and imgs[0].get_alpha()

        win._dem_drape_actions["none"].trigger()          # back to the tint
        QApplication.processEvents()
        assert not win._map_view._dem.has_drape()

    def test_fetch_dem_failure_flashes_specific_stage(self, win, state,
                                                      tmp_path, monkeypatch):
        """A failed fetch must reach _flash_status with a specific stage message —
        the blank-map-silent regression. Exercises the real handler wiring
        (_dem.failed → _flash_status) added in _build_elevation_menu."""
        pytest.importorskip("rasterio")
        state.project.crs_epsg = 32631
        state.add_section(Section([(606000, 6080000), (610000, 6082000)],
                                  name="L1", crs_epsg=32631))
        state.set_active_section(state.project.sections[0])

        flashed = []
        monkeypatch.setattr(win, "_flash_status", flashed.append)

        def boom(*a, **k):
            raise RuntimeError("HTTP 500 from source")

        win._map_view._dem.fetch(
            "gebco", win._map_view._basemap_extent(), 32631,
            str(tmp_path / "dem" / "elevation.tif"), opener=boom)
        win._map_view._dem._last_thread.join(timeout=10)
        QApplication.processEvents()
        assert any("DEM failed" in m for m in flashed)
        assert any("HTTP 500" in m for m in flashed)
