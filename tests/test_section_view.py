"""Tests for section_tool.views.section_view.SectionView."""

import sys

import numpy as np
import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick, Surface
from section_tool.core.wells import Well
from section_tool.io.project import SeismicRef
from section_tool.views.section_view import SectionView


# ---------------------------------------------------------------------------
# Session-scoped QApplication (QWidget requires one)
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
def view(qapp, state):
    return SectionView(state)


def _east_section(name="L1", length=1000.0):
    return Section([(0.0, 0.0), (length, 0.0)], name=name)


def _horizon_pick(name="TopSand"):
    return HorizonPick([0.0, 500.0, 1000.0], [100.0, 200.0, 150.0],
                       name=name, color="#ff0000")


def _surface(name="Horizon"):
    xc = np.linspace(0, 1000, 5)
    yc = np.linspace(-50, 50, 5)
    xx, yy = np.meshgrid(xc, yc)
    return Surface.from_grid(xc, yc, xx * 0.1 + 300, name=name)


def _well(name="W1"):
    return Well(name=name, x=500.0, y=0.0, kb=10.0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_is_qwidget(self, view):
        assert view.isWidgetType()

    def test_has_canvas(self, view):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        assert isinstance(view.canvas, FigureCanvasQTAgg)

    def test_has_figure(self, view):
        from matplotlib.figure import Figure
        assert isinstance(view.figure, Figure)

    def test_has_axes(self, view):
        assert view.axes is not None

    def test_has_toolbar(self, view):
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
        # Toolbar exists but is hidden — canvas is the only widget
        assert isinstance(view._toolbar, NavigationToolbar2QT)
        assert not view._toolbar.isVisible()

    def test_picking_off_by_default(self, view):
        assert not view._picking_active

    def test_display_mode_default(self, view):
        assert view.display_mode == "variable_density"

    def test_seismic_cache_empty(self, view):
        assert view._seismic_cache == {}


# ---------------------------------------------------------------------------
# render() — no crash guarantee
# ---------------------------------------------------------------------------

class TestRenderNoCrash:
    def test_render_no_active_section(self, view):
        view.render()  # must not raise

    def test_render_with_section(self, view, state):
        state.add_section(_east_section())
        state.set_active_section(state.project.sections[0])
        view.render()

    def test_render_with_horizon_pick(self, view, state):
        state.add_section(_east_section())
        state.set_active_section(state.project.sections[0])
        state.add_horizon_pick(_horizon_pick())
        view.render()

    def test_render_with_surface(self, view, state):
        state.add_section(_east_section())
        state.set_active_section(state.project.sections[0])
        state.add_surface(_surface())
        view.render()

    def test_render_with_well(self, view, state):
        state.add_section(_east_section())
        state.set_active_section(state.project.sections[0])
        state.add_well(_well())
        view.render()

    def test_render_with_well_and_formation_tops(self, view, state):
        state.add_section(_east_section())
        state.set_active_section(state.project.sections[0])
        well = _well()
        well.add_formation_top("TopA", 500.0)
        state.add_well(well)
        view.render()

    def test_render_with_bad_seismic_ref_no_crash(self, view, state):
        state.add_section(_east_section())
        state.set_active_section(state.project.sections[0])
        state.add_seismic_ref(SeismicRef(path="/nonexistent/file.segy", name="Bad"))
        view.render()  # _get_or_load_seismic must catch the error

    def test_render_all_types(self, view, state):
        state.add_section(_east_section())
        state.set_active_section(state.project.sections[0])
        state.add_horizon_pick(_horizon_pick())
        state.add_surface(_surface())
        state.add_well(_well())
        view.render()

    def test_render_dogleg_section(self, view, state):
        sec = Section([(0, 0), (500, 0), (500, 500)], name="Dogleg")
        state.add_section(sec)
        state.set_active_section(sec)
        state.add_horizon_pick(_horizon_pick())
        view.render()

    def test_render_twice_no_crash(self, view, state):
        state.add_section(_east_section())
        state.set_active_section(state.project.sections[0])
        view.render()
        view.render()


# ---------------------------------------------------------------------------
# Axes state after render
# ---------------------------------------------------------------------------

class TestAxesState:
    def test_axes_cleared_on_no_section(self, view, state):
        state.add_section(_east_section())
        state.set_active_section(state.project.sections[0])
        view.render()
        state.set_active_section(None)
        view.render()
        # After clearing active section, axes should have no lines
        assert len(view.axes.lines) == 0

    def test_title_set_to_section_name(self, view, state):
        state.add_section(_east_section(name="Dip Line"))
        state.set_active_section(state.project.sections[0])
        view.render()
        # Title is now in the header label, not the axes title
        assert "Dip Line" in view._section_name_label.text()

    def test_xlabel_set(self, view, state):
        # Axis labels are now shown by the HUD scale bar (not matplotlib axes).
        state.add_section(_east_section())
        state.set_active_section(state.project.sections[0])
        view.render()
        # xlabel is empty — all label rendering moved to HUD widgets.
        assert view.axes.get_xlabel() == ""

    def test_ylabel_twt(self, view, state):
        # Y-axis label rendered by HUD depth ruler, not matplotlib.
        sec = Section([(0, 0), (1000, 0)], depth_domain="twt")
        state.add_section(sec)
        state.set_active_section(sec)
        view.render()
        assert view.axes.get_ylabel() == ""

    def test_ylabel_depth(self, view, state):
        # Y-axis label rendered by HUD depth ruler, not matplotlib.
        sec = Section([(0, 0), (1000, 0)], depth_domain="depth", depth_units="m")
        state.add_section(sec)
        state.set_active_section(sec)
        view.render()
        assert view.axes.get_ylabel() == ""

    def test_xlim_matches_section_length(self, view, state):
        state.add_section(_east_section(length=2500.0))
        state.set_active_section(state.project.sections[0])
        view.render()
        xmin, xmax = view.axes.get_xlim()
        assert pytest.approx(xmin) == 0.0
        assert pytest.approx(xmax) == 2500.0

    def test_horizon_pick_adds_line(self, view, state):
        state.add_section(_east_section())
        state.set_active_section(state.project.sections[0])
        n_lines_before = len(view.axes.lines)
        state.add_horizon_pick(_horizon_pick())
        view.render()
        assert len(view.axes.lines) > n_lines_before

    def test_well_adds_track_line(self, view, state):
        state.add_section(_east_section())
        state.set_active_section(state.project.sections[0])
        state.add_well(_well())
        view.render()
        assert len(view.axes.lines) > 0

    def test_surface_adds_line(self, view, state):
        state.add_section(_east_section())
        state.set_active_section(state.project.sections[0])
        state.add_surface(_surface())
        view.render()
        assert len(view.axes.lines) > 0


# ---------------------------------------------------------------------------
# State-change → auto-render
# ---------------------------------------------------------------------------

class TestAutoRender:
    def test_active_section_change_triggers_render(self, view, state):
        """Setting active section should trigger a re-render (title changes)."""
        sec1 = _east_section(name="First")
        sec2 = _east_section(name="Second")
        state.add_section(sec1)
        state.add_section(sec2)
        state.set_active_section(sec1)
        QTest.qWait(100)  # flush 50ms debounce timer
        assert "First" in view._section_name_label.text()
        state.set_active_section(sec2)
        QTest.qWait(100)
        assert "Second" in view._section_name_label.text()

    def test_project_changed_triggers_render(self, view, state):
        state.add_section(_east_section(name="Before"))
        state.set_active_section(state.project.sections[0])
        # new_project emits project_changed → view re-renders (no active section)
        state.new_project()
        assert view._section_name_label.text() in ("", "— no section —", "Section View")

    def test_horizon_pick_added_triggers_render(self, view, state):
        sec = _east_section()
        state.add_section(sec)
        state.set_active_section(sec)
        QTest.qWait(100)
        n_before = len(view.axes.lines)
        state.add_horizon_pick(_horizon_pick())
        QTest.qWait(100)
        assert len(view.axes.lines) > n_before

    def test_horizon_pick_removed_triggers_render(self, view, state):
        sec = _east_section()
        state.add_section(sec)
        state.set_active_section(sec)
        pick = _horizon_pick()
        state.add_horizon_pick(pick)
        QTest.qWait(100)
        n_with = len(view.axes.lines)
        state.remove_horizon_pick(pick)
        QTest.qWait(100)
        assert len(view.axes.lines) < n_with

    def test_well_added_triggers_render(self, view, state):
        sec = _east_section()
        state.add_section(sec)
        state.set_active_section(sec)
        QTest.qWait(100)
        n_before = len(view.axes.lines)
        state.add_well(_well())
        QTest.qWait(100)
        assert len(view.axes.lines) > n_before

    def test_surface_added_triggers_render(self, view, state):
        sec = _east_section()
        state.add_section(sec)
        state.set_active_section(sec)
        QTest.qWait(100)
        n_before = len(view.axes.lines)
        state.add_surface(_surface())
        QTest.qWait(100)
        assert len(view.axes.lines) > n_before

    def test_seismic_ref_removed_clears_cache(self, view, state):
        ref = SeismicRef(path="/fake.segy", name="Fake")
        state.add_seismic_ref(ref)
        view._seismic_cache["/fake.segy"] = None  # fake cached entry
        state.remove_seismic_ref(ref)
        assert "/fake.segy" not in view._seismic_cache


# ---------------------------------------------------------------------------
# Picking mode
# ---------------------------------------------------------------------------

class TestPickingMode:
    def test_set_picking_active(self, view):
        view.set_picking_active(True)
        assert view._picking_active

    def test_set_picking_inactive(self, view):
        view.set_picking_active(True)
        view.set_picking_active(False)
        assert not view._picking_active

    def test_set_fault_picking(self, view):
        view.set_fault_picking(True)
        assert view._fault_picking
        assert not view._picking_active

    def test_picking_click_adds_to_active_target(self, view, state):
        """When picking active + target set, click writes to horizon pick."""
        state.add_section(_east_section())
        state.set_active_section(state.project.sections[0])
        hp = HorizonPick([500.0], [1000.0], name="H1")
        state.add_horizon_pick(hp)
        state.set_active_pick_target("Horizons", 0)
        view.set_picking_active(True)

        class FakeEvent:
            button = 1
            inaxes = view.axes
            x = 300.0
            y = 100.0
            xdata = 300.0
            ydata = 500.0

        view._on_sv_press(FakeEvent())
        assert state.project.horizon_picks[0].n_picks == 2

    def test_click_outside_axes_noop(self, view, state):
        state.add_section(_east_section())
        state.set_active_section(state.project.sections[0])
        view.set_picking_active(True)
        n_before = len(state.project.horizon_picks)

        class FakeEvent:
            button = 1
            inaxes = None
            x = 300.0; y = 100.0
            xdata = 300.0; ydata = 500.0

        view._on_sv_press(FakeEvent())
        assert len(state.project.horizon_picks) == n_before

    def test_click_with_none_coords_noop(self, view, state):
        state.add_section(_east_section())
        state.set_active_section(state.project.sections[0])
        hp = HorizonPick([500.0], [1000.0])
        state.add_horizon_pick(hp)
        state.set_active_pick_target("Horizons", 0)
        view.set_picking_active(True)

        class FakeEvent:
            button = 1
            inaxes = view.axes
            x = None; y = None
            xdata = None; ydata = None

        view._on_sv_press(FakeEvent())
        # Pick count unchanged
        assert state.project.horizon_picks[0].n_picks == 1


# ---------------------------------------------------------------------------
# Display mode
# ---------------------------------------------------------------------------

class TestDisplayMode:
    def test_set_wiggle_mode(self, view):
        view.set_display_mode("wiggle")
        assert view.display_mode == "wiggle"

    def test_set_variable_density_mode(self, view):
        view.set_display_mode("wiggle")
        view.set_display_mode("variable_density")
        assert view.display_mode == "variable_density"

    def test_set_mode_triggers_render(self, view, state):
        state.add_section(_east_section(name="Wiggle Test"))
        state.set_active_section(state.project.sections[0])
        view.set_display_mode("wiggle")
        assert "Wiggle Test" in view._section_name_label.text()


# ---------------------------------------------------------------------------
# Seismic cache
# ---------------------------------------------------------------------------

class TestSeismicCache:
    def test_cache_cleared_by_method(self, view):
        view._seismic_cache["/a.segy"] = None
        view._seismic_cache["/b.segy"] = None
        view.clear_seismic_cache()
        assert view._seismic_cache == {}

    def test_bad_path_returns_none(self, view):
        ref = SeismicRef(path="/does_not_exist.segy")
        result = view._get_or_load_seismic(ref)
        assert result is None

    def test_bad_path_not_cached(self, view):
        ref = SeismicRef(path="/also_missing.segy")
        view._get_or_load_seismic(ref)
        assert "/also_missing.segy" not in view._seismic_cache


# ---------------------------------------------------------------------------
# Wiggle rendering helper
# ---------------------------------------------------------------------------

class TestWiggleRendering:
    def test_wiggle_render_no_crash(self, view, state):
        sec = _east_section()
        state.add_section(sec)
        state.set_active_section(sec)
        view.set_display_mode("wiggle")
        # no seismic data — just check it doesn't crash
        view.render()

    def test_wiggle_with_synthetic_data(self, view, state):
        sec = _east_section()
        state.add_section(sec)
        state.set_active_section(sec)
        view.set_display_mode("wiggle")

        # Manually call the wiggle renderer with synthetic data
        n_traces, n_samples = 5, 20
        distances = np.linspace(0, 1000, n_traces)
        data = np.random.default_rng(0).standard_normal((n_traces, n_samples)).astype(np.float32)
        samples = np.linspace(0, 400, n_samples)
        view._render_wiggle(distances, data, samples)
        # Should have added lines
        assert len(view.axes.lines) > 0


# ---------------------------------------------------------------------------
# Polygon preflight
# ---------------------------------------------------------------------------

class TestPolygonPreflight:
    def test_polygon_finish_preflight(self, view, state):
        """Polygon creation uses preflight settings."""
        view.set_polygon_preflight(
            name="TestPoly", formation="Sand",
            color="#ff0000", opacity=0.5
        )
        # Simulate adding vertices then finishing
        view._polygon_drawing = True
        view._polygon_vertices = [(100, 200), (500, 200), (300, 400)]
        # polygon_finished signal emits to state.add_polygon
        state.polygon_added.connect(lambda p: None)  # ensure signal exists
        received = []
        view.polygon_finished.connect(lambda p: received.append(p))
        view.finish_polygon()
        assert len(received) == 1
        p = received[0]
        assert p.name == "TestPoly"
        assert p.fill_color == "#ff0000"
        assert p.fill_alpha == 0.5
        assert p.formation == "Sand"

    def test_preflight_cleared_after_finish(self, view):
        """Preflight dict is cleared after polygon finishes."""
        view.set_polygon_preflight(name="X", formation="", color="#aabbcc", opacity=0.7)
        view._polygon_drawing = True
        view._polygon_vertices = [(0, 0), (100, 0), (50, 100)]
        view.finish_polygon()
        assert view._poly_preflight == {}

    def test_finish_too_few_vertices(self, view):
        """finish_polygon with < 3 vertices clears without emitting."""
        view.set_polygon_preflight(name="Short", formation="", color="#000000", opacity=1.0)
        view._polygon_drawing = True
        view._polygon_vertices = [(0, 0), (100, 0)]
        received = []
        view.polygon_finished.connect(lambda p: received.append(p))
        view.finish_polygon()
        assert len(received) == 0
        assert view._poly_preflight == {}


# ---------------------------------------------------------------------------
# Segment intersection helper
# ---------------------------------------------------------------------------

class TestSegIntersect:
    def test_crossing_segments(self):
        from section_tool.views.section_view import _seg_intersect
        p = _seg_intersect(0, 0, 10, 0, 5, -5, 5, 5)  # horizontal meets vertical
        assert p is not None
        assert abs(p[0] - 5.0) < 0.01
        assert abs(p[1] - 0.0) < 0.01

    def test_non_intersecting(self):
        from section_tool.views.section_view import _seg_intersect
        assert _seg_intersect(0, 0, 5, 0, 6, -5, 6, 5) is None  # no overlap

    def test_parallel_segments(self):
        from section_tool.views.section_view import _seg_intersect
        assert _seg_intersect(0, 0, 10, 0, 0, 5, 10, 5) is None  # parallel horizontal
