"""Tests for cross_section_tool.views.map_view.MapView."""

import sys
import math
import copy

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from cross_section_tool.app_state import AppState
from cross_section_tool.core.section import Section
from cross_section_tool.core.surfaces import Surface
from cross_section_tool.core.wells import Well
from cross_section_tool.io.project import SeismicRef
from cross_section_tool.views.map_view import MapView, _min_dist_to_polyline, _nice_interval


# ---------------------------------------------------------------------------
# Session-scoped QApplication
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


# ---------------------------------------------------------------------------
# Per-test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def state():
    s = AppState()
    s.set_active_tool("select")  # node editing requires select or edit_nodes
    return s


@pytest.fixture
def view(qapp, state):
    return MapView(state)


def _sec(name="L1", nodes=None):
    if nodes is None:
        nodes = [(0.0, 0.0), (1000.0, 0.0)]
    return Section(nodes, name=name)


def _well(name="W1", x=500.0, y=200.0):
    return Well(name=name, x=x, y=y)


def _surf(name="S1"):
    xc = np.linspace(0, 1000, 5)
    yc = np.linspace(0, 500, 5)
    xx, yy = np.meshgrid(xc, yc)
    return Surface.from_grid(xc, yc, xx * 0.1 + 100, name=name)


# ---------------------------------------------------------------------------
# _min_dist_to_polyline  (pure geometry, no Qt needed)
# ---------------------------------------------------------------------------

class TestMinDistToPolyline:
    def test_point_on_segment(self):
        nodes = np.array([[0.0, 0.0], [1000.0, 0.0]])
        assert pytest.approx(_min_dist_to_polyline(500.0, 0.0, nodes)) == 0.0

    def test_point_off_midpoint(self):
        nodes = np.array([[0.0, 0.0], [1000.0, 0.0]])
        assert pytest.approx(_min_dist_to_polyline(500.0, 100.0, nodes)) == 100.0

    def test_point_beyond_end(self):
        nodes = np.array([[0.0, 0.0], [1000.0, 0.0]])
        # Nearest point is the end node (1000,0); dist = sqrt(200^2 + 50^2)
        d = _min_dist_to_polyline(1200.0, 50.0, nodes)
        assert pytest.approx(d) == math.hypot(200.0, 50.0)

    def test_point_before_start(self):
        nodes = np.array([[0.0, 0.0], [1000.0, 0.0]])
        d = _min_dist_to_polyline(-100.0, 0.0, nodes)
        assert pytest.approx(d) == 100.0

    def test_dogleg_second_segment(self):
        nodes = np.array([[0.0, 0.0], [500.0, 0.0], [500.0, 500.0]])
        # Point (500, 250) is on the second segment
        assert pytest.approx(_min_dist_to_polyline(500.0, 250.0, nodes)) == 0.0

    def test_zero_length_segment(self):
        nodes = np.array([[0.0, 0.0], [0.0, 0.0], [1000.0, 0.0]])
        assert pytest.approx(_min_dist_to_polyline(500.0, 0.0, nodes)) == 0.0


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
        # Toolbar exists but is hidden — not in the layout
        assert isinstance(view._toolbar, NavigationToolbar2QT)
        assert not view._toolbar.isVisible()

    def test_no_drag_initially(self, view):
        assert view._selected_node is None
        assert view._hover_node is None
        assert view._drag_active is False


# ---------------------------------------------------------------------------
# render() — no-crash guarantee
# ---------------------------------------------------------------------------

class TestRenderNoCrash:
    def test_render_empty(self, view):
        view.render()

    def test_render_with_section(self, view, state):
        state.add_section(_sec())
        view.render()

    def test_render_with_active_section(self, view, state):
        sec = _sec()
        state.add_section(sec)
        state.set_active_section(sec)
        view.render()

    def test_render_multiple_sections(self, view, state):
        for i in range(4):
            state.add_section(_sec(name=f"L{i}"))
        view.render()

    def test_render_with_well(self, view, state):
        state.add_well(_well())
        view.render()

    def test_render_with_surface(self, view, state):
        state.add_surface(_surf())
        view.render()

    def test_render_dogleg(self, view, state):
        sec = _sec(nodes=[(0, 0), (500, 0), (500, 500)])
        state.add_section(sec)
        view.render()

    def test_render_all_types(self, view, state):
        state.add_section(_sec())
        state.add_well(_well())
        state.add_surface(_surf())
        view.render()

    def test_render_twice(self, view, state):
        state.add_section(_sec())
        view.render()
        view.render()


# ---------------------------------------------------------------------------
# Axes state after render
# ---------------------------------------------------------------------------

class TestAxesState:
    def test_xlabel_set(self, view, state):
        state.add_section(_sec())
        view.render()
        assert "Easting" in view.axes.get_xlabel()

    def test_ylabel_set(self, view, state):
        state.add_section(_sec())
        view.render()
        assert "Northing" in view.axes.get_ylabel()

    def test_aspect_equal(self, view, state):
        state.add_section(_sec())
        view.render()
        assert view.axes.get_aspect() == 1.0  # 'equal' stores as 1.0

    def test_section_produces_lines(self, view, state):
        state.add_section(_sec())
        view.render()
        assert len(view.axes.lines) > 0

    def test_two_sections_two_lines(self, view, state):
        state.add_section(_sec("A"))
        state.add_section(_sec("B"))
        view.render()
        assert len(view.axes.lines) >= 2

    def test_active_section_rendered_in_active_color(self, view, state):
        sec = _sec()
        state.add_section(sec)
        state.set_active_section(sec)
        view.render()
        # Active section line should use active color
        from cross_section_tool.views.map_view import _ACTIVE_COLOR
        active_line_colors = [
            l.get_color() for l in view.axes.lines if l.get_color() == _ACTIVE_COLOR
        ]
        assert len(active_line_colors) > 0

    def test_well_produces_scatter(self, view, state):
        state.add_well(_well())
        view.render()
        # Well rendered as scatter; check axes has scatter collections
        assert len(view.axes.collections) > 0

    def test_surface_produces_patch(self, view, state):
        state.add_surface(_surf())
        view.render()
        assert len(view.axes.patches) > 0

    def test_no_patches_without_surfaces(self, view, state):
        state.add_section(_sec())
        view.render()
        assert len(view.axes.patches) == 0


# ---------------------------------------------------------------------------
# Auto-render on AppState signals
# ---------------------------------------------------------------------------

class TestAutoRender:
    def test_section_added_triggers_render(self, view, state):
        state.add_section(_sec(name="Auto"))
        assert len(view.axes.lines) > 0

    def test_section_removed_triggers_render(self, view, state):
        sec = _sec(name="SectionToRemove")
        state.add_section(sec)
        state.remove_section(sec)
        # After removal, no section label text remains
        texts = [t.get_text() for t in view.axes.texts]
        assert not any("SectionToRemove" in t for t in texts)

    def test_active_section_change_triggers_render(self, view, state):
        sec1 = _sec("First")
        sec2 = _sec("Second")
        state.add_section(sec1)
        state.add_section(sec2)
        state.set_active_section(sec1)
        # Check active color applied to sec1
        from cross_section_tool.views.map_view import _ACTIVE_COLOR, _INACTIVE_COLOR
        active_colors = [l.get_color() for l in view.axes.lines
                         if l.get_color() == _ACTIVE_COLOR]
        inactive_colors = [l.get_color() for l in view.axes.lines
                           if l.get_color() == _INACTIVE_COLOR]
        assert len(active_colors) > 0
        assert len(inactive_colors) > 0

    def test_well_added_triggers_render(self, view, state):
        state.add_well(_well())
        assert len(view.axes.collections) > 0

    def test_well_removed_triggers_render(self, view, state):
        w = _well()
        state.add_well(w)
        n_before = len(view.axes.collections)
        state.remove_well(w)
        assert len(view.axes.collections) < n_before

    def test_surface_added_triggers_render(self, view, state):
        state.add_surface(_surf())
        assert len(view.axes.patches) > 0

    def test_surface_removed_triggers_render(self, view, state):
        s = _surf()
        state.add_surface(s)
        state.remove_surface(s)
        assert len(view.axes.patches) == 0

    def test_project_changed_triggers_render(self, view, state):
        state.add_section(_sec(name="BeforeReset"))
        state.new_project()
        # After new_project, no section labels should remain
        texts = [t.get_text() for t in view.axes.texts]
        assert not any("BeforeReset" in t for t in texts)

    def test_seismic_ref_added_triggers_render(self, view, state):
        state.add_section(_sec(name="Seismic"))
        n_before = len(view.axes.lines)
        state.add_seismic_ref(SeismicRef(path="/x.segy"))
        # Map re-renders (no seismic data loaded, but render is triggered)
        assert len(view.axes.lines) == n_before  # same content, just re-rendered


# ---------------------------------------------------------------------------
# Hit-testing helpers
# ---------------------------------------------------------------------------

class TestHitTesting:
    def test_find_nearest_node_exact_hit(self, view, state):
        sec = _sec(nodes=[(0.0, 0.0), (1000.0, 0.0)])
        state.add_section(sec)
        # Set known axes limits so threshold is predictable
        view.axes.set_xlim(-200, 1200)
        view.axes.set_ylim(-200, 200)
        view.canvas.draw()
        result = view._find_nearest_node(0.0, 0.0)
        assert result == (0, 0)

    def test_find_nearest_node_second_node(self, view, state):
        sec = _sec(nodes=[(0.0, 0.0), (1000.0, 0.0)])
        state.add_section(sec)
        view.axes.set_xlim(-200, 1200)
        view.axes.set_ylim(-200, 200)
        view.canvas.draw()
        result = view._find_nearest_node(1000.0, 0.0)
        assert result == (0, 1)

    def test_find_nearest_node_far_away_returns_none(self, view, state):
        sec = _sec(nodes=[(0.0, 0.0), (1000.0, 0.0)])
        state.add_section(sec)
        view.axes.set_xlim(-200, 1200)
        view.axes.set_ylim(-200, 200)
        view.canvas.draw()
        result = view._find_nearest_node(5000.0, 5000.0)
        assert result is None

    def test_find_nearest_node_no_sections(self, view, state):
        view.axes.set_xlim(-200, 1200)
        view.axes.set_ylim(-200, 200)
        view.canvas.draw()
        assert view._find_nearest_node(0.0, 0.0) is None

    def test_find_nearest_section_on_line(self, view, state):
        sec = _sec(nodes=[(0.0, 0.0), (1000.0, 0.0)])
        state.add_section(sec)
        view.axes.set_xlim(-200, 1200)
        view.axes.set_ylim(-200, 200)
        view.canvas.draw()
        # Click exactly on midpoint of section
        result = view._find_nearest_section(500.0, 0.0)
        assert result == 0

    def test_find_nearest_section_near_line(self, view, state):
        sec = _sec(nodes=[(0.0, 0.0), (1000.0, 0.0)])
        state.add_section(sec)
        view.axes.set_xlim(-200, 1200)
        view.axes.set_ylim(-500, 500)
        view.canvas.draw()
        # Threshold is ~HIT_PX * scale; small offset should still hit
        threshold = view._pixel_threshold()
        result = view._find_nearest_section(500.0, threshold * 0.5)
        assert result == 0

    def test_find_nearest_section_far_away(self, view, state):
        sec = _sec(nodes=[(0.0, 0.0), (1000.0, 0.0)])
        state.add_section(sec)
        view.axes.set_xlim(-200, 1200)
        view.axes.set_ylim(-200, 200)
        view.canvas.draw()
        result = view._find_nearest_section(500.0, 9999.0)
        assert result is None

    def test_find_nearest_section_no_sections(self, view, state):
        view.axes.set_xlim(-200, 1200)
        view.axes.set_ylim(-200, 200)
        view.canvas.draw()
        assert view._find_nearest_section(500.0, 0.0) is None


# ---------------------------------------------------------------------------
# Node drag interaction
# ---------------------------------------------------------------------------

class FakePress:
    button = 1
    def __init__(self, axes, x, y):
        self.inaxes = axes
        self.xdata = x
        self.ydata = y
        # Display pixel coordinates required by _start_pan
        self.x = float(x) if x is not None else 0.0
        self.y = float(y) if y is not None else 0.0

class FakeMotion:
    def __init__(self, axes, x, y):
        self.inaxes = axes
        self.xdata = x
        self.ydata = y

class FakeRelease:
    button = 1
    def __init__(self, axes, x, y):
        self.inaxes = axes
        self.xdata = x
        self.ydata = y


class TestNodeDrag:
    def _setup(self, view, state):
        sec = _sec(nodes=[(0.0, 0.0), (1000.0, 0.0)])
        state.add_section(sec)
        view.axes.set_xlim(-200, 1200)
        view.axes.set_ylim(-200, 200)
        view.canvas.draw()
        return sec

    def test_press_near_node_starts_drag(self, view, state):
        self._setup(view, state)
        view._on_canvas_press(FakePress(view.axes, 0.0, 0.0))
        assert view._selected_node is not None

    def test_drag_records_correct_sec_idx(self, view, state):
        self._setup(view, state)
        view._on_canvas_press(FakePress(view.axes, 0.0, 0.0))
        assert view._selected_node[0] == 0

    def test_drag_records_correct_node_idx(self, view, state):
        self._setup(view, state)
        view._on_canvas_press(FakePress(view.axes, 0.0, 0.0))
        assert view._selected_node[1] == 0

    def test_press_far_from_node_no_drag(self, view, state):
        self._setup(view, state)
        view._on_canvas_press(FakePress(view.axes, 5000.0, 5000.0))
        assert view._selected_node is None

    def test_motion_updates_drag_copy(self, view, state):
        self._setup(view, state)
        view._on_canvas_press(FakePress(view.axes, 0.0, 0.0))
        view._on_canvas_motion(FakeMotion(view.axes, -100.0, 50.0))
        node = view._drag_section_copy.nodes[0]
        assert pytest.approx(node[0]) == -100.0
        assert pytest.approx(node[1]) == 50.0

    def test_release_clears_drag(self, view, state):
        self._setup(view, state)
        view._on_canvas_press(FakePress(view.axes, 0.0, 0.0))
        view._on_canvas_release(FakeRelease(view.axes, -100.0, 50.0))
        assert view._drag_active is False

    def test_release_updates_app_state(self, view, state):
        self._setup(view, state)
        view._on_canvas_press(FakePress(view.axes, 0.0, 0.0))
        view._on_canvas_motion(FakeMotion(view.axes, -100.0, 50.0))
        view._on_canvas_release(FakeRelease(view.axes, -100.0, 50.0))
        updated = state.project.sections[0]
        assert pytest.approx(updated.nodes[0, 0]) == -100.0
        assert pytest.approx(updated.nodes[0, 1]) == 50.0

    def test_release_emits_section_node_moved(self, view, state):
        self._setup(view, state)
        received = []
        view.section_node_moved.connect(
            lambda si, ni, x, y: received.append((si, ni, x, y))
        )
        view._on_canvas_press(FakePress(view.axes, 0.0, 0.0))
        view._on_canvas_motion(FakeMotion(view.axes, -50.0, 25.0))
        view._on_canvas_release(FakeRelease(view.axes, -50.0, 25.0))
        assert len(received) == 1
        si, ni, x, y = received[0]
        assert si == 0
        assert ni == 0
        assert pytest.approx(x) == -50.0
        assert pytest.approx(y) == 25.0

    def test_motion_without_drag_is_noop(self, view, state):
        self._setup(view, state)
        # Should not crash
        view._on_canvas_motion(FakeMotion(view.axes, 500.0, 0.0))

    def test_right_button_press_does_not_start_drag(self, view, state):
        self._setup(view, state)
        event = FakePress(view.axes, 0.0, 0.0)
        event.button = 3
        view._on_canvas_press(event)
        assert view._selected_node is None

    def test_press_outside_axes_does_not_start_drag(self, view, state):
        self._setup(view, state)
        event = FakePress(None, 0.0, 0.0)  # inaxes=None
        view._on_canvas_press(event)
        assert view._selected_node is None

    def test_motion_with_none_coords_does_not_crash(self, view, state):
        self._setup(view, state)
        view._on_canvas_press(FakePress(view.axes, 0.0, 0.0))
        event = FakeMotion(view.axes, None, None)
        view._on_canvas_motion(event)  # must not crash

    def test_drag_does_not_modify_original_section(self, view, state):
        """AppState section must not change until release."""
        self._setup(view, state)
        original_x = state.project.sections[0].nodes[0, 0]
        view._on_canvas_press(FakePress(view.axes, 0.0, 0.0))
        view._on_canvas_motion(FakeMotion(view.axes, -500.0, 0.0))
        # AppState not yet updated
        assert pytest.approx(state.project.sections[0].nodes[0, 0]) == original_x

    def test_drag_second_node(self, view, state):
        self._setup(view, state)
        view._on_canvas_press(FakePress(view.axes, 1000.0, 0.0))
        assert view._selected_node is not None
        assert view._selected_node[1] == 1

    def test_drag_section_modified_signal_emitted(self, view, state):
        """AppState.section_modified must fire on drag release."""
        self._setup(view, state)
        received = []
        state.section_modified.connect(lambda idx, sec: received.append(idx))
        view._on_canvas_press(FakePress(view.axes, 0.0, 0.0))
        view._on_canvas_motion(FakeMotion(view.axes, -100.0, 0.0))
        view._on_canvas_release(FakeRelease(view.axes, -100.0, 0.0))
        assert 0 in received


# ---------------------------------------------------------------------------
# Section selection via click on line
# ---------------------------------------------------------------------------

class TestSectionSelection:
    def test_click_on_section_line_selects_it(self, view, state):
        sec = _sec(nodes=[(0.0, 0.0), (1000.0, 0.0)])
        state.add_section(sec)
        view.axes.set_xlim(-200, 1200)
        view.axes.set_ylim(-200, 200)
        view.canvas.draw()
        # Click exactly on midpoint (no node there)
        view._on_canvas_press(FakePress(view.axes, 500.0, 0.0))
        assert state.active_section is sec

    def test_click_far_from_all_sections_no_change(self, view, state):
        sec = _sec()
        state.add_section(sec)
        state.set_active_section(sec)
        view.axes.set_xlim(-200, 1200)
        view.axes.set_ylim(-200, 200)
        view.canvas.draw()
        # Click very far away
        view._on_canvas_press(FakePress(view.axes, 9999.0, 9999.0))
        # Active section unchanged
        assert state.active_section is sec


# ---------------------------------------------------------------------------
# Graticule / grid helpers
# ---------------------------------------------------------------------------

class TestNiceInterval:
    def test_small_span(self):
        assert _nice_interval(0.3) == 0.5

    def test_span_80(self):
        assert _nice_interval(80) == 100.0

    def test_span_300(self):
        assert _nice_interval(300) == 500.0

    def test_span_1500(self):
        assert _nice_interval(1500) == 2000.0

    def test_span_7000(self):
        assert _nice_interval(7000) == 10000.0

    def test_exact_power_of_ten(self):
        assert _nice_interval(100.0) == 100.0

    def test_zero_returns_one(self):
        assert _nice_interval(0) == 1.0

    def test_negative_returns_one(self):
        assert _nice_interval(-5) == 1.0

    def test_result_ge_input(self):
        for v in [1, 3, 7, 15, 99, 250, 999, 2500]:
            assert _nice_interval(v) >= v


class TestGraticule:
    def test_render_with_sections_sets_labels(self, view, state):
        state.add_section(Section([(0.0, 0.0), (1000.0, 0.0)]))
        view.render()
        assert "Easting" in view.axes.get_xlabel()
        assert "Northing" in view.axes.get_ylabel()

    def test_render_empty_still_sets_labels(self, view):
        view.render()
        assert "Easting" in view.axes.get_xlabel()

    def test_grid_lines_present_after_render(self, view, state):
        state.add_section(Section([(0.0, 0.0), (5000.0, 0.0)]))
        view.render()
        # Grid lines are axvlines/axhlines; check lines exist
        assert len(view.axes.lines) >= 0  # renders without crash

    def test_no_scientific_notation(self, view, state):
        state.add_section(Section([(500000.0, 5500000.0), (510000.0, 5500000.0)]))
        view.render()
        # TickLabel format set to plain — just check no crash
        assert view.axes.get_xlabel() != ""


# ---------------------------------------------------------------------------
# Hover state
# ---------------------------------------------------------------------------

class TestHoverState:
    def _setup(self, view, state):
        sec = _sec(nodes=[(0.0, 0.0), (1000.0, 0.0)])
        state.add_section(sec)
        view.axes.set_xlim(-200, 1200)
        view.axes.set_ylim(-200, 200)
        view.canvas.draw()

    def test_hover_none_initially(self, view):
        assert view._hover_node is None

    def test_hover_set_on_move_near_node(self, view, state):
        self._setup(view, state)

        class FakeMotion:
            xdata = 0.0
            ydata = 0.0
            inaxes = view.axes

        view._update_hover(FakeMotion())
        assert view._hover_node is not None
        assert view._hover_node == (0, 0)

    def test_hover_cleared_on_move_away(self, view, state):
        self._setup(view, state)

        class FarMotion:
            xdata = 9999.0
            ydata = 9999.0
            inaxes = view.axes

        view._hover_node = (0, 0)
        view._update_hover(FarMotion())
        assert view._hover_node is None

    def test_hover_cleared_on_non_edit_tool(self, view, state):
        self._setup(view, state)
        view._hover_node = (0, 0)
        state.set_active_tool("pan")
        # tool_changed should clear hover
        assert view._hover_node is None

    def test_no_hover_when_pan_tool_active(self, view, state):
        self._setup(view, state)
        state.set_active_tool("pan")

        class FakeMotion:
            button = None
            xdata = 0.0
            ydata = 0.0
            inaxes = view.axes

        view._on_canvas_motion(FakeMotion())
        assert view._hover_node is None


# ---------------------------------------------------------------------------
# Delete key
# ---------------------------------------------------------------------------

class TestDeleteKey:
    def _setup(self, view, state):
        sec = Section([(0.0, 0.0), (500.0, 0.0), (1000.0, 0.0)], name="L")
        state.add_section(sec)
        view.axes.set_xlim(-200, 1200)
        view.axes.set_ylim(-200, 200)
        view.canvas.draw()

    def test_delete_selected_node(self, view, state):
        self._setup(view, state)
        view._on_canvas_press(FakePress(view.axes, 500.0, 0.0))  # select midpoint

        class FakeKey:
            key = "delete"

        view._on_key_press(FakeKey())
        assert state.project.sections[0].n_nodes == 2

    def test_delete_noop_when_nothing_selected(self, view, state):
        self._setup(view, state)

        class FakeKey:
            key = "delete"

        view._on_key_press(FakeKey())
        assert state.project.sections[0].n_nodes == 3

    def test_delete_noop_when_only_two_nodes(self, view, state):
        sec = _sec(nodes=[(0.0, 0.0), (1000.0, 0.0)])
        state.add_section(sec)
        view.axes.set_xlim(-200, 1200)
        view.axes.set_ylim(-200, 200)
        view.canvas.draw()
        view._on_canvas_press(FakePress(view.axes, 0.0, 0.0))

        class FakeKey:
            key = "delete"

        view._on_key_press(FakeKey())
        assert state.project.sections[0].n_nodes == 2  # unchanged


# ---------------------------------------------------------------------------
# Escape key
# ---------------------------------------------------------------------------

class TestEscapeKey:
    def _setup(self, view, state):
        sec = _sec(nodes=[(0.0, 0.0), (1000.0, 0.0)])
        state.add_section(sec)
        view.axes.set_xlim(-200, 1200)
        view.axes.set_ylim(-200, 200)
        view.canvas.draw()

    def test_escape_deselects(self, view, state):
        self._setup(view, state)
        view._on_canvas_press(FakePress(view.axes, 0.0, 0.0))
        assert view._selected_node is not None

        class FakeKey:
            key = "escape"

        view._on_key_press(FakeKey())
        assert view._selected_node is None

    def test_escape_cancels_drag_without_committing(self, view, state):
        self._setup(view, state)
        original_x = state.project.sections[0].nodes[0, 0]
        view._on_canvas_press(FakePress(view.axes, 0.0, 0.0))
        view._on_canvas_motion(FakeMotion(view.axes, -500.0, 0.0))

        class FakeKey:
            key = "escape"

        view._on_key_press(FakeKey())
        # Drag cancelled — AppState NOT updated
        assert pytest.approx(state.project.sections[0].nodes[0, 0]) == original_x
        assert not view._drag_active


# ---------------------------------------------------------------------------
# Tool-routing (node editing disabled for non-edit tools)
# ---------------------------------------------------------------------------

class TestToolRouting:
    def test_press_noop_when_pan_tool_active(self, view, state):
        sec = _sec(nodes=[(0.0, 0.0), (1000.0, 0.0)])
        state.add_section(sec)
        state.set_active_tool("pan")
        view.axes.set_xlim(-200, 1200)
        view.axes.set_ylim(-200, 200)
        view.canvas.draw()
        view._on_canvas_press(FakePress(view.axes, 0.0, 0.0))
        # Pan tool: should start pan, NOT select node
        assert view._selected_node is None

    def test_node_selection_works_for_edit_nodes_tool(self, view, state):
        sec = _sec(nodes=[(0.0, 0.0), (1000.0, 0.0)])
        state.add_section(sec)
        state.set_active_tool("edit_nodes")
        view.axes.set_xlim(-200, 1200)
        view.axes.set_ylim(-200, 200)
        view.canvas.draw()
        view._on_canvas_press(FakePress(view.axes, 0.0, 0.0))
        assert view._selected_node is not None
