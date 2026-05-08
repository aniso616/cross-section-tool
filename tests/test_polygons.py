"""Tests for SectionPolygon, AppState polygon signals, and SectionView drawing."""

import sys
import math

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from cross_section_tool.app_state import AppState
from cross_section_tool.core.polygons import SectionPolygon
from cross_section_tool.core.section import Section
from cross_section_tool.io.project import Project
from cross_section_tool.views.section_view import SectionView


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


@pytest.fixture
def state():
    return AppState()


@pytest.fixture
def view(qapp, state):
    return SectionView(state)


def _rect_poly(**kw) -> SectionPolygon:
    verts = [(0, 100), (500, 100), (500, 800), (0, 800)]
    return SectionPolygon(verts, **kw)


def _sec():
    return Section([(0.0, 0.0), (1000.0, 0.0)], name="L1")


# ---------------------------------------------------------------------------
# SectionPolygon — construction
# ---------------------------------------------------------------------------

class TestSectionPolygonConstruction:
    def test_basic(self):
        p = _rect_poly(name="Test")
        assert p.name == "Test"
        assert p.n_vertices == 4

    def test_too_few_vertices_raises(self):
        with pytest.raises(ValueError):
            SectionPolygon([(0, 0), (100, 0)])

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            SectionPolygon([(0, 0, 0), (100, 0, 0), (100, 100, 0)])

    def test_vertices_copied(self):
        raw = [[0, 0], [100, 0], [100, 100]]
        p = SectionPolygon(raw)
        raw[0][0] = 999
        assert p.vertices[0, 0] == 0.0

    def test_default_colors(self):
        p = _rect_poly()
        assert p.fill_color.startswith("#")
        assert p.edge_color.startswith("#")

    def test_fill_alpha_clamped(self):
        p = SectionPolygon([[0,0],[1,0],[1,1]], fill_alpha=1.5)
        assert p.fill_alpha == 1.0

    def test_fill_alpha_clamped_low(self):
        p = SectionPolygon([[0,0],[1,0],[1,1]], fill_alpha=-0.5)
        assert p.fill_alpha == 0.0

    def test_repr(self):
        p = _rect_poly(name="Shale")
        assert "Shale" in repr(p)


# ---------------------------------------------------------------------------
# SectionPolygon — coordinate conversion
# ---------------------------------------------------------------------------

class TestSectionPolygonCoords:
    def test_closed_distances_has_n_plus_1(self):
        p = _rect_poly()
        assert len(p.closed_distances()) == p.n_vertices + 1

    def test_closed_depths_has_n_plus_1(self):
        p = _rect_poly()
        assert len(p.closed_depths()) == p.n_vertices + 1

    def test_closing_vertex_matches_first(self):
        p = _rect_poly()
        assert pytest.approx(p.closed_distances()[-1]) == p.closed_distances()[0]
        assert pytest.approx(p.closed_depths()[-1]) == p.closed_depths()[0]

    def test_to_map_coords_length(self):
        sec = _sec()
        p = _rect_poly()
        coords = p.to_map_coords(sec)
        assert len(coords) == p.n_vertices

    def test_to_map_coords_depth_negated(self):
        sec = _sec()
        p = SectionPolygon([[0, 500], [100, 500], [100, 800]])
        coords = p.to_map_coords(sec)
        for _, _, z in coords:
            assert z <= 0.0

    def test_to_map_coords_on_east_section(self):
        sec = _sec()  # east section from (0,0) to (1000,0)
        p = SectionPolygon([[200, 500], [400, 500], [400, 800]])
        coords = p.to_map_coords(sec)
        # distance=200 → x=200, y=0
        assert pytest.approx(coords[0][0]) == 200.0
        assert pytest.approx(coords[0][1]) == 0.0
        assert pytest.approx(coords[0][2]) == -500.0


# ---------------------------------------------------------------------------
# AppState — polygon signals
# ---------------------------------------------------------------------------

class TestAppStatePolygonSignals:
    def test_add_polygon_emits_signal(self, state):
        received = []
        state.polygon_added.connect(lambda p: received.append(p))
        poly = _rect_poly()
        state.add_polygon(poly)
        assert received == [poly]

    def test_add_polygon_sets_modified(self, state):
        state.add_polygon(_rect_poly())
        assert state.is_modified

    def test_remove_polygon_emits_signal(self, state):
        poly = _rect_poly()
        state.add_polygon(poly)
        received = []
        state.polygon_removed.connect(lambda p: received.append(p))
        state.remove_polygon(poly)
        assert received == [poly]

    def test_update_polygon_emits_signal(self, state):
        state.add_polygon(_rect_poly(name="Old"))
        received = []
        state.polygon_modified.connect(lambda i, p: received.append((i, p)))
        new = _rect_poly(name="New")
        state.update_polygon(0, new)
        assert len(received) == 1
        assert received[0][1].name == "New"

    def test_polygon_in_project_after_add(self, state):
        poly = _rect_poly()
        state.add_polygon(poly)
        assert poly in state.project.polygons

    def test_polygon_removed_from_project(self, state):
        poly = _rect_poly()
        state.add_polygon(poly)
        state.remove_polygon(poly)
        assert poly not in state.project.polygons

    def test_new_project_clears_polygons(self, state):
        state.add_polygon(_rect_poly())
        state.new_project()
        assert state.project.polygons == []


# ---------------------------------------------------------------------------
# Project — polygon save/load round-trip
# ---------------------------------------------------------------------------

class TestPolygonProjectRoundtrip:
    def test_roundtrip_vertices(self, tmp_path):
        p = Project()
        poly = _rect_poly(name="RT")
        p.polygons.append(poly)
        path = tmp_path / "poly.h5"
        p.save(path)
        loaded = Project.load(path)
        assert len(loaded.polygons) == 1
        np.testing.assert_allclose(loaded.polygons[0].vertices, poly.vertices)

    def test_roundtrip_name(self, tmp_path):
        p = Project()
        p.polygons.append(_rect_poly(name="SavedPoly"))
        path = tmp_path / "p.h5"
        p.save(path)
        assert Project.load(path).polygons[0].name == "SavedPoly"

    def test_roundtrip_fill_color(self, tmp_path):
        p = Project()
        p.polygons.append(_rect_poly(fill_color="#112233"))
        path = tmp_path / "p.h5"
        p.save(path)
        assert Project.load(path).polygons[0].fill_color == "#112233"

    def test_roundtrip_fill_alpha(self, tmp_path):
        p = Project()
        p.polygons.append(SectionPolygon([[0,0],[1,0],[1,1]], fill_alpha=0.4))
        path = tmp_path / "p.h5"
        p.save(path)
        assert pytest.approx(Project.load(path).polygons[0].fill_alpha) == 0.4

    def test_empty_polygons_roundtrip(self, tmp_path):
        p = Project()
        path = tmp_path / "empty.h5"
        p.save(path)
        assert Project.load(path).polygons == []

    def test_multiple_polygons_order(self, tmp_path):
        p = Project()
        for name in ["A", "B", "C"]:
            p.polygons.append(_rect_poly(name=name))
        path = tmp_path / "multi.h5"
        p.save(path)
        names = [poly.name for poly in Project.load(path).polygons]
        assert names == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# SectionView — polygon drawing
# ---------------------------------------------------------------------------

class TestSectionViewPolygonDrawing:
    def test_polygon_drawing_off_by_default(self, view):
        assert not view._polygon_drawing

    def test_set_polygon_drawing(self, view):
        view.set_polygon_drawing(True)
        assert view._polygon_drawing

    def test_set_polygon_drawing_clears_picking(self, view):
        view.set_picking_active(True)
        view.set_polygon_drawing(True)
        assert not view._picking_active

    def test_set_picking_clears_polygon_drawing(self, view):
        view.set_polygon_drawing(True)
        view.set_picking_active(True)
        assert not view._polygon_drawing

    def test_click_adds_vertex_when_drawing(self, view, state):
        state.add_section(_sec())
        state.set_active_section(state.project.sections[0])
        view.set_polygon_drawing(True)

        class FakeEvent:
            button = 1
            inaxes = view.axes
            xdata = 200.0
            ydata = 500.0

        view._on_canvas_click(FakeEvent())
        assert len(view._polygon_vertices) == 1
        assert view._polygon_vertices[0] == (200.0, 500.0)

    def test_multiple_clicks_add_vertices(self, view, state):
        state.add_section(_sec())
        state.set_active_section(state.project.sections[0])
        view.set_polygon_drawing(True)

        class FakeEvent:
            button = 1
            inaxes = view.axes
            def __init__(self, x, y):
                self.xdata = x
                self.ydata = y

        for x, y in [(100, 200), (400, 200), (400, 600)]:
            view._on_canvas_click(FakeEvent(x, y))
        assert len(view._polygon_vertices) == 3

    def test_vertex_added_signal_emitted(self, view, state):
        state.add_section(_sec())
        state.set_active_section(state.project.sections[0])
        view.set_polygon_drawing(True)
        received = []
        view.polygon_vertex_added.connect(lambda d, z: received.append((d, z)))

        class FakeEvent:
            button = 1
            inaxes = view.axes
            xdata = 300.0
            ydata = 700.0

        view._on_canvas_click(FakeEvent())
        assert received == [(300.0, 700.0)]

    def test_finish_polygon_emits_signal(self, view, state):
        state.add_section(_sec())
        state.set_active_section(state.project.sections[0])
        view.set_polygon_drawing(True)
        view._polygon_vertices = [(0,100),(500,100),(500,800),(0,800)]
        received = []
        view.polygon_finished.connect(lambda p: received.append(p))
        view.finish_polygon()
        assert len(received) == 1
        assert isinstance(received[0], SectionPolygon)

    def test_finish_polygon_clears_vertices(self, view, state):
        view.set_polygon_drawing(True)
        view._polygon_vertices = [(0,100),(500,100),(500,800)]
        received = []
        view.polygon_finished.connect(lambda p: received.append(p))
        view.finish_polygon()
        assert len(view._polygon_vertices) == 0

    def test_finish_polygon_too_few_vertices_no_signal(self, view):
        view.set_polygon_drawing(True)
        view._polygon_vertices = [(0, 100), (500, 100)]  # only 2 vertices
        received = []
        view.polygon_finished.connect(lambda p: received.append(p))
        view.finish_polygon()
        assert received == []

    def test_render_with_polygon(self, view, state):
        state.add_section(_sec())
        state.set_active_section(state.project.sections[0])
        state.add_polygon(_rect_poly(name="TestPoly"))
        view.render()  # must not crash

    def test_render_polygon_in_progress(self, view, state):
        state.add_section(_sec())
        state.set_active_section(state.project.sections[0])
        view.set_polygon_drawing(True)
        view._polygon_vertices = [(100, 200), (400, 200)]
        view.render()  # must not crash; shows in-progress lines


# ---------------------------------------------------------------------------
# Tool palette state machine tests
# ---------------------------------------------------------------------------

class TestToolPaletteStateMachine:
    def test_only_one_tool_active_across_many_activations(self, qapp):
        from cross_section_tool.views.tool_palette import ToolPalette, _TOOL_IDS
        p = ToolPalette()
        for tid in _TOOL_IDS:
            p.set_active_tool(tid)
            checked = [t for t, btn in p._buttons.items() if btn.isChecked()]
            assert checked == [tid], f"After activating {tid}: {checked}"

    def test_rapid_switching_emits_each_change(self, qapp):
        from cross_section_tool.views.tool_palette import ToolPalette
        p = ToolPalette()
        sequence = ["pan", "zoom", "select", "horizon_pick", "pan"]
        received = []
        p.tool_changed.connect(lambda t: received.append(t))
        prev = p.active_tool
        for tid in sequence:
            p.set_active_tool(tid)
            if tid != prev:
                prev = tid
        assert received == sequence  # each activation is new vs previous

    def test_deactivate_polygon_via_select(self, qapp):
        from cross_section_tool.views.tool_palette import ToolPalette
        p = ToolPalette()
        p.set_active_tool("polygon")
        p.set_active_tool("select")
        assert p.active_tool == "select"
        assert not p._buttons["polygon"].isChecked()
