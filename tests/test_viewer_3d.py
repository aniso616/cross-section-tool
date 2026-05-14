"""Tests for cross_section_tool.views.viewer_3d."""

import sys
import math

import numpy as np
import pytest
import pyvista as pv
from PySide6.QtWidgets import QApplication

from cross_section_tool.app_state import AppState
from cross_section_tool.core.section import Section
from cross_section_tool.core.surfaces import HorizonPick, Surface
from cross_section_tool.core.wells import DeviationSurvey, Well
from cross_section_tool.views.viewer_3d import (
    Viewer3D,
    build_horizon_pick_3d,
    build_section_curtain,
    build_surface_mesh,
    build_well_track,
)


# ---------------------------------------------------------------------------
# Session-scoped QApplication
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def state():
    return AppState()


@pytest.fixture
def viewer(qapp, state):
    v = Viewer3D(state)
    yield v
    try:
        v.plotter.close()
    except Exception:
        pass


def _sec(name="L1", nodes=None):
    if nodes is None:
        nodes = [(0.0, 0.0), (1000.0, 0.0)]
    return Section(nodes, name=name)


def _grid_surf(name="S1"):
    xc = np.linspace(0, 1000, 5)
    yc = np.linspace(-100, 100, 5)
    xx, yy = np.meshgrid(xc, yc)
    return Surface.from_grid(xc, yc, xx * 0.1 + 500, name=name)


def _scatter_surf(name="Scat"):
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1000, 30)
    y = rng.uniform(-200, 200, 30)
    z = x * 0.1 + 400
    return Surface(x, y, z, name=name)


def _well(name="W1", x=500.0, y=0.0):
    return Well(name=name, x=x, y=y, kb=10.0)


def _deviated_well(name="Dev"):
    dev = DeviationSurvey(
        [0.0, 1000.0], [45.0, 45.0], [90.0, 90.0], 0.0, 0.0
    )
    return Well(name=name, x=0.0, y=0.0, deviation=dev)


def _pick(name="TopSand"):
    return HorizonPick(
        [0.0, 500.0, 1000.0], [500.0, 700.0, 600.0],
        name=name, color="#ff0000"
    )


# ===========================================================================
# Mesh builder tests (no Qt, no pyvista plotter needed)
# ===========================================================================

class TestBuildSectionCurtain:
    def test_returns_polydata(self):
        sec = _sec()
        mesh = build_section_curtain(sec, max_depth=2000.0)
        assert isinstance(mesh, pv.PolyData)

    def test_none_for_single_node(self):
        # Can't construct Section with 1 node, so test with a degenerate copy
        sec = _sec()
        sec._nodes = sec._nodes[:1]  # manually reduce to 1 node
        assert build_section_curtain(sec, 2000.0) is None

    def test_point_count_straight(self):
        # 2 nodes → 2 top + 2 bottom = 4 points
        sec = _sec(nodes=[(0.0, 0.0), (1000.0, 0.0)])
        mesh = build_section_curtain(sec, 1000.0)
        assert mesh.n_points == 4

    def test_point_count_dogleg(self):
        # 3 nodes → 3 top + 3 bottom = 6 points
        sec = _sec(nodes=[(0.0, 0.0), (500.0, 0.0), (500.0, 500.0)])
        mesh = build_section_curtain(sec, 1000.0)
        assert mesh.n_points == 6

    def test_top_nodes_at_z_zero(self):
        sec = _sec(nodes=[(0.0, 0.0), (1000.0, 0.0)])
        mesh = build_section_curtain(sec, 2000.0)
        pts = mesh.points
        top_z = pts[:2, 2]
        np.testing.assert_allclose(top_z, 0.0)

    def test_bottom_nodes_at_neg_max_depth(self):
        sec = _sec(nodes=[(0.0, 0.0), (1000.0, 0.0)])
        mesh = build_section_curtain(sec, 3000.0)
        pts = mesh.points
        bottom_z = pts[2:, 2]
        np.testing.assert_allclose(bottom_z, -3000.0)

    def test_xy_coords_match_section_nodes(self):
        sec = _sec(nodes=[(100.0, 200.0), (900.0, 400.0)])
        mesh = build_section_curtain(sec, 1000.0)
        pts = mesh.points
        assert pytest.approx(pts[0, 0]) == 100.0
        assert pytest.approx(pts[0, 1]) == 200.0
        assert pytest.approx(pts[1, 0]) == 900.0
        assert pytest.approx(pts[1, 1]) == 400.0

    def test_face_count_straight(self):
        # 2 nodes → 1 segment → 1 quad face
        sec = _sec(nodes=[(0.0, 0.0), (1000.0, 0.0)])
        mesh = build_section_curtain(sec, 1000.0)
        assert mesh.n_faces == 1

    def test_face_count_dogleg(self):
        # 3 nodes → 2 segments → 2 quad faces
        sec = _sec(nodes=[(0.0, 0.0), (500.0, 0.0), (500.0, 500.0)])
        mesh = build_section_curtain(sec, 1000.0)
        assert mesh.n_faces == 2

    def test_default_max_depth(self):
        from cross_section_tool.views.viewer_3d import _DEFAULT_DEPTH
        sec = _sec()
        mesh = build_section_curtain(sec)
        pts = mesh.points
        assert pytest.approx(pts[2:, 2].min()) == -_DEFAULT_DEPTH


class TestBuildSurfaceMesh:
    def test_grid_surface_returns_structured_grid(self):
        surf = _grid_surf()
        mesh = build_surface_mesh(surf)
        assert isinstance(mesh, pv.StructuredGrid)

    def test_scattered_surface_returns_polydata(self):
        surf = _scatter_surf()
        mesh = build_surface_mesh(surf)
        assert isinstance(mesh, pv.PolyData)

    def test_grid_z_is_negated(self):
        # z=500 at x=0 → 3D z should be -500
        xc = np.array([0.0, 1.0])
        yc = np.array([0.0, 1.0])
        zg = np.array([[500.0, 500.0], [500.0, 500.0]])
        surf = Surface.from_grid(xc, yc, zg)
        mesh = build_surface_mesh(surf)
        np.testing.assert_allclose(mesh.points[:, 2], -500.0)

    def test_grid_xy_preserved(self):
        xc = np.array([0.0, 100.0])
        yc = np.array([0.0, 50.0])
        zg = np.ones((2, 2)) * 300.0
        surf = Surface.from_grid(xc, yc, zg)
        mesh = build_surface_mesh(surf)
        x_vals = mesh.points[:, 0]
        assert set(np.round(x_vals).astype(int)) == {0, 100}

    def test_collinear_scattered_returns_none(self):
        # Points on a straight line → Delaunay 2D produces empty mesh → None
        surf = Surface([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [10.0, 10.0, 10.0])
        assert build_surface_mesh(surf) is None

    def test_scattered_point_count(self):
        surf = _scatter_surf()
        mesh = build_surface_mesh(surf)
        assert mesh is not None
        assert mesh.n_points > 0


class TestBuildWellTrack:
    def test_vertical_well_returns_polydata(self):
        well = _well()
        mesh = build_well_track(well)
        assert isinstance(mesh, pv.PolyData)

    def test_vertical_well_x_constant(self):
        well = _well(x=300.0, y=150.0)
        mesh = build_well_track(well)
        np.testing.assert_allclose(mesh.points[:, 0], 300.0)
        np.testing.assert_allclose(mesh.points[:, 1], 150.0)

    def test_vertical_well_z_nonpositive(self):
        well = _well()
        mesh = build_well_track(well)
        assert np.all(mesh.points[:, 2] <= 0.0)

    def test_deviated_well_x_changes(self):
        well = _deviated_well()
        mesh = build_well_track(well)
        # x at start is 0, x at end should be > 0 (kicking east)
        assert mesh.points[-1, 0] > mesh.points[0, 0]

    def test_two_station_minimum(self):
        dev = DeviationSurvey.vertical(0.0, 0.0, td=100.0)
        well = Well("W", 0.0, 0.0, deviation=dev)
        mesh = build_well_track(well)
        assert mesh is not None
        assert mesh.n_points == 2

    def test_z_equals_neg_tvd(self):
        dev = DeviationSurvey.vertical(0.0, 0.0, td=1000.0)
        well = Well("W", 0.0, 0.0, deviation=dev)
        mesh = build_well_track(well)
        # TVD at first station = 0, so z[0] = 0
        assert pytest.approx(mesh.points[0, 2]) == 0.0
        # TVD at last station = 1000, so z[-1] = -1000
        assert pytest.approx(mesh.points[-1, 2]) == -1000.0


class TestBuildHorizonPick3D:
    def test_returns_polydata(self):
        sec = _sec(nodes=[(0.0, 0.0), (1000.0, 0.0)])
        pick = _pick()
        mesh = build_horizon_pick_3d(pick, sec)
        assert isinstance(mesh, pv.PolyData)

    def test_z_values_are_negated_depths(self):
        # Pick at constant depth 500m → z should all be -500
        sec = _sec(nodes=[(0.0, 0.0), (1000.0, 0.0)])
        pick = HorizonPick([0.0, 1000.0], [500.0, 500.0])
        mesh = build_horizon_pick_3d(pick, sec)
        np.testing.assert_allclose(mesh.points[:, 2], -500.0, atol=1e-6)

    def test_x_matches_section_to_map(self):
        sec = _sec(nodes=[(0.0, 0.0), (1000.0, 0.0)])
        pick = HorizonPick([0.0, 1000.0], [500.0, 500.0])
        mesh = build_horizon_pick_3d(pick, sec)
        # All y-coords should be 0 (section goes east along y=0)
        np.testing.assert_allclose(mesh.points[:, 1], 0.0, atol=1e-9)

    def test_nan_depths_excluded(self):
        sec = _sec(nodes=[(0.0, 0.0), (1000.0, 0.0)])
        # Pick only in [200, 800] — outside that it's nan
        pick = HorizonPick([200.0, 800.0], [500.0, 600.0])
        mesh = build_horizon_pick_3d(pick, sec)
        # All z should be valid (no nan)
        assert not np.any(np.isnan(mesh.points))

    def test_returns_none_when_all_nan(self):
        sec = _sec(nodes=[(0.0, 0.0), (1000.0, 0.0)])
        # Pick entirely outside section range
        pick = HorizonPick([2000.0, 3000.0], [500.0, 600.0])
        mesh = build_horizon_pick_3d(pick, sec)
        assert mesh is None

    def test_point_count_reasonable(self):
        sec = _sec(nodes=[(0.0, 0.0), (1000.0, 0.0)])
        pick = _pick()
        mesh = build_horizon_pick_3d(pick, sec)
        # 200 sample points, all valid → n_points should be ~200
        assert mesh.n_points > 100

    def test_dogleg_section(self):
        sec = _sec(nodes=[(0.0, 0.0), (500.0, 0.0), (500.0, 500.0)])
        pick = _pick()
        mesh = build_horizon_pick_3d(pick, sec)
        assert mesh is not None
        assert not np.any(np.isnan(mesh.points))


# ===========================================================================
# Viewer3D widget tests (require QApplication)
# ===========================================================================

class TestViewer3DConstruction:
    def test_is_qwidget(self, viewer):
        assert viewer.isWidgetType()

    def test_plotter_starts_uninitialized(self, viewer):
        # Plotter is lazy — None until user clicks Enable
        assert viewer.plotter is None

    def test_has_enable_button(self, viewer):
        from PySide6.QtWidgets import QPushButton
        assert isinstance(viewer._enable_btn, QPushButton)

    def test_has_placeholder_label(self, viewer):
        from PySide6.QtWidgets import QLabel
        assert isinstance(viewer._placeholder, QLabel)


class TestViewer3DRenderNoCrash:
    def test_render_empty(self, viewer):
        viewer.render()

    def test_render_with_section(self, viewer, state):
        state.add_section(_sec())
        viewer.render()

    def test_render_with_active_section(self, viewer, state):
        sec = _sec()
        state.add_section(sec)
        state.set_active_section(sec)
        viewer.render()

    def test_render_with_surface_grid(self, viewer, state):
        state.add_surface(_grid_surf())
        viewer.render()

    def test_render_with_surface_scattered(self, viewer, state):
        state.add_surface(_scatter_surf())
        viewer.render()

    def test_render_with_well(self, viewer, state):
        state.add_well(_well())
        viewer.render()

    def test_render_with_deviated_well(self, viewer, state):
        state.add_well(_deviated_well())
        viewer.render()

    def test_render_with_well_and_tops(self, viewer, state):
        well = _well()
        well.add_formation_top("TopA", 500.0)
        well.add_formation_top("TopB", 1200.0)
        state.add_well(well)
        viewer.render()

    def test_render_with_horizon_pick_and_section(self, viewer, state):
        sec = _sec()
        state.add_section(sec)
        state.set_active_section(sec)
        state.add_horizon_pick(_pick())
        viewer.render()

    def test_render_pick_without_active_section(self, viewer, state):
        # Should not crash — just skips picks
        state.add_horizon_pick(_pick())
        viewer.render()

    def test_render_all_types(self, viewer, state):
        sec = _sec()
        state.add_section(sec)
        state.set_active_section(sec)
        state.add_surface(_grid_surf())
        state.add_well(_well())
        state.add_horizon_pick(_pick())
        viewer.render()

    def test_render_twice(self, viewer, state):
        state.add_section(_sec())
        viewer.render()
        viewer.render()

    def test_render_dogleg_section(self, viewer, state):
        sec = _sec(nodes=[(0, 0), (500, 0), (500, 500)])
        state.add_section(sec)
        viewer.render()


class TestViewer3DComputeMaxDepth:
    def test_default_when_no_data(self, viewer):
        from cross_section_tool.views.viewer_3d import _DEFAULT_DEPTH
        d = viewer._compute_max_depth()
        assert pytest.approx(d) == _DEFAULT_DEPTH

    def test_uses_well_tvd(self, viewer, state):
        dev = DeviationSurvey.vertical(0.0, 0.0, td=8000.0)
        well = Well("W", 0.0, 0.0, deviation=dev)
        state.add_well(well)
        assert pytest.approx(viewer._compute_max_depth()) == 8000.0

    def test_uses_surface_depth(self, viewer, state):
        surf = Surface([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [6000.0, 6000.0, 6000.0])
        state.add_surface(surf)
        assert pytest.approx(viewer._compute_max_depth()) == 6000.0

    def test_uses_pick_depth(self, viewer, state):
        pick = HorizonPick([0.0, 1000.0], [7500.0, 7500.0])
        state.add_horizon_pick(pick)
        assert pytest.approx(viewer._compute_max_depth()) == 7500.0

    def test_takes_maximum_across_all_sources(self, viewer, state):
        dev = DeviationSurvey.vertical(0.0, 0.0, td=3000.0)
        state.add_well(Well("W", 0.0, 0.0, deviation=dev))
        # Surface depth 8000 > default 5000 → must win
        surf = Surface([0.0, 500.0, 1000.0, 500.0], [0.0, 500.0, 0.0, -500.0],
                       [8000.0, 8000.0, 8000.0, 8000.0])
        state.add_surface(surf)
        pick = HorizonPick([0.0, 1.0], [2000.0, 2000.0])
        state.add_horizon_pick(pick)
        assert pytest.approx(viewer._compute_max_depth()) == 8000.0


class TestViewer3DAutoRender:
    def test_section_added_triggers_render(self, viewer, state):
        # Just verify it doesn't crash
        state.add_section(_sec())

    def test_section_removed_triggers_render(self, viewer, state):
        sec = _sec()
        state.add_section(sec)
        state.remove_section(sec)

    def test_surface_added_triggers_render(self, viewer, state):
        state.add_surface(_grid_surf())

    def test_surface_removed_triggers_render(self, viewer, state):
        surf = _grid_surf()
        state.add_surface(surf)
        state.remove_surface(surf)

    def test_well_added_triggers_render(self, viewer, state):
        state.add_well(_well())

    def test_well_removed_triggers_render(self, viewer, state):
        well = _well()
        state.add_well(well)
        state.remove_well(well)

    def test_pick_added_triggers_render(self, viewer, state):
        state.add_horizon_pick(_pick())

    def test_project_changed_triggers_render(self, viewer, state):
        state.add_section(_sec())
        state.new_project()

    def test_active_section_change_triggers_render(self, viewer, state):
        sec = _sec()
        state.add_section(sec)
        state.set_active_section(sec)
        state.set_active_section(None)
