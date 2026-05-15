"""Tests for section_tool.core.surfaces — Surface and HorizonPick."""

import math

import numpy as np
import pytest

from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick, Surface


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def tilted_plane_scattered(nx: int = 10, ny: int = 10) -> Surface:
    """z = 2*x + 3*y on a 10x10 scattered point cloud over [0,100]x[0,100]."""
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, nx * ny)
    y = rng.uniform(0, 100, nx * ny)
    z = 2.0 * x + 3.0 * y
    return Surface(x, y, z, name="plane_scattered")


def tilted_plane_grid(nx: int = 11, ny: int = 11) -> Surface:
    """z = 2*x + 3*y on a regular [0,100]x[0,100] grid."""
    x_coords = np.linspace(0.0, 100.0, nx)
    y_coords = np.linspace(0.0, 100.0, ny)
    xx, yy = np.meshgrid(x_coords, y_coords)
    z_grid = 2.0 * xx + 3.0 * yy
    return Surface.from_grid(x_coords, y_coords, z_grid, name="plane_grid")


def east_section(length: float = 100.0) -> Section:
    """Section going east from origin."""
    return Section([(0.0, 0.0), (length, 0.0)])


def north_section(length: float = 100.0) -> Section:
    """Section going north from origin."""
    return Section([(0.0, 0.0), (0.0, length)])


# ---------------------------------------------------------------------------
# Surface — construction
# ---------------------------------------------------------------------------

class TestSurfaceConstruction:
    def test_basic_construction(self):
        surf = Surface([0, 1, 2], [0, 0, 0], [10, 20, 30])
        assert surf.n_picks if hasattr(surf, 'n_picks') else True  # just alive
        assert repr(surf)  # doesn't crash

    def test_too_few_points(self):
        with pytest.raises(ValueError):
            Surface([0, 1], [0, 1], [0, 1])

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            Surface([0, 1, 2], [0, 1], [0, 1, 2])

    def test_metadata_defaults(self):
        surf = Surface([0, 1, 2], [0, 1, 2], [0, 1, 2])
        assert surf.kind == "horizon"
        assert surf.z_units == "m"

    def test_metadata_custom(self):
        surf = Surface(
            [0, 1, 2], [0, 1, 2], [0, 1, 2],
            name="Top Cretaceous",
            kind="unconformity",
            z_units="ft",
            crs_epsg=4326,
        )
        assert surf.name == "Top Cretaceous"
        assert surf.kind == "unconformity"
        assert surf.z_units == "ft"
        assert surf.crs_epsg == 4326

    def test_from_grid_shape_mismatch(self):
        with pytest.raises(ValueError):
            Surface.from_grid([0, 1, 2], [0, 1], np.zeros((3, 3)))

    def test_from_grid_correct_shape(self):
        z = np.zeros((2, 3))
        surf = Surface.from_grid([0, 1, 2], [0, 1], z)
        assert surf._is_grid

    def test_repr_scattered(self):
        surf = tilted_plane_scattered()
        assert "scattered" in repr(surf)

    def test_repr_grid(self):
        surf = tilted_plane_grid()
        assert "grid" in repr(surf)


# ---------------------------------------------------------------------------
# Surface — extent
# ---------------------------------------------------------------------------

class TestSurfaceExtent:
    def test_extent_matches_data(self):
        surf = Surface([10, 20, 30], [5, 15, 25], [0, 0, 0])
        xmin, xmax, ymin, ymax = surf.extent()
        assert xmin == 10.0
        assert xmax == 30.0
        assert ymin == 5.0
        assert ymax == 25.0

    def test_extent_grid(self):
        surf = tilted_plane_grid()
        xmin, xmax, ymin, ymax = surf.extent()
        assert pytest.approx(xmin) == 0.0
        assert pytest.approx(xmax) == 100.0
        assert pytest.approx(ymin) == 0.0
        assert pytest.approx(ymax) == 100.0


# ---------------------------------------------------------------------------
# Surface — sample (grid)
# ---------------------------------------------------------------------------

class TestSurfaceSampleGrid:
    def test_sample_at_grid_node(self):
        surf = tilted_plane_grid()
        # z = 2*x + 3*y at (0, 0) = 0
        assert pytest.approx(surf.sample(0.0, 0.0)) == 0.0

    def test_sample_at_far_corner(self):
        surf = tilted_plane_grid()
        # z = 2*100 + 3*100 = 500
        assert pytest.approx(surf.sample(100.0, 100.0)) == 500.0

    def test_sample_interpolated_interior(self):
        surf = tilted_plane_grid()
        # Tilted plane → interpolation is exact for linear surfaces
        assert pytest.approx(surf.sample(50.0, 0.0), rel=1e-6) == 100.0
        assert pytest.approx(surf.sample(0.0, 50.0), rel=1e-6) == 150.0
        assert pytest.approx(surf.sample(30.0, 40.0), rel=1e-6) == 180.0

    def test_sample_outside_bounds_is_nan(self):
        surf = tilted_plane_grid()
        assert math.isnan(surf.sample(-1.0, 50.0))
        assert math.isnan(surf.sample(50.0, 200.0))

    def test_sample_many_grid(self):
        surf = tilted_plane_grid()
        xs = np.array([0.0, 50.0, 100.0])
        ys = np.array([0.0, 0.0, 0.0])
        zs = surf.sample_many(xs, ys)
        assert zs.shape == (3,)
        assert pytest.approx(zs[0]) == 0.0
        assert pytest.approx(zs[1], rel=1e-6) == 100.0
        assert pytest.approx(zs[2]) == 200.0

    def test_sample_many_outside_is_nan(self):
        surf = tilted_plane_grid()
        xs = np.array([50.0, -10.0])
        ys = np.array([50.0, 50.0])
        zs = surf.sample_many(xs, ys)
        assert not math.isnan(zs[0])
        assert math.isnan(zs[1])


# ---------------------------------------------------------------------------
# Surface — sample (scattered)
# ---------------------------------------------------------------------------

class TestSurfaceSampleScattered:
    def test_sample_at_exact_point(self):
        # Three-point scattered surface: easy to reason about
        surf = Surface([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 2.0, 3.0])
        # At origin
        assert pytest.approx(surf.sample(0.0, 0.0), abs=1e-6) == 0.0

    def test_sample_interior_interpolated(self):
        surf = tilted_plane_scattered()
        # z = 2*x + 3*y; interior point — within convex hull, near data
        z = surf.sample(45.0, 45.0)
        assert pytest.approx(z, rel=0.05) == 2 * 45 + 3 * 45  # allow 5% error

    def test_sample_outside_convex_hull_is_nan(self):
        # Tiny triangle; far-away point should be nan
        surf = Surface([0.0, 1.0, 0.5], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
        assert math.isnan(surf.sample(100.0, 100.0))


# ---------------------------------------------------------------------------
# Surface — profile_along_section
# ---------------------------------------------------------------------------

class TestSurfaceProfile:
    def test_profile_shape(self):
        surf = tilted_plane_grid()
        sec = east_section(100.0)
        distances, z_values = surf.profile_along_section(sec, n_samples=50)
        assert distances.shape == (50,)
        assert z_values.shape == (50,)

    def test_profile_start_end(self):
        surf = tilted_plane_grid()
        sec = east_section(100.0)
        distances, z_values = surf.profile_along_section(sec, n_samples=10)
        assert pytest.approx(distances[0]) == 0.0
        assert pytest.approx(distances[-1]) == 100.0

    def test_profile_linear_east(self):
        """Along an east section (y=0), z = 2*x. Distance == x for east section."""
        surf = tilted_plane_grid()
        sec = east_section(100.0)
        distances, z_values = surf.profile_along_section(sec, n_samples=11)
        expected = 2.0 * distances  # z = 2*x + 3*0
        np.testing.assert_allclose(z_values, expected, rtol=1e-6)

    def test_profile_linear_north(self):
        """Along a north section (x=0), z = 3*y. Distance == y for north section."""
        surf = tilted_plane_grid()
        sec = north_section(100.0)
        distances, z_values = surf.profile_along_section(sec, n_samples=11)
        expected = 3.0 * distances  # z = 2*0 + 3*y
        np.testing.assert_allclose(z_values, expected, rtol=1e-6)

    def test_profile_dogleg(self):
        """Dogleg section: east then north. Profile should kink at bend."""
        surf = tilted_plane_grid()
        sec = Section([(0.0, 0.0), (50.0, 0.0), (50.0, 50.0)])
        distances, z_values = surf.profile_along_section(sec, n_samples=101)
        # At distance 50 (bend): (x=50, y=0) → z = 100
        idx_bend = 50
        assert pytest.approx(distances[idx_bend]) == 50.0
        assert pytest.approx(z_values[idx_bend], rel=1e-5) == 100.0

    def test_profile_n_samples_too_small(self):
        surf = tilted_plane_grid()
        sec = east_section()
        with pytest.raises(ValueError):
            surf.profile_along_section(sec, n_samples=1)


# ---------------------------------------------------------------------------
# HorizonPick — construction
# ---------------------------------------------------------------------------

class TestHorizonPickConstruction:
    def test_basic(self):
        hp = HorizonPick([0.0, 100.0, 200.0], [1000.0, 1100.0, 1200.0])
        assert hp.n_picks == 3

    def test_auto_sort(self):
        hp = HorizonPick([200.0, 0.0, 100.0], [1200.0, 1000.0, 1100.0])
        assert list(hp.distances) == [0.0, 100.0, 200.0]
        assert list(hp.depths) == [1000.0, 1100.0, 1200.0]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            HorizonPick([], [])

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            HorizonPick([0.0, 100.0], [1000.0])

    def test_single_pick_allowed(self):
        hp = HorizonPick([500.0], [2500.0])
        assert hp.n_picks == 1

    def test_metadata_defaults(self):
        hp = HorizonPick([0.0, 1.0], [0.0, 1.0])
        assert hp.z_units == "m"
        assert hp.color.startswith("#")

    def test_metadata_custom(self):
        hp = HorizonPick(
            [0.0, 1.0], [0.0, 1.0], name="Top Sand", z_units="ft", color="#ff0000"
        )
        assert hp.name == "Top Sand"
        assert hp.z_units == "ft"
        assert hp.color == "#ff0000"

    def test_distances_are_copies(self):
        d = np.array([0.0, 100.0])
        hp = HorizonPick(d, [10.0, 20.0])
        d[0] = 999.0
        assert hp.distances[0] == 0.0

    def test_repr(self):
        hp = HorizonPick([0.0, 1000.0], [500.0, 600.0], name="Horizon A")
        assert "Horizon A" in repr(hp)


# ---------------------------------------------------------------------------
# HorizonPick — sampling
# ---------------------------------------------------------------------------

class TestHorizonPickSample:
    def test_sample_at_first_pick(self):
        hp = HorizonPick([0.0, 500.0, 1000.0], [100.0, 200.0, 300.0])
        assert pytest.approx(hp.sample(0.0)) == 100.0

    def test_sample_at_last_pick(self):
        hp = HorizonPick([0.0, 500.0, 1000.0], [100.0, 200.0, 300.0])
        assert pytest.approx(hp.sample(1000.0)) == 300.0

    def test_sample_interpolated(self):
        hp = HorizonPick([0.0, 1000.0], [100.0, 300.0])
        assert pytest.approx(hp.sample(500.0)) == 200.0

    def test_sample_quarter(self):
        hp = HorizonPick([0.0, 1000.0], [100.0, 500.0])
        assert pytest.approx(hp.sample(250.0)) == 200.0

    def test_sample_before_range_is_nan(self):
        hp = HorizonPick([100.0, 500.0], [10.0, 50.0])
        assert math.isnan(hp.sample(0.0))

    def test_sample_after_range_is_nan(self):
        hp = HorizonPick([0.0, 500.0], [10.0, 50.0])
        assert math.isnan(hp.sample(600.0))

    def test_sample_single_pick_at_exact(self):
        hp = HorizonPick([500.0], [2500.0])
        assert pytest.approx(hp.sample(500.0)) == 2500.0

    def test_sample_single_pick_off_is_nan(self):
        hp = HorizonPick([500.0], [2500.0])
        assert math.isnan(hp.sample(501.0))

    def test_sample_many(self):
        hp = HorizonPick([0.0, 1000.0], [0.0, 1000.0])
        zs = hp.sample_many(np.array([0.0, 250.0, 500.0, 750.0, 1000.0]))
        expected = np.array([0.0, 250.0, 500.0, 750.0, 1000.0])
        np.testing.assert_allclose(zs, expected)

    def test_sample_many_nan_outside(self):
        hp = HorizonPick([100.0, 900.0], [10.0, 90.0])
        zs = hp.sample_many(np.array([0.0, 500.0, 1000.0]))
        assert math.isnan(zs[0])
        assert not math.isnan(zs[1])
        assert math.isnan(zs[2])


# ---------------------------------------------------------------------------
# HorizonPick — pick operations
# ---------------------------------------------------------------------------

class TestHorizonPickOperations:
    def test_insert_pick_maintains_order(self):
        hp = HorizonPick([0.0, 1000.0], [100.0, 300.0])
        hp.insert_pick(500.0, 200.0)
        assert hp.n_picks == 3
        assert list(hp.distances) == [0.0, 500.0, 1000.0]
        assert list(hp.depths) == [100.0, 200.0, 300.0]

    def test_insert_pick_at_start(self):
        hp = HorizonPick([500.0, 1000.0], [50.0, 100.0])
        hp.insert_pick(0.0, 0.0)
        assert hp.distances[0] == 0.0

    def test_insert_pick_at_end(self):
        hp = HorizonPick([0.0, 500.0], [0.0, 50.0])
        hp.insert_pick(1000.0, 100.0)
        assert hp.distances[-1] == 1000.0

    def test_insert_makes_sample_continuous(self):
        hp = HorizonPick([0.0, 1000.0], [100.0, 300.0])
        hp.insert_pick(500.0, 200.0)
        assert pytest.approx(hp.sample(750.0)) == 250.0

    def test_delete_pick(self):
        hp = HorizonPick([0.0, 500.0, 1000.0], [100.0, 200.0, 300.0])
        hp.delete_pick(1)
        assert hp.n_picks == 2
        assert list(hp.distances) == [0.0, 1000.0]

    def test_delete_first_pick(self):
        hp = HorizonPick([0.0, 500.0, 1000.0], [100.0, 200.0, 300.0])
        hp.delete_pick(0)
        assert hp.distances[0] == 500.0

    def test_delete_refuses_last_pick(self):
        hp = HorizonPick([500.0, 1000.0], [10.0, 20.0])
        hp.delete_pick(0)
        assert hp.n_picks == 1
        with pytest.raises(ValueError):
            hp.delete_pick(0)

    def test_delete_out_of_range(self):
        hp = HorizonPick([0.0, 1000.0], [0.0, 100.0])
        with pytest.raises(IndexError):
            hp.delete_pick(99)

    def test_move_pick_same_position(self):
        hp = HorizonPick([0.0, 500.0, 1000.0], [100.0, 200.0, 300.0])
        hp.move_pick(1, 500.0, 250.0)
        assert hp.n_picks == 3
        assert pytest.approx(hp.sample(500.0)) == 250.0

    def test_move_pick_reorders(self):
        hp = HorizonPick([0.0, 500.0, 1000.0], [100.0, 200.0, 300.0])
        # Move index 1 (at 500) to 750
        hp.move_pick(1, 750.0, 225.0)
        assert hp.n_picks == 3
        assert list(hp.distances) == [0.0, 750.0, 1000.0]

    def test_move_pick_out_of_range(self):
        hp = HorizonPick([0.0, 1000.0], [0.0, 100.0])
        with pytest.raises(IndexError):
            hp.move_pick(99, 0.0, 0.0)


# ---------------------------------------------------------------------------
# HorizonPick — coordinate transforms
# ---------------------------------------------------------------------------

class TestHorizonPickCoords:
    def test_to_map_coords_straight_section(self):
        sec = east_section(1000.0)
        hp = HorizonPick([0.0, 500.0, 1000.0], [100.0, 200.0, 300.0])
        pts = hp.to_map_coords(sec)
        assert len(pts) == 3
        # At distance 0: map = (0, 0)
        x, y, z = pts[0]
        assert pytest.approx(x) == 0.0
        assert pytest.approx(y) == 0.0
        assert pytest.approx(z) == 100.0
        # At distance 500: map = (500, 0)
        x, y, z = pts[1]
        assert pytest.approx(x) == 500.0
        assert pytest.approx(z) == 200.0

    def test_to_map_coords_dogleg(self):
        sec = Section([(0.0, 0.0), (1000.0, 0.0), (1000.0, 1000.0)])
        hp = HorizonPick([0.0, 1000.0, 2000.0], [50.0, 150.0, 250.0])
        pts = hp.to_map_coords(sec)
        # At distance 1000 (bend node): map = (1000, 0)
        x, y, z = pts[1]
        assert pytest.approx(x) == 1000.0
        assert pytest.approx(y) == 0.0
        # At distance 2000 (end): map = (1000, 1000)
        x, y, z = pts[2]
        assert pytest.approx(x) == 1000.0
        assert pytest.approx(y) == 1000.0

    def test_snap_to_surface(self):
        """snap_to_surface should replace depths with sampled surface z-values."""
        surf = tilted_plane_grid()  # z = 2*x + 3*y
        sec = east_section(100.0)   # y=0 along section, so z = 2*x = 2*distance
        hp = HorizonPick([0.0, 50.0, 100.0], [999.0, 999.0, 999.0])
        snapped = hp.snap_to_surface(surf, sec)
        assert snapped.n_picks == 3
        assert pytest.approx(snapped.depths[0], rel=1e-5) == 0.0    # 2*0
        assert pytest.approx(snapped.depths[1], rel=1e-5) == 100.0  # 2*50
        assert pytest.approx(snapped.depths[2], rel=1e-5) == 200.0  # 2*100

    def test_snap_preserves_metadata(self):
        surf = tilted_plane_grid()
        sec = east_section(100.0)
        hp = HorizonPick([0.0, 100.0], [0.0, 0.0], name="MyHorizon", color="#aabbcc")
        snapped = hp.snap_to_surface(surf, sec)
        assert snapped.name == "MyHorizon"
        assert snapped.color == "#aabbcc"

    def test_snap_does_not_mutate_original(self):
        surf = tilted_plane_grid()
        sec = east_section(100.0)
        hp = HorizonPick([0.0, 100.0], [999.0, 999.0])
        _ = hp.snap_to_surface(surf, sec)
        assert pytest.approx(hp.depths[0]) == 999.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_surface_single_column_grid(self):
        """A 1-column grid (nx=1) should still work for vertical profiles."""
        z = np.array([[0.0], [10.0], [20.0]])
        surf = Surface.from_grid([5.0], [0.0, 50.0, 100.0], z)
        assert pytest.approx(surf.sample(5.0, 50.0)) == 10.0

    def test_horizon_pick_two_identical_distances(self):
        """Duplicate distances are allowed; sort is stable."""
        hp = HorizonPick([100.0, 100.0], [10.0, 20.0])
        assert hp.n_picks == 2
        assert hp.distances[0] == 100.0

    def test_profile_along_section_length_1_sample(self):
        surf = tilted_plane_grid()
        sec = east_section(100.0)
        with pytest.raises(ValueError):
            surf.profile_along_section(sec, n_samples=1)

    def test_profile_two_samples_boundary(self):
        surf = tilted_plane_grid()
        sec = east_section(100.0)
        distances, zs = surf.profile_along_section(sec, n_samples=2)
        assert pytest.approx(distances[0]) == 0.0
        assert pytest.approx(distances[1]) == 100.0

    def test_sample_many_empty_arrays(self):
        surf = tilted_plane_grid()
        zs = surf.sample_many(np.array([]), np.array([]))
        assert len(zs) == 0

    def test_horizon_pick_sample_many_empty(self):
        hp = HorizonPick([0.0, 1000.0], [0.0, 100.0])
        zs = hp.sample_many(np.array([]))
        assert len(zs) == 0
