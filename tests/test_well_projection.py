"""Tests for well-section projection, elevation helpers, and large-coord handling."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from section_tool.core.section import (
    Section,
    WellSectionProjection,
    depth_to_elevation,
    elevation_to_depth,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ew_section(cx: float = 606554.0, cy: float = 6080126.0,
                    half: float = 5_000.0, name: str = "EW") -> Section:
    """E-W section 10 km long centred on (cx, cy)."""
    return Section([(cx - half, cy), (cx + half, cy)], name=name)


def make_well(name: str = "W", x: float = 606554.0, y: float = 6080126.0):
    return SimpleNamespace(name=name, x=x, y=y)


# ---------------------------------------------------------------------------
# 1. Elevation helpers
# ---------------------------------------------------------------------------

class TestElevationHelpers:
    def test_depth_to_elevation_positive(self):
        assert depth_to_elevation(1000.0) == pytest.approx(-1000.0)

    def test_depth_to_elevation_zero(self):
        assert depth_to_elevation(0.0) == pytest.approx(0.0)

    def test_depth_to_elevation_negative_depth(self):
        # Negative depth = above datum → positive elevation
        assert depth_to_elevation(-50.0) == pytest.approx(50.0)

    def test_elevation_to_depth_negative_elev(self):
        assert elevation_to_depth(-1000.0) == pytest.approx(1000.0)

    def test_elevation_to_depth_positive_elev(self):
        assert elevation_to_depth(50.0) == pytest.approx(-50.0)

    @pytest.mark.parametrize("d", [0.0, 100.0, 3150.0, 5000.0])
    def test_round_trip(self, d):
        assert elevation_to_depth(depth_to_elevation(d)) == pytest.approx(d)

    def test_module_level_import(self):
        """Both helpers are importable from section module."""
        from section_tool.core.section import depth_to_elevation, elevation_to_depth
        assert depth_to_elevation(1.0) == -1.0
        assert elevation_to_depth(-1.0) == 1.0


# ---------------------------------------------------------------------------
# 2. project_point — E-W section through F02-01
# ---------------------------------------------------------------------------

class TestProjectPointEW:
    """All tests use the 10 km E-W section at UTM ~(601554, 6080126)–(611554, 6080126)."""

    @pytest.fixture
    def section(self):
        return make_ew_section()

    def test_well_on_section_midpoint(self, section):
        dist, perp = section.project_point(606554.0, 6080126.0)
        assert dist == pytest.approx(5000.0, abs=1e-6)
        assert perp == pytest.approx(0.0, abs=1e-6)

    def test_well_at_start_node(self, section):
        dist, perp = section.project_point(601554.0, 6080126.0)
        assert dist == pytest.approx(0.0, abs=1e-6)
        assert perp == pytest.approx(0.0, abs=1e-6)

    def test_well_at_end_node(self, section):
        dist, perp = section.project_point(611554.0, 6080126.0)
        assert dist == pytest.approx(10000.0, abs=1e-6)
        assert perp == pytest.approx(0.0, abs=1e-6)

    def test_well_500m_north(self, section):
        dist, perp = section.project_point(606554.0, 6080626.0)
        assert dist == pytest.approx(5000.0, abs=1e-6)
        assert perp == pytest.approx(500.0, abs=1e-6)   # north = left of E-W = positive

    def test_well_500m_south(self, section):
        dist, perp = section.project_point(606554.0, 6079626.0)
        assert dist == pytest.approx(5000.0, abs=1e-6)
        assert perp == pytest.approx(-500.0, abs=1e-6)  # south = right = negative

    def test_well_past_east_end(self, section):
        """Distance along > section total_length (10 000 m)."""
        dist, perp = section.project_point(615000.0, 6080126.0)
        assert dist > section.total_length()
        assert perp == pytest.approx(0.0, abs=1e-6)
        # 615000 - 601554 (section start) = 13446 m from node 0
        assert dist == pytest.approx(13446.0, abs=1e-3)

    def test_well_before_west_start(self, section):
        """Distance along < 0 (before section start)."""
        dist, perp = section.project_point(598000.0, 6080126.0)
        assert dist < 0.0
        assert perp == pytest.approx(0.0, abs=1e-6)

    def test_large_utm_coordinates_preserved(self, section):
        """Projection of a point near but off the section gives sane numbers."""
        dist, perp = section.project_point(606554.0 + 100, 6080126.0 + 200)
        assert math.isfinite(dist)
        assert math.isfinite(perp)
        assert abs(perp) == pytest.approx(200.0, abs=1.0)

    def test_distance_and_perp_independent(self, section):
        """Offset in Y must not contaminate distance_along."""
        d0, _ = section.project_point(606554.0, 6080126.0)
        d1, _ = section.project_point(606554.0, 6080126.0 + 999.0)
        assert d0 == pytest.approx(d1, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. project_point — N-S section
# ---------------------------------------------------------------------------

class TestProjectPointNS:
    @pytest.fixture
    def section(self):
        # N-S section 10 km from cy-5000 to cy+5000
        cy = 6080126.0
        cx = 606554.0
        return Section([(cx, cy - 5000), (cx, cy + 5000)], name="NS")

    def test_well_on_midpoint(self, section):
        dist, perp = section.project_point(606554.0, 6080126.0)
        assert dist == pytest.approx(5000.0, abs=1e-6)
        assert perp == pytest.approx(0.0, abs=1e-6)

    def test_well_east_of_section(self, section):
        dist, perp = section.project_point(606554.0 + 300, 6080126.0)
        assert dist == pytest.approx(5000.0, abs=1e-6)
        # East = right of northward travel = negative
        assert perp == pytest.approx(-300.0, abs=1e-6)

    def test_well_west_of_section(self, section):
        dist, perp = section.project_point(606554.0 - 300, 6080126.0)
        assert dist == pytest.approx(5000.0, abs=1e-6)
        # West = left of northward travel = positive
        assert perp == pytest.approx(300.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. project_point — dogleg section
# ---------------------------------------------------------------------------

class TestProjectPointDogleg:
    @pytest.fixture
    def section(self):
        # L-shaped section: (0,0)→(5000,0)→(5000,5000)
        return Section([(0, 0), (5000, 0), (5000, 5000)], name="L")

    def test_point_on_first_segment(self, section):
        dist, perp = section.project_point(2500.0, 0.0)
        assert dist == pytest.approx(2500.0, abs=1e-6)
        assert perp == pytest.approx(0.0, abs=1e-6)

    def test_point_on_second_segment(self, section):
        dist, perp = section.project_point(5000.0, 3000.0)
        assert dist == pytest.approx(5000.0 + 3000.0, abs=1e-6)
        assert perp == pytest.approx(0.0, abs=1e-6)

    def test_point_at_corner(self, section):
        dist, perp = section.project_point(5000.0, 0.0)
        assert dist == pytest.approx(5000.0, abs=1e-6)
        assert perp == pytest.approx(0.0, abs=1e-6)

    def test_point_past_last_node(self, section):
        dist, perp = section.project_point(5000.0, 7000.0)
        total = section.total_length()  # 5000 + 5000 = 10000
        assert dist > total
        assert dist == pytest.approx(12000.0, abs=1e-6)

    def test_point_projects_to_nearest_segment(self, section):
        """A point equidistant between extended segments picks the nearest segment."""
        # (2500, 100): 100m north of segment 1 mid-point
        dist, perp = section.project_point(2500.0, 100.0)
        assert dist == pytest.approx(2500.0, abs=1e-6)
        assert perp == pytest.approx(100.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 5. WellSectionProjection — tiers
# ---------------------------------------------------------------------------

class TestWellSectionProjectionTiers:
    @pytest.fixture
    def section(self):
        return make_ew_section()

    def test_well_on_plane(self, section):
        w = make_well(x=606554.0, y=6080126.0)        # exactly on section
        proj = WellSectionProjection.compute(w, section)
        assert proj.display_tier == "on_plane"
        assert proj.perpendicular_offset == pytest.approx(0.0, abs=1e-6)

    def test_well_50m_off_is_on_plane(self, section):
        w = make_well(x=606554.0, y=6080126.0 + 50)   # 50m < 100m threshold
        proj = WellSectionProjection.compute(w, section)
        assert proj.display_tier == "on_plane"

    def test_well_99m_off_is_on_plane(self, section):
        w = make_well(x=606554.0, y=6080126.0 + 99)
        proj = WellSectionProjection.compute(w, section)
        assert proj.display_tier == "on_plane"

    def test_well_500m_off_is_near(self, section):
        w = make_well(x=606554.0, y=6080126.0 + 500)  # 500m < 2000m tolerance
        proj = WellSectionProjection.compute(w, section)
        assert proj.display_tier == "near"
        assert abs(proj.perpendicular_offset) == pytest.approx(500.0, abs=1e-6)

    def test_well_1999m_off_is_near(self, section):
        w = make_well(x=606554.0, y=6080126.0 + 1999)
        proj = WellSectionProjection.compute(w, section)
        assert proj.display_tier == "near"

    def test_well_3000m_off_is_far(self, section):
        w = make_well(x=606554.0, y=6080126.0 + 3000)  # 3000m < 5000m (2000*2.5)
        proj = WellSectionProjection.compute(w, section)
        assert proj.display_tier == "far"

    def test_well_4999m_off_is_far(self, section):
        w = make_well(x=606554.0, y=6080126.0 + 4999)
        proj = WellSectionProjection.compute(w, section)
        assert proj.display_tier == "far"

    def test_well_10000m_off_is_hidden(self, section):
        w = make_well(x=606554.0, y=6080126.0 + 10_000)
        proj = WellSectionProjection.compute(w, section)
        assert proj.display_tier == "hidden"

    def test_projection_stores_names(self, section):
        w = make_well(name="F02-01", x=606554.0, y=6080126.0)
        proj = WellSectionProjection.compute(w, section)
        assert proj.well_name == "F02-01"
        assert proj.section_name == "EW"

    def test_projection_stores_distance(self, section):
        w = make_well(x=606554.0, y=6080126.0)
        proj = WellSectionProjection.compute(w, section)
        assert proj.distance_along == pytest.approx(5000.0, abs=1e-6)

    def test_custom_tolerance_changes_tiers(self, section):
        w = make_well(x=606554.0, y=6080126.0 + 1500)
        proj_default = WellSectionProjection.compute(w, section)             # tol=2000
        proj_tight   = WellSectionProjection.compute(w, section, tolerance=1000)
        assert proj_default.display_tier == "near"
        assert proj_tight.display_tier == "far"

    def test_negative_offset_same_tier(self, section):
        """Tier depends on |offset|, not sign."""
        w_north = make_well(x=606554.0, y=6080126.0 + 500)
        w_south = make_well(x=606554.0, y=6080126.0 - 500)
        p_n = WellSectionProjection.compute(w_north, section)
        p_s = WellSectionProjection.compute(w_south, section)
        assert p_n.display_tier == p_s.display_tier == "near"
        assert p_n.perpendicular_offset > 0
        assert p_s.perpendicular_offset < 0


# ---------------------------------------------------------------------------
# 6. Large UTM coordinates — numerical precision
# ---------------------------------------------------------------------------

class TestLargeUTMCoordinates:
    """Verify there is no precision loss with real-world UTM coordinates."""

    @pytest.fixture
    def section(self):
        return make_ew_section()  # centred near F02-01

    def test_midpoint_precision(self, section):
        dist, perp = section.project_point(606554.0, 6080126.0)
        assert dist == pytest.approx(5000.0, abs=1e-4)
        assert abs(perp) < 1e-4

    def test_small_offset_preserved(self, section):
        """A 1-metre offset should not be swamped by large base coordinates."""
        dist, perp = section.project_point(606554.0, 6080127.0)  # 1 m north
        assert abs(perp) == pytest.approx(1.0, abs=1e-4)

    def test_quarter_metre_offset(self, section):
        dist, perp = section.project_point(606554.0, 6080126.25)
        assert abs(perp) == pytest.approx(0.25, abs=1e-4)

    def test_section_length_preserved(self, section):
        assert section.total_length() == pytest.approx(10_000.0, abs=1e-6)

    def test_map_to_section_vs_project_point_interior(self, section):
        """For an interior point, map_to_section and project_point agree."""
        pt = (606554.0, 6080126.0 + 300.0)
        s1, p1 = section.map_to_section(*pt)
        s2, p2 = section.project_point(*pt)
        assert s1 == pytest.approx(s2, abs=1e-6)
        assert p1 == pytest.approx(p2, abs=1e-6)

    def test_map_to_section_clamps_past_end(self, section):
        """map_to_section clamps; project_point does not."""
        pt = (615000.0, 6080126.0)
        s_clamped, _ = section.map_to_section(*pt)
        s_unclamped, _ = section.project_point(*pt)
        assert s_clamped == pytest.approx(section.total_length(), abs=1e-6)
        assert s_unclamped > section.total_length()
