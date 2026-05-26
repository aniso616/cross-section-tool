"""Tests for section_tool.core.geometry."""
from __future__ import annotations

import pytest
import numpy as np

from section_tool.core.geometry import find_section_intersection, sample_pick_at_distance
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(nodes):
    return Section(nodes, name=f"sec_{id(nodes)}")


def _pick_with_data(section_name: str, distances, depths) -> HorizonPick:
    hp = HorizonPick.empty(name="H1")
    for d, z in zip(distances, depths):
        hp.insert_pick(float(d), float(z), section_name)
    return hp


# ---------------------------------------------------------------------------
# find_section_intersection
# ---------------------------------------------------------------------------

class TestFindSectionIntersection:
    def test_perpendicular_cross(self):
        """Two straight sections crossing at right angles."""
        # Horizontal: y=0, x from 0 to 100
        a = _section([[0, 0], [100, 0]])
        # Vertical: x=50, y from -50 to 50
        b = _section([[50, -50], [50, 50]])
        result = find_section_intersection(a, b)
        assert result is not None
        s_a, s_b = result
        assert abs(s_a - 50.0) < 1e-6
        assert abs(s_b - 50.0) < 1e-6

    def test_diagonal_cross(self):
        """Two diagonal sections crossing at their midpoints."""
        a = _section([[0, 0], [100, 100]])
        b = _section([[0, 100], [100, 0]])
        result = find_section_intersection(a, b)
        assert result is not None
        s_a, s_b = result
        # Both sections are 100*sqrt(2) long; intersection at midpoint
        half = 50.0 * np.sqrt(2)
        assert abs(s_a - half) < 1e-4
        assert abs(s_b - half) < 1e-4

    def test_parallel_sections_no_intersection(self):
        """Parallel sections do not intersect."""
        a = _section([[0, 0], [100, 0]])
        b = _section([[0, 10], [100, 10]])
        assert find_section_intersection(a, b) is None

    def test_non_overlapping_collinear(self):
        """Collinear but non-overlapping segments do not intersect."""
        a = _section([[0, 0], [40, 0]])
        b = _section([[60, 0], [100, 0]])
        assert find_section_intersection(a, b) is None

    def test_t_intersection(self):
        """One section ends exactly on the other (T-junction)."""
        a = _section([[0, 0], [100, 0]])
        b = _section([[50, -50], [50, 0]])
        result = find_section_intersection(a, b)
        assert result is not None
        s_a, s_b = result
        assert abs(s_a - 50.0) < 1e-6
        # b is 50 m long, intersection at the far end
        assert abs(s_b - 50.0) < 1e-6

    def test_polyline_sections(self):
        """Polyline (3-node) sections that cross midway."""
        a = _section([[0, 0], [50, 0], [100, 0]])    # straight but two segments
        b = _section([[50, -50], [50, 50]])
        result = find_section_intersection(a, b)
        assert result is not None
        s_a, s_b = result
        assert abs(s_a - 50.0) < 1e-6

    def test_same_section_no_self_intersection(self):
        """A section does not intersect with itself in this context."""
        a = _section([[0, 0], [100, 0]])
        b = _section([[0, 0], [100, 0]])
        # Identical, parallel — denom ≈ 0 for all pairs → no intersection
        # (segments are collinear; denom=0 path, skipped)
        assert find_section_intersection(a, b) is None


# ---------------------------------------------------------------------------
# sample_pick_at_distance
# ---------------------------------------------------------------------------

class TestSamplePickAtDistance:
    def test_exact_node(self):
        """Sample at an existing node distance returns its depth."""
        hp = _pick_with_data("sec", [0.0, 50.0, 100.0], [200.0, 300.0, 400.0])
        assert sample_pick_at_distance(hp, 50.0, "sec") == pytest.approx(300.0)

    def test_interpolated(self):
        """Sample between two nodes interpolates linearly."""
        hp = _pick_with_data("sec", [0.0, 100.0], [100.0, 200.0])
        result = sample_pick_at_distance(hp, 25.0, "sec")
        assert result == pytest.approx(125.0)

    def test_before_range_returns_none(self):
        hp = _pick_with_data("sec", [10.0, 50.0], [100.0, 200.0])
        assert sample_pick_at_distance(hp, 5.0, "sec") is None

    def test_after_range_returns_none(self):
        hp = _pick_with_data("sec", [10.0, 50.0], [100.0, 200.0])
        assert sample_pick_at_distance(hp, 60.0, "sec") is None

    def test_no_picks_on_section_returns_none(self):
        hp = _pick_with_data("other_sec", [0.0, 100.0], [100.0, 200.0])
        assert sample_pick_at_distance(hp, 50.0, "sec") is None

    def test_single_node_within_range(self):
        """Exactly one node: sample at its distance returns depth."""
        hp = _pick_with_data("sec", [50.0], [300.0])
        assert sample_pick_at_distance(hp, 50.0, "sec") == pytest.approx(300.0)

    def test_unordered_distances_still_interpolates(self):
        """Picks added in reverse distance order are sorted before interpolation."""
        hp = _pick_with_data("sec", [100.0, 0.0], [200.0, 100.0])
        result = sample_pick_at_distance(hp, 50.0, "sec")
        assert result == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# Integration: ghost pick geometry
# ---------------------------------------------------------------------------

class TestGhostPickGeometry:
    def test_ghost_depth_at_crossing(self):
        """End-to-end: two sections cross; correct depth is sampled from other section."""
        sec_a = Section([[0, 0], [100, 0]], name="A")
        sec_b = Section([[50, -50], [50, 50]], name="B")

        # Pick on section B at the intersection distance with depth=500
        hp = HorizonPick.empty(name="Top")
        # Section B is 100 m long; intersection is at s_b=50 (midpoint)
        hp.insert_pick(50.0, 500.0, "B")

        result = find_section_intersection(sec_a, sec_b)
        assert result is not None
        s_a, s_b = result

        depth = sample_pick_at_distance(hp, s_b, "B")
        assert depth == pytest.approx(500.0)

        # Ghost marker should appear at (s_a=50, depth=500) on section A
        assert abs(s_a - 50.0) < 1e-6
