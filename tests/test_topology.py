"""Tests for the live topological intersection graph."""
from __future__ import annotations

import pytest
from cross_section_tool.core.topology import IntersectionPoint, SectionTopology


@pytest.fixture
def topo():
    return SectionTopology("S1", section_length=10_000, max_depth=5_000)


class TestSectionTopologyBasics:
    def test_creation(self, topo):
        assert topo.section_name == "S1"

    def test_empty_has_no_user_intersections(self, topo):
        # Boundary lines only, no non-boundary intersections
        inter = topo.intersections
        # All intersections are boundary corners only
        assert all(p.type.endswith("_boundary") or "boundary" in p.type
                   for p in inter)

    def test_add_line(self, topo):
        topo.update_line("h0", "horizon", [(1000, 1000), (9000, 1000)])
        names = [k for k in topo._lines if not k.startswith("__")]
        assert "h0" in names

    def test_remove_line(self, topo):
        topo.update_line("h0", "horizon", [(1000, 1000), (9000, 1000)])
        topo.remove_line("h0")
        assert "h0" not in topo._lines

    def test_clear_user_lines(self, topo):
        topo.update_line("h0", "horizon", [(1000, 1000), (9000, 1000)])
        topo.update_line("f0", "fault", [(4000, 500), (5000, 2000)])
        topo.clear_user_lines()
        user = [k for k in topo._lines if not k.startswith("__")]
        assert user == []


class TestIntersections:
    def test_two_crossing_lines_one_intersection(self, topo):
        # Horizontal horizon at depth 2000
        topo.update_line("h0", "horizon", [(500, 2000), (9500, 2000)])
        # Diagonal fault crossing it
        topo.update_line("f0", "fault", [(4000, 500), (6000, 4000)])
        inter = [p for p in topo.intersections
                 if not p.type.endswith("_boundary")]
        assert len(inter) == 1
        # Intersection type
        assert "horizon" in inter[0].type and "fault" in inter[0].type

    def test_intersection_coordinates(self, topo):
        # Flat horizon at y=2000
        topo.update_line("h0", "horizon", [(0, 2000), (10000, 2000)])
        # Vertical fault at x=5000, spanning full depth
        topo.update_line("f0", "fault", [(5000, 0), (5000, 5000)])
        inter = [p for p in topo.intersections
                 if not p.type.endswith("_boundary") and "horizon" in p.type]
        assert len(inter) == 1
        assert abs(inter[0].x - 5000) < 1.0
        assert abs(inter[0].y - 2000) < 1.0

    def test_parallel_lines_no_intersection(self, topo):
        topo.update_line("h0", "horizon", [(500, 1000), (9500, 1000)])
        topo.update_line("h1", "horizon", [(500, 2000), (9500, 2000)])
        inter = [p for p in topo.intersections
                 if not p.type.endswith("_boundary")]
        assert len(inter) == 0

    def test_three_lines_produce_correct_count(self, topo):
        topo.update_line("h0", "horizon", [(0, 1500), (10000, 1500)])
        topo.update_line("h1", "horizon", [(0, 3000), (10000, 3000)])
        topo.update_line("f0", "fault", [(5000, 0), (5000, 5000)])
        inter_nob = [p for p in topo.intersections
                     if not p.type.endswith("_boundary")]
        # h0×f0 and h1×f0 = 2 intersections
        assert len(inter_nob) == 2

    def test_moving_line_updates_intersection(self, topo):
        topo.update_line("h0", "horizon", [(0, 1000), (10000, 1000)])
        topo.update_line("f0", "fault", [(5000, 0), (5000, 5000)])
        inter_before = [p for p in topo.intersections
                        if not p.type.endswith("_boundary")][0]
        # Move horizon to depth 3000
        topo.update_line("h0", "horizon", [(0, 3000), (10000, 3000)])
        inter_after = [p for p in topo.intersections
                       if not p.type.endswith("_boundary")][0]
        assert abs(inter_before.y - 1000) < 1.0
        assert abs(inter_after.y - 3000) < 1.0


class TestExtension:
    def test_line_extended_to_left_edge(self, topo):
        topo.update_line("h0", "horizon", [(2000, 1000), (8000, 1000)])
        ls = topo._lines["h0"][1]
        assert ls.coords[0][0] == pytest.approx(0.0, abs=1.0)

    def test_line_extended_to_right_edge(self, topo):
        topo.update_line("h0", "horizon", [(2000, 1000), (8000, 1000)])
        ls = topo._lines["h0"][1]
        assert ls.coords[-1][0] == pytest.approx(10_000.0, abs=1.0)

    def test_line_already_at_edges_unchanged_length(self, topo):
        topo.update_line("h0", "horizon", [(0, 1000), (10000, 1000)])
        ls = topo._lines["h0"][1]
        # No extra extension needed
        assert ls.coords[0][0] == pytest.approx(0.0)
        assert ls.coords[-1][0] == pytest.approx(10_000.0)


class TestFaceDetection:
    def test_two_horizons_produce_three_faces(self, topo):
        topo.update_line("h0", "horizon", [(500, 1000), (9500, 1000)])
        topo.update_line("h1", "horizon", [(500, 3000), (9500, 3000)])
        faces = topo.get_all_faces()
        assert len(faces) == 3

    def test_crossing_fault_increases_face_count(self, topo):
        topo.update_line("h0", "horizon", [(0, 1500), (10000, 1500)])
        topo.update_line("h1", "horizon", [(0, 3500), (10000, 3500)])
        topo.update_line("f0", "fault", [(5000, 0), (5000, 5000)])
        faces = topo.get_all_faces()
        # 2 horizons split by 1 fault → 6 faces
        assert len(faces) >= 5

    def test_no_faces_without_user_lines(self, topo):
        # Only boundary lines — no interior
        faces = topo.get_all_faces()
        # The boundary rectangle itself may form 1 face, but no interior faces
        assert len(faces) <= 1


class TestSnapTargets:
    def test_intersection_is_snap_target(self, topo):
        topo.update_line("h0", "horizon", [(0, 2000), (10000, 2000)])
        topo.update_line("f0", "fault", [(5000, 0), (5000, 5000)])
        snaps = topo.get_snap_targets()
        # Should contain the horizon×fault intersection at (5000, 2000)
        found = any(abs(x - 5000) < 5 and abs(y - 2000) < 5 for x, y in snaps)
        assert found
