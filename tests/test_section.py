"""Tests for cross_section_tool.core.section.Section."""

import math

import numpy as np
import pytest

from cross_section_tool.core.section import Section


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def straight_section(length: float = 1000.0) -> Section:
    """Horizontal (east-ward) straight section of *length* metres."""
    return Section([(0.0, 0.0), (length, 0.0)])


def dogleg_section() -> Section:
    """L-shaped section: 1000 m east, then 1000 m north."""
    return Section([(0.0, 0.0), (1000.0, 0.0), (1000.0, 1000.0)])


# ---------------------------------------------------------------------------
# Construction and metadata
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_minimum_two_nodes(self):
        with pytest.raises(ValueError):
            Section([(0.0, 0.0)])

    def test_wrong_shape(self):
        with pytest.raises(ValueError):
            Section([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])

    def test_nodes_are_copied(self):
        raw = np.array([[0.0, 0.0], [1.0, 0.0]])
        sec = Section(raw)
        raw[0, 0] = 99.0
        assert sec.nodes[0, 0] == 0.0

    def test_default_metadata(self):
        sec = Section([(0.0, 0.0), (1.0, 0.0)])
        assert sec.depth_domain == "depth"
        assert sec.depth_units == "m"
        assert sec.vertical_exaggeration == 1.0

    def test_custom_metadata(self):
        sec = Section(
            [(0, 0), (1, 0)],
            name="Testline",
            depth_domain="twt",
            depth_units="ft",
            vertical_exaggeration=2.5,
            crs_epsg=4326,
        )
        assert sec.name == "Testline"
        assert sec.depth_domain == "twt"
        assert sec.depth_units == "ft"
        assert sec.vertical_exaggeration == 2.5
        assert sec.crs_epsg == 4326

    def test_repr_contains_name(self):
        sec = Section([(0, 0), (100, 0)], name="MyLine")
        assert "MyLine" in repr(sec)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class TestGeometry:
    def test_segment_lengths_straight(self):
        sec = straight_section(500.0)
        lengths = sec.segment_lengths()
        assert lengths.shape == (1,)
        assert pytest.approx(lengths[0]) == 500.0

    def test_segment_lengths_dogleg(self):
        sec = dogleg_section()
        lengths = sec.segment_lengths()
        assert lengths.shape == (2,)
        assert pytest.approx(lengths[0]) == 1000.0
        assert pytest.approx(lengths[1]) == 1000.0

    def test_cumulative_distances(self):
        sec = dogleg_section()
        cum = sec.cumulative_distances()
        assert cum.shape == (3,)
        assert pytest.approx(cum[0]) == 0.0
        assert pytest.approx(cum[1]) == 1000.0
        assert pytest.approx(cum[2]) == 2000.0

    def test_total_length(self):
        sec = dogleg_section()
        assert pytest.approx(sec.total_length()) == 2000.0

    def test_azimuth_east(self):
        # (0,0) → (1,0): east = 90° from north
        sec = Section([(0.0, 0.0), (1.0, 0.0)])
        az = sec.segment_azimuths()
        assert pytest.approx(az[0]) == 90.0

    def test_azimuth_north(self):
        # (0,0) → (0,1): north = 0°
        sec = Section([(0.0, 0.0), (0.0, 1.0)])
        az = sec.segment_azimuths()
        assert pytest.approx(az[0]) == 0.0

    def test_azimuth_south(self):
        # (0,0) → (0,-1): south = 180°
        sec = Section([(0.0, 0.0), (0.0, -1.0)])
        az = sec.segment_azimuths()
        assert pytest.approx(az[0]) == 180.0

    def test_azimuth_west(self):
        # (0,0) → (-1,0): west = 270°
        sec = Section([(0.0, 0.0), (-1.0, 0.0)])
        az = sec.segment_azimuths()
        assert pytest.approx(az[0]) == 270.0

    def test_azimuth_northeast(self):
        # 45° NE
        sec = Section([(0.0, 0.0), (1.0, 1.0)])
        az = sec.segment_azimuths()
        assert pytest.approx(az[0]) == 45.0

    def test_dogleg_azimuths(self):
        sec = dogleg_section()
        az = sec.segment_azimuths()
        assert pytest.approx(az[0]) == 90.0   # east
        assert pytest.approx(az[1]) == 0.0    # north


# ---------------------------------------------------------------------------
# map_to_section
# ---------------------------------------------------------------------------

class TestMapToSection:
    def test_on_start_node(self):
        sec = straight_section(1000.0)
        s, perp = sec.map_to_section(0.0, 0.0)
        assert pytest.approx(s) == 0.0
        assert pytest.approx(perp) == 0.0

    def test_on_end_node(self):
        sec = straight_section(1000.0)
        s, perp = sec.map_to_section(1000.0, 0.0)
        assert pytest.approx(s) == 1000.0
        assert pytest.approx(perp) == 0.0

    def test_midpoint_on_line(self):
        sec = straight_section(1000.0)
        s, perp = sec.map_to_section(500.0, 0.0)
        assert pytest.approx(s) == 500.0
        assert pytest.approx(perp) == 0.0

    def test_perpendicular_offset_positive(self):
        # Straight east section; point 100 m north = to the LEFT = positive
        sec = straight_section(1000.0)
        s, perp = sec.map_to_section(500.0, 100.0)
        assert pytest.approx(s) == 500.0
        assert pytest.approx(perp) == 100.0

    def test_perpendicular_offset_negative(self):
        # Point south of an east-going line = to the right = negative
        sec = straight_section(1000.0)
        s, perp = sec.map_to_section(500.0, -75.0)
        assert pytest.approx(s) == 500.0
        assert pytest.approx(perp) == -75.0

    def test_point_beyond_end_clamps(self):
        # x > total length should clamp to the last node
        sec = straight_section(1000.0)
        s, perp = sec.map_to_section(1500.0, 0.0)
        assert pytest.approx(s) == 1000.0

    def test_point_before_start_clamps(self):
        sec = straight_section(1000.0)
        s, perp = sec.map_to_section(-200.0, 0.0)
        assert pytest.approx(s) == 0.0

    def test_dogleg_bend_node(self):
        # The bend node (1000, 0) sits exactly at distance 1000
        sec = dogleg_section()
        s, perp = sec.map_to_section(1000.0, 0.0)
        assert pytest.approx(s) == 1000.0
        assert pytest.approx(perp, abs=1e-9) == 0.0

    def test_dogleg_second_segment_midpoint(self):
        # Mid-point of second segment: (1000, 500) → distance 1500
        sec = dogleg_section()
        s, perp = sec.map_to_section(1000.0, 500.0)
        assert pytest.approx(s) == 1500.0
        assert pytest.approx(perp, abs=1e-9) == 0.0

    def test_dogleg_offset_on_second_segment(self):
        # 200 m west of mid-point on second segment
        # Second segment goes north (azimuth 0°); left is west (negative x)
        sec = dogleg_section()
        s, perp = sec.map_to_section(800.0, 500.0)
        assert pytest.approx(s, abs=1.0) == 1500.0
        # Perpendicular should be -200 (to the right / east side)
        # direction of travel: north (+y); right-hand rule: left = west = -x
        # perp sign: positive = left = west; point is 200 m west → +200
        assert pytest.approx(perp, abs=1.0) == 200.0

    def test_point_nearest_to_inner_bend(self):
        # A point at the inner corner of the bend should snap to the bend node
        sec = dogleg_section()
        s, perp = sec.map_to_section(900.0, 100.0)
        # Closest point is somewhere on segment 0 or 1, distance ≥ 0
        assert s >= 0.0
        assert s <= sec.total_length()


# ---------------------------------------------------------------------------
# section_to_map (back-calculation)
# ---------------------------------------------------------------------------

class TestSectionToMap:
    def test_start(self):
        sec = straight_section(1000.0)
        x, y = sec.section_to_map(0.0)
        assert pytest.approx(x) == 0.0
        assert pytest.approx(y) == 0.0

    def test_end(self):
        sec = straight_section(1000.0)
        x, y = sec.section_to_map(1000.0)
        assert pytest.approx(x) == 1000.0
        assert pytest.approx(y) == 0.0

    def test_midpoint(self):
        sec = straight_section(1000.0)
        x, y = sec.section_to_map(500.0)
        assert pytest.approx(x) == 500.0
        assert pytest.approx(y) == 0.0

    def test_dogleg_bend_node(self):
        sec = dogleg_section()
        x, y = sec.section_to_map(1000.0)
        assert pytest.approx(x) == 1000.0
        assert pytest.approx(y) == 0.0

    def test_dogleg_second_segment(self):
        sec = dogleg_section()
        x, y = sec.section_to_map(1500.0)
        assert pytest.approx(x) == 1000.0
        assert pytest.approx(y) == 500.0

    def test_clamp_beyond_end(self):
        sec = straight_section(1000.0)
        x, y = sec.section_to_map(2000.0)
        assert pytest.approx(x) == 1000.0
        assert pytest.approx(y) == 0.0

    def test_clamp_before_start(self):
        sec = straight_section(1000.0)
        x, y = sec.section_to_map(-100.0)
        assert pytest.approx(x) == 0.0
        assert pytest.approx(y) == 0.0

    def test_roundtrip_on_section(self):
        """Points on the section line should round-trip exactly."""
        sec = dogleg_section()
        for s_target in [0.0, 250.0, 1000.0, 1750.0, 2000.0]:
            x, y = sec.section_to_map(s_target)
            s_back, perp = sec.map_to_section(x, y)
            assert pytest.approx(s_back, abs=1e-8) == s_target
            assert pytest.approx(perp, abs=1e-8) == 0.0

    def test_dogleg_three_segment(self):
        """Three-segment dogleg: east, north, east."""
        sec = Section([(0, 0), (1000, 0), (1000, 1000), (2000, 1000)])
        # 500 m into segment 3 (at distance 2500)
        x, y = sec.section_to_map(2500.0)
        assert pytest.approx(x) == 1500.0
        assert pytest.approx(y) == 1000.0


# ---------------------------------------------------------------------------
# Node operations
# ---------------------------------------------------------------------------

class TestNodeOperations:
    def test_add_node(self):
        sec = straight_section()
        sec.add_node(2000.0, 0.0)
        assert sec.n_nodes == 3
        assert pytest.approx(sec.nodes[-1, 0]) == 2000.0

    def test_insert_node_at_index_0(self):
        sec = straight_section(1000.0)
        sec.insert_node(0, -500.0, 0.0)
        assert sec.n_nodes == 3
        assert pytest.approx(sec.nodes[0, 0]) == -500.0

    def test_insert_node_at_middle(self):
        sec = dogleg_section()
        sec.insert_node(1, 500.0, 0.0)
        assert sec.n_nodes == 4
        assert sec.nodes[1] == pytest.approx([500.0, 0.0])

    def test_insert_node_out_of_range(self):
        sec = straight_section()
        with pytest.raises(IndexError):
            sec.insert_node(99, 0.0, 0.0)

    def test_insert_node_on_segment(self):
        sec = dogleg_section()
        # Insert midpoint on segment 0
        sec.insert_node_on_segment(0, 500.0, 0.0)
        assert sec.n_nodes == 4
        assert sec.nodes[1] == pytest.approx([500.0, 0.0])

    def test_insert_node_on_segment_bad_index(self):
        sec = straight_section()
        with pytest.raises(IndexError):
            sec.insert_node_on_segment(5, 0.0, 0.0)

    def test_delete_node(self):
        sec = dogleg_section()
        sec.delete_node(1)
        assert sec.n_nodes == 2
        assert sec.nodes[0] == pytest.approx([0.0, 0.0])
        assert sec.nodes[1] == pytest.approx([1000.0, 1000.0])

    def test_delete_node_refuses_two_nodes(self):
        sec = straight_section()
        with pytest.raises(ValueError):
            sec.delete_node(0)

    def test_delete_node_out_of_range(self):
        sec = dogleg_section()
        with pytest.raises(IndexError):
            sec.delete_node(99)

    def test_move_node(self):
        sec = straight_section(1000.0)
        sec.move_node(1, 1500.0, 50.0)
        assert pytest.approx(sec.nodes[1, 0]) == 1500.0
        assert pytest.approx(sec.nodes[1, 1]) == 50.0

    def test_move_node_updates_geometry(self):
        sec = straight_section(1000.0)
        sec.move_node(1, 2000.0, 0.0)
        assert pytest.approx(sec.total_length()) == 2000.0

    def test_move_node_out_of_range(self):
        sec = straight_section()
        with pytest.raises(IndexError):
            sec.move_node(99, 0.0, 0.0)


# ---------------------------------------------------------------------------
# from_azimuth_length constructor
# ---------------------------------------------------------------------------

class TestFromAzimuthLength:
    def test_single_segment_north(self):
        sec = Section.from_azimuth_length(0.0, 0.0, [(0.0, 1000.0)])
        assert sec.n_nodes == 2
        assert pytest.approx(sec.nodes[1, 0], abs=1e-9) == 0.0
        assert pytest.approx(sec.nodes[1, 1]) == 1000.0

    def test_single_segment_east(self):
        sec = Section.from_azimuth_length(0.0, 0.0, [(90.0, 1000.0)])
        assert pytest.approx(sec.nodes[1, 0]) == 1000.0
        assert pytest.approx(sec.nodes[1, 1], abs=1e-9) == 0.0

    def test_two_segment_dogleg(self):
        sec = Section.from_azimuth_length(0.0, 0.0, [(90.0, 1000.0), (0.0, 1000.0)])
        assert sec.n_nodes == 3
        assert pytest.approx(sec.nodes[1, 0]) == 1000.0
        assert pytest.approx(sec.nodes[1, 1], abs=1e-9) == 0.0
        assert pytest.approx(sec.nodes[2, 0]) == 1000.0
        assert pytest.approx(sec.nodes[2, 1]) == 1000.0

    def test_azimuth_west(self):
        sec = Section.from_azimuth_length(0.0, 0.0, [(270.0, 500.0)])
        assert pytest.approx(sec.nodes[1, 0]) == -500.0
        assert pytest.approx(sec.nodes[1, 1], abs=1e-9) == 0.0

    def test_azimuth_length_matches_geometry(self):
        sec = Section.from_azimuth_length(500.0, 200.0, [(45.0, 1000.0)])
        az = sec.segment_azimuths()
        assert pytest.approx(az[0]) == 45.0
        assert pytest.approx(sec.segment_lengths()[0]) == 1000.0

    def test_kwargs_forwarded(self):
        sec = Section.from_azimuth_length(
            0.0, 0.0, [(0.0, 100.0)], name="Test", crs_epsg=32654
        )
        assert sec.name == "Test"
        assert sec.crs_epsg == 32654


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_diagonal_roundtrip(self):
        """45-degree section round-trips correctly."""
        sec = Section([(0.0, 0.0), (1000.0, 1000.0)])
        diag_len = 1000.0 * math.sqrt(2)
        s, perp = sec.map_to_section(500.0, 500.0)
        assert pytest.approx(s, rel=1e-6) == diag_len / 2
        assert pytest.approx(perp, abs=1e-9) == 0.0
        x, y = sec.section_to_map(s)
        assert pytest.approx(x, abs=1e-9) == 500.0
        assert pytest.approx(y, abs=1e-9) == 500.0

    def test_many_nodes(self):
        """Section with 10 nodes runs without error."""
        nodes = [(float(i * 100), float(i * 50)) for i in range(10)]
        sec = Section(nodes)
        assert sec.n_segments == 9
        s, _ = sec.map_to_section(450.0, 225.0)
        assert 0.0 <= s <= sec.total_length()

    def test_zero_length_segment_safe(self):
        """Duplicate nodes (zero-length segment) don't cause a crash."""
        sec = Section([(0.0, 0.0), (0.0, 0.0), (1000.0, 0.0)])
        s, _ = sec.map_to_section(500.0, 0.0)
        assert pytest.approx(s, abs=1.0) == 500.0

    def test_perp_sign_consistency(self):
        """For a north-going line left is west (negative x) → positive perp."""
        sec = Section([(0.0, 0.0), (0.0, 1000.0)])
        _, perp_left = sec.map_to_section(-100.0, 500.0)   # west of north line = left
        _, perp_right = sec.map_to_section(100.0, 500.0)   # east of north line = right
        assert perp_left > 0
        assert perp_right < 0
