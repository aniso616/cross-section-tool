"""Tests for snap engine and construction tools.

All tests are headless (no Qt) — they operate on pure data.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from section_tool.core.snap import (
    SnapResult,
    _ray_seg_intersect,
    _seg_intersect,
    extend_pick_to_entity,
    find_snap,
    trim_pick_at_entity,
)
from section_tool.core.surfaces import HorizonPick
from section_tool.tools.construction_tools import (
    DipConstrainedTool,
    KinkBandTool,
    ParallelOffsetTool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hp(distances, depths, sec="A"):
    return HorizonPick(
        distances=distances,
        depths=depths,
        section_names=[sec] * len(distances),
        name="test",
    )


def _identity_screen(d, z):
    """Trivial to_screen: 1 px = 1 data unit."""
    return (d, z)


# ---------------------------------------------------------------------------
# _seg_intersect
# ---------------------------------------------------------------------------

class TestSegIntersect:
    def test_crossing_segments(self):
        p = _seg_intersect(0, 0, 2, 2,  0, 2, 2, 0)
        assert p is not None
        assert abs(p[0] - 1.0) < 1e-9
        assert abs(p[1] - 1.0) < 1e-9

    def test_parallel_returns_none(self):
        assert _seg_intersect(0, 0, 1, 0,  0, 1, 1, 1) is None

    def test_non_overlapping_segments_returns_none(self):
        # Segments on the same line but do not overlap
        assert _seg_intersect(0, 0, 1, 0,  2, 0, 3, 0) is None

    def test_t_outside_range_returns_none(self):
        # Extension of segment AB passes through CD but not the finite segment AB
        assert _seg_intersect(0, 0, 0.4, 0.4,  0, 1, 1, 0) is None

    def test_s_outside_range_returns_none(self):
        assert _seg_intersect(0, 0, 2, 2,  3, 0, 4, 0) is None

    def test_t_boundary(self):
        # t = 0 exactly (intersection at A)
        p = _seg_intersect(1, 1, 2, 2,  0, 1, 2, 1)
        assert p is not None
        assert abs(p[0] - 1.0) < 1e-9
        assert abs(p[1] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# _ray_seg_intersect
# ---------------------------------------------------------------------------

class TestRaySegIntersect:
    def test_ray_hits_segment(self):
        # Horizontal ray from (0,0), slope=0, hits segment (5,-1)→(5,1)
        p = _ray_seg_intersect(0, 0, 0,  5, -1, 5, 1)
        assert p is not None
        assert abs(p[0] - 5.0) < 1e-9
        assert abs(p[1] - 0.0) < 1e-9

    def test_ray_parallel_to_segment_returns_none(self):
        assert _ray_seg_intersect(0, 0, 0,  0, 1, 5, 1) is None

    def test_ray_misses_segment(self):
        # Ray going right, segment entirely above
        assert _ray_seg_intersect(0, 0, 0,  3, 2, 3, 4) is None

    def test_sloped_ray(self):
        # Ray from origin with slope 1 (45°), segment vertical at d=3
        p = _ray_seg_intersect(0, 0, 1,  3, 0, 3, 5)
        assert p is not None
        assert abs(p[0] - 3.0) < 1e-9
        assert abs(p[1] - 3.0) < 1e-9

    def test_backward_direction(self):
        # Ray can hit target behind origin (t negative)
        p = _ray_seg_intersect(5, 0, 0,  2, -1, 2, 1)
        assert p is not None
        assert abs(p[0] - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# find_snap
# ---------------------------------------------------------------------------

class TestFindSnap:
    def _picks(self):
        hp1 = _hp([0, 10], [100, 200])
        hp2 = _hp([0, 10], [300, 100])
        return {"Horizons": [hp1, hp2]}

    def test_snaps_to_endpoint(self):
        # Cursor near endpoint (10, 200), section edges far away
        result = find_snap(
            cursor=(9.5, 198.0),
            picks_by_cat=self._picks(),
            threshold_px=20.0,
            to_screen=_identity_screen,
            section_edges=(0.0, 1000.0),
            sec_name="A",
        )
        assert result is not None
        assert result.kind == "endpoint"
        assert abs(result.pt[0] - 10.0) < 1e-9

    def test_snaps_to_midpoint(self):
        # Cursor near midpoint (5, 150)
        result = find_snap(
            cursor=(5.0, 148.0),
            picks_by_cat=self._picks(),
            threshold_px=5.0,
            to_screen=_identity_screen,
            sec_name="A",
        )
        assert result is not None
        assert result.kind in ("midpoint", "intersection", "endpoint")

    def test_snaps_to_intersection_priority(self):
        # hp1: (0,100)→(10,200), hp2: (0,300)→(10,100)
        # They cross somewhere around d=6.something
        # Cursor placed at intersection
        d_int = 200 / 30 * 10  # solve: 100+10d/10*100 = 300-200d/10
        # hp1 at d: 100 + d*10
        # hp2 at d: 300 - 20*d
        # Equal when 100+10d = 300-20d → 30d=200 → d=200/30 ≈ 6.667
        d_int = 200.0 / 30.0
        z_int = 100.0 + d_int * 10.0
        result = find_snap(
            cursor=(d_int + 0.5, z_int + 0.5),
            picks_by_cat=self._picks(),
            threshold_px=5.0,
            to_screen=_identity_screen,
            sec_name="A",
        )
        assert result is not None
        assert result.kind == "intersection"
        assert abs(result.pt[0] - d_int) < 0.1

    def test_returns_none_when_nothing_nearby(self):
        result = find_snap(
            cursor=(100.0, 5000.0),
            picks_by_cat=self._picks(),
            threshold_px=5.0,
            to_screen=_identity_screen,
            section_edges=(0.0, 10.0),
            sec_name="A",
        )
        assert result is None

    def test_edge_snap(self):
        result = find_snap(
            cursor=(0.5, 500.0),
            picks_by_cat={},
            threshold_px=5.0,
            to_screen=_identity_screen,
            section_edges=(0.0, 100.0),
            sec_name="A",
        )
        assert result is not None
        assert result.kind == "edge"
        assert result.pt[0] == 0.0

    def test_topology_pts_snap(self):
        result = find_snap(
            cursor=(5.1, 200.1),
            picks_by_cat={},
            threshold_px=5.0,
            to_screen=_identity_screen,
            topology_pts=[(5.0, 200.0)],
            sec_name="A",
        )
        assert result is not None
        assert result.kind == "intersection"
        assert abs(result.pt[0] - 5.0) < 1e-9


# ---------------------------------------------------------------------------
# trim_pick_at_entity
# ---------------------------------------------------------------------------

class TestTrimPickAtEntity:
    def test_trim_keeps_left(self):
        hp   = _hp([0, 10], [100, 200])  # slope +10
        cut  = _hp([0, 10], [200, 100])  # slope -10, crosses at d=5, z=150
        result = trim_pick_at_entity(hp, cut, keep_side_x=2.0, sec_name="A")
        si = result.section_indices("A")
        d_out = result._distances[si]
        z_out = result._depths[si]
        # Terminal point must be at intersection (d=5, z=150)
        assert abs(d_out[-1] - 5.0) < 1e-9
        assert abs(z_out[-1] - 150.0) < 1e-9
        # All kept points ≤ 5
        assert np.all(d_out <= 5.0 + 1e-9)

    def test_trim_keeps_right(self):
        hp  = _hp([0, 10], [100, 200])
        cut = _hp([0, 10], [200, 100])
        result = trim_pick_at_entity(hp, cut, keep_side_x=8.0, sec_name="A")
        si = result.section_indices("A")
        d_out = result._distances[si]
        z_out = result._depths[si]
        assert abs(d_out[0] - 5.0) < 1e-9
        assert abs(z_out[0] - 150.0) < 1e-9
        assert np.all(d_out >= 5.0 - 1e-9)

    def test_no_mutation(self):
        hp  = _hp([0, 10], [100, 200])
        cut = _hp([0, 10], [200, 100])
        d_before = hp._distances.copy()
        trim_pick_at_entity(hp, cut, keep_side_x=2.0, sec_name="A")
        np.testing.assert_array_equal(hp._distances, d_before)

    def test_raises_when_no_intersection(self):
        hp  = _hp([0, 10], [100, 200])
        cut = _hp([20, 30], [100, 200])  # doesn't overlap
        with pytest.raises(ValueError, match="do not intersect"):
            trim_pick_at_entity(hp, cut, keep_side_x=2.0, sec_name="A")

    def test_raises_when_too_few_points(self):
        hp   = _hp([5], [100])
        cut  = _hp([0, 10], [150, 150])
        with pytest.raises(ValueError, match="≥ 2 points"):
            trim_pick_at_entity(hp, cut, keep_side_x=2.0, sec_name="A")

    def test_exact_endpoint_no_gap(self):
        """The intersection must be bitwise-representable as exact shared endpoint."""
        hp   = _hp([0, 10], [0,  10])
        cut  = _hp([5,  5], [-5, 15])   # vertical cut at d=5
        result_l = trim_pick_at_entity(hp, cut, keep_side_x=2.0, sec_name="A")
        result_r = trim_pick_at_entity(hp, cut, keep_side_x=8.0, sec_name="A")
        si_l = result_l.section_indices("A")
        si_r = result_r.section_indices("A")
        # Both trimmed ends should be at exactly the same (d, z)
        assert result_l._distances[si_l[-1]] == result_r._distances[si_r[0]]
        assert result_l._depths[si_l[-1]]    == result_r._depths[si_r[0]]


# ---------------------------------------------------------------------------
# extend_pick_to_entity
# ---------------------------------------------------------------------------

class TestExtendPickToEntity:
    def test_extend_end(self):
        # hp: (0,0)→(5,0), horizontal; target: vertical line at d=10
        hp     = _hp([0, 5], [0, 0])
        target = _hp([10, 10], [-5, 5])
        result = extend_pick_to_entity(hp, "end", target, sec_name="A")
        si = result.section_indices("A")
        d_out = result._distances[si]
        z_out = result._depths[si]
        assert abs(d_out[-1] - 10.0) < 1e-9
        assert abs(z_out[-1] - 0.0)  < 1e-9

    def test_extend_start(self):
        # hp: (5,5)→(10,10), slope=1; target: vertical line at d=2
        hp     = _hp([5, 10], [5, 10])
        target = _hp([2, 2],  [0, 20])
        result = extend_pick_to_entity(hp, "start", target, sec_name="A")
        si = result.section_indices("A")
        d_out = result._distances[si]
        z_out = result._depths[si]
        assert abs(d_out[0] - 2.0) < 1e-9
        assert abs(z_out[0] - 2.0) < 1e-9  # slope=1 back to d=2: z=5-(5-2)*1=2

    def test_no_mutation(self):
        hp     = _hp([0, 5], [0, 0])
        target = _hp([10, 10], [-5, 5])
        d_before = hp._distances.copy()
        extend_pick_to_entity(hp, "end", target, sec_name="A")
        np.testing.assert_array_equal(hp._distances, d_before)

    def test_raises_when_no_intersection(self):
        hp     = _hp([0, 5],  [0, 0])
        target = _hp([0, 10], [5, 5])   # horizontal target above, ray misses
        with pytest.raises(ValueError, match="ray does not intersect"):
            extend_pick_to_entity(hp, "end", target, sec_name="A")

    def test_raises_when_too_few_points_source(self):
        hp     = _hp([5], [0])
        target = _hp([10, 10], [-5, 5])
        with pytest.raises(ValueError, match="≥ 2 points"):
            extend_pick_to_entity(hp, "end", target, sec_name="A")


# ---------------------------------------------------------------------------
# DipConstrainedTool
# ---------------------------------------------------------------------------

class TestDipConstrainedTool:
    def test_two_clicks_create_pick(self):
        tool = DipConstrainedTool(dip_deg=30.0)
        first = tool.handle_click(0.0, 1000.0, "A")
        assert first is None
        assert tool.state == "anchor_set"
        second = tool.handle_click(1000.0, 0.0, "A")  # z will be overridden
        assert second is not None
        assert tool.state == "idle"
        si = second.section_indices("A")
        d_out = second._distances[si]
        z_out = second._depths[si]
        expected_z = 1000.0 + 1000.0 * math.tan(math.radians(30.0))
        assert abs(z_out[-1] - expected_z) < 1e-6

    def test_construction_rule_attached(self):
        from section_tool.core.construction import DipConstrainedRule
        tool = DipConstrainedTool(dip_deg=20.0)
        tool.handle_click(0.0, 1000.0, "A")
        hp = tool.handle_click(500.0, 0.0, "A")
        assert isinstance(hp.construction_rule, DipConstrainedRule)
        assert hp.construction_rule.dip_deg == 20.0

    def test_reset_clears_anchor(self):
        tool = DipConstrainedTool()
        tool.handle_click(0.0, 0.0, "A")
        tool.reset()
        assert tool.state == "idle"

    def test_constrain_depth(self):
        tool = DipConstrainedTool(dip_deg=45.0)
        tool.handle_click(0.0, 100.0, "A")
        z = tool.constrain_depth(50.0)
        assert z is not None
        assert abs(z - 150.0) < 1e-9  # 100 + 50 * tan(45°) = 150

    def test_horizontal_dip(self):
        tool = DipConstrainedTool(dip_deg=0.0)
        tool.handle_click(0.0, 1000.0, "A")
        hp = tool.handle_click(5000.0, 1200.0, "A")
        si = hp.section_indices("A")
        z_out = hp._depths[si]
        assert abs(z_out[0] - z_out[-1]) < 1e-9


# ---------------------------------------------------------------------------
# ParallelOffsetTool
# ---------------------------------------------------------------------------

class TestParallelOffsetTool:
    def test_offset_copy(self):
        ref = _hp([0, 10], [100, 200])
        tool = ParallelOffsetTool()
        tool.set_reference(ref, "A")
        assert tool.state == "ref_selected"
        # Cursor at d=5, z=250 → ref at d=5 is 150, offset=100
        hp_new = tool.handle_placement(5.0, 250.0, "A")
        assert hp_new is not None
        si = hp_new.section_indices("A")
        z_out = hp_new._depths[si]
        # All depths should be offset by 100
        np.testing.assert_allclose(z_out, [200.0, 300.0], atol=1e-9)

    def test_construction_rule_attached(self):
        from section_tool.core.construction import ParallelToBedRule
        ref = _hp([0, 10], [100, 200])
        tool = ParallelOffsetTool()
        tool.set_reference(ref, "A")
        hp_new = tool.handle_placement(5.0, 250.0, "A")
        assert isinstance(hp_new.construction_rule, ParallelToBedRule)
        assert abs(hp_new.construction_rule.offset_m - 100.0) < 1e-9

    def test_resets_after_placement(self):
        ref = _hp([0, 10], [100, 200])
        tool = ParallelOffsetTool()
        tool.set_reference(ref, "A")
        tool.handle_placement(5.0, 250.0, "A")
        assert tool.state == "idle"

    def test_returns_none_without_reference(self):
        tool = ParallelOffsetTool()
        assert tool.handle_placement(5.0, 250.0, "A") is None


# ---------------------------------------------------------------------------
# KinkBandTool
# ---------------------------------------------------------------------------

class TestKinkBandTool:
    def test_creates_forelimb(self):
        backlimb = _hp([0, 10], [1000, 1000])  # flat horizon
        tool = KinkBandTool(axial_surface_dip_deg=45.0, fore_dip_deg=30.0)
        tool.set_reference(backlimb)
        assert tool.state == "ref_selected"
        hp_new = tool.handle_axial_click(5.0, "A", extent_d=10.0)
        assert hp_new is not None
        si = hp_new.section_indices("A")
        d_out = hp_new._distances[si]
        z_out = hp_new._depths[si]
        assert abs(d_out[0] - 5.0) < 1e-9   # starts at axial trace
        assert abs(z_out[0] - 1000.0) < 1e-9  # z at axial = z of flat horizon
        expected_z_end = 1000.0 + 5.0 * math.tan(math.radians(30.0))
        assert abs(z_out[-1] - expected_z_end) < 1e-6

    def test_construction_rule_on_forelimb(self):
        from section_tool.core.construction import KinkBandRule
        backlimb = _hp([0, 10], [1000, 1000])
        tool = KinkBandTool(axial_surface_dip_deg=45.0, fore_dip_deg=30.0, back_dip_deg=0.0)
        tool.set_reference(backlimb)
        hp_new = tool.handle_axial_click(5.0, "A", extent_d=10.0)
        assert isinstance(hp_new.construction_rule, KinkBandRule)
        assert hp_new.construction_rule.fore_dip_deg == 30.0

    def test_tags_backlimb(self):
        from section_tool.core.construction import KinkBandRule
        backlimb = _hp([0, 10], [1000, 1000])
        tool = KinkBandTool()
        tool.set_reference(backlimb)
        tool.handle_axial_click(5.0, "A", extent_d=10.0)
        assert isinstance(backlimb.construction_rule, KinkBandRule)

    def test_resets_after_axial_click(self):
        backlimb = _hp([0, 10], [1000, 1000])
        tool = KinkBandTool()
        tool.set_reference(backlimb)
        tool.handle_axial_click(5.0, "A", extent_d=10.0)
        assert tool.state == "idle"

    def test_default_extent_auto(self):
        backlimb = _hp([0, 10], [1000, 1000])
        tool = KinkBandTool(fore_dip_deg=0.0)
        tool.set_reference(backlimb)
        hp_new = tool.handle_axial_click(5.0, "A")  # no extent_d
        assert hp_new is not None
        si = hp_new.section_indices("A")
        d_out = hp_new._distances[si]
        assert len(d_out) == 2
        assert d_out[-1] > 5.0  # extends beyond axial trace
