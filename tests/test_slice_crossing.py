"""Tests for slice_crossing — the generic slice×slice dispatcher (4 pairings)."""
from __future__ import annotations

import numpy as np
import pytest

from section_tool.core.geometry import slice_crossing, SliceCrossing
from section_tool.core.section import Section
from section_tool.core.slices import HorizontalSlice
from section_tool.core.surfaces import HorizonPick


def _pick(dists, depths, name="P", sec="L1"):
    return HorizonPick(np.array(dists, float), np.array(depths, float),
                       name=name, section_names=[sec] * len(dists))


# ---------------------------------------------------------------------------
# section × section — must reproduce the existing ghost geometry
# ---------------------------------------------------------------------------

class TestSectionSection:
    def test_crossing_point_and_depth(self):
        a = Section([(0.0, 0.0), (1000.0, 0.0)], name="A")          # E-W
        b = Section([(500.0, -500.0), (500.0, 500.0)], name="B")    # N-S crossing A at x=500
        hp = _pick([0, 1000], [1000, 1200], name="H", sec="B")      # on B
        sc = slice_crossing(a, b, [hp])
        assert sc.locus_kind == "v_line"
        assert sc.locus == pytest.approx(500.0)                     # s_a = 500 along A
        assert len(sc.piercings) == 1
        p = sc.piercings[0]
        assert p.u == pytest.approx(500.0)                          # rendered at s_a in A
        # B's pick depth at its crossing distance (b crossing is at s_b=500 -> depth 1100)
        assert p.v == pytest.approx(1100.0)

    def test_no_crossing_when_parallel(self):
        a = Section([(0.0, 0.0), (1000.0, 0.0)], name="A")
        b = Section([(0.0, 500.0), (1000.0, 500.0)], name="B")      # parallel, never meets
        sc = slice_crossing(a, b, [])
        assert sc.locus_kind == "none"
        assert sc.piercings == []


# ---------------------------------------------------------------------------
# section × horizontal — the new "window" content on a section
# ---------------------------------------------------------------------------

class TestSectionHorizontal:
    def test_h_line_and_piercing(self):
        sec = Section([(0.0, 0.0), (1000.0, 0.0)], name="L1")
        hs = HorizontalSlice(name="Z-1500", elevation=-1500.0)
        hp = _pick([0, 1000], [1000, 2000], name="F1", sec="L1")    # crosses 1500 at d=500
        sc = slice_crossing(sec, hs, [hp])
        assert sc.locus_kind == "h_line"
        assert sc.locus == pytest.approx(1500.0)                    # depth = -z0
        assert len(sc.piercings) == 1
        assert sc.piercings[0].u == pytest.approx(500.0)            # distance
        assert sc.piercings[0].v == pytest.approx(1500.0)

    def test_folded_pick_two_piercings(self):
        sec = Section([(0.0, 0.0), (2000.0, 0.0)], name="L1")
        hs = HorizontalSlice(name="Z-1500", elevation=-1500.0)
        hp = _pick([0, 1000, 2000], [1000, 2000, 1000], name="F1", sec="L1")
        sc = slice_crossing(sec, hs, [hp])
        assert len(sc.piercings) == 2


# ---------------------------------------------------------------------------
# horizontal × section — symmetric (for the future z-slice plan view)
# ---------------------------------------------------------------------------

class TestHorizontalSection:
    def test_polyline_locus_and_world_piercing(self):
        sec = Section([(0.0, 0.0), (1000.0, 0.0)], name="L1")
        hs = HorizontalSlice(name="Z-1500", elevation=-1500.0)
        hp = _pick([0, 1000], [1000, 2000], name="F1", sec="L1")
        sc = slice_crossing(hs, sec, [hp])
        assert sc.locus_kind == "polyline"
        assert sc.locus == [(0.0, 0.0), (1000.0, 0.0)]              # section trace in plan
        assert len(sc.piercings) == 1
        # piercing in plan coords = world (x, y) of the d=500 crossing on an E-W trace
        assert (sc.piercings[0].u, sc.piercings[0].v) == pytest.approx((500.0, 0.0))
        assert sc.piercings[0].z == pytest.approx(-1500.0)


# ---------------------------------------------------------------------------
# horizontal × horizontal — parallel planes
# ---------------------------------------------------------------------------

class TestHorizontalHorizontal:
    def test_different_elevation_is_none(self):
        a = HorizontalSlice(name="Z-1500", elevation=-1500.0)
        b = HorizontalSlice(name="Z-2000", elevation=-2000.0)
        assert slice_crossing(a, b, []).locus_kind == "none"

    def test_equal_elevation_is_coincident(self):
        a = HorizontalSlice(name="A", elevation=-1500.0)
        b = HorizontalSlice(name="B", elevation=-1500.0)
        assert slice_crossing(a, b, []).locus_kind == "coincident"
