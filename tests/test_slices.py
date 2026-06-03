"""Tests for the Slice abstraction: Section (vertical) + HorizontalSlice (plan)."""
from __future__ import annotations

import numpy as np
import pytest

from section_tool.core.section import Section
from section_tool.core.slices import HorizontalSlice, Slice


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

class TestProtocolConformance:
    def test_section_is_a_slice(self):
        sec = Section([(0.0, 0.0), (1000.0, 0.0)], name="L1")
        assert isinstance(sec, Slice)
        assert sec.kind == "section"

    def test_horizontal_is_a_slice(self):
        hs = HorizontalSlice(name="Z-1500", elevation=-1500.0)
        assert isinstance(hs, Slice)
        assert hs.kind == "horizontal"


# ---------------------------------------------------------------------------
# Section (vertical slice) transforms
# ---------------------------------------------------------------------------

class TestSectionSliceTransforms:
    def test_to_world_uses_trace_and_negates_depth(self):
        # E-W section from (0,0) to (1000,0); slice coords (distance, depth)
        sec = Section([(0.0, 0.0), (1000.0, 0.0)], name="L1")
        x, y, z = sec.to_world(400.0, 1500.0)
        assert (x, y) == pytest.approx((400.0, 0.0))
        assert z == pytest.approx(-1500.0)        # depth -> elevation

    def test_from_world_round_trips_on_plane(self):
        sec = Section([(0.0, 0.0), (1000.0, 0.0)], name="L1")
        u, v, residual = sec.from_world(400.0, 0.0, -1500.0)
        assert u == pytest.approx(400.0)          # distance_along
        assert v == pytest.approx(1500.0)         # depth
        assert residual == pytest.approx(0.0)     # on the trace

    def test_from_world_reports_off_plane_residual(self):
        sec = Section([(0.0, 0.0), (1000.0, 0.0)], name="L1")
        _, _, residual = sec.from_world(400.0, 250.0, -1500.0)
        assert residual == pytest.approx(250.0)   # 250 m off the trace


# ---------------------------------------------------------------------------
# HorizontalSlice transforms (trivial: slice coords ARE world easting/northing)
# ---------------------------------------------------------------------------

class TestHorizontalSliceTransforms:
    def test_to_world_is_identity_xy_fixed_z(self):
        hs = HorizontalSlice(name="Z-1500", elevation=-1500.0)
        assert hs.to_world(612340.0, 6086000.0) == pytest.approx(
            (612340.0, 6086000.0, -1500.0))

    def test_from_world_round_trips_on_plane(self):
        hs = HorizontalSlice(name="Z-1500", elevation=-1500.0)
        u, v, residual = hs.from_world(612340.0, 6086000.0, -1500.0)
        assert (u, v) == pytest.approx((612340.0, 6086000.0))
        assert residual == pytest.approx(0.0)

    def test_from_world_residual_is_vertical_distance(self):
        hs = HorizontalSlice(name="Z-1500", elevation=-1500.0)
        _, _, residual = hs.from_world(612340.0, 6086000.0, -1200.0)
        assert residual == pytest.approx(300.0)   # 300 m above the slice

    def test_world_round_trip(self):
        hs = HorizontalSlice(name="Z0", elevation=-800.0)
        e, n = 500000.0, 5800000.0
        x, y, z = hs.to_world(e, n)
        u, v, residual = hs.from_world(x, y, z)
        assert (u, v) == pytest.approx((e, n))
        assert residual == pytest.approx(0.0)
