"""Tests for sample_pick_at_elevation — the 'window into 3D' keystone primitive."""
from __future__ import annotations

import numpy as np
import pytest

from section_tool.core.geometry import sample_pick_at_elevation, LevelCrossing
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick


def _pick(dists, depths, name="P", sec="L1"):
    return HorizonPick(np.array(dists, float), np.array(depths, float),
                       name=name, section_names=[sec] * len(dists))


# ---------------------------------------------------------------------------
# Crossing counts (the four cases the audit called out)
# ---------------------------------------------------------------------------

class TestCrossingCounts:
    def test_dipping_pick_crosses_once(self):
        # depth 1000 -> 2000 over distance 0 -> 1000; level z0=-1500 -> depth 1500
        cr = sample_pick_at_elevation(_pick([0, 1000], [1000, 2000]), -1500.0)
        assert len(cr) == 1
        assert cr[0].distance == pytest.approx(500.0)
        assert cr[0].depth == pytest.approx(1500.0)

    def test_flat_coincident_pick_crosses_once(self):
        # flat pick AT depth 1500; level z0=-1500 -> coincident -> single crossing
        cr = sample_pick_at_elevation(_pick([0, 1000], [1500, 1500]), -1500.0)
        assert len(cr) == 1
        assert cr[0].depth == pytest.approx(1500.0)

    def test_folded_pick_crosses_twice(self):
        # syncline: 1000 -> 2000 -> 1000; level depth 1500 crossed on each limb
        cr = sample_pick_at_elevation(_pick([0, 1000, 2000], [1000, 2000, 1000]), -1500.0)
        assert len(cr) == 2
        assert cr[0].distance == pytest.approx(500.0)
        assert cr[1].distance == pytest.approx(1500.0)
        assert all(c.depth == pytest.approx(1500.0) for c in cr)

    def test_pick_entirely_above_crosses_zero(self):
        # all shallower than the level (depths 500,800 < 1500) -> no crossing
        assert sample_pick_at_elevation(_pick([0, 1000], [500, 800]), -1500.0) == []

    def test_pick_entirely_below_crosses_zero(self):
        assert sample_pick_at_elevation(_pick([0, 1000], [2000, 2500]), -1500.0) == []


# ---------------------------------------------------------------------------
# Edge handling
# ---------------------------------------------------------------------------

class TestEdges:
    def test_vertex_on_level_not_double_counted(self):
        # monotonic through a vertex sitting exactly on the level -> one crossing
        cr = sample_pick_at_elevation(_pick([0, 1000, 2000], [2000, 1500, 1000]), -1500.0)
        assert len(cr) == 1
        assert cr[0].distance == pytest.approx(1000.0)

    def test_single_point_pick_no_crossing(self):
        assert sample_pick_at_elevation(_pick([0], [1500]), -1500.0) == []


# ---------------------------------------------------------------------------
# World coords via the owning section
# ---------------------------------------------------------------------------

class TestWorldCoords:
    def test_world_coords_filled_with_section(self):
        sec = Section([(0.0, 0.0), (1000.0, 0.0)], name="L1")
        cr = sample_pick_at_elevation(_pick([0, 1000], [1000, 2000], sec="L1"),
                                      -1500.0, section=sec)
        assert len(cr) == 1
        # crossing at distance 500 on an E-W trace -> world (500, 0, -1500)
        assert (cr[0].x, cr[0].y, cr[0].z) == pytest.approx((500.0, 0.0, -1500.0))

    def test_world_nan_without_section(self):
        cr = sample_pick_at_elevation(_pick([0, 1000], [1000, 2000]), -1500.0)
        assert np.isnan(cr[0].x) and np.isnan(cr[0].y) and np.isnan(cr[0].z)

    def test_returns_levelcrossing_namedtuple(self):
        cr = sample_pick_at_elevation(_pick([0, 1000], [1000, 2000]), -1500.0)
        assert isinstance(cr[0], LevelCrossing)
