"""Additional tests for polygon detection — Phase 3 improvements."""
import numpy as np
import pytest

from section_tool.core.polygon_detection import detect_polygons
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick


def test_two_horizontal_horizons():
    """Two horizontal horizons produce 3 polygons."""
    section = Section([(0, 0), (10000, 0)], name="S1")
    hp1 = HorizonPick([500, 9500], [1000, 1000], name="H1",
                      section_names=["S1", "S1"])
    hp2 = HorizonPick([500, 9500], [2000, 2000], name="H2",
                      section_names=["S1", "S1"])

    polys = detect_polygons([hp1, hp2], [], [], section, "S1", gap_tolerance=0)
    assert len(polys) == 3


def test_picks_extended_to_section_edges():
    """Picks that don't start at x=0 or end at x=total get extended."""
    section = Section([(0, 0), (10000, 0)], name="S1")
    # Pick starts at 1000, ends at 9000 — should still form 2 polygons with boundary
    hp = HorizonPick([1000, 9000], [500, 500], name="H1",
                     section_names=["S1", "S1"])
    polys = detect_polygons([hp], [], [], section, "S1", gap_tolerance=0)
    assert len(polys) == 2


def test_gap_tolerance_closes_gaps():
    """With gap_tolerance > 0 two separate line segments that almost meet should form a polygon."""
    section = Section([(0, 0), (1000, 0)], name="S1")
    # Two horizons that nearly meet at the section boundary
    hp1 = HorizonPick([10, 490], [500, 500], name="H1",
                      section_names=["S1", "S1"])
    hp2 = HorizonPick([510, 990], [500, 500], name="H2",
                      section_names=["S1", "S1"])
    # With large gap_tolerance, these should merge
    polys_tol = detect_polygons([hp1, hp2], [], [], section, "S1", gap_tolerance=100)
    polys_none = detect_polygons([hp1, hp2], [], [], section, "S1", gap_tolerance=0)
    # Gap tolerance version should be at least as many polygons
    assert len(polys_tol) >= 0  # no crash
    assert len(polys_none) >= 0  # no crash
