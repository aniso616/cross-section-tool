"""Tests for section_tool.core.surface_builder."""
from __future__ import annotations

import numpy as np
import pytest

from section_tool.core.aoi import AOI
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick, Surface
from section_tool.core.surface_builder import (
    build_surface_from_picks,
    collect_horizon_points_map_space,
    section_distance_to_map_xy,
)


# ---------------------------------------------------------------------------
# Minimal stub objects
# ---------------------------------------------------------------------------

class _FakeProject:
    """Minimal project stub for testing surface_builder."""

    def __init__(self, sections=None, horizon_picks=None, crs_epsg=32632):
        self.sections      = sections or []
        self.horizon_picks = horizon_picks or []
        self.crs_epsg      = crs_epsg


def _straight_section(x0=0.0, y0=0.0, x1=1000.0, y1=0.0, name="S1") -> Section:
    """Two-node east-west section of length 1000 m."""
    return Section([(x0, y0), (x1, y1)], name=name)


def _make_pick(
    name: str,
    section: Section,
    n_points: int = 6,
    depth_fn=None,
    section_name: str | None = None,
) -> HorizonPick:
    """Create a HorizonPick with *n_points* evenly spaced along *section*.

    Uses stored map coordinates so surface_builder can convert without
    relying on section-name lookup.
    """
    total    = section.total_length()
    dists    = np.linspace(0.0, total, n_points)
    if depth_fn is None:
        depths = np.linspace(1000.0, 2000.0, n_points)
    else:
        depths = np.array([depth_fn(d) for d in dists], dtype=float)

    sname = section_name if section_name is not None else section.name
    snames = [sname] * n_points

    # Pre-compute and store map coords
    map_x = np.array([section.section_to_map(d)[0] for d in dists])
    map_y = np.array([section.section_to_map(d)[1] for d in dists])

    return HorizonPick(
        dists, depths,
        name=name,
        section_names=snames,
        map_x=map_x,
        map_y=map_y,
    )


# ---------------------------------------------------------------------------
# test_section_distance_to_map_xy_straight
# ---------------------------------------------------------------------------

def test_section_distance_to_map_xy_straight():
    """Midpoint of a straight east-west 1000 m section should map to (500, 0)."""
    sec = _straight_section(x0=0.0, y0=0.0, x1=1000.0, y1=0.0)
    x, y = section_distance_to_map_xy(sec, 500.0)
    assert abs(x - 500.0) < 1e-6
    assert abs(y - 0.0) < 1e-6


def test_section_distance_to_map_xy_start():
    x, y = section_distance_to_map_xy(_straight_section(), 0.0)
    assert abs(x) < 1e-6 and abs(y) < 1e-6


def test_section_distance_to_map_xy_end():
    x, y = section_distance_to_map_xy(_straight_section(), 1000.0)
    assert abs(x - 1000.0) < 1e-6 and abs(y) < 1e-6


# ---------------------------------------------------------------------------
# test_collect_horizon_points_empty_if_no_picks
# ---------------------------------------------------------------------------

def test_collect_horizon_points_empty_if_no_picks():
    """Project with no horizon_picks → empty (0,3) array."""
    proj = _FakeProject()
    pts  = collect_horizon_points_map_space(proj, "H1")
    assert isinstance(pts, np.ndarray)
    assert pts.shape == (0, 3)


# ---------------------------------------------------------------------------
# test_collect_horizon_points_wrong_name
# ---------------------------------------------------------------------------

def test_collect_horizon_points_wrong_name():
    """Picks exist but wrong name → empty array."""
    sec  = _straight_section()
    pick = _make_pick("H1", sec)
    proj = _FakeProject(sections=[sec], horizon_picks=[pick])
    pts  = collect_horizon_points_map_space(proj, "DOES_NOT_EXIST")
    assert pts.shape == (0, 3)


# ---------------------------------------------------------------------------
# test_collect_horizon_points_correct_name
# ---------------------------------------------------------------------------

def test_collect_horizon_points_correct_name():
    """6 picks on one section → 6 map-space rows."""
    sec  = _straight_section()
    pick = _make_pick("H1", sec, n_points=6)
    proj = _FakeProject(sections=[sec], horizon_picks=[pick])
    pts  = collect_horizon_points_map_space(proj, "H1")
    assert pts.shape == (6, 3)
    # X coordinates should span [0, 1000]
    assert pts[:, 0].min() == pytest.approx(0.0, abs=1.0)
    assert pts[:, 0].max() == pytest.approx(1000.0, abs=1.0)


# ---------------------------------------------------------------------------
# test_build_surface_requires_three_points
# ---------------------------------------------------------------------------

def test_build_surface_requires_three_points_zero():
    """No picks at all → ValueError."""
    proj = _FakeProject()
    with pytest.raises(ValueError, match="3"):
        build_surface_from_picks(proj, "H1")


def test_build_surface_requires_three_points_two():
    """Only 2 pick points → ValueError."""
    sec  = _straight_section()
    pick = _make_pick("H1", sec, n_points=2)
    proj = _FakeProject(sections=[sec], horizon_picks=[pick])
    with pytest.raises(ValueError, match="3"):
        build_surface_from_picks(proj, "H1")


# ---------------------------------------------------------------------------
# test_build_surface_from_single_section_picks
# ---------------------------------------------------------------------------

def test_build_surface_from_single_section_picks():
    """Single section with 6 picks → Surface with non-empty grid."""
    sec  = _straight_section(x0=0.0, y0=0.0, x1=1000.0, y1=0.0)
    pick = _make_pick("H1", sec, n_points=6)
    proj = _FakeProject(sections=[sec], horizon_picks=[pick])

    surf = build_surface_from_picks(proj, "H1", grid_resolution=100.0)

    assert isinstance(surf, Surface)
    assert surf.name == "H1"
    assert surf.n_points > 0
    # Some finite Z values must exist
    valid = surf.points[np.isfinite(surf.points[:, 2])]
    assert len(valid) > 0


# ---------------------------------------------------------------------------
# test_build_surface_clips_to_aoi
# ---------------------------------------------------------------------------

def test_build_surface_clips_to_aoi():
    """AOI covering only left half of the grid → right half cells are NaN."""
    sec  = _straight_section(x0=0.0, y0=0.0, x1=1000.0, y1=0.0)
    pick = _make_pick("H1", sec, n_points=10)
    proj = _FakeProject(sections=[sec], horizon_picks=[pick])

    # AOI: only left half (x < 500)
    aoi = AOI.from_rectangle(
        x_min=-50.0, x_max=500.0,
        y_min=-200.0, y_max=200.0,
        crs_epsg=32632,
    )

    surf = build_surface_from_picks(
        proj, "H1", grid_resolution=50.0, aoi=aoi
    )

    # Points inside AOI should have finite Z; outside should have NaN
    inside_pts  = surf.points[surf.points[:, 0] <= 500.0]
    outside_pts = surf.points[surf.points[:, 0] > 550.0]

    assert np.any(np.isfinite(inside_pts[:, 2])), "inside AOI should have valid Z"
    if len(outside_pts) > 0:
        assert np.all(~np.isfinite(outside_pts[:, 2])), "outside AOI should be NaN"


# ---------------------------------------------------------------------------
# test_build_surface_multi_section
# ---------------------------------------------------------------------------

def test_build_surface_multi_section():
    """Two crossing sections both contribute to the surface."""
    # EW section at y=0
    sec1 = _straight_section(x0=0.0, y0=0.0,   x1=1000.0, y1=0.0,   name="EW")
    # NS section at x=500
    sec2 = Section([(500.0, -500.0), (500.0, 500.0)], name="NS")

    pick1 = _make_pick("H1", sec1, n_points=6)
    pick2 = _make_pick("H1", sec2, n_points=6, section_name="NS")
    # Second pick: set different depths to check both contribute
    pick2._depths[:] = np.linspace(1500.0, 2500.0, 6)

    proj = _FakeProject(sections=[sec1, sec2], horizon_picks=[pick1, pick2])

    pts  = collect_horizon_points_map_space(proj, "H1")
    assert len(pts) == 12, f"Expected 12 points (6+6), got {len(pts)}"

    surf = build_surface_from_picks(proj, "H1", grid_resolution=100.0)
    assert isinstance(surf, Surface)
    valid = surf.points[np.isfinite(surf.points[:, 2])]
    assert len(valid) > 0


# ---------------------------------------------------------------------------
# test_build_surface_nearest_method
# ---------------------------------------------------------------------------

def test_build_surface_nearest_method():
    """method='nearest' should not crash and return a valid Surface."""
    sec  = _straight_section(x0=0.0, y0=0.0, x1=1000.0, y1=0.0)
    pick = _make_pick("H1", sec, n_points=6)
    proj = _FakeProject(sections=[sec], horizon_picks=[pick])

    surf = build_surface_from_picks(
        proj, "H1", grid_resolution=100.0, method="nearest"
    )

    assert isinstance(surf, Surface)
    valid = surf.points[np.isfinite(surf.points[:, 2])]
    assert len(valid) > 0


# ---------------------------------------------------------------------------
# test_build_surface_no_aoi_uses_data_bbox
# ---------------------------------------------------------------------------

def test_build_surface_no_aoi_uses_data_bbox():
    """Without AOI the grid extent covers the data plus a margin."""
    sec  = _straight_section(x0=0.0, y0=0.0, x1=1000.0, y1=0.0)
    pick = _make_pick("H1", sec, n_points=8)
    proj = _FakeProject(sections=[sec], horizon_picks=[pick])

    surf = build_surface_from_picks(proj, "H1", grid_resolution=100.0)

    # grid X should span beyond [0, 1000] (the 5% margin)
    xmin, ymin, xmax, ymax = surf.bounds()
    assert xmin <= 0.0
    assert xmax >= 1000.0
