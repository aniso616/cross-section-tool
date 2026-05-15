"""Unit tests for SectionTopology auto-polygon generation.

Verifies that get_all_faces() always produces the correct closed faces given
various combinations of horizons, faults, and boundary conditions.
"""
from __future__ import annotations

import pytest
from section_tool.core.topology import SectionTopology

L = 10_000.0  # section length (m)
D = 5_000.0   # max depth (m)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_topo(*user_lines) -> SectionTopology:
    """Create a topology with the given (name, type, coords) tuples."""
    t = SectionTopology("S1", section_length=L, max_depth=D)
    for name, ltype, coords in user_lines:
        t.update_line(name, ltype, coords)
    return t


def areas(topo) -> list[float]:
    return sorted(p.area for p in topo.get_all_faces())


# ---------------------------------------------------------------------------
# 1. No horizons → 1 polygon covering the whole section
# ---------------------------------------------------------------------------

def test_no_horizons_yields_one_face():
    t = SectionTopology("S1", section_length=L, max_depth=D)
    faces = t.get_all_faces()
    assert len(faces) == 1
    assert faces[0].area == pytest.approx(L * D, rel=1e-6)


def test_no_horizons_face_covers_full_bbox():
    t = SectionTopology("S1", section_length=L, max_depth=D)
    face = t.get_all_faces()[0]
    minx, miny, maxx, maxy = face.bounds
    assert minx == pytest.approx(0.0)
    assert miny == pytest.approx(0.0)
    assert maxx == pytest.approx(L)
    assert maxy == pytest.approx(D)


# ---------------------------------------------------------------------------
# 2. Single horizon → 2 polygons
# ---------------------------------------------------------------------------

def test_single_full_horizon_yields_two_faces():
    t = make_topo(
        ("h0", "horizon", [(0, 2000), (L, 2000)])
    )
    faces = t.get_all_faces()
    assert len(faces) == 2
    total = sum(f.area for f in faces)
    assert total == pytest.approx(L * D, rel=1e-6)


def test_single_horizon_areas_sum_to_bbox():
    t = make_topo(
        ("h0", "horizon", [(1000, 1500), (9000, 1500)])  # partial — extended
    )
    faces = t.get_all_faces()
    assert len(faces) == 2
    total = sum(f.area for f in faces)
    assert total == pytest.approx(L * D, rel=1e-3)


# ---------------------------------------------------------------------------
# 3. Two horizontal horizons across full section → 3 polygons
# ---------------------------------------------------------------------------

def test_two_full_horizons_yield_three_faces():
    t = make_topo(
        ("h0", "horizon", [(0, 1000), (L, 1000)]),
        ("h1", "horizon", [(0, 3000), (L, 3000)]),
    )
    faces = t.get_all_faces()
    assert len(faces) == 3
    total = sum(f.area for f in faces)
    assert total == pytest.approx(L * D, rel=1e-6)


def test_two_horizons_correct_areas():
    t = make_topo(
        ("h0", "horizon", [(0, 1000), (L, 1000)]),
        ("h1", "horizon", [(0, 3000), (L, 3000)]),
    )
    a = areas(t)
    # Top band 0→1000, mid band 1000→3000, bottom band 3000→5000
    assert a[0] == pytest.approx(L * 1000, rel=1e-6)  # smallest: top band
    assert a[1] == pytest.approx(L * 2000, rel=1e-6)
    assert a[2] == pytest.approx(L * 2000, rel=1e-6)


# ---------------------------------------------------------------------------
# 4. Horizon that doesn't span full width — extended to edges
# ---------------------------------------------------------------------------

def test_partial_horizon_extended_gives_two_faces():
    """Picks only cover 20%–80% of the section; topology must extend to edges."""
    t = make_topo(
        ("h0", "horizon", [(2000, 1500), (8000, 1500)])
    )
    faces = t.get_all_faces()
    assert len(faces) == 2
    total = sum(f.area for f in faces)
    # After extension, horizon is at y=1500 across full width
    assert total == pytest.approx(L * D, rel=1e-3)


def test_partial_horizon_areas():
    t = make_topo(
        ("h0", "horizon", [(2000, 2000), (8000, 2000)])
    )
    a = areas(t)
    # Horizon extends to y=2000 across full width → top L*2000, bottom L*3000
    assert a[0] == pytest.approx(L * 2000, rel=1e-3)
    assert a[1] == pytest.approx(L * 3000, rel=1e-3)


def test_partial_tilted_horizon_two_faces():
    """Tilted horizon: extrapolation to edges must still form two faces."""
    # Horizon goes from (2000, 1000) to (8000, 2000) — tilted 1/6 rise
    t = make_topo(
        ("h0", "horizon", [(2000, 1000), (8000, 2000)])
    )
    faces = t.get_all_faces()
    assert len(faces) == 2
    total = sum(f.area for f in faces)
    assert total == pytest.approx(L * D, rel=1e-3)


# ---------------------------------------------------------------------------
# 5. Two horizons + one diagonal fault → more than 3 polygons
# ---------------------------------------------------------------------------

def test_two_horizons_one_vertical_fault():
    t = make_topo(
        ("h0", "horizon", [(0, 1000), (L, 1000)]),
        ("h1", "horizon", [(0, 3000), (L, 3000)]),
        ("f0", "fault",   [(5000, 0), (5000, D)]),   # vertical at mid-section
    )
    faces = t.get_all_faces()
    # 4 quadrants split by fault + 2 bands above/below horizons = 6 polygons
    assert len(faces) >= 5
    total = sum(f.area for f in faces)
    assert total == pytest.approx(L * D, rel=1e-3)


def test_two_horizons_diagonal_fault():
    t = make_topo(
        ("h0", "horizon", [(0, 1000), (L, 1000)]),
        ("h1", "horizon", [(0, 3000), (L, 3000)]),
        ("f0", "fault",   [(2000, 0), (8000, D)]),   # diagonal fault
    )
    faces = t.get_all_faces()
    assert len(faces) >= 5
    total = sum(f.area for f in faces)
    assert total == pytest.approx(L * D, rel=1e-3)


# ---------------------------------------------------------------------------
# 6. Areas always sum to section bounding box
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_horizons,depths", [
    (0, []),
    (1, [2500.0]),
    (2, [1000.0, 3000.0]),
    (3, [1000.0, 2000.0, 4000.0]),
])
def test_areas_sum_to_bbox(n_horizons, depths):
    lines = [
        (f"h{i}", "horizon", [(0.0, d), (L, d)])
        for i, d in enumerate(depths)
    ]
    t = make_topo(*lines)
    total = sum(f.area for f in t.get_all_faces())
    assert total == pytest.approx(L * D, rel=1e-6)


# ---------------------------------------------------------------------------
# 7. Line extension correctness
# ---------------------------------------------------------------------------

def test_extension_horizontal_line():
    t = SectionTopology("S1", section_length=L, max_depth=D)
    extended = t._extend_to_edges([(2000, 1000), (8000, 1000)])
    assert extended[0][0] == pytest.approx(0.0)     # left edge
    assert extended[-1][0] == pytest.approx(L)      # right edge
    assert extended[0][1] == pytest.approx(1000.0)  # flat extrapolation
    assert extended[-1][1] == pytest.approx(1000.0)


def test_extension_tilted_line():
    # Rise from (2000, 1000) to (8000, 2000): slope = 1000/6000 ≈ 0.1667
    t = SectionTopology("S1", section_length=L, max_depth=D)
    extended = t._extend_to_edges([(2000, 1000), (8000, 2000)])
    # Left at x=0: z = 1000 + (2000-1000)*(0-2000)/(8000-2000) = 1000 - 333.3 ≈ 666.7
    assert extended[0][0] == pytest.approx(0.0)
    assert extended[0][1] == pytest.approx(1000 - 1000 * 2000 / 6000, abs=1.0)
    # Right at x=10000: z = 1000 + 1000*(10000-2000)/6000 ≈ 2333.3
    assert extended[-1][0] == pytest.approx(L)
    assert extended[-1][1] == pytest.approx(1000 + 1000 * 8000 / 6000, abs=1.0)


def test_extension_clamps_to_depth_bounds():
    """Steep extrapolation must not go below max_depth or above 0."""
    t = SectionTopology("S1", section_length=L, max_depth=D)
    # Very steep line — extrapolation would go way out of bounds
    extended = t._extend_to_edges([(4000, 100), (6000, 4900)])
    for _, z in extended:
        assert 0.0 <= z <= D


def test_extension_already_at_edges():
    """Line already reaches both edges — no points should be added."""
    t = SectionTopology("S1", section_length=L, max_depth=D)
    coords = [(0.0, 1000.0), (L, 1000.0)]
    extended = t._extend_to_edges(coords)
    assert extended[0][0] == pytest.approx(0.0)
    assert extended[-1][0] == pytest.approx(L)
    # No extra points prepended or appended
    assert len(extended) == 2


def test_extension_single_point_held_constant():
    """Single-pick horizon must be extended horizontally."""
    t = SectionTopology("S1", section_length=L, max_depth=D)
    extended = t._extend_to_edges([(5000, 2000)])
    assert extended[0][0] == pytest.approx(0.0)
    assert extended[-1][0] == pytest.approx(L)
    # Constant depth
    assert all(z == pytest.approx(2000.0) for _, z in extended)


# ---------------------------------------------------------------------------
# 8. Bounds update preserves faces
# ---------------------------------------------------------------------------

def test_bounds_update_and_faces():
    t = make_topo(
        ("h0", "horizon", [(0, 1000), (L, 1000)]),
    )
    t.update_bounds(L * 2, D * 2)
    faces = t.get_all_faces()
    assert len(faces) == 2
    total = sum(f.area for f in faces)
    assert total == pytest.approx(L * 2 * D * 2, rel=1e-3)
