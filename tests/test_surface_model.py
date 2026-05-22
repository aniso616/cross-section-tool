"""Tests for the Surface data model and readers."""
import numpy as np
import pytest
from section_tool.core.surfaces import Surface, GridInfo, detect_grid


def test_surface_from_points():
    pts = np.array([[0,0,100],[1000,0,110],[0,1000,90],[1000,1000,105]], dtype=float)
    s = Surface(name="test", points=pts)
    assert s.n_points == 4
    assert s.bounds() == (0.0, 0.0, 1000.0, 1000.0)


def test_surface_z_range():
    pts = np.array([[0,0,90],[1000,0,110],[0,1000,100]], dtype=float)
    assert Surface(name="t", points=pts).z_range() == (90.0, 110.0)


def test_z_at_interpolation():
    pts = np.array([[0,0,100],[1000,0,110],[0,1000,90],[1000,1000,105]], dtype=float)
    s = Surface(name="test", points=pts)
    z = s.sample(500, 500)
    assert not np.isnan(z)
    assert 90 <= z <= 110


def test_z_outside_bounds_nan():
    pts = np.array([[0,0,100],[100,0,100],[0,100,100]], dtype=float)
    assert np.isnan(Surface(name="t", points=pts).sample(1000, 1000))


def test_display_color_hex():
    s = Surface(name="t", points=np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=float),
                color=(255, 128, 0))
    assert s.display_color == "#ff8000"


def test_extent_backward_compat():
    pts = np.array([[0,0,0],[1000,0,0],[0,2000,0]], dtype=float)
    xmn, xmx, ymn, ymx = Surface(name="t", points=pts).extent()
    assert (xmn, xmx, ymn, ymx) == (0.0, 1000.0, 0.0, 2000.0)


def test_intersect_section():
    xs = np.linspace(0, 1000, 21); ys = np.linspace(0, 1000, 21)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel(), (X / 10.0).ravel()])
    surf = Surface(name="plane", points=pts)

    class MockSection:
        def total_length(self): return 1000.0
        def section_to_map(self, d): return (d, 500.0)
        depth_domain = "depth"

    distances, zs = surf.intersect_section(MockSection(), n_samples=11)
    valid = ~np.isnan(zs); assert valid.sum() >= 5
    assert zs[valid][-1] > zs[valid][0]


def test_sample_many():
    pts = np.array([[0,0,10],[1,0,20],[0,1,30],[1,1,40]], dtype=float)
    zs = Surface(name="t", points=pts).sample_many(np.array([0.5, 0.0]), np.array([0.5, 0.0]))
    assert not np.isnan(zs[0])
    assert abs(zs[1] - 10.0) < 1e-6


def test_detect_regular_grid():
    xs = np.arange(0, 1000, 100); ys = np.arange(0, 1000, 100)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    g = detect_grid(pts)
    assert g is not None and g.nx == 10 and g.ny == 10


def test_detect_irregular_none():
    rng = np.random.default_rng(0)
    assert detect_grid(rng.uniform(0, 1000, (100, 3))) is None


# ------------------------------------------------------------------
# XYZReader
# ------------------------------------------------------------------

def test_xyz_reader_basic(tmp_path):
    from section_tool.io.surface_readers.xyz_reader import XYZReader
    f = tmp_path / "t.xyz"
    f.write_text("# comment\n100.0 200.0 50.0\n101.0 200.0 51.0\n100.0 201.0 50.5\n")
    reader = XYZReader()
    assert reader.can_read(str(f))
    s = reader.read(str(f))
    assert s.n_points == 3 and s.z_range() == (50.0, 51.0)


def test_xyz_reader_csv(tmp_path):
    from section_tool.io.surface_readers.xyz_reader import XYZReader
    f = tmp_path / "t.csv"
    f.write_text("x,y,z\n1,2,3\n4,5,6\n7,8,9\n")
    assert XYZReader().read(str(f)).n_points == 3


def test_xyz_strips_null(tmp_path):
    from section_tool.io.surface_readers.xyz_reader import XYZReader
    f = tmp_path / "t.xyz"
    f.write_text("0 0 100\n1 0 -999\n0 1 102\n1 1 101\n")
    s = XYZReader().read(str(f))
    assert s.n_points == 3
    assert -999.0 not in s.points[:, 2]


def test_read_surface_dispatch(tmp_path):
    from section_tool.io.surface_readers import read_surface
    f = tmp_path / "horizon.xyz"
    f.write_text("0 0 500\n1000 0 510\n0 1000 490\n1000 1000 505\n")
    s = read_surface(str(f))
    assert s.n_points == 4 and s.source_format == "XYZ ASCII"


def test_read_surface_unknown_raises(tmp_path):
    from section_tool.io.surface_readers import read_surface
    f = tmp_path / "file.zzz"
    f.write_text("not a surface\n")
    with pytest.raises(ValueError, match="No surface reader"):
        read_surface(str(f))
