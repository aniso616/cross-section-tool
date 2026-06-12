"""DEM core: windowed reproject to the project CRS (datum round-trip pinned),
hillshade shape/dtype, ocean-absent fill flag, provenance round-trip, and the
fetch orchestration with an injected opener (no network)."""
from __future__ import annotations

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_bounds as _affine_from_bounds  # noqa: E402

from section_tool.core import dem as D  # noqa: E402
from section_tool.core.crs import transform_points  # noqa: E402

_PROJ = 32631  # UTM 31N (F3)
# A lon/lat box inside UTM zone 31N (lon 0–6 E).
_LON0, _LON1, _LAT0, _LAT1 = 4.0, 5.0, 54.0, 55.0


def _make_fixture_4326(path, nodata_corner=False):
    """A 40×40 EPSG:4326 DEM whose elevation ramps with longitude (predictable)."""
    h = w = 40
    lons = np.linspace(_LON0, _LON1, w)
    elev = np.tile((lons - _LON0) * 1000.0, (h, 1)).astype("float32")  # 0..1000 m W→E
    nodata = -9999.0
    if nodata_corner:
        elev[:10, :10] = nodata                         # an "absent ocean" block
    transform = _affine_from_bounds(_LON0, _LAT0, _LON1, _LAT1, w, h)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1,
                       dtype="float32", crs="EPSG:4326", transform=transform,
                       nodata=nodata) as ds:
        ds.write(elev, 1)
    return str(path)


def _proj_bounds():
    xs = [_LON0, _LON1, _LON0, _LON1]
    ys = [_LAT0, _LAT0, _LAT1, _LAT1]
    ex, ny = transform_points(xs, ys, 4326, _PROJ)
    return (float(min(ex)), float(max(ex)), float(min(ny)), float(max(ny)))


# ---------------------------------------------------------------------------
# Reproject to project CRS — with a datum round-trip so a slip can't hide.
# ---------------------------------------------------------------------------

def test_reproject_lands_in_project_crs(tmp_path):
    src_path = _make_fixture_4326(tmp_path / "src.tif")
    with rasterio.open(src_path) as src:
        dem = D.reproject_to_project(src, _proj_bounds(), _PROJ)
    assert dem.epsg == _PROJ
    assert dem.data.ndim == 2 and dem.data.size > 0
    left, right, bottom, top = dem.extent
    assert 1e5 < left < 9e5 and 5e6 < bottom < 7e6        # UTM 31N range, not lon/lat


def test_datum_roundtrip_value_at_known_point(tmp_path):
    """A known lon/lat samples the same elevation after warping to UTM."""
    src_path = _make_fixture_4326(tmp_path / "src.tif")
    with rasterio.open(src_path) as src:
        dem = D.reproject_to_project(src, _proj_bounds(), _PROJ)
    # Pick lon 4.5 (ramp ⇒ ~500 m); convert to UTM; sample the warped grid there.
    lon, lat = 4.5, 54.5
    ex, ny = transform_points([lon], [lat], 4326, _PROJ)
    px = (ex[0] - dem.transform.c) / dem.transform.a
    py = (ny[0] - dem.transform.f) / dem.transform.e
    val = dem.data[int(round(py)), int(round(px))]
    assert val == pytest.approx(500.0, abs=40.0)          # ramp value, interp tol


# ---------------------------------------------------------------------------
# Hillshade
# ---------------------------------------------------------------------------

def test_hillshade_shape_and_range(tmp_path):
    src_path = _make_fixture_4326(tmp_path / "src.tif")
    with rasterio.open(src_path) as src:
        dem = D.reproject_to_project(src, _proj_bounds(), _PROJ)
    dx, dy = D.pixel_size(dem)
    hs = D.hillshade(dem.data, dx=dx, dy=dy)
    assert hs.shape == dem.data.shape
    assert hs.dtype.kind == "f"
    assert float(hs.min()) >= 0.0 and float(hs.max()) <= 1.0


# ---------------------------------------------------------------------------
# Ocean-absent handling: nodata → fill + flag.
# ---------------------------------------------------------------------------

def test_ocean_absent_filled_and_flagged(tmp_path):
    src_path = _make_fixture_4326(tmp_path / "src.tif", nodata_corner=True)
    with rasterio.open(src_path) as src:
        dem = D.reproject_to_project(src, _proj_bounds(), _PROJ, fill_value=0.0)
    assert dem.ocean_filled is True
    assert np.all(np.isfinite(dem.data))                  # no NaN leaks through
    assert float(dem.data.min()) == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Provenance round-trip through the stored GeoTIFF.
# ---------------------------------------------------------------------------

def test_geotiff_provenance_roundtrip(tmp_path):
    src_path = _make_fixture_4326(tmp_path / "src.tif")
    with rasterio.open(src_path) as src:
        dem = D.reproject_to_project(src, _proj_bounds(), _PROJ)
    prov = D.make_provenance("copernicus", _proj_bounds(), _PROJ, dem.ocean_filled)
    dest = tmp_path / "dem" / "elevation.tif"
    D.write_dem_geotiff(dem, str(dest), prov)
    data, extent, epsg, prov2 = D.load_dem_geotiff(str(dest))
    assert epsg == _PROJ
    assert prov2["source"] == "copernicus" and prov2["dataset"] == "COP30"
    assert "fetch_date" in prov2 and prov2["crs"] == f"EPSG:{_PROJ}"
    assert np.allclose(data, dem.data, atol=1e-3)


# ---------------------------------------------------------------------------
# fetch_dem orchestration with an injected opener (no network).
# ---------------------------------------------------------------------------

def test_fetch_dem_with_injected_opener(tmp_path):
    src_path = _make_fixture_4326(tmp_path / "src.tif")

    class _Ctx:
        def __enter__(self): self._ds = rasterio.open(src_path); return self._ds
        def __exit__(self, *a): self._ds.close()

    calls = {"n": 0}

    def opener(source_key, bounds_ll, api_key):
        calls["n"] += 1
        assert source_key == "copernicus"
        assert len(bounds_ll) == 4                        # (w,s,e,n) lon/lat
        return _Ctx()

    dest = tmp_path / "dem" / "elevation.tif"
    res = D.fetch_dem("copernicus", _proj_bounds(), _PROJ, str(dest), opener=opener)
    assert calls["n"] == 1
    assert res.path == str(dest)
    assert res.provenance["source"] == "copernicus"
    assert dest.exists()


def test_source_registry():
    assert D.DEM_SOURCE_ORDER == ("copernicus", "gebco", "eudtm")
    assert D.DEM_SOURCES["copernicus"].needs_key is False
    assert D.DEM_SOURCES["gebco"].needs_key is True
    assert "DSM" in D.DEM_SOURCES["copernicus"].note
