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


# ---------------------------------------------------------------------------
# Boundary instrumentation: the request bbox must be WGS84 degrees (not metres),
# and the warp must announce its coverage so a "200 but blank" is never silent.
# ---------------------------------------------------------------------------

def _make_flat_4326(path, value=-40.0):
    """A constant-bathymetry EPSG:4326 DEM — the low-relief GEBCO case."""
    h = w = 40
    elev = np.full((h, w), value, dtype="float32")
    transform = _affine_from_bounds(_LON0, _LAT0, _LON1, _LAT1, w, h)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1,
                       dtype="float32", crs="EPSG:4326", transform=transform,
                       nodata=-9999.0) as ds:
        ds.write(elev, 1)
    return str(path)


def test_fetch_logs_request_bbox_in_degrees_not_metres(tmp_path, caplog):
    """The request window fetch_dem hands the opener must read in WGS84 degrees
    (~4–5 / 54–55), proving the project-CRS extent is reprojected before the
    request — the Step-3 'app sends metres' suspect, pinned by a test."""
    src_path = _make_fixture_4326(tmp_path / "src.tif")

    class _Ctx:
        def __enter__(self): self._ds = rasterio.open(src_path); return self._ds
        def __exit__(self, *a): self._ds.close()

    seen = {}

    def opener(source_key, bounds_ll, api_key):
        seen["bbox"] = bounds_ll
        return _Ctx()

    dest = tmp_path / "dem" / "elevation.tif"
    with caplog.at_level("INFO", logger="section_tool.core.dem"):
        D.fetch_dem("gebco", _proj_bounds(), _PROJ, str(dest), opener=opener)
    w, s, e, n = seen["bbox"]
    assert 3.0 < w < 6.0 and 3.0 < e < 6.0              # degrees, not 4e5 metres
    assert 53.0 < s < 56.0 and 53.0 < n < 56.0
    assert "DEM request" in caplog.text and "WGS84" in caplog.text
    assert "DEM warp" in caplog.text                    # warp boundary announced


def test_auto_vert_exag_lifts_low_relief_and_clamps():
    """Gentle relief over coarse pixels gets exaggerated into the marine band;
    a flat grid falls back to the documented constant; the result is recorded."""
    gentle = np.full((20, 20), -40.0, dtype="float32")
    gentle[:, 10:] = -36.0                                  # 4 m of relief
    ve = D.auto_vert_exag(gentle, dx=300.0, dy=300.0)
    assert 1.0 < ve <= 40.0                                # lifted, capped at hi
    # A perfectly flat grid can't be shaded — documented fallback, not a blow-up.
    flat = np.full((20, 20), -40.0, dtype="float32")
    assert D.auto_vert_exag(flat, dx=300.0, dy=300.0) == D.DEFAULT_MARINE_VERT_EXAG
    # Steep land needs no lift: clamped down to 1.0, never de-exaggerated below it.
    steep = np.tile(np.linspace(0, 3000, 20), (20, 1)).astype("float32")
    assert D.auto_vert_exag(steep, dx=30.0, dy=30.0) == pytest.approx(1.0)


def test_planar_bathymetry_tint_carries_depth_when_relief_flat():
    """A tilted plane has constant gradient, so its hillshade is a single flat
    value at ANY vert_exag (the live F3 blank — vert_exag can't fix a plane). The
    elevation tint must vary across the slope so depth still reads as colour."""
    h = w = 40
    _yy, xx = np.mgrid[0:h, 0:w]
    plane = (-50.0 + xx * (20.0 / w)).astype("float32")    # smooth -50 -> -30 m
    dx = dy = 100.0
    ve = D.auto_vert_exag(plane, dx=dx, dy=dy)
    grey = D.hillshade(plane, dx=dx, dy=dy, vert_exag=ve)
    assert float(grey.max() - grey.min()) < 1e-3           # relief shading is flat
    rgb = D.shaded_relief(plane, dx=dx, dy=dy, vert_exag=ve)
    assert rgb.shape[:2] == plane.shape and rgb.shape[2] == 4
    assert float(rgb[..., :3].std()) > 0.05                # tint varies = depth
    assert not np.allclose(rgb[:, 0, :3], rgb[:, -1, :3], atol=0.05)
    # Alpha must be fully opaque — a zero-alpha RGBA would draw as background
    # with no error (the invisible-DEM suspect).
    assert float(rgb[..., 3].min()) >= 0.99


def test_warp_flags_constant_surface(tmp_path, caplog):
    """An all-equal warp result (elevation lost / flat seabed) is logged loudly —
    it is the silent-blank break point, computed on finite cells before fill."""
    src_path = _make_flat_4326(tmp_path / "src.tif", value=-40.0)
    with caplog.at_level("WARNING", logger="section_tool.core.dem"):
        with rasterio.open(src_path) as src:
            D.reproject_to_project(src, _proj_bounds(), _PROJ)
    assert "constant surface" in caplog.text


# ---------------------------------------------------------------------------
# Download hardening: a flaky short read must be retried/validated, never passed
# half-decoded into the warp (the real GEBCO blank — transient read timeouts).
# ---------------------------------------------------------------------------

def _valid_tiff_bytes():
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds
    with MemoryFile() as mf:
        with mf.open(driver="GTiff", height=8, width=8, count=1, dtype="int16",
                     crs="EPSG:4326", transform=from_bounds(4, 54, 5, 55, 8, 8),
                     compress="lzw", nodata=-32767) as ds:
            ds.write(np.full((8, 8), -40, "int16"), 1)
        return mf.read()


class _Resp:
    def __init__(self, body, clen="__match__"):
        self._body = body
        self.headers = {} if clen is None else {
            "Content-Length": str(len(body) if clen == "__match__" else clen)}

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _opener_from(sequence):
    """A fake urlopen yielding each item in turn (Exception → raised, else _Resp)."""
    state = {"n": 0}

    def opener(url, timeout=None):
        item = sequence[min(state["n"], len(sequence) - 1)]
        state["n"] += 1
        if isinstance(item, Exception):
            raise item
        return item

    opener.state = state
    return opener


def test_download_retries_transient_then_writes_real_file(tmp_path):
    good = _valid_tiff_bytes()
    op = _opener_from([TimeoutError("read timed out"), _Resp(good)])
    dest = str(tmp_path / "dem" / "source.tif")
    out = D.download_validated_tiff("http://x", dest, opener=op, sleep=lambda *_: None)
    assert out == dest and op.state["n"] == 2          # retried once, then OK
    with rasterio.open(out) as ds:                     # the warp will open this same file
        assert ds.read(1).shape == (8, 8)


def test_download_rejects_error_body_with_message(tmp_path):
    op = _opener_from([_Resp(b"<html>API rate limit exceeded</html>", clen=None)])
    with pytest.raises(RuntimeError, match="non-GeoTIFF"):
        D.download_validated_tiff("http://x", str(tmp_path / "s.tif"),
                                  opener=op, sleep=lambda *_: None)
    assert op.state["n"] == 1                          # no retry on a permanent error


def test_download_retries_truncated_body_then_fails(tmp_path):
    good = _valid_tiff_bytes()
    truncated = good[:len(good) // 2]                  # valid magic, short body
    op = _opener_from([_Resp(truncated, clen=len(good))])   # declared > delivered
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        D.download_validated_tiff("http://x", str(tmp_path / "s.tif"),
                                  opener=op, retries=3, sleep=lambda *_: None)
    assert op.state["n"] == 3                          # all three attempts made


def test_download_corrupt_tiff_caught_by_real_file_decode(tmp_path):
    # TIFF magic present and Content-Length matches, so only decoding the written
    # file can reject it — the "header parses, tiles won't" failure that otherwise
    # blew up in the warp.
    body = b"II*\x00" + b"\x00" * 60                   # magic, but not a real TIFF
    op = _opener_from([_Resp(body, clen="__match__")])
    with pytest.raises(RuntimeError, match="failed after"):
        D.download_validated_tiff("http://x", str(tmp_path / "s.tif"),
                                  opener=op, retries=2, sleep=lambda *_: None)
    assert op.state["n"] == 2


def test_dem_cmap_registry():
    assert D.DEM_CMAP_ORDER[0] == "terrain"             # bathy default leads
    assert {"gray", "viridis", "gist_earth", "ocean"} <= set(D.DEM_CMAP_ORDER)
    assert all(D.is_dem_cmap(k) for k in D.DEM_CMAP_ORDER)
    assert not D.is_dem_cmap("not-a-cmap")
    # Grayscale is honestly an elevation ramp, never mislabelled a slope hillshade.
    assert "ramp" in D.DEM_CMAPS["gray"].lower()
    assert "hillshade" not in D.DEM_CMAPS["gray"].lower()


def test_source_registry():
    assert D.DEM_SOURCE_ORDER == ("copernicus", "gebco", "eudtm")
    assert D.DEM_SOURCES["copernicus"].needs_key is False
    assert D.DEM_SOURCES["gebco"].needs_key is True
    assert "DSM" in D.DEM_SOURCES["copernicus"].note
