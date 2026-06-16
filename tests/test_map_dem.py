"""DEM hillshade map layer: renders under the data at the right z-order, honours
the visibility toggle, keeps the map 1:1, and loads via an off-thread fetch with
an injected opener (no network)."""
from __future__ import annotations

import sys

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_bounds as _affine_from_bounds  # noqa: E402

from section_tool.app_state import AppState  # noqa: E402
from section_tool.core.section import Section  # noqa: E402
from section_tool.core import dem as D  # noqa: E402
from section_tool.core.crs import transform_points  # noqa: E402
from section_tool.views.map_view import MapView  # noqa: E402

_PROJ = 32631
_LON0, _LON1, _LAT0, _LAT1 = 4.0, 5.0, 54.0, 55.0


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


def _fixture_4326(path):
    h = w = 40
    lons = np.linspace(_LON0, _LON1, w)
    elev = np.tile((lons - _LON0) * 1000.0, (h, 1)).astype("float32")
    transform = _affine_from_bounds(_LON0, _LAT0, _LON1, _LAT1, w, h)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1,
                       dtype="float32", crs="EPSG:4326", transform=transform,
                       nodata=-9999.0) as ds:
        ds.write(elev, 1)
    return str(path)


def _proj_bounds():
    ex, ny = transform_points([_LON0, _LON1, _LON0, _LON1],
                              [_LAT0, _LAT0, _LAT1, _LAT1], 4326, _PROJ)
    return (float(min(ex)), float(max(ex)), float(min(ny)), float(max(ny)))


def _store_dem(tmp_path):
    """Make a project-CRS DEM GeoTIFF and return its path."""
    src = _fixture_4326(tmp_path / "src.tif")
    with rasterio.open(src) as ds:
        dem = D.reproject_to_project(ds, _proj_bounds(), _PROJ)
    dest = tmp_path / "dem" / "elevation.tif"
    D.write_dem_geotiff(dem, str(dest),
                        D.make_provenance("copernicus", _proj_bounds(), _PROJ,
                                          dem.ocean_filled))
    return str(dest)


def _map_with_section(qapp):
    state = AppState()
    state.project.crs_epsg = _PROJ
    mv = MapView(state)
    state.add_section(Section([(_proj_bounds()[0], _proj_bounds()[2]),
                               (_proj_bounds()[1], _proj_bounds()[3])],
                              name="L1", crs_epsg=_PROJ))
    state.set_active_section(state.project.sections[0])
    return state, mv


def test_hillshade_renders_under_data(qapp, tmp_path):
    state, mv = _map_with_section(qapp)
    assert mv._dem.load_geotiff(_store_dem(tmp_path))
    mv.render()
    imgs = mv._ax.get_images()
    assert imgs, "hillshade image should be present"
    assert any(-10 < im.get_zorder() < 0 for im in imgs)   # under the data
    # 1:1 aspect is preserved (regression, like the basemap/VE fixes).
    mv._ax.apply_aspect()
    assert mv._ax.get_aspect() == "equal" or float(mv._ax.get_aspect()) == pytest.approx(1.0)


def test_visibility_toggle(qapp, tmp_path):
    state, mv = _map_with_section(qapp)
    mv._dem.load_geotiff(_store_dem(tmp_path))
    mv.set_hillshade_visible(False)
    assert mv._ax.get_images() == []                       # nothing drawn when hidden
    mv.set_hillshade_visible(True)
    assert mv._ax.get_images()                              # back


def _fixture_gebco_4326(path):
    """Negative, gently-sloping bathymetry — the offshore (GEBCO) case the demo
    actually fetches. Unlike the positive 0..1000 m ramp, every cell is below sea
    level and the relief is low (−50 → −30 m), which is where the live blank showed.
    """
    h = w = 40
    lons = np.linspace(_LON0, _LON1, w)
    elev = np.tile(-50.0 + (lons - _LON0) * 20.0, (h, 1)).astype("float32")
    transform = _affine_from_bounds(_LON0, _LAT0, _LON1, _LAT1, w, h)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1,
                       dtype="float32", crs="EPSG:4326", transform=transform,
                       nodata=-9999.0) as ds:
        ds.write(elev, 1)
    return str(path)


def test_planar_seabed_renders_tinted_relief_not_blank(qapp, tmp_path, caplog):
    """The regression that would have caught the live F3 blank: a near-planar
    seabed (constant gradient) shades to ONE flat grey value, so pure hillshade
    is invisible at any vert_exag. The layer must render an elevation-TINTED
    relief that still varies across the slope (carries depth) — and flag that the
    grey relief alone is flat. (Pure-grey + vert_exag-only would NOT close this.)"""
    dest = tmp_path / "dem" / "elevation.tif"
    dest.parent.mkdir(parents=True)
    h = w = 48
    _yy, xx = np.mgrid[0:h, 0:w]
    plane = (-50.0 + xx * (20.0 / w)).astype("float32")    # smooth tilted seabed
    with rasterio.open(dest, "w", driver="GTiff", height=h, width=w, count=1,
                       dtype="float32", crs="EPSG:32631",
                       transform=_affine_from_bounds(600000, 6080000,
                                                     610000, 6090000, w, h),
                       nodata=None) as ds:
        ds.write(plane, 1)

    state, mv = _map_with_section(qapp)
    with caplog.at_level("WARNING", logger="section_tool.views.map_dem_layer"):
        assert mv._dem.load_geotiff(str(dest))

    # Pure grey relief is flat (the blank) and is flagged...
    assert float(mv._dem._hs.max() - mv._dem._hs.min()) < 1e-3
    assert "near-planar" in caplog.text
    assert mv._dem._vert_exag is not None and mv._dem._vert_exag > 1.0
    # ...but the tinted RGB carries depth: it varies across the slope.
    rgb = mv._dem._rgb
    assert rgb is not None and float(rgb[..., :3].std()) > 0.05
    assert not np.allclose(rgb[:, 0, :3], rgb[:, -1, :3], atol=0.05)
    # and it actually draws an image under the data.
    mv.render()
    imgs = mv._ax.get_images()
    assert imgs and any(-10 < im.get_zorder() < 0 for im in imgs)


def test_negative_bathymetry_warps_and_hillshades(qapp, tmp_path):
    """GEBCO-style negative elevations must warp + hillshade to a valid layer
    (not all-zeroed, not NaN, in [0,1]). Locks the offshore case on a fixture."""
    src = _fixture_gebco_4326(tmp_path / "src.tif")
    with rasterio.open(src) as ds:
        dem = D.reproject_to_project(ds, _proj_bounds(), _PROJ)
    assert float(np.nanmin(dem.data)) < -25.0          # bathymetry survived the warp
    dest = tmp_path / "dem" / "elevation.tif"
    D.write_dem_geotiff(dem, str(dest),
                        D.make_provenance("gebco", _proj_bounds(), _PROJ,
                                          dem.ocean_filled))
    state, mv = _map_with_section(qapp)
    ok, detail = mv._dem._load_geotiff_diagnosed(str(dest))
    assert ok and mv._dem.has_hillshade()
    hs = mv._dem._hs
    assert np.all(np.isfinite(hs)) and hs.min() >= 0.0 and hs.max() <= 1.0
    assert "EPSG:" in detail and "hillshade[" in detail


def test_flat_dem_loads_but_flags_constant_hillshade(qapp, tmp_path, caplog):
    """A uniform DEM loads fine but its hillshade is a flat grey wash that reads
    as blank. The layer must FLAG that (the suspected low-relief cause), never go
    silent — this is the durable diagnostic regardless of the live root cause."""
    dest = tmp_path / "dem" / "elevation.tif"
    dest.parent.mkdir(parents=True)
    h = w = 32
    data = np.full((h, w), -40.0, dtype="float32")     # constant bathymetry
    with rasterio.open(dest, "w", driver="GTiff", height=h, width=w, count=1,
                       dtype="float32", crs="EPSG:32631",
                       transform=_affine_from_bounds(600000, 6080000,
                                                     610000, 6090000, w, h),
                       nodata=None) as ds:
        ds.write(data, 1)
    state, mv = _map_with_section(qapp)
    with caplog.at_level("WARNING", logger="section_tool.views.map_dem_layer"):
        ok, detail = mv._dem._load_geotiff_diagnosed(str(dest))
    assert ok and mv._dem.has_hillshade()              # not a hard failure
    assert "near-constant" in caplog.text              # but flagged, not silent
    assert "hillshade[" in detail


def test_corrupt_dem_reports_specific_stage_not_silence(qapp, tmp_path):
    """A non-GeoTIFF body (e.g. an OpenTopography HTML error written to disk)
    must surface a specific 'could not read' message, not a silent blank."""
    bad = tmp_path / "dem" / "elevation.tif"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"<html>OpenTopography error: invalid demtype</html>")
    state, mv = _map_with_section(qapp)
    ok, detail = mv._dem._load_geotiff_diagnosed(str(bad))
    assert ok is False
    assert "could not read" in detail.lower()
    assert not mv._dem.has_hillshade()


def test_fetch_stage_a_failure_emits_specific_message(qapp, tmp_path):
    """When fetch/warp/store raises, fetch() emits a stage-tagged failed message
    carrying the cause — not the old generic 'could not be loaded'."""
    from PySide6.QtWidgets import QApplication
    state, mv = _map_with_section(qapp)
    got: list[str] = []
    mv._dem.failed.connect(got.append)

    def boom(*a, **k):
        raise RuntimeError("HTTP 500 from source")

    mv._dem.fetch("gebco", _proj_bounds(), _PROJ,
                  str(tmp_path / "dem" / "elevation.tif"), opener=boom)
    mv._dem._last_thread.join(timeout=10)
    QApplication.processEvents()
    assert any("fetch/warp failed" in m for m in got)
    assert any("HTTP 500" in m for m in got)


def test_dem_cmap_default_is_terrain(qapp):
    _state, mv = _map_with_section(qapp)
    assert mv.dem_cmap() == "terrain"


def test_set_cmap_retints_cached_elevation_without_reload(qapp, tmp_path, monkeypatch):
    """Changing colormap re-tints the cached warped DEM — no disk read, no fetch."""
    state, mv = _map_with_section(qapp)
    assert mv._dem.load_geotiff(_store_dem(tmp_path))   # loads + tints (terrain)
    rgb_terrain = mv._dem._rgb.copy()

    # Break the loader: a re-tint must not touch disk (proves the elevation cache).
    monkeypatch.setattr(D, "load_dem_geotiff",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("reloaded")))
    assert mv._dem.set_cmap("gray") is True
    rgb_gray = mv._dem._rgb
    assert rgb_gray.shape == rgb_terrain.shape
    assert not np.allclose(rgb_gray[..., :3], rgb_terrain[..., :3])   # tint changed
    assert float(rgb_gray[..., 3].min()) >= 0.99                      # still opaque
    assert mv._dem.set_cmap("gray") is False            # no-op when unchanged
    assert mv._dem.set_cmap("not-a-cmap") is False      # rejects unknown


def test_dem_artist_on_axes_after_fetch(qapp, tmp_path):
    """Render-path regression: once a fetch completes, the tinted DEM artist must
    actually be on the map axes — visible, non-zero alpha, correct project-CRS
    extent, under the data (zorder −9). Guards the loaded→render→imshow wiring."""
    state, mv = _map_with_section(qapp)
    src = _fixture_gebco_4326(tmp_path / "src.tif")    # valid negative bathymetry

    class _Ctx:
        def __enter__(self): self._ds = rasterio.open(src); return self._ds
        def __exit__(self, *a): self._ds.close()

    dest = str(tmp_path / "dem" / "elevation.tif")
    mv._dem.fetch("gebco", _proj_bounds(), _PROJ, dest, opener=lambda *a, **k: _Ctx())
    mv._dem._last_thread.join(timeout=10)
    assert mv._dem.has_hillshade() and mv._dem._rgb is not None

    mv.render()
    imgs = [im for im in mv._ax.get_images() if -10 < im.get_zorder() < 0]
    assert imgs, "DEM artist must be on the map axes under the data"
    im = imgs[0]
    assert im.get_visible()
    assert im.get_alpha() and im.get_alpha() > 0
    ext = im.get_extent()
    assert 1e5 < min(ext[:2]) and 5e6 < min(ext[2:]) < 7e6          # UTM 31N metres
    assert sorted(round(v) for v in ext) == sorted(round(v) for v in mv._dem._extent)


def test_fetch_into_layer_with_injected_opener(qapp, tmp_path):
    state, mv = _map_with_section(qapp)
    src = _fixture_4326(tmp_path / "src.tif")

    class _Ctx:
        def __enter__(self): self._ds = rasterio.open(src); return self._ds
        def __exit__(self, *a): self._ds.close()

    opener_calls = {"n": 0}

    def opener(source_key, bounds_ll, api_key):
        opener_calls["n"] += 1
        return _Ctx()

    # Redirect the layer's storage into tmp so we don't touch the real project dir.
    dest = str(tmp_path / "dem" / "elevation.tif")
    mv._dem.fetch("copernicus", _proj_bounds(), _PROJ, dest, opener=opener)
    mv._dem._last_thread.join(timeout=10)
    assert opener_calls["n"] == 1
    assert mv._dem.has_hillshade()
    assert mv._dem.provenance["source"] == "copernicus"
