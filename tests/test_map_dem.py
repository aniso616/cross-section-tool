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
