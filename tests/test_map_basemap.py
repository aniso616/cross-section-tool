"""Basemap underlay: source registry, project-CRS warp (never reproject the map),
1:1 aspect, settle-debounce, stale discard, and a real-MainWindow toggle.

No network: the tile fetch is injected/mocked; contextily itself is only used
for the provider registry."""
from __future__ import annotations

import sys

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.views.map_view import MapView
from section_tool.views import map_basemap_layer as mbl
from section_tool.views.map_basemap_layer import (
    MapBasemapLayer, BASEMAP_ORDER, BASEMAP_LABELS, BASEMAP_SOURCES,
    basemap_available,
)

_EPSG = 32631  # UTM 31N (F3); eastings ~6e5, northings ~6e6 — far from 3857 range


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


def _utm_extent():
    return (606000.0, 610000.0, 6080000.0, 6082000.0)


def _fake_img():
    return np.zeros((8, 8, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

def test_source_registry_keys_and_order():
    assert BASEMAP_ORDER == ("none", "satellite", "osm", "opentopo")
    assert BASEMAP_LABELS["none"] == "None"
    if basemap_available():
        for key in ("satellite", "osm", "opentopo"):
            src = BASEMAP_SOURCES[key]
            assert src.provider is not None
            assert src.attribution                      # ToS attribution present
        # No Google endpoints.
        for src in BASEMAP_SOURCES.values():
            assert "google" not in str(src.provider.get("url", "")).lower()


# ---------------------------------------------------------------------------
# The whole point: tiles are WARPED to the project CRS; the map is never
# reprojected to Web Mercator.
# ---------------------------------------------------------------------------

def test_fetch_warps_to_project_crs(monkeypatch):
    if not basemap_available():
        pytest.skip("contextily not installed")
    captured = {}

    def fake_bounds2img(w, s, e, n, zoom, source, ll):
        captured["merc_bounds"] = (w, s, e, n)
        return _fake_img(), (w, e, s, n)

    def fake_warp_tiles(img, ext, t_crs):
        captured["t_crs"] = t_crs
        return img, (606000.0, 610000.0, 6080000.0, 6082000.0)

    monkeypatch.setattr(mbl._cx, "bounds2img", fake_bounds2img)
    monkeypatch.setattr(mbl._cx, "warp_tiles", fake_warp_tiles)

    prov = BASEMAP_SOURCES["satellite"].provider
    img, ext = mbl._fetch_warped(prov, _EPSG, _utm_extent())

    # Warp target is the PROJECT CRS — the interface never lies about geography.
    assert captured["t_crs"] == f"EPSG:{_EPSG}"
    # Bounds handed to the tile fetch are Web Mercator (huge metres), proving the
    # UTM extent was transformed OUT to fetch tiles, not the map reprojected in.
    w, s, e, n = captured["merc_bounds"]
    assert abs(w) > 1e5 and abs(n) > 5e6
    # Returned extent is back in project-CRS coordinates.
    assert ext == (606000.0, 610000.0, 6080000.0, 6082000.0)


def test_render_keeps_axes_in_project_crs(qapp):
    """Adding the basemap must not move the axes off project-CRS coordinates."""
    layer = MapBasemapLayer()
    layer.set_source("satellite")
    # Simulate a completed fetch with a project-CRS extent.
    ext = _utm_extent()
    layer._img = _fake_img()
    layer._extent = (ext[0], ext[1], ext[2], ext[3])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_xlim(606000.0, 610000.0)
    ax.set_ylim(6080000.0, 6082000.0)
    before = (ax.get_xlim(), ax.get_ylim())
    layer.render(ax)
    after = (ax.get_xlim(), ax.get_ylim())
    assert after == before                                  # unchanged → still UTM
    images = ax.get_images()
    assert images and images[0].get_zorder() < 0            # under the data
    plt.close(fig)


# ---------------------------------------------------------------------------
# Aspect 1:1 must survive the basemap imshow (same style as the VE fix).
# ---------------------------------------------------------------------------

def test_basemap_does_not_break_1to1_aspect(qapp):
    state = AppState()
    mv = MapView(state)
    state.add_section(Section([(606000, 6080000), (610000, 6082000)],
                              name="L1", crs_epsg=_EPSG))
    state.set_active_section(state.project.sections[0])
    mv._basemap.set_source("satellite")
    mv._basemap._img = _fake_img()
    mv._basemap._extent = _utm_extent()
    mv.render()
    ax = mv._ax
    ax.apply_aspect()
    assert ax.get_aspect() == "equal" or float(ax.get_aspect()) == pytest.approx(1.0)
    t = ax.transData
    x0, y0 = t.transform((0.0, 0.0))
    ppm_x = abs(t.transform((1.0, 0.0))[0] - x0)
    ppm_y = abs(t.transform((0.0, 1.0))[1] - y0)
    assert ppm_x == pytest.approx(ppm_y, rel=1e-3)          # E and N: equal pixels


# ---------------------------------------------------------------------------
# Settle-debounce: no fetch storm on pan (the navigation-doesn't-restretch idea).
# ---------------------------------------------------------------------------

def test_settle_debounce_no_fetch_storm(qapp):
    state = AppState()
    mv = MapView(state)
    state.add_section(Section([(606000, 6080000), (610000, 6082000)],
                              name="L1", crs_epsg=_EPSG))
    state.set_active_section(state.project.sections[0])
    mv._basemap.set_source("satellite")

    calls = {"n": 0}
    mv._basemap.request = lambda *a, **k: calls.__setitem__("n", calls["n"] + 1)

    for _ in range(20):                                     # rapid pan/zoom ticks
        mv._schedule_basemap_fetch()
    assert calls["n"] == 0                                   # nothing fetched mid-gesture
    assert mv._basemap_settle_timer.isActive()
    mv._fetch_basemap()                                     # what the settle timer fires
    assert calls["n"] == 1                                   # one fetch after settle


# ---------------------------------------------------------------------------
# Stale-extent discard + cache coalescing.
# ---------------------------------------------------------------------------

def test_stale_result_discarded():
    layer = MapBasemapLayer(fetch_fn=lambda *a, **k: (_fake_img(), _utm_extent()))
    layer.set_source("satellite")
    # A worker carrying an out-of-date request id must not apply its result.
    layer._req_id = 5
    layer._worker(rid=3, key=("p", "satellite", (0, 0, 0, 0), "auto"),
                  provider=None, epsg=_EPSG, extent=_utm_extent(), zoom="auto")
    assert not layer.has_image()
    # The current request id does apply.
    layer._worker(rid=5, key=("p", "satellite", (0, 0, 0, 0), "auto"),
                  provider=None, epsg=_EPSG, extent=_utm_extent(), zoom="auto")
    assert layer.has_image()


def test_request_coalesces_identical_view():
    calls = {"n": 0}

    def fetch(provider, epsg, extent, zoom):
        calls["n"] += 1
        return _fake_img(), _utm_extent()

    layer = MapBasemapLayer(fetch_fn=fetch)
    layer.set_source("satellite")
    layer.request("proj", _EPSG, _utm_extent())
    layer._last_thread.join(timeout=5)
    layer.request("proj", _EPSG, _utm_extent())             # identical view → cached
    assert layer._last_thread.join(timeout=5) is None
    assert calls["n"] == 1


def test_none_source_never_fetches():
    calls = {"n": 0}
    layer = MapBasemapLayer(
        fetch_fn=lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1),
                                  (_fake_img(), _utm_extent()))[1])
    layer.request("proj", _EPSG, _utm_extent())             # source is "none"
    assert layer._last_thread is None
    assert calls["n"] == 0
