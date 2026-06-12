"""Web-tile basemap underlay, warped to the project CRS.

The interface never lies about geography: the map is *never* reprojected to Web
Mercator. Instead contextily fetches tiles and ``warp_tiles`` warps them INTO the
project's authoritative projected CRS, so the axes keep their easting/northing
coordinates and 1:1 aspect. Fetching happens off the UI thread on extent-settle;
stale results are discarded. Imports are gated so the app still runs (reporting
"basemap unavailable") when contextily is not installed.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal

from section_tool.core.crs import transform_points

try:
    import contextily as _cx
    _HAVE_CX = True
    _IMPORT_ERROR = ""
except Exception as _exc:                       # pragma: no cover - env dependent
    _cx = None
    _HAVE_CX = False
    _IMPORT_ERROR = str(_exc)

_WEB_MERCATOR = 3857
_BASEMAP_ZORDER = -10                            # under the grid (0) and all data


@dataclass(frozen=True)
class BasemapSource:
    key: str
    label: str
    provider: object            # xyzservices TileProvider (None when cx absent)
    attribution: str


def _build_sources() -> "dict[str, BasemapSource]":
    if not _HAVE_CX:
        return {}
    p = _cx.providers
    out: dict[str, BasemapSource] = {}

    def add(key, label, prov):
        out[key] = BasemapSource(key, label, prov, str(prov.get("attribution", "")))

    # No Google endpoints (ToS). Satellite default.
    add("satellite", "Satellite (Esri)", p.Esri.WorldImagery)
    add("osm", "OpenStreetMap", p.OpenStreetMap.Mapnik)
    add("opentopo", "OpenTopoMap", p.OpenTopoMap)
    return out


BASEMAP_SOURCES = _build_sources()
# Menu order; "none" is always first and the default (no surprise network on open).
BASEMAP_ORDER = ("none", "satellite", "osm", "opentopo")
BASEMAP_LABELS = {"none": "None",
                  **{k: s.label for k, s in BASEMAP_SOURCES.items()}}


def basemap_available() -> bool:
    return _HAVE_CX


def unavailable_reason() -> str:
    return "basemap unavailable — install contextily" if not _HAVE_CX else ""


def _fetch_warped(provider, epsg: int, extent, zoom="auto"):
    """Fetch tiles covering *extent* (project-CRS x0,x1,y0,y1) and warp to EPSG:*epsg*.

    Returns ``(img, (left, right, bottom, top))`` with the extent in project-CRS
    metres — ready to imshow on the map axes without touching its limits.
    """
    x0, x1, y0, y1 = extent
    xs = [x0, x1, x0, x1]
    ys = [y0, y0, y1, y1]
    mx, my = transform_points(xs, ys, epsg, _WEB_MERCATOR)
    w, e = float(min(mx)), float(max(mx))
    s, n = float(min(my)), float(max(my))
    img, ext3857 = _cx.bounds2img(w, s, e, n, zoom=zoom, source=provider, ll=False)
    warped, wext = _cx.warp_tiles(img, ext3857, t_crs=f"EPSG:{epsg}")
    # warp_tiles returns extent as (left, right, bottom, top)
    return warped, tuple(float(v) for v in wext)


class MapBasemapLayer(QObject):
    """Current basemap source + a cached warped-tile image fetched off-thread.

    The map rebuilds every render, so :meth:`render` re-blits the cached image
    under the data layers; the network fetch only runs on extent-settle via
    :meth:`request`. Cache keys carry the **project identity** so two projects
    never collide on the same view.
    """

    updated = Signal()          # cache changed → map should request a re-render

    def __init__(self, fetch_fn=None, parent=None) -> None:
        super().__init__(parent)
        self._source_key = "none"
        self._fetch_fn = fetch_fn or _fetch_warped
        self._cache_key = None
        self._img = None
        self._extent = None         # (left, right, bottom, top) in project CRS
        self._req_id = 0
        self._lock = threading.Lock()
        self._last_thread = None        # test hook: join the most recent worker

    # ---- source selection ------------------------------------------------

    @property
    def source_key(self) -> str:
        return self._source_key

    def set_source(self, key: str) -> None:
        if key not in BASEMAP_LABELS:
            key = "none"
        if key != self._source_key:
            self._source_key = key
            with self._lock:
                self._img = None
                self._extent = None
                self._cache_key = None
                self._req_id += 1       # invalidate any in-flight fetch

    def attribution(self) -> str:
        s = BASEMAP_SOURCES.get(self._source_key)
        return s.attribution if s else ""

    def has_image(self) -> bool:
        return self._img is not None

    # ---- render (re-blit cached image; called every map render) ----------

    def render(self, ax) -> None:
        if self._source_key == "none" or self._img is None or self._extent is None:
            return
        left, right, bottom, top = self._extent
        # aspect='auto' so this imshow does NOT impose its own axes aspect — the
        # map's set_aspect('equal', datalim) governs (basemap must not fight 1:1).
        ax.imshow(self._img, extent=(left, right, bottom, top), origin="upper",
                  zorder=_BASEMAP_ZORDER, interpolation="bilinear", aspect="auto")
        attr = self.attribution()
        if attr:
            ax.text(0.995, 0.005, attr, transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=5.5, color="#e0e0e0",
                    zorder=50, clip_on=False,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#000000",
                              alpha=0.4, edgecolor="none"))

    # ---- fetch on settle -------------------------------------------------

    def request(self, project_id: str, epsg: int, extent, zoom="auto") -> None:
        """Fetch tiles for *extent* unless the exact view is already cached.

        Spawns a daemon worker; the result is applied via :attr:`updated` only if
        no newer request superseded it (stale-extent discard).
        """
        if self._source_key == "none" or not _HAVE_CX or not epsg:
            return
        key = (str(project_id), self._source_key,
               tuple(round(float(v), -1) for v in extent), zoom)
        if key == self._cache_key:
            return
        with self._lock:
            self._req_id += 1
            rid = self._req_id
        provider = BASEMAP_SOURCES[self._source_key].provider
        t = threading.Thread(
            target=self._worker, args=(rid, key, provider, int(epsg), tuple(extent), zoom),
            daemon=True)
        self._last_thread = t
        t.start()

    def _worker(self, rid, key, provider, epsg, extent, zoom) -> None:
        try:
            img, wext = self._fetch_fn(provider, epsg, extent, zoom)
        except Exception:
            return          # network / tile error — keep the previous basemap
        with self._lock:
            if rid != self._req_id:
                return      # a newer request (or a source change) superseded this
            self._img = img
            self._extent = wext
            self._cache_key = key
        self.updated.emit()
