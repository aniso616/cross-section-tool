"""AOI elevation — fetch a DEM once for an extent, reproject to the project CRS,
store it as a GeoTIFF in the project folder with provenance, and hillshade it.

Network access is isolated behind a source *opener* callable, so the rasterio
mechanics (windowed read, warp to project CRS, ocean fill, hillshade, provenance)
are all tested on a local fixture GeoTIFF with no network.

The map extent is the 75% stand-in for the future first-class AOI polygon — the
extent→AOI seam is left deliberately (callers pass an x0,x1,y0,y1 extent).
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

log = logging.getLogger(__name__)

import numpy as np
from matplotlib.colors import LightSource

from section_tool.core.crs import transform_points

try:
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.warp import (
        reproject, transform_bounds, calculate_default_transform)
    from rasterio.windows import from_bounds as _window_from_bounds
    from rasterio.enums import Resampling
    _HAVE_RIO = True
    _IMPORT_ERROR = ""
except Exception as _exc:                       # pragma: no cover - env dependent
    rasterio = None
    _HAVE_RIO = False
    _IMPORT_ERROR = str(_exc)

DEFAULT_AZIMUTH = 315.0
DEFAULT_ALTITUDE = 45.0
_WGS84 = 4326
_PROVENANCE_TAG = "SECTION_DEM_PROVENANCE"     # GeoTIFF tag holding the JSON below


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DEMSource:
    key: str
    label: str
    needs_key: bool
    dataset: str            # provenance dataset id
    note: str               # caveat surfaced in metadata / tooltip


DEM_SOURCES: "dict[str, DEMSource]" = {
    "copernicus": DEMSource(
        "copernicus", "Copernicus GLO-30 (no key)", False, "COP30",
        "Surface model (DSM) — includes canopy/buildings; ocean tiles absent."),
    "gebco": DEMSource(
        "gebco", "GEBCO topo+bathymetry (OpenTopography key)", True, "GEBCOIceTopo",
        "Global topography + bathymetry; the offshore DEM for the F3 demo."),
    "eudtm": DEMSource(
        "eudtm", "Europe DTM 30m (OpenTopography key)", True, "EU_DTM",
        "Bare-earth terrain model (DTM); onshore Europe only."),
}
DEM_SOURCE_ORDER = ("copernicus", "gebco", "eudtm")


def dem_available() -> bool:
    return _HAVE_RIO


def unavailable_reason() -> str:
    return "elevation unavailable — install rasterio" if not _HAVE_RIO else ""


# ---------------------------------------------------------------------------
# Core mechanics (testable on a fixture; no network)
# ---------------------------------------------------------------------------

@dataclass
class DEMArray:
    """A DEM reprojected into the project CRS, ready to store/hillshade."""
    data: np.ndarray                 # 2D float32, rows = north→south (origin upper)
    transform: object                # affine in project CRS
    epsg: int
    extent: tuple                    # (left, right, bottom, top) project CRS
    ocean_filled: bool               # any nodata replaced with fill_value


def reproject_to_project(src, bounds_proj, project_epsg, *,
                         fill_value: float = 0.0,
                         resolution: float | None = None) -> DEMArray:
    """Windowed-read *src* over *bounds_proj* and warp to EPSG:*project_epsg*.

    *src* is an open rasterio dataset (a COG URL in production, a fixture file in
    tests). *bounds_proj* is ``(x0, x1, y0, y1)`` in the project CRS. Nodata /
    absent ocean cells are replaced with *fill_value* and the result flags it.
    """
    if not _HAVE_RIO:
        raise RuntimeError(unavailable_reason())
    x0, x1, y0, y1 = bounds_proj
    left, right = min(x0, x1), max(x0, x1)
    bottom, top = min(y0, y1), max(y0, y1)

    # Project bounds → source CRS, to size the destination grid.
    s_w, s_s, s_e, s_n = transform_bounds(
        f"EPSG:{project_epsg}", src.crs, left, bottom, right, top)
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src.crs, f"EPSG:{project_epsg}", src.width, src.height,
        left=s_w, bottom=s_s, right=s_e, top=s_n,
        resolution=resolution)

    dst = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
    src_band = src.read(1).astype(np.float32)
    src_nodata = src.nodata
    if src_nodata is not None:
        src_band = np.where(src_band == src_nodata, np.nan, src_band)

    reproject(
        source=src_band,
        destination=dst,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs=f"EPSG:{project_epsg}",
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

    # Warp-boundary diagnostics: an all-nodata or all-equal result here is the
    # break point for a "silent blank" map. Captured BEFORE the ocean fill so
    # the nodata fraction is the true warp coverage, not the filled value.
    finite = np.isfinite(dst)
    nodata_frac = float(np.mean(~finite))
    if np.any(finite):
        w_min, w_max = float(np.min(dst[finite])), float(np.max(dst[finite]))
    else:
        w_min = w_max = float("nan")
    log.info(
        "DEM warp: dst=%dx%d src.crs=%s dst=EPSG:%s nodata_frac=%.3f "
        "elev[min=%.1f max=%.1f]",
        dst_w, dst_h, src.crs, project_epsg, nodata_frac, w_min, w_max)
    if nodata_frac >= 0.999:
        log.warning("DEM warp produced ~all-nodata — check bbox / src-dst CRS")
    elif np.isfinite(w_min) and w_max - w_min < 1e-6:
        log.warning("DEM warp produced a constant surface (%.1f) — elevation lost", w_min)

    ocean_filled = bool(np.any(~finite))
    if ocean_filled:
        dst = np.where(finite, dst, float(fill_value)).astype(np.float32)

    # Extent from the destination affine (origin upper-left).
    e_left = dst_transform.c
    e_top = dst_transform.f
    e_right = e_left + dst_transform.a * dst_w
    e_bottom = e_top + dst_transform.e * dst_h
    return DEMArray(
        data=dst, transform=dst_transform, epsg=int(project_epsg),
        extent=(e_left, e_right, e_bottom, e_top), ocean_filled=ocean_filled)


def write_dem_geotiff(dem: DEMArray, dest_path: str, provenance: dict) -> None:
    """Write *dem* to a single-band float32 GeoTIFF with provenance in a tag."""
    if not _HAVE_RIO:
        raise RuntimeError(unavailable_reason())
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    h, w = dem.data.shape
    with rasterio.open(
        dest_path, "w", driver="GTiff", height=h, width=w, count=1,
        dtype="float32", crs=f"EPSG:{dem.epsg}", transform=dem.transform,
        nodata=None, compress="deflate",
    ) as ds:
        ds.write(dem.data, 1)
        ds.update_tags(**{_PROVENANCE_TAG: json.dumps(provenance)})


def load_dem_geotiff(path: str) -> "tuple[np.ndarray, tuple, int, dict]":
    """Return (data2d, extent(left,right,bottom,top), epsg, provenance)."""
    if not _HAVE_RIO:
        raise RuntimeError(unavailable_reason())
    with rasterio.open(path) as ds:
        data = ds.read(1).astype(np.float32)
        b = ds.bounds
        epsg = ds.crs.to_epsg() if ds.crs else 0
        tag = ds.tags().get(_PROVENANCE_TAG, "")
        prov = json.loads(tag) if tag else {}
    return data, (b.left, b.right, b.bottom, b.top), int(epsg or 0), prov


def hillshade(dem2d: np.ndarray, *, dx: float = 30.0, dy: float = 30.0,
              azimuth: float = DEFAULT_AZIMUTH,
              altitude: float = DEFAULT_ALTITUDE,
              vert_exag: float = 1.0) -> np.ndarray:
    """Greyscale hillshade in [0, 1] via matplotlib's LightSource.

    *vert_exag* scales the elevation against the (metric) pixel spacing before
    shading. ``1.0`` is right for steep land at native resolution but flattens
    gentle seabed sampled at hundreds-of-metres/pixel to a featureless wash —
    callers on low-relief grids should size it with :func:`auto_vert_exag`.
    """
    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    arr = np.asarray(dem2d, dtype=float)
    return ls.hillshade(arr, dx=dx, dy=dy, vert_exag=vert_exag)


DEFAULT_MARINE_VERT_EXAG = 20.0   # documented fallback when relief is ~flat


def auto_vert_exag(dem2d: np.ndarray, *, dx: float, dy: float,
                   target_relief_px: float = 3.0,
                   lo: float = 1.0, hi: float = 40.0) -> float:
    """A vertical exaggeration that lifts low-gradient relief into visible shading.

    ``vert_exag=1`` suits steep terrain at native resolution but renders gentle
    bathymetry (a few metres of relief over hundreds-of-metres pixels — the
    GEBCO/F3 case) as a flat grey wash. Scale so the data's elevation range maps
    to roughly *target_relief_px* pixels of apparent height::

        vert_exag = target_relief_px * mean_pixel_size / elevation_range

    clamped to ``[lo, hi]``: *lo*=1 never de-exaggerates already-steep land
    (onshore DEMs), *hi*=40 caps the lift for gentle marine relief. A truly flat
    grid
    (range ≈ 0) can't be shaded into relief, so it falls back to
    :data:`DEFAULT_MARINE_VERT_EXAG`. The chosen value is returned so callers can
    record it.
    """
    arr = np.asarray(dem2d, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return DEFAULT_MARINE_VERT_EXAG
    rng = float(finite.max() - finite.min())
    if rng < 1e-6:
        return DEFAULT_MARINE_VERT_EXAG
    mean_px = (abs(dx) + abs(dy)) / 2.0 or 1.0
    ve = target_relief_px * mean_px / rng
    return float(min(max(ve, lo), hi))


DEFAULT_DEM_CMAP = "terrain"   # topo+bathy ramp: depth reads as colour, not relief


def _get_cmap(name: str):
    try:
        from matplotlib import colormaps
        return colormaps[name]
    except Exception:                                   # pragma: no cover - old mpl
        from matplotlib.cm import get_cmap
        return get_cmap(name)


def shaded_relief(data: np.ndarray, *, dx: float, dy: float, vert_exag: float,
                  cmap: str = DEFAULT_DEM_CMAP,
                  azimuth: float = DEFAULT_AZIMUTH,
                  altitude: float = DEFAULT_ALTITUDE,
                  blend_mode: str = "soft") -> np.ndarray:
    """Elevation tinted by *cmap* and lit by a hillshade — an RGBA image in [0,1].

    A pure greyscale hillshade collapses to a single flat value on near-planar
    bathymetry (constant gradient → one shading direction), which is exactly why
    the F3 seabed read as blank no matter the exaggeration. Tinting by elevation
    keeps depth legible even where the relief shading is flat. Built on
    matplotlib's :meth:`LightSource.shade`.
    """
    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    arr = np.asarray(data, dtype=float)
    return ls.shade(arr, cmap=_get_cmap(cmap), blend_mode=blend_mode,
                    vert_exag=vert_exag, dx=dx, dy=dy)


def pixel_size(dem: DEMArray) -> "tuple[float, float]":
    """(dx, dy) ground sample distance in project-CRS units."""
    return abs(dem.transform.a), abs(dem.transform.e)


# ---------------------------------------------------------------------------
# Provenance + storage location
# ---------------------------------------------------------------------------

def make_provenance(source_key: str, bounds_proj, project_epsg: int,
                    ocean_filled: bool) -> dict:
    src = DEM_SOURCES[source_key]
    return {
        "source": source_key,
        "label": src.label,
        "dataset": src.dataset,
        "note": src.note,
        "fetch_date": datetime.date.today().isoformat(),
        "extent_proj": [float(v) for v in bounds_proj],
        "crs": f"EPSG:{int(project_epsg)}",
        "ocean_filled": bool(ocean_filled),
    }


def dem_path_for_project(project_path: str | None) -> str:
    """Where the project's DEM GeoTIFF lives (a ``dem/`` folder in the project)."""
    root = project_path or os.path.join(os.path.expanduser("~"), ".section_dem")
    return os.path.join(root, "dem", "elevation.tif")


# ---------------------------------------------------------------------------
# Fetch orchestration (opener injected; default openers do the network)
# ---------------------------------------------------------------------------

@dataclass
class DEMResult:
    path: str
    provenance: dict
    ocean_filled: bool


def fetch_dem(source_key: str, bounds_proj, project_epsg: int, dest_path: str, *,
              api_key: str | None = None, opener=None,
              fill_value: float = 0.0) -> DEMResult:
    """Fetch → reproject → store a DEM for *bounds_proj*, returning a DEMResult.

    *opener(source_key, bounds_lonlat, api_key)* yields an open rasterio dataset
    (context manager). The default opener does the network; tests inject one that
    opens a fixture. Absent ocean tiles surface as a zero-filled DEM with a flag
    rather than an error.
    """
    if not _HAVE_RIO:
        raise RuntimeError(unavailable_reason())
    if source_key not in DEM_SOURCES:
        raise ValueError(f"unknown DEM source {source_key!r}")
    opener = opener or _default_opener
    x0, x1, y0, y1 = bounds_proj
    # Project extent → lon/lat for the source request window.
    xs = [x0, x1, x0, x1]
    ys = [y0, y0, y1, y1]
    lons, lats = transform_points(xs, ys, project_epsg, _WGS84)
    bounds_ll = (float(min(lons)), float(min(lats)),
                 float(max(lons)), float(max(lats)))   # (w, s, e, n)
    # The request window the opener actually sends. Logged so a "200 but blank"
    # can be checked against the known-good curl bbox: this must read in WGS84
    # degrees (e.g. F3 ~4–5°E / 54–54.3°N), never project-CRS metres.
    log.info("DEM request: source=%s bbox(WGS84 w/s/e/n)=%.4f/%.4f/%.4f/%.4f "
             "from project bounds=%s EPSG:%s",
             source_key, *bounds_ll, bounds_proj, project_epsg)

    with opener(source_key, bounds_ll, api_key) as src:
        dem = reproject_to_project(src, bounds_proj, project_epsg,
                                   fill_value=fill_value)
    prov = make_provenance(source_key, bounds_proj, project_epsg, dem.ocean_filled)
    write_dem_geotiff(dem, dest_path, prov)
    return DEMResult(path=dest_path, provenance=prov,
                     ocean_filled=dem.ocean_filled)


_FETCH_RETRIES = 3
_FETCH_TIMEOUT = 60


def _looks_like_tiff(body: bytes) -> bool:
    return body[:4] in (b"II*\x00", b"MM\x00*")


def download_validated_tiff(url: str, dest: str, *, opener=None,
                            retries: int = _FETCH_RETRIES,
                            timeout: int = _FETCH_TIMEOUT, sleep=time.sleep) -> str:
    """Download *url* to the real file *dest*, validated to decode. Returns *dest*.

    Two things made the DEM blank intermittently. (1) The OpenTopography body is
    tiny (a few KB), so a dropped/short read yields a TIFF whose header parses but
    whose LZW tiles won't decode. (2) Reading those bytes through an in-memory
    ``MemoryFile`` (``/vsimem``) is unreliable for these tiled-LZW tiffs — two
    separate opens of the *same* bytes can disagree, so a trial decode passes and
    the warp then throws ("scanline size zero"). Writing to a real file (what a
    plain ``curl`` does) and validating *that* file — the very file the warp will
    open — removes both. API error pages are rejected with their message (no
    retry); transient failures (timeout, truncation, bad decode) retry with
    backoff.
    """
    opener = opener or urllib.request.urlopen
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    last = None
    for attempt in range(1, retries + 1):
        try:
            with opener(url, timeout=timeout) as resp:
                body = resp.read()
                headers = getattr(resp, "headers", None)
                clen = headers.get("Content-Length") if headers else None
        except urllib.error.HTTPError as exc:           # 4xx = client error, no retry
            detail = ""
            try:
                detail = exc.read()[:200].decode("utf-8", "replace").strip()
            except Exception:
                pass
            if 400 <= exc.code < 500:
                raise RuntimeError(
                    f"DEM source error {exc.code}: {detail or exc.reason}") from exc
            last = exc
            log.warning("DEM download attempt %d/%d HTTP %s", attempt, retries, exc.code)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            log.warning("DEM download attempt %d/%d failed: %s", attempt, retries, exc)
        else:
            # A non-TIFF body is an API error page (bad key, rate limit, bad
            # bbox) — surface its message; retrying will not help.
            if not _looks_like_tiff(body):
                snippet = body[:200].decode("utf-8", "replace").strip()
                raise RuntimeError(
                    f"DEM source returned a non-GeoTIFF response: {snippet!r}")
            if clen is not None and str(clen).isdigit() and int(clen) != len(body):
                last = RuntimeError(f"short read: {len(body)}/{clen} bytes")
                log.warning("DEM download attempt %d/%d truncated (%d/%s)",
                            attempt, retries, len(body), clen)
            else:
                with open(dest, "wb") as fh:
                    fh.write(body)
                try:
                    with rasterio.open(dest) as ds:     # validate the REAL file
                        ds.read(1)
                    log.info("DEM download OK: %d bytes -> %s (attempt %d)",
                             len(body), dest, attempt)
                    return dest
                except Exception as exc:                # truncated/corrupt tiles
                    last = exc
                    log.warning("DEM download attempt %d/%d would not decode: %s",
                                attempt, retries, exc)
        if attempt < retries:
            sleep(min(2 ** (attempt - 1), 5))
    raise RuntimeError(f"DEM download failed after {retries} attempts: {last}")


def _default_opener(source_key: str, bounds_ll, api_key):   # pragma: no cover - network
    """Open the real source as a rasterio dataset. Network — not run in tests.

    Copernicus GLO-30 reads straight from the AWS COG; GEBCO / EU DTM come from
    the OpenTopography global-DEM API (key required). The key is passed in by the
    caller (entered in settings, stored locally, never committed).
    """
    import tempfile
    w, s, e, n = bounds_ll
    src = DEM_SOURCES[source_key]
    if src.needs_key:
        if not api_key:
            raise RuntimeError(
                f"{src.label} needs an OpenTopography API key (set it in settings).")
        url = ("https://portal.opentopography.org/API/globaldem"
               f"?demtype={src.dataset}&south={s}&north={n}&west={w}&east={e}"
               f"&outputFormat=GTiff&API_Key={api_key}")
        # Download to a real file and open THAT for the warp — never /vsimem — so a
        # flaky short read can't slip a half-decodable tiff into the reproject.
        tmp = os.path.join(tempfile.mkdtemp(prefix="section_dem_"), "source.tif")
        download_validated_tiff(url, tmp)
        return rasterio.open(tmp)
    # Copernicus GLO-30 mosaic COG on AWS Open Data (no key); /vsicurl/ windowed.
    cog = ("/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/"
           "Copernicus_DSM_COG_10_mosaic.vrt")
    return rasterio.open(cog)
