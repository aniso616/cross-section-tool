"""Hillshaded-DEM underlay for the map.

Unlike the basemap, the DEM is fetched only on explicit user request (never on
pan/settle). This layer loads the stored project GeoTIFF, computes a greyscale
hillshade once, and re-blits it under the data layers on every render (the map
rebuilds each render). The network fetch runs off the UI thread; completion is
marshalled back via a Qt signal.
"""
from __future__ import annotations

import logging
import os
import threading

import numpy as np
from PySide6.QtCore import QObject, Signal

from section_tool.core import dem as _dem

log = logging.getLogger(__name__)

_DEM_ZORDER = -9            # above the basemap (-10), under the grid (0) and data


class MapDemLayer(QObject):
    """Holds the current hillshade image + visibility, with an off-thread fetch."""

    loaded = Signal()                  # a fetch completed → map should re-render
    failed = Signal(str)               # fetch error message for status feedback
    draped = Signal()                  # an imagery drape completed → re-render
    drape_failed = Signal(str)         # drape fetch/composite error

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._visible = True
        self._hs = None                # 2D greyscale hillshade in [0, 1] (diagnostic)
        self._rgb = None               # RGBA elevation-tinted relief (the tint)
        self._elev = None              # warped project-CRS elevation (re-tint cache)
        self._dx = self._dy = None     # metric pixel size of the warped grid
        self._epsg = None              # project CRS of the warped DEM grid
        self._extent = None            # (left, right, bottom, top) project CRS
        self._provenance = {}
        self._vert_exag = None         # vertical exaggeration used for the shading
        self._cmap = _dem.DEFAULT_DEM_CMAP
        # Imagery drape (imagery × hillshade) — occupies this layer's slot when on.
        self._drape_source = "none"
        self._drape_grid_rgb = None    # imagery resampled onto the DEM grid
        self._composite = None         # RGBA draped composite (what renders when on)
        self._drape_provenance = {}
        self._last_thread = None       # test hook
        self._last_drape_thread = None # test hook

    # ---- visibility ------------------------------------------------------

    @property
    def visible(self) -> bool:
        return self._visible

    def set_visible(self, on: bool) -> None:
        self._visible = bool(on)

    def has_hillshade(self) -> bool:
        return self._hs is not None

    @property
    def provenance(self) -> dict:
        return dict(self._provenance)

    # ---- colormap (tint only — never re-fetches) -------------------------

    @property
    def cmap(self) -> str:
        return self._cmap

    def set_cmap(self, name: str) -> bool:
        """Select the elevation colormap. Returns True if it changed.

        The warped DEM is cached in memory, so a colormap change only re-runs the
        tint step — no disk read, no network. Sets the name even with no DEM yet
        loaded, so a later fetch tints with it.
        """
        if not _dem.is_dem_cmap(name) or name == self._cmap:
            return False
        self._cmap = name
        self._recompute_tint()
        return True

    def _recompute_tint(self) -> None:
        """Re-run shaded_relief on the cached elevation with the current cmap."""
        if self._elev is None:
            return
        try:
            self._rgb = _dem.shaded_relief(self._elev, dx=self._dx, dy=self._dy,
                                           vert_exag=self._vert_exag, cmap=self._cmap)
        except Exception:
            log.exception("DEM re-tint failed for cmap %s", self._cmap)

    # ---- loading ---------------------------------------------------------

    def load_geotiff(self, path: str) -> bool:
        """Load a stored DEM GeoTIFF and compute its hillshade. Returns success."""
        ok, _detail = self._load_geotiff_diagnosed(path)
        return ok

    def _load_geotiff_diagnosed(self, path: str) -> "tuple[bool, str]":
        """Load + hillshade with stage-specific diagnostics. Never raises.

        Returns ``(ok, detail)`` where *detail* is a specific message for the
        failed stage (read / hillshade) on failure, or the layer stats on
        success. The old single bare ``except`` here swallowed the real cause,
        which is exactly how a blank map stayed silent — so every exit reports.
        """
        try:
            data, extent, epsg, prov = _dem.load_dem_geotiff(path)
        except Exception as exc:                       # the actual read error
            log.exception("DEM load failed: %s", path)
            return False, f"could not read stored DEM: {exc}"
        h, w = data.shape
        if h == 0 or w == 0:
            return False, f"stored DEM has empty dimensions ({w}x{h})"
        # Ground sample distance from the stored extent — real project-CRS metres
        # per pixel (EPSG:metres). These drive the shading: a default of 1.0 would
        # itself flatten the hillshade, so a non-positive size is a real fault.
        dx = abs(extent[1] - extent[0]) / max(w, 1)
        dy = abs(extent[3] - extent[2]) / max(h, 1)
        if dx <= 0 or dy <= 0:
            log.warning("DEM pixel size non-positive (dx=%.3f dy=%.3f) — extent "
                        "or dimensions are degenerate; shading will flatten", dx, dy)
            dx, dy = dx or 1.0, dy or 1.0
        # Size the vertical exaggeration to the data so gentle (marine) relief
        # shades visibly instead of washing out — the low-relief blank fix.
        vert_exag = _dem.auto_vert_exag(data, dx=dx, dy=dy)
        try:
            hs = _dem.hillshade(data, dx=dx, dy=dy, vert_exag=vert_exag)
            rgb = _dem.shaded_relief(data, dx=dx, dy=dy, vert_exag=vert_exag,
                                     cmap=self._cmap)
        except Exception as exc:
            log.exception("DEM shading failed")
            return False, f"shading computation failed: {exc}"

        self._hs = hs                      # greyscale relief — kept for diagnostics
        self._rgb = rgb                    # elevation-tinted relief — what renders
        self._elev = data                  # cached so a cmap change re-tints offline
        self._dx, self._dy = dx, dy
        self._epsg = int(epsg) if epsg else self._epsg
        self._extent = extent
        self._provenance = prov or {}
        self._vert_exag = vert_exag        # recorded: the exaggeration actually used
        # A new DEM grid invalidates any existing drape composite (stale extent);
        # the source choice is kept so the map can re-apply it for the new grid.
        self._composite = None
        self._drape_grid_rgb = None

        elev_min, elev_max = float(np.nanmin(data)), float(np.nanmax(data))
        hs_min, hs_max = float(hs.min()), float(hs.max())
        flat = (hs_max - hs_min) < 1e-3
        detail = (f"DEM layer: {w}x{h} EPSG:{epsg} "
                  f"extent={tuple(round(v) for v in extent)} "
                  f"px=({dx:.1f},{dy:.1f})m vert_exag={vert_exag:.1f} "
                  f"elev[{elev_min:.1f},{elev_max:.1f}] "
                  f"hillshade[{hs_min:.3f},{hs_max:.3f}] "
                  f"tinted=yes visible={self._visible} z={_DEM_ZORDER}")
        log.info(detail)
        if flat:
            # The relief shading alone is a flat wash here (near-planar terrain —
            # one shading direction at any exaggeration). Not a failure: the
            # elevation tint still carries depth, which is why we render the
            # tinted RGB rather than pure grey. Logged so the cause is on record.
            log.warning("DEM relief shading near-constant (range %.4f) — near-planar "
                        "terrain; depth carried by the elevation tint instead",
                        hs_max - hs_min)
        return True, detail

    def clear(self) -> None:
        self._hs = None
        self._rgb = None
        self._elev = None
        self._dx = self._dy = None
        self._epsg = None
        self._extent = None
        self._provenance = {}
        self._drape_source = "none"
        self._drape_grid_rgb = None
        self._composite = None
        self._drape_provenance = {}

    # ---- imagery drape (imagery × hillshade on the DEM grid) -------------

    @property
    def drape_source(self) -> str:
        return self._drape_source

    def has_drape(self) -> bool:
        return self._composite is not None

    @property
    def drape_provenance(self) -> dict:
        return dict(self._drape_provenance)

    def clear_drape(self) -> None:
        self._drape_source = "none"
        self._drape_grid_rgb = None
        self._composite = None
        self._drape_provenance = {}

    def apply_drape(self, src_rgb, src_transform, src_crs, *,
                    source: str = "imported", provenance=None) -> bool:
        """Resample *src_rgb* onto the DEM grid and composite it over the relief.

        *src_transform*/*src_crs* describe the imagery; it is warped + resampled to
        this DEM's grid (elevation authoritative), then draped. Returns success.
        """
        if self._elev is None or self._extent is None:
            return False
        from rasterio.transform import from_bounds
        h, w = self._elev.shape
        left, right, bottom, top = self._extent
        dst_transform = from_bounds(left, bottom, right, top, w, h)
        grid = _dem.resample_rgb_to_grid(
            src_rgb, src_transform, src_crs,
            dst_transform=dst_transform, dst_crs=f"EPSG:{self._epsg}",
            dst_shape=(h, w))
        self._drape_grid_rgb = grid
        self._composite = _dem.drape_rgb(grid, self._elev, dx=self._dx,
                                         dy=self._dy, vert_exag=self._vert_exag)
        self._drape_source = source
        self._drape_provenance = dict(provenance or {"drape": source})
        return True

    def fetch_satellite_drape(self, provider, epsg: int, *, fetch_fn,
                              provenance=None) -> None:
        """Fetch satellite tiles for the DEM extent off-thread, then drape them."""
        if self._elev is None or self._extent is None:
            self.drape_failed.emit("fetch a DEM before draping imagery onto it")
            return
        extent = self._extent
        self._drape_source = "satellite"

        def _work():
            try:
                img, ext = fetch_fn(provider, int(epsg), extent)
            except Exception as exc:
                log.exception("drape tile fetch failed")
                self.drape_failed.emit(f"drape fetch failed: {exc}")
                return
            try:
                from rasterio.transform import from_bounds
                left, right, bottom, top = ext
                tr = from_bounds(left, bottom, right, top, img.shape[1], img.shape[0])
                ok = self.apply_drape(img, tr, f"EPSG:{int(epsg)}",
                                      source="satellite",
                                      provenance=provenance or {"drape": "satellite"})
            except Exception as exc:
                log.exception("drape composite failed")
                self.drape_failed.emit(f"drape composite failed: {exc}")
                return
            if ok:
                self.draped.emit()

        t = threading.Thread(target=_work, daemon=True)
        self._last_drape_thread = t
        t.start()

    # ---- render (re-blit each map render) --------------------------------

    def render(self, ax) -> None:
        # When a drape is active, render the imagery×hillshade composite in this
        # layer's slot; otherwise the elevation tint. Both at z −9 (over basemap,
        # under data), so an active drape supersedes a redundant satellite basemap.
        img = self._composite if self._composite is not None else self._rgb
        if not self._visible or img is None or self._extent is None:
            return
        left, right, bottom, top = self._extent
        ax.imshow(img, extent=(left, right, bottom, top), origin="upper",
                  zorder=_DEM_ZORDER, interpolation="bilinear",
                  aspect="auto", alpha=0.9)

    # ---- off-thread fetch ------------------------------------------------

    def fetch(self, source_key: str, bounds_proj, project_epsg: int,
              dest_path: str, *, api_key: str | None = None, opener=None) -> None:
        """Fetch + store + load a DEM off the UI thread; emit :attr:`loaded`.

        The post-fetch path is split into named stages so a blank map can never
        again be silent about *which* stage broke: fetch/warp/store, the stored
        file, then load+hillshade each emit a specific :attr:`failed` message.
        """
        def _work():
            # Stage A — fetch + warp + store (network + rasterio in fetch_dem,
            # which logs the request bbox and warp stats at the boundaries).
            try:
                res = _dem.fetch_dem(source_key, bounds_proj, int(project_epsg),
                                     dest_path, api_key=api_key, opener=opener)
            except Exception as exc:
                log.exception("DEM fetch/warp/store failed")
                self.failed.emit(f"fetch/warp failed: {exc}")
                return

            # Stage A check — a real file with bytes, not an error body silently
            # written as a zero-length or HTML "GeoTIFF".
            try:
                size = os.path.getsize(res.path)
            except OSError as exc:
                self.failed.emit(f"stored DEM missing: {exc}")
                return
            if size <= 0:
                self.failed.emit(f"stored DEM is 0 bytes ({res.path})")
                return
            log.info("DEM stored: %s (%d bytes) ocean_filled=%s",
                     res.path, size, res.ocean_filled)

            # Stage B/C — load + hillshade, with stats and a specific message.
            ok, detail = self._load_geotiff_diagnosed(res.path)
            if not ok:
                self.failed.emit(detail)
                return
            self.loaded.emit()

        t = threading.Thread(target=_work, daemon=True)
        self._last_thread = t
        t.start()
