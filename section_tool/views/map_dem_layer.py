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

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._visible = True
        self._hs = None                # 2D hillshade in [0, 1]
        self._extent = None            # (left, right, bottom, top) project CRS
        self._provenance = {}
        self._last_thread = None       # test hook

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
        # Ground sample distance from the stored extent (project-CRS metres).
        dx = abs(extent[1] - extent[0]) / max(w, 1)
        dy = abs(extent[3] - extent[2]) / max(h, 1)
        try:
            hs = _dem.hillshade(data, dx=dx or 1.0, dy=dy or 1.0)
        except Exception as exc:
            log.exception("DEM hillshade failed")
            return False, f"hillshade computation failed: {exc}"

        self._hs = hs
        self._extent = extent
        self._provenance = prov or {}

        elev_min, elev_max = float(np.nanmin(data)), float(np.nanmax(data))
        hs_min, hs_max = float(hs.min()), float(hs.max())
        flat = (hs_max - hs_min) < 1e-3
        detail = (f"DEM layer: {w}x{h} EPSG:{epsg} "
                  f"extent={tuple(round(v) for v in extent)} "
                  f"elev[{elev_min:.1f},{elev_max:.1f}] "
                  f"hillshade[{hs_min:.3f},{hs_max:.3f}] "
                  f"visible={self._visible} z={_DEM_ZORDER}")
        log.info(detail)
        if flat:
            # Not a hard failure (the data loaded), but a flat hillshade reads as
            # a blank/uniform wash — surface it instead of letting it look broken.
            log.warning("DEM hillshade is near-constant (range %.4f) — low relief "
                        "or vert_exag too small for this terrain", hs_max - hs_min)
        return True, detail

    def clear(self) -> None:
        self._hs = None
        self._extent = None
        self._provenance = {}

    # ---- render (re-blit each map render) --------------------------------

    def render(self, ax) -> None:
        if not self._visible or self._hs is None or self._extent is None:
            return
        left, right, bottom, top = self._extent
        ax.imshow(self._hs, extent=(left, right, bottom, top), origin="upper",
                  zorder=_DEM_ZORDER, cmap="gray", vmin=0.0, vmax=1.0,
                  interpolation="bilinear", aspect="auto", alpha=0.85)

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
