"""Hillshaded-DEM underlay for the map.

Unlike the basemap, the DEM is fetched only on explicit user request (never on
pan/settle). This layer loads the stored project GeoTIFF, computes a greyscale
hillshade once, and re-blits it under the data layers on every render (the map
rebuilds each render). The network fetch runs off the UI thread; completion is
marshalled back via a Qt signal.
"""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal

from section_tool.core import dem as _dem

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
        try:
            data, extent, _epsg, prov = _dem.load_dem_geotiff(path)
        except Exception:
            return False
        # Ground sample distance from the stored extent (project-CRS metres).
        h, w = data.shape
        dx = abs(extent[1] - extent[0]) / max(w, 1)
        dy = abs(extent[3] - extent[2]) / max(h, 1)
        self._hs = _dem.hillshade(data, dx=dx or 1.0, dy=dy or 1.0)
        self._extent = extent
        self._provenance = prov or {}
        return True

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
        """Fetch + store + load a DEM off the UI thread; emit :attr:`loaded`."""
        def _work():
            try:
                _dem.fetch_dem(source_key, bounds_proj, int(project_epsg),
                               dest_path, api_key=api_key, opener=opener)
            except Exception as exc:
                self.failed.emit(str(exc))
                return
            if self.load_geotiff(dest_path):
                self.loaded.emit()
            else:
                self.failed.emit("DEM stored but could not be loaded.")

        t = threading.Thread(target=_work, daemon=True)
        self._last_thread = t
        t.start()
