"""Area of Interest — project-level spatial clip polygon."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AOI:
    """Single polygon in project CRS that bounds spatial queries.

    Stored as WKT in SQLite; all spatial tests use shapely.
    If the project has no AOI, ``project.aoi`` is ``None`` and everything
    is unbounded.
    """

    name: str
    polygon_wkt: str
    crs_epsg: int

    # ------------------------------------------------------------------
    # Cached shapely geometry (not serialised)
    # ------------------------------------------------------------------

    _polygon: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self._polygon = None  # lazy — built on first access

    @property
    def polygon(self):
        """Return the shapely Polygon, building it lazily."""
        if self._polygon is None:
            from shapely import from_wkt
            self._polygon = from_wkt(self.polygon_wkt)
        return self._polygon

    # ------------------------------------------------------------------
    # Spatial predicates
    # ------------------------------------------------------------------

    def contains_xy(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Return boolean array — True where (xs[i], ys[i]) is inside the AOI."""
        import shapely as shp
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        pts = shp.points(xs, ys)
        return shp.contains(self.polygon, pts)

    def bbox(self) -> tuple[float, float, float, float]:
        """Return (xmin, ymin, xmax, ymax) bounding box."""
        b = self.polygon.bounds   # (minx, miny, maxx, maxy) from shapely
        return b[0], b[1], b[2], b[3]

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_rectangle(
        cls,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        crs_epsg: int,
        name: str = "AOI",
    ) -> "AOI":
        """Construct from an axis-aligned bounding box."""
        from shapely.geometry import box
        poly = box(x_min, y_min, x_max, y_max)
        aoi = cls(name=name, polygon_wkt=poly.wkt, crs_epsg=crs_epsg)
        aoi._polygon = poly
        return aoi

    @classmethod
    def from_polygon_wkt(
        cls,
        wkt: str,
        crs_epsg: int,
        name: str = "AOI",
    ) -> "AOI":
        return cls(name=name, polygon_wkt=wkt, crs_epsg=crs_epsg)
