from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator


class Surface:
    """Map-space 2.5D surface: (x, y) → z, stored as scattered or regular-grid data."""

    def __init__(
        self,
        x: list | np.ndarray,
        y: list | np.ndarray,
        z: list | np.ndarray,
        name: str = "",
        kind: Literal["horizon", "fault", "unconformity"] = "horizon",
        z_units: Literal["m", "ft", "ms", "km", "s"] = "m",
        crs_epsg: int = 32632,
        display_color: str = "#E87722",
        source_file: str | None = None,
        source_format: str | None = None,
    ) -> None:
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        z = np.asarray(z, dtype=float).ravel()
        if not (len(x) == len(y) == len(z)):
            raise ValueError("x, y, z must have the same length")
        if len(x) < 3:
            raise ValueError("Surface requires at least 3 points")
        self._x = x
        self._y = y
        self._z = z
        self._is_grid = False
        self._interp = None
        self._hull = None   # lazy shapely convex hull
        self.name = name
        self.kind: Literal["horizon", "fault", "unconformity"] = kind
        self.z_units: Literal["m", "ft", "ms", "km", "s"] = z_units
        self.crs_epsg = int(crs_epsg)
        self.display_color: str = display_color
        self.source_file: str | None = source_file
        self.source_format: str | None = source_format

    @property
    def z_domain(self):
        """Return a :class:`~section_tool.core.zdomain.ZDomain` for this surface."""
        from section_tool.core.zdomain import ZDomain
        _map = {"ft": ZDomain.DEPTH_FT, "km": ZDomain.DEPTH_KM,
                "ms": ZDomain.TWT_MS,   "s":  ZDomain.TWT_S}
        return _map.get(self.z_units, ZDomain.DEPTH_M)

    @property
    def convex_hull(self):
        """Shapely Polygon of the data point convex hull (lazy, cached)."""
        if self._hull is None:
            import shapely
            self._hull = shapely.convex_hull(
                shapely.multipoints(np.column_stack([self._x, self._y]))
            )
        return self._hull

    @property
    def n_points(self) -> int:
        return len(self._x)

    @classmethod
    def from_grid(
        cls,
        x_coords: list | np.ndarray,
        y_coords: list | np.ndarray,
        z_grid: list | np.ndarray,
        **kwargs,
    ) -> "Surface":
        """Construct from a regular grid.

        x_coords: 1D sorted array of x values, shape (nx,)
        y_coords: 1D sorted array of y values, shape (ny,)
        z_grid:   2D array of z values, shape (ny, nx) — z_grid[i, j] is at
                  (x_coords[j], y_coords[i])
        """
        x_coords = np.asarray(x_coords, dtype=float)
        y_coords = np.asarray(y_coords, dtype=float)
        z_grid = np.asarray(z_grid, dtype=float)
        if z_grid.shape != (len(y_coords), len(x_coords)):
            raise ValueError(
                f"z_grid shape {z_grid.shape} must be (ny={len(y_coords)}, nx={len(x_coords)})"
            )
        xx, yy = np.meshgrid(x_coords, y_coords)
        inst = cls(xx.ravel(), yy.ravel(), z_grid.ravel(), **kwargs)
        inst._is_grid = True
        inst._grid_x = x_coords
        inst._grid_y = y_coords
        inst._grid_z = z_grid
        inst._hull = None
        return inst

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_interpolator(self) -> None:
        if self._is_grid:
            self._interp = RegularGridInterpolator(
                (self._grid_y, self._grid_x),
                self._grid_z,
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
        else:
            self._interp = LinearNDInterpolator(
                np.column_stack([self._x, self._y]),
                self._z,
                fill_value=np.nan,
            )

    def _ensure_interp(self) -> None:
        if self._interp is None:
            self._build_interpolator()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, x: float, y: float) -> float:
        """Return z at (x, y). Returns nan outside the data extent / convex hull."""
        self._ensure_interp()
        if self._is_grid:
            return float(self._interp([[y, x]])[0])
        return float(self._interp([[x, y]])[0])

    def sample_many(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Return z values at arrays of (x, y) coordinates."""
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        self._ensure_interp()
        if self._is_grid:
            pts = np.column_stack([ys, xs])
        else:
            pts = np.column_stack([xs, ys])
        return self._interp(pts).astype(float)

    def profile_along_section(
        self,
        section,
        n_samples: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample the surface along *section*, returning (distances, z_values).

        Both arrays have length *n_samples*, evenly spaced along the polyline.
        z_values will be nan where the surface has no data.
        """
        if n_samples < 2:
            raise ValueError("n_samples must be at least 2")
        distances = np.linspace(0.0, section.total_length(), n_samples)
        map_pts = np.array([section.section_to_map(d) for d in distances])
        z_values = self.sample_many(map_pts[:, 0], map_pts[:, 1])
        return distances, z_values

    # ------------------------------------------------------------------
    # Metadata / geometry helpers
    # ------------------------------------------------------------------

    def extent(self) -> tuple[float, float, float, float]:
        """Return (xmin, xmax, ymin, ymax) bounding box of the data points."""
        return (
            float(self._x.min()),
            float(self._x.max()),
            float(self._y.min()),
            float(self._y.max()),
        )

    def __repr__(self) -> str:
        storage = "grid" if self._is_grid else "scattered"
        return (
            f"Surface(name={self.name!r}, n_points={self.n_points}, "
            f"kind={self.kind!r}, z_units={self.z_units!r}, storage={storage!r})"
        )


# ---------------------------------------------------------------------------


class HorizonPick:
    """An interpreted horizon in section-space: ordered (distance, depth) pairs.

    Distances are always kept in ascending sorted order.
    """

    def __init__(
        self,
        distances: list | np.ndarray,
        depths: list | np.ndarray,
        name: str = "",
        z_units: Literal["m", "ft", "ms"] = "m",
        color: str = "#1f77b4",
        line_width: float = 1.5,
        line_style: str = "solid",
        section_names: list | np.ndarray | None = None,
        map_x: list | np.ndarray | None = None,
        map_y: list | np.ndarray | None = None,
        # Phase A: horizon / contact attributes
        contact_type: str = "conformable",
        formation_above: str = "",
        formation_below: str = "",
        age_ma: float | None = None,
        confidence: float = 1.0,
        event_id: int | None = None,
        # Phase B: fault attributes
        fault_type: str = "normal",
        dip_direction: str = "right",
        sense_of_slip: str = "dip_slip",
        displacement: float | None = None,
        age_activation_ma: float | None = None,
        age_cessation_ma: float | None = None,
    ) -> None:
        distances = np.asarray(distances, dtype=float)
        depths = np.asarray(depths, dtype=float)
        if distances.ndim != 1 or depths.ndim != 1:
            raise ValueError("distances and depths must be 1D arrays")
        if len(distances) != len(depths):
            raise ValueError("distances and depths must have the same length")
        if len(distances) == 0:
            raise ValueError("HorizonPick requires at least one point")
        if section_names is None:
            snames = np.array([""] * len(distances), dtype=object)
        else:
            snames = np.asarray(section_names, dtype=object).ravel()
            if len(snames) != len(distances):
                raise ValueError("section_names must have the same length as distances")
        order = np.argsort(distances, kind="stable")
        self._distances = distances[order].copy()
        self._depths = depths[order].copy()
        self._section_names: np.ndarray = snames[order].copy()
        n = len(distances)
        # Map-space source of truth — NaN when not yet set (legacy picks)
        if map_x is None:
            self._map_x: np.ndarray = np.full(n, np.nan)
            self._map_y: np.ndarray = np.full(n, np.nan)
        else:
            mx = np.asarray(map_x, dtype=float).ravel()
            my = np.asarray(map_y, dtype=float).ravel()
            self._map_x = mx[order].copy()
            self._map_y = my[order].copy()
        # Phase 3: per-point metadata
        n = len(distances)
        self._confidence: np.ndarray = np.ones(n, dtype=float)
        self._quality: np.ndarray    = np.array(["picked"] * n, dtype=object)
        self._note: np.ndarray       = np.array([""] * n, dtype=object)
        self.name = name
        self.z_units: Literal["m", "ft", "ms"] = z_units
        self.color = color
        self.line_width: float = float(line_width)
        self.line_style: str = str(line_style)
        # Phase A: horizon / contact attributes
        self.contact_type:    str            = str(contact_type)
        self.formation_above: str            = str(formation_above)
        self.formation_below: str            = str(formation_below)
        self.age_ma:          float | None   = age_ma
        self.confidence:      float          = float(confidence)
        self.event_id:        int | None     = event_id
        # Phase B: fault attributes
        self.fault_type:       str          = str(fault_type)
        self.dip_direction:    str          = str(dip_direction)
        self.sense_of_slip:    str          = str(sense_of_slip)
        self.displacement:     float | None = displacement
        self.age_activation_ma: float | None = age_activation_ma
        self.age_cessation_ma:  float | None = age_cessation_ma

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def distances(self) -> np.ndarray:
        return self._distances.copy()

    @property
    def depths(self) -> np.ndarray:
        return self._depths.copy()

    @property
    def n_picks(self) -> int:
        return len(self._distances)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, distance: float) -> float:
        """Linearly interpolate depth at *distance*. Returns nan outside the pick range."""
        return float(
            np.interp(distance, self._distances, self._depths, left=np.nan, right=np.nan)
        )

    def sample_many(self, distances: np.ndarray) -> np.ndarray:
        """Linearly interpolate depths at an array of distances."""
        distances = np.asarray(distances, dtype=float)
        return np.interp(
            distances, self._distances, self._depths, left=np.nan, right=np.nan
        ).astype(float)

    # ------------------------------------------------------------------
    # Pick operations
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Per-section access
    # ------------------------------------------------------------------

    def section_names(self) -> list[str]:
        """Return sorted unique non-empty section names across all picks."""
        return sorted(set(
            str(s) for s in self._section_names if str(s) != ""
        ))

    def section_indices(self, section_name: str) -> np.ndarray:
        """Full-array indices for picks on *section_name* or global picks ('')."""
        mask = (self._section_names == section_name) | (self._section_names == "")
        return np.where(mask)[0]

    def picks_for_section(
        self, section_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (distances, depths) for picks visible on *section_name*."""
        idxs = self.section_indices(section_name)
        return self._distances[idxs], self._depths[idxs]

    def n_picks_for_section(self, section_name: str) -> int:
        return int(len(self.section_indices(section_name)))

    # ------------------------------------------------------------------
    # Pick operations
    # ------------------------------------------------------------------

    def insert_pick(self, distance: float, depth: float,
                    section_name: str = "",
                    confidence: float = 1.0,
                    quality: str = "picked",
                    note: str = "",
                    map_x: float = float("nan"),
                    map_y: float = float("nan")) -> None:
        """Insert a pick, maintaining ascending distance order."""
        idx = int(np.searchsorted(self._distances, distance))
        self._distances     = np.insert(self._distances, idx, distance)
        self._depths        = np.insert(self._depths, idx, depth)
        self._section_names = np.insert(self._section_names, idx, section_name)
        self._confidence    = np.insert(self._confidence, idx, confidence)
        self._quality       = np.insert(self._quality, idx, quality)
        self._note          = np.insert(self._note, idx, note)
        self._map_x         = np.insert(self._map_x, idx, map_x)
        self._map_y         = np.insert(self._map_y, idx, map_y)

    def delete_pick(self, index: int) -> None:
        """Delete the pick at *index*."""
        if self.n_picks <= 1:
            raise ValueError("Cannot delete: HorizonPick must retain at least one pick")
        if not (0 <= index < self.n_picks):
            raise IndexError(f"index {index} out of range for {self.n_picks} picks")
        for attr in ("_distances", "_depths", "_section_names",
                     "_confidence", "_quality", "_note", "_map_x", "_map_y"):
            setattr(self, attr, np.delete(getattr(self, attr), index))

    def move_pick(self, index: int, distance: float, depth: float,
                  map_x: float = float("nan"), map_y: float = float("nan")) -> None:
        """Move the pick at *index* to (distance, depth), re-sorting as needed."""
        if not (0 <= index < self.n_picks):
            raise IndexError(f"index {index} out of range for {self.n_picks} picks")
        sec  = self._section_names[index]
        conf = self._confidence[index]
        qual = self._quality[index]
        note = self._note[index]
        for attr in ("_distances", "_depths", "_section_names",
                     "_confidence", "_quality", "_note", "_map_x", "_map_y"):
            setattr(self, attr, np.delete(getattr(self, attr), index))
        idx = int(np.searchsorted(self._distances, distance))
        self._distances     = np.insert(self._distances, idx, distance)
        self._depths        = np.insert(self._depths, idx, depth)
        self._section_names = np.insert(self._section_names, idx, sec)
        self._confidence    = np.insert(self._confidence, idx, conf)
        self._quality       = np.insert(self._quality, idx, qual)
        self._note          = np.insert(self._note, idx, note)
        self._map_x         = np.insert(self._map_x, idx, map_x)
        self._map_y         = np.insert(self._map_y, idx, map_y)

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def recompute_distances(self, section) -> None:
        """Recompute *_distances* from stored map coordinates after the section geometry changes.

        Only updates picks that have valid (non-NaN) map_x/map_y.  Legacy picks
        without map coordinates are left unchanged.
        """
        valid = ~(np.isnan(self._map_x) | np.isnan(self._map_y))
        if not np.any(valid):
            return
        idxs = np.where(valid)[0]
        for i in idxs:
            dist, _ = section.project_point(float(self._map_x[i]), float(self._map_y[i]))
            self._distances[i] = dist
        # Re-sort after distances changed
        order = np.argsort(self._distances, kind="stable")
        for attr in ("_distances", "_depths", "_section_names",
                     "_confidence", "_quality", "_note", "_map_x", "_map_y"):
            setattr(self, attr, getattr(self, attr)[order])

    def to_map_coords(self, section) -> list[tuple[float, float, float]]:
        """Convert picks to (x, y, z) map coordinates via *section*."""
        return [
            (*section.section_to_map(float(d)), float(z))
            for d, z in zip(self._distances, self._depths)
        ]

    def snap_to_surface(self, surface: Surface, section) -> "HorizonPick":
        """Return a new HorizonPick with depths replaced by *surface* z-values at pick locations."""
        map_pts = np.array([section.section_to_map(float(d)) for d in self._distances])
        new_depths = surface.sample_many(map_pts[:, 0], map_pts[:, 1])
        return HorizonPick(
            self._distances.copy(),
            new_depths,
            name=self.name,
            z_units=self.z_units,
            color=self.color,
            line_width=self.line_width,
            line_style=self.line_style,
            section_names=self._section_names.tolist(),
            contact_type=self.contact_type,
            formation_above=self.formation_above,
            formation_below=self.formation_below,
            age_ma=self.age_ma,
            confidence=self.confidence,
            event_id=self.event_id,
            fault_type=self.fault_type,
            dip_direction=self.dip_direction,
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    @classmethod
    def empty(
        cls,
        name: str = "",
        z_units: Literal["m", "ft", "ms"] = "m",
        color: str = "#1f77b4",
        line_width: float = 1.5,
        line_style: str = "solid",
    ) -> "HorizonPick":
        """Create a HorizonPick with no picks (for a newly-added horizon/fault)."""
        obj = object.__new__(cls)
        obj._distances     = np.array([], dtype=float)
        obj._depths        = np.array([], dtype=float)
        obj._section_names = np.array([], dtype=object)
        obj._confidence    = np.array([], dtype=float)
        obj._quality       = np.array([], dtype=object)
        obj._note          = np.array([], dtype=object)
        obj._map_x         = np.array([], dtype=float)
        obj._map_y         = np.array([], dtype=float)
        obj.name       = name
        obj.z_units    = z_units
        obj.color      = color
        obj.line_width = float(line_width)
        obj.line_style = str(line_style)
        # Phase A/B defaults
        obj.contact_type     = "conformable"
        obj.formation_above  = ""
        obj.formation_below  = ""
        obj.age_ma           = None
        obj.confidence       = 1.0
        obj.event_id         = None
        obj.fault_type       = "normal"
        obj.dip_direction    = "right"
        obj.sense_of_slip    = "dip_slip"
        obj.displacement     = None
        obj.age_activation_ma = None
        obj.age_cessation_ma  = None
        return obj

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"HorizonPick(name={self.name!r}, n_picks={self.n_picks}, "
            f"dist_range={[round(self._distances[0],1), round(self._distances[-1],1)] if self.n_picks else '[]'})"
        )
