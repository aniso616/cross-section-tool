"""3D geological surface data model with section intersection.

Primary representation: irregular point cloud (N, 3) with on-demand
Delaunay interpolation. Regular grids are detected automatically and
use bilinear lookup for fast section intersection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Grid metadata
# ---------------------------------------------------------------------------

@dataclass
class GridInfo:
    """Optional metadata for surfaces that happen to lie on a regular grid."""
    origin: tuple           # (x0, y0)
    step_x: tuple           # (dx_x, dx_y) — supports rotated grids
    step_y: tuple           # (dy_x, dy_y)
    nx: int
    ny: int
    inline_range: tuple = None   # (il_min, il_max) — populated by survey readers
    xline_range: tuple = None    # (xl_min, xl_max)


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------

@dataclass
class Surface:
    """3D surface stored as irregular points with on-demand interpolation.

    All coordinates in geographic CRS. Z values may be depth, TWT, or
    elevation depending on z_domain. Section views handle domain conversion.
    """

    name: str
    points: np.ndarray          # shape (N, 3): X, Y, Z columns
    crs_epsg: int = 0
    z_domain: str = "depth_m"   # 'depth_m', 'twt_ms', 'twt_s', 'elevation_m'
    z_units: str = "m"
    color: tuple = (255, 165, 0)  # RGB 0-255, orange default
    line_width: float = 1.5
    source_file: str | None = None
    source_format: str | None = None
    interpolation: str = "linear"   # 'linear', 'nearest'
    visible: bool = True
    grid_info: Optional[GridInfo] = None
    kind: str = "horizon"           # 'horizon', 'fault', 'unconformity'
    map_display: str = "bbox"       # 'bbox', 'contours', 'points', 'none'

    # Private lazy-built interpolator — excluded from repr/compare/init
    _interpolator: object = field(default=None, repr=False, compare=False, init=False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_points(self) -> int:
        return len(self.points) if self.points is not None else 0

    @property
    def display_color(self) -> str:
        """Hex colour string (for matplotlib)."""
        r, g, b = self.color
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

    def bounds(self) -> tuple[float, float, float, float]:
        """Return (xmin, ymin, xmax, ymax) bounding box."""
        if self.n_points == 0:
            return (0.0, 0.0, 0.0, 0.0)
        return (
            float(self.points[:, 0].min()),
            float(self.points[:, 1].min()),
            float(self.points[:, 0].max()),
            float(self.points[:, 1].max()),
        )

    def extent(self) -> tuple[float, float, float, float]:
        """Return (xmin, xmax, ymin, ymax) — backward-compat alias for bounds()."""
        b = self.bounds()
        return b[0], b[2], b[1], b[3]

    def z_range(self) -> tuple[float, float]:
        """Return (zmin, zmax) of valid (finite) Z values."""
        if self.n_points == 0:
            return (0.0, 0.0)
        z = self.points[:, 2]
        valid = z[np.isfinite(z)]
        if len(valid) == 0:
            return (0.0, 0.0)
        return (float(valid.min()), float(valid.max()))

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def _build_interpolator(self) -> None:
        """Build lazy scipy interpolator from point cloud."""
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        xy = self.points[:, :2]
        z  = self.points[:, 2]
        if self.interpolation == "nearest":
            self._interpolator = NearestNDInterpolator(xy, z)
        else:
            self._interpolator = LinearNDInterpolator(xy, z, fill_value=np.nan)

    def z_along_polyline(self, xy_points: np.ndarray) -> np.ndarray:
        """Interpolate Z at an (M, 2) array of (x, y) positions.

        Returns an (M,) array; NaN where the surface has no data coverage.
        """
        if self.n_points < 3:
            return np.full(len(xy_points), np.nan)
        if self._interpolator is None:
            self._build_interpolator()
        return self._interpolator(xy_points).astype(float)

    def sample_many(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Backward-compat wrapper: interpolate Z at separate x/y arrays."""
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        return self.z_along_polyline(np.column_stack([xs, ys]))

    def sample(self, x: float, y: float) -> float:
        """Return Z at a single (x, y). Returns NaN outside data extent."""
        return float(self.z_along_polyline(np.array([[x, y]]))[0])

    # ------------------------------------------------------------------
    # Section intersection
    # ------------------------------------------------------------------

    def intersect_section(
        self, section, n_samples: int = 200
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (distances, z_values) along *section*'s polyline.

        Evenly spaced at *n_samples* positions from 0 to section.total_length().
        Z is NaN where the surface has no coverage.
        """
        total = section.total_length()
        distances = np.linspace(0.0, total, n_samples)
        xy = np.array([section.section_to_map(d) for d in distances])
        z_values = self.z_along_polyline(xy)
        return distances, z_values

    # profile_along_section — backward-compat alias
    def profile_along_section(
        self, section, n_samples: int = 200
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.intersect_section(section, n_samples)

    # ------------------------------------------------------------------
    # Cache invalidation
    # ------------------------------------------------------------------

    def invalidate_cache(self) -> None:
        """Call when points are modified externally."""
        self._interpolator = None

    def __repr__(self) -> str:
        return (
            f"Surface(name={self.name!r}, n_points={self.n_points}, "
            f"z_domain={self.z_domain!r}, visible={self.visible})"
        )


# ---------------------------------------------------------------------------
# Grid detection helper
# ---------------------------------------------------------------------------

def detect_grid(
    points: np.ndarray, tolerance: float = 0.01
) -> Optional[GridInfo]:
    """Return a GridInfo if *points* lie on a regular axis-aligned grid, else None."""
    if len(points) < 4:
        return None
    xs = np.unique(points[:, 0])
    ys = np.unique(points[:, 1])
    if len(xs) * len(ys) != len(points):
        return None
    dx = np.diff(xs)
    dy = np.diff(ys)
    if len(dx) == 0 or len(dy) == 0:
        return None
    if dx.std() / max(abs(dx.mean()), 1e-12) > tolerance:
        return None
    if dy.std() / max(abs(dy.mean()), 1e-12) > tolerance:
        return None
    return GridInfo(
        origin=(float(xs[0]), float(ys[0])),
        step_x=(float(dx.mean()), 0.0),
        step_y=(0.0, float(dy.mean())),
        nx=len(xs),
        ny=len(ys),
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
