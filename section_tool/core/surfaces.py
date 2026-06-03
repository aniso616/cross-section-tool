"""3D geological surface data model with section intersection.

Primary representation: irregular point cloud (N, 3) with on-demand
Delaunay interpolation. Regular grids are detected automatically and
use bilinear lookup for fast section intersection.
"""
from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


def new_entity_uuid() -> str:
    """Return a fresh stable identity (UUID4 string) for a horizon/fault entity."""
    return str(_uuid.uuid4())


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
    # Classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_xyz(
        cls,
        x: list | np.ndarray,
        y: list | np.ndarray,
        z: list | np.ndarray,
        *,
        name: str = "",
        crs_epsg: int = 0,
        z_domain: str = "depth_m",
        z_units: str = "m",
        color: tuple = (255, 165, 0),
        kind: str = "horizon",
        **kwargs,
    ) -> "Surface":
        """Create a scattered Surface from separate x, y, z arrays.

        Raises ValueError for fewer than 3 points or mismatched lengths.
        """
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        z = np.asarray(z, dtype=float).ravel()
        if not (len(x) == len(y) == len(z)):
            raise ValueError(
                f"x, y, z must have the same length (got {len(x)}, {len(y)}, {len(z)})"
            )
        if len(x) < 3:
            raise ValueError(f"Surface requires at least 3 points, got {len(x)}")
        points = np.column_stack([x, y, z])
        return cls(
            name=name,
            points=points,
            crs_epsg=crs_epsg,
            z_domain=z_domain,
            z_units=z_units,
            color=color,
            kind=kind,
            **kwargs,
        )

    @classmethod
    def from_grid(
        cls,
        xs: list | np.ndarray,
        ys: list | np.ndarray,
        z_grid: np.ndarray,
        *,
        name: str = "",
        crs_epsg: int = 0,
        z_domain: str = "depth_m",
        z_units: str = "m",
        color: tuple = (255, 165, 0),
        kind: str = "horizon",
        **kwargs,
    ) -> "Surface":
        """Create a Surface from 1-D coordinate axes and a 2-D Z array.

        Parameters
        ----------
        xs : 1-D array of X coordinates (length nx)
        ys : 1-D array of Y coordinates (length ny)
        z_grid : 2-D array, shape (ny, nx)
        """
        xs = np.asarray(xs, dtype=float).ravel()
        ys = np.asarray(ys, dtype=float).ravel()
        z_grid = np.asarray(z_grid, dtype=float)
        if z_grid.shape != (len(ys), len(xs)):
            raise ValueError(
                f"z_grid shape {z_grid.shape} does not match (len(ys)={len(ys)}, len(xs)={len(xs)})"
            )
        xx, yy = np.meshgrid(xs, ys)
        points = np.column_stack([xx.ravel(), yy.ravel(), z_grid.ravel()])
        gi = GridInfo(
            origin=(float(xs[0]), float(ys[0])),
            step_x=(float(xs[1] - xs[0]) if len(xs) > 1 else 1.0, 0.0),
            step_y=(0.0, float(ys[1] - ys[0]) if len(ys) > 1 else 1.0),
            nx=len(xs),
            ny=len(ys),
        )
        return cls(
            name=name,
            points=points,
            crs_epsg=crs_epsg,
            z_domain=z_domain,
            z_units=z_units,
            color=color,
            kind=kind,
            grid_info=gi,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _is_grid(self) -> bool:
        """True when the surface was created from a regular grid."""
        return self.grid_info is not None

    @property
    def _x(self) -> np.ndarray:
        return self.points[:, 0] if self.points is not None and len(self.points) else np.array([])

    @property
    def _y(self) -> np.ndarray:
        return self.points[:, 1] if self.points is not None and len(self.points) else np.array([])

    @property
    def _z(self) -> np.ndarray:
        return self.points[:, 2] if self.points is not None and len(self.points) else np.array([])

    @property
    def _grid_x(self) -> np.ndarray:
        """Unique x-axis coordinates for grid surfaces (1-D)."""
        if self.grid_info is not None:
            gi = self.grid_info
            return np.array([gi.origin[0] + i * gi.step_x[0] for i in range(gi.nx)])
        return np.unique(self._x)

    @property
    def _grid_y(self) -> np.ndarray:
        """Unique y-axis coordinates for grid surfaces (1-D)."""
        if self.grid_info is not None:
            gi = self.grid_info
            return np.array([gi.origin[1] + j * gi.step_y[1] for j in range(gi.ny)])
        return np.unique(self._y)

    @property
    def _grid_z(self) -> np.ndarray:
        """2-D Z array, shape (ny, nx), for grid surfaces."""
        gi = self.grid_info
        if gi is None:
            return self._z.reshape(-1, 1)
        xs = self._grid_x
        ys = self._grid_y
        z_2d = np.full((gi.ny, gi.nx), np.nan)
        for pt in self.points:
            ix = int(np.round((pt[0] - xs[0]) / gi.step_x[0])) if gi.step_x[0] != 0 else 0
            iy = int(np.round((pt[1] - ys[0]) / gi.step_y[1])) if gi.step_y[1] != 0 else 0
            if 0 <= ix < gi.nx and 0 <= iy < gi.ny:
                z_2d[iy, ix] = pt[2]
        return z_2d

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
            return
        try:
            interp = LinearNDInterpolator(xy, z, fill_value=np.nan)
            # Probe the centroid to detect degenerate (collinear) point sets
            centroid = np.array([[xy[:, 0].mean(), xy[:, 1].mean()]])
            if not np.isfinite(interp(centroid)[0]):
                self._interpolator = NearestNDInterpolator(xy, z)
            else:
                self._interpolator = interp
        except Exception:
            self._interpolator = NearestNDInterpolator(xy, z)

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
        if n_samples < 2:
            raise ValueError(f"n_samples must be ≥ 2, got {n_samples}")
        return self.intersect_section(section, n_samples)

    # ------------------------------------------------------------------
    # Cache invalidation
    # ------------------------------------------------------------------

    def invalidate_cache(self) -> None:
        """Call when points are modified externally."""
        self._interpolator = None

    def __repr__(self) -> str:
        kind_str = "grid" if self._is_grid else "scattered"
        return (
            f"Surface(name={self.name!r}, n_points={self.n_points}, "
            f"kind={kind_str!r}, z_domain={self.z_domain!r}, visible={self.visible})"
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
        slice_kinds: list | np.ndarray | None = None,
        uuid: str | None = None,
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
        # Per-point slice discriminant: 'section' (default) | 'horizontal'.
        # Parallel to _section_names, which holds the slice REFERENCE string
        # (a Section name or a HorizontalSlice name); _slice_kinds says which.
        if slice_kinds is None:
            skinds = np.array(["section"] * len(distances), dtype=object)
        else:
            skinds = np.asarray(slice_kinds, dtype=object).ravel()
            if len(skinds) != len(distances):
                raise ValueError("slice_kinds must have the same length as distances")
        order = np.argsort(distances, kind="stable")
        self._distances = distances[order].copy()
        self._depths = depths[order].copy()
        self._section_names: np.ndarray = snames[order].copy()
        self._slice_kinds: np.ndarray = skinds[order].copy()
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
        # Stable entity identity — rename-safe. Generated on creation; preserved
        # across save/reload and deepcopy. Distinct from the (mutable) name.
        self.uuid: str = uuid if uuid else new_entity_uuid()
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
        # Kinematic restoration: how this element was constructed
        self.construction_rule = None  # ConstructionRule | None

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
        """Sorted unique non-empty *section-kind* slice refs across all picks.

        Kind-aware: a horizontal-slice ref is NOT a section name and is excluded,
        so the section write/render paths never mistake one for the other.
        """
        return sorted(set(
            str(self._section_names[i]) for i in range(len(self._section_names))
            if self._slice_kinds[i] == "section" and str(self._section_names[i]) != ""
        ))

    def section_indices(self, section_name: str) -> np.ndarray:
        """Full-array indices for section-kind picks on *section_name* or globals ('')."""
        sec = (self._slice_kinds == "section")
        mask = sec & ((self._section_names == section_name) | (self._section_names == ""))
        return np.where(mask)[0]

    # ------------------------------------------------------------------
    # Slice-agnostic access (sections + horizontal slices)
    # ------------------------------------------------------------------

    def slice_keys(self) -> list[tuple[str, str]]:
        """Sorted distinct ``(slice_kind, slice_ref)`` pairs (non-empty ref)."""
        return sorted(set(
            (str(self._slice_kinds[i]), str(self._section_names[i]))
            for i in range(len(self._section_names))
            if str(self._section_names[i]) != ""
        ))

    def horizontal_slice_refs(self) -> list[str]:
        """Sorted unique non-empty horizontal-slice refs across all picks."""
        return sorted(set(
            str(self._section_names[i]) for i in range(len(self._section_names))
            if self._slice_kinds[i] == "horizontal" and str(self._section_names[i]) != ""
        ))

    def indices_for_slice(self, kind: str, ref: str) -> np.ndarray:
        """Full-array indices for picks on slice ``(kind, ref)`` exactly."""
        mask = (self._slice_kinds == kind) & (self._section_names == ref)
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
                    map_y: float = float("nan"),
                    slice_kind: str = "section") -> None:
        """Insert a pick, maintaining ascending distance order."""
        idx = int(np.searchsorted(self._distances, distance))
        self._distances     = np.insert(self._distances, idx, distance)
        self._depths        = np.insert(self._depths, idx, depth)
        self._section_names = np.insert(self._section_names, idx, section_name)
        self._slice_kinds   = np.insert(self._slice_kinds, idx, slice_kind)
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
        for attr in ("_distances", "_depths", "_section_names", "_slice_kinds",
                     "_confidence", "_quality", "_note", "_map_x", "_map_y"):
            setattr(self, attr, np.delete(getattr(self, attr), index))

    def move_pick(self, index: int, distance: float, depth: float,
                  map_x: float = float("nan"), map_y: float = float("nan")) -> None:
        """Move the pick at *index* to (distance, depth), re-sorting as needed."""
        if not (0 <= index < self.n_picks):
            raise IndexError(f"index {index} out of range for {self.n_picks} picks")
        sec  = self._section_names[index]
        kind = self._slice_kinds[index]
        conf = self._confidence[index]
        qual = self._quality[index]
        note = self._note[index]
        for attr in ("_distances", "_depths", "_section_names", "_slice_kinds",
                     "_confidence", "_quality", "_note", "_map_x", "_map_y"):
            setattr(self, attr, np.delete(getattr(self, attr), index))
        idx = int(np.searchsorted(self._distances, distance))
        self._distances     = np.insert(self._distances, idx, distance)
        self._depths        = np.insert(self._depths, idx, depth)
        self._section_names = np.insert(self._section_names, idx, sec)
        self._slice_kinds   = np.insert(self._slice_kinds, idx, kind)
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
        for attr in ("_distances", "_depths", "_section_names", "_slice_kinds",
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
        obj._slice_kinds   = np.array([], dtype=object)
        obj._confidence    = np.array([], dtype=float)
        obj._quality       = np.array([], dtype=object)
        obj._note          = np.array([], dtype=object)
        obj._map_x         = np.array([], dtype=float)
        obj._map_y         = np.array([], dtype=float)
        obj.name       = name
        obj.uuid       = new_entity_uuid()
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
        # Kinematic restoration — match __init__ so the attribute always exists
        # (a pickless entity is freehand until a construction tool sets a rule).
        obj.construction_rule = None
        return obj

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"HorizonPick(name={self.name!r}, n_picks={self.n_picks}, "
            f"dist_range={[round(self._distances[0],1), round(self._distances[-1],1)] if self.n_picks else '[]'})"
        )
