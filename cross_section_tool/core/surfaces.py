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
        z_units: Literal["m", "ft", "ms"] = "m",
        crs_epsg: int = 32632,
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
        self.name = name
        self.kind: Literal["horizon", "fault", "unconformity"] = kind
        self.z_units: Literal["m", "ft", "ms"] = z_units
        self.crs_epsg = int(crs_epsg)

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
            f"Surface(name={self.name!r}, n_points={len(self._x)}, "
            f"kind={self.kind!r}, storage={storage!r})"
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
        self.name = name
        self.z_units: Literal["m", "ft", "ms"] = z_units
        self.color = color
        self.line_width: float = float(line_width)
        self.line_style: str = str(line_style)

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
                    section_name: str = "") -> None:
        """Insert a pick, maintaining ascending distance order."""
        idx = int(np.searchsorted(self._distances, distance))
        self._distances = np.insert(self._distances, idx, distance)
        self._depths = np.insert(self._depths, idx, depth)
        self._section_names = np.insert(self._section_names, idx, section_name)

    def delete_pick(self, index: int) -> None:
        """Delete the pick at *index*.

        Raises ValueError if fewer than two picks remain (would leave < 1).
        """
        if self.n_picks <= 1:
            raise ValueError("Cannot delete: HorizonPick must retain at least one pick")
        if not (0 <= index < self.n_picks):
            raise IndexError(f"index {index} out of range for {self.n_picks} picks")
        self._distances = np.delete(self._distances, index)
        self._depths = np.delete(self._depths, index)
        self._section_names = np.delete(self._section_names, index)

    def move_pick(self, index: int, distance: float, depth: float) -> None:
        """Move the pick at *index* to (distance, depth), re-sorting as needed."""
        if not (0 <= index < self.n_picks):
            raise IndexError(f"index {index} out of range for {self.n_picks} picks")
        sec = self._section_names[index]
        self._distances = np.delete(self._distances, index)
        self._depths = np.delete(self._depths, index)
        self._section_names = np.delete(self._section_names, index)
        idx = int(np.searchsorted(self._distances, distance))
        self._distances = np.insert(self._distances, idx, distance)
        self._depths = np.insert(self._depths, idx, depth)
        self._section_names = np.insert(self._section_names, idx, sec)

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

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
        obj.name       = name
        obj.z_units    = z_units
        obj.color      = color
        obj.line_width = float(line_width)
        obj.line_style = str(line_style)
        return obj

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"HorizonPick(name={self.name!r}, n_picks={self.n_picks}, "
            f"dist_range={[round(self._distances[0],1), round(self._distances[-1],1)] if self.n_picks else '[]'})"
        )
