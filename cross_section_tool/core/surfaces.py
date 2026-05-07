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
    ) -> None:
        distances = np.asarray(distances, dtype=float)
        depths = np.asarray(depths, dtype=float)
        if distances.ndim != 1 or depths.ndim != 1:
            raise ValueError("distances and depths must be 1D arrays")
        if len(distances) != len(depths):
            raise ValueError("distances and depths must have the same length")
        if len(distances) == 0:
            raise ValueError("HorizonPick requires at least one point")
        order = np.argsort(distances, kind="stable")
        self._distances = distances[order].copy()
        self._depths = depths[order].copy()
        self.name = name
        self.z_units: Literal["m", "ft", "ms"] = z_units
        self.color = color

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

    def insert_pick(self, distance: float, depth: float) -> None:
        """Insert a pick, maintaining ascending distance order."""
        idx = int(np.searchsorted(self._distances, distance))
        self._distances = np.insert(self._distances, idx, distance)
        self._depths = np.insert(self._depths, idx, depth)

    def delete_pick(self, index: int) -> None:
        """Delete the pick at *index*.

        Raises ValueError if only one pick remains.
        """
        if self.n_picks <= 1:
            raise ValueError("Cannot delete: HorizonPick must retain at least one pick")
        if not (0 <= index < self.n_picks):
            raise IndexError(f"index {index} out of range for {self.n_picks} picks")
        self._distances = np.delete(self._distances, index)
        self._depths = np.delete(self._depths, index)

    def move_pick(self, index: int, distance: float, depth: float) -> None:
        """Move the pick at *index* to (distance, depth), re-sorting as needed."""
        if not (0 <= index < self.n_picks):
            raise IndexError(f"index {index} out of range for {self.n_picks} picks")
        self._distances = np.delete(self._distances, index)
        self._depths = np.delete(self._depths, index)
        idx = int(np.searchsorted(self._distances, distance))
        self._distances = np.insert(self._distances, idx, distance)
        self._depths = np.insert(self._depths, idx, depth)

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
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"HorizonPick(name={self.name!r}, n_picks={self.n_picks}, "
            f"dist_range=[{self._distances[0]:.1f}, {self._distances[-1]:.1f}])"
        )
