"""Build a gridded Surface from HorizonPick data collected across sections.

Public API
----------
collect_horizon_points_map_space(project, horizon_name) -> np.ndarray  (N, 3)
build_surface_from_picks(project, horizon_name, *, grid_resolution, method, aoi) -> Surface
"""
from __future__ import annotations

import numpy as np

from section_tool.core.surfaces import Surface


# ---------------------------------------------------------------------------
# Section-distance → map-space helper
# ---------------------------------------------------------------------------

def section_distance_to_map_xy(section, distance: float) -> tuple[float, float]:
    """Return (x, y) map coordinates for *distance* along *section*.

    Delegates to ``section.section_to_map``, which performs arc-length
    polyline interpolation and clamps to [0, total_length].

    Parameters
    ----------
    section : Section
        Any Section object with a ``section_to_map(distance)`` method.
    distance : float
        Distance along section in the same units as the section CRS.

    Returns
    -------
    (x, y) : tuple[float, float]
    """
    # section.section_to_map already exists and handles clamping / polyline
    # interpolation along section._nodes.
    return section.section_to_map(float(distance))


# ---------------------------------------------------------------------------
# Collect horizon points
# ---------------------------------------------------------------------------

def collect_horizon_points_map_space(project, horizon_name: str) -> np.ndarray:
    """Collect all (easting, northing, depth) triples for *horizon_name*.

    Iterates ``project.horizon_picks`` (list[HorizonPick]), finds picks whose
    ``name`` matches *horizon_name*, converts section-space (distance, depth)
    to map-space (x, y) via the stored ``_map_x`` / ``_map_y`` arrays when
    available, falling back to ``section.section_to_map`` via the section
    registry on the project.

    Parameters
    ----------
    project : Project
        Project instance with ``horizon_picks`` and ``sections`` lists.
    horizon_name : str
        Exact name to match against ``HorizonPick.name``.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Columns are X (easting), Y (northing), Z (depth).
        Returns an empty (0, 3) array when no matching picks are found.
    """
    pts: list[tuple[float, float, float]] = []

    # Build section lookup by name for fast access
    section_map: dict[str, object] = {s.name: s for s in project.sections}

    for pick in project.horizon_picks:
        if pick.name != horizon_name:
            continue

        n = pick.n_picks
        if n == 0:
            continue

        for i in range(n):
            dist  = float(pick._distances[i])
            depth = float(pick._depths[i])
            mx    = float(pick._map_x[i])
            my    = float(pick._map_y[i])

            if np.isfinite(mx) and np.isfinite(my):
                # Use stored map coordinates (most accurate)
                pts.append((mx, my, depth))
            else:
                # Fall back: convert via section geometry
                sname = str(pick._section_names[i])
                sec   = section_map.get(sname)
                if sec is None:
                    # Try any section if sname is blank / missing
                    if section_map:
                        sec = next(iter(section_map.values()))
                    else:
                        continue
                x, y = section_distance_to_map_xy(sec, dist)
                pts.append((x, y, depth))

    if not pts:
        return np.zeros((0, 3), dtype=float)

    return np.array(pts, dtype=float)


# ---------------------------------------------------------------------------
# Build gridded surface
# ---------------------------------------------------------------------------

def build_surface_from_picks(
    project,
    horizon_name: str,
    *,
    grid_resolution: float = 100.0,
    method: str = "linear",
    aoi=None,
) -> Surface:
    """Interpolate a regular-grid Surface from horizon picks across sections.

    Parameters
    ----------
    project : Project
    horizon_name : str
        Name of the horizon to surface.
    grid_resolution : float
        Approximate grid cell size in CRS units (default 100 m).
    method : {'linear', 'nearest'}
        Interpolation method.
    aoi : AOI | None
        If supplied, grid cells outside the AOI polygon are set to NaN.

    Returns
    -------
    Surface
        A gridded :class:`Surface` whose ``grid_info`` is populated.

    Raises
    ------
    ValueError
        If fewer than 3 unique map-space points are available.
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    pts = collect_horizon_points_map_space(project, horizon_name)
    if len(pts) < 3:
        raise ValueError(
            f"Surface requires at least 3 pick points; "
            f"got {len(pts)} for horizon {horizon_name!r}."
        )

    x_data = pts[:, 0]
    y_data = pts[:, 1]
    z_data = pts[:, 2]

    # ------------------------------------------------------------------
    # Determine grid extent
    # ------------------------------------------------------------------
    if aoi is not None:
        xmin, ymin, xmax, ymax = aoi.bbox()
    else:
        margin_x = (x_data.max() - x_data.min()) * 0.05 or grid_resolution
        margin_y = (y_data.max() - y_data.min()) * 0.05 or grid_resolution
        xmin = x_data.min() - margin_x
        xmax = x_data.max() + margin_x
        ymin = y_data.min() - margin_y
        ymax = y_data.max() + margin_y

    # ------------------------------------------------------------------
    # Build coordinate axes (aspect-ratio-aware)
    # ------------------------------------------------------------------
    span_x = xmax - xmin
    span_y = ymax - ymin
    # Ensure at least 2 cells per axis
    nx = max(2, int(np.round(span_x / grid_resolution)))
    ny = max(2, int(np.round(span_y / grid_resolution)))

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)

    # ------------------------------------------------------------------
    # Interpolate
    # ------------------------------------------------------------------
    xy = np.column_stack([x_data, y_data])
    xx, yy = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])

    if method == "nearest":
        interp = NearestNDInterpolator(xy, z_data)
        z_flat = interp(grid_pts).astype(float)
    else:
        try:
            lin = LinearNDInterpolator(xy, z_data, fill_value=np.nan)
            # Probe centroid to detect collinear / degenerate point sets
            centroid = np.array([[xy[:, 0].mean(), xy[:, 1].mean()]])
            probe = lin(centroid)[0]
            if np.isfinite(probe):
                interp = lin
            else:
                interp = NearestNDInterpolator(xy, z_data)
        except Exception:
            interp = NearestNDInterpolator(xy, z_data)
        z_flat = interp(grid_pts).astype(float)

    z_grid = z_flat.reshape(ny, nx)

    # ------------------------------------------------------------------
    # AOI clip — set out-of-polygon cells to NaN
    # ------------------------------------------------------------------
    if aoi is not None:
        inside = aoi.contains_xy(xx.ravel(), yy.ravel())
        outside = ~inside
        z_grid.ravel()[outside] = np.nan

    # ------------------------------------------------------------------
    # Determine CRS from project
    # ------------------------------------------------------------------
    crs_epsg = getattr(project, "crs_epsg", 0)

    return Surface.from_grid(
        xs,
        ys,
        z_grid,
        name=horizon_name,
        crs_epsg=crs_epsg,
        z_domain="depth_m",
        z_units="m",
        kind="horizon",
    )
