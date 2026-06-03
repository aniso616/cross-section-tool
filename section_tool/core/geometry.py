"""Cross-section geometry utilities.

find_section_intersection   -- polyline-polyline intersection in map space
sample_pick_at_distance     -- interpolate a pick's depth at a section distance
sample_pick_at_elevation    -- find where a pick's polyline crosses a level z0
slice_crossing              -- generic "where does slice A pierce slice B"
"""
from __future__ import annotations

import math
from typing import NamedTuple


class LevelCrossing(NamedTuple):
    """One crossing of a pick polyline with a horizontal level.

    distance / depth — in the pick's section (distance_along, depth) coords.
    x / y / z        — world coordinates (NaN when no section was supplied).
    """
    distance: float
    depth: float
    x: float
    y: float
    z: float


def find_section_intersection(
    section_a,
    section_b,
) -> tuple[float, float] | None:
    """Return (s_a, s_b) where two section polylines cross, or None.

    s_a and s_b are distances along *section_a* and *section_b* respectively,
    measured from node 0.  Uses exact segment-segment intersection in map space
    (no tolerance: segments must share a point within floating-point precision).

    Parameters
    ----------
    section_a, section_b:
        :class:`~section_tool.core.section.Section` instances.

    Returns
    -------
    (s_a, s_b) or None
        First intersection found (iterating section_a segments outer loop).
    """
    cum_a = section_a.cumulative_distances()
    cum_b = section_b.cumulative_distances()
    nodes_a = section_a._nodes
    nodes_b = section_b._nodes

    for i in range(section_a.n_segments):
        ax1, ay1 = float(nodes_a[i, 0]),     float(nodes_a[i, 1])
        ax2, ay2 = float(nodes_a[i + 1, 0]), float(nodes_a[i + 1, 1])
        dax, day = ax2 - ax1, ay2 - ay1

        for j in range(section_b.n_segments):
            bx1, by1 = float(nodes_b[j, 0]),     float(nodes_b[j, 1])
            bx2, by2 = float(nodes_b[j + 1, 0]), float(nodes_b[j + 1, 1])
            dbx, dby = bx2 - bx1, by2 - by1

            # Solve: A1 + t*(A2-A1) = B1 + u*(B2-B1)
            denom = dax * dby - day * dbx
            if abs(denom) < 1e-12:
                continue  # parallel

            dx = bx1 - ax1
            dy = by1 - ay1
            t = (dx * dby - dy * dbx) / denom
            u = (dx * day - dy * dax) / denom

            if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                s_a = cum_a[i] + t * math.hypot(dax, day)
                s_b = cum_b[j] + u * math.hypot(dbx, dby)
                return (s_a, s_b)

    return None


def sample_pick_at_distance(pick, s: float, section_name: str) -> float | None:
    """Return the interpolated depth of *pick* at distance *s* on *section_name*.

    Parameters
    ----------
    pick:
        A :class:`~section_tool.core.surfaces.HorizonPick` instance.
    s:
        Distance along the section (same units as the pick distances).
    section_name:
        Which section's picks to use.

    Returns
    -------
    float or None
        Interpolated depth, or None if the pick has fewer than 1 node on the
        section or *s* is outside the pick's distance range.
    """
    import numpy as np

    idxs = pick.section_indices(section_name)
    if len(idxs) < 1:
        return None

    distances = pick._distances[idxs]
    depths    = pick._depths[idxs]

    order = np.argsort(distances)
    distances = distances[order]
    depths    = depths[order]

    if s < distances[0] or s > distances[-1]:
        return None

    return float(np.interp(s, distances, depths))


def sample_pick_at_elevation(pick, elevation: float, section=None) -> list[LevelCrossing]:
    """Find where *pick*'s polyline crosses the horizontal level at *elevation*.

    The pick lives in ``(distance_along, depth)`` coords; the target level is
    ``depth = -elevation`` (depth is positive-down, elevation positive-up). A
    pick may cross a level zero, one, or *many* times (folds), so this returns a
    list of all crossings ordered by distance — not just the first.

    Parameters
    ----------
    pick:
        A HorizonPick (used for faults too).
    elevation:
        Target elevation z0 (positive up). Crossing is at ``depth = -elevation``.
    section:
        Optional owning Section. When given, only the pick's points on that
        section are used and each crossing's world ``(x, y, z)`` is filled via
        ``section.to_world``; otherwise all points are used and world is NaN.

    Returns
    -------
    list[LevelCrossing]
        Empty when the pick never reaches the level (entirely above/below) or
        has fewer than 2 points to form a segment.
    """
    import numpy as np

    target_depth = -float(elevation)
    if section is not None:
        idxs = pick.section_indices(section.name)
    else:
        idxs = np.arange(pick.n_picks)
    if len(idxs) < 2:
        return []

    d = pick._distances[idxs].astype(float)
    z = pick._depths[idxs].astype(float)
    order = np.argsort(d, kind="stable")
    d, z = d[order], z[order]

    raw: list[float] = []   # crossing distances
    for i in range(len(d) - 1):
        z0, z1, d0, d1 = z[i], z[i + 1], d[i], d[i + 1]
        if z0 == z1:
            # Segment lies exactly on the level → one coincident crossing (start).
            if z0 == target_depth:
                raw.append(d0)
            continue
        # Transverse crossing: endpoints straddle the level (inclusive).
        if (z0 - target_depth) * (z1 - target_depth) <= 0.0:
            t = (target_depth - z0) / (z1 - z0)
            raw.append(d0 + t * (d1 - d0))

    # Dedup crossings coincident within tolerance (shared vertices on the level).
    crossings: list[float] = []
    for dc in raw:
        if not crossings or abs(dc - crossings[-1]) > 1e-9:
            crossings.append(dc)

    out: list[LevelCrossing] = []
    for dc in crossings:
        if section is not None:
            x, y, wz = section.to_world(dc, target_depth)
        else:
            x = y = wz = float("nan")
        out.append(LevelCrossing(distance=float(dc), depth=target_depth,
                                  x=float(x), y=float(y), z=float(wz)))
    return out
