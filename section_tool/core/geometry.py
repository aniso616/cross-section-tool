"""Cross-section geometry utilities.

find_section_intersection   -- polyline-polyline intersection in map space
sample_pick_at_distance     -- interpolate a pick's depth at a section distance
"""
from __future__ import annotations

import math


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
