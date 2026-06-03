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


class Piercing(NamedTuple):
    """A point where a pick pierces the rendered slice, in that slice's coords.

    u, v       — slice_a's native 2D coords (distance/depth for a section,
                 easting/northing for a horizontal slice).
    x, y, z    — world coordinates of the piercing.
    """
    pick: object
    u: float
    v: float
    x: float
    y: float
    z: float


class SliceCrossing(NamedTuple):
    """Where slice_b crosses slice_a, expressed in slice_a's coords.

    locus_kind — 'v_line' | 'h_line' | 'polyline' | 'coincident' | 'none'.
    locus      — the crossing locus payload in slice_a coords:
                 v_line   → float u (a vertical line at distance u, all depths)
                 h_line   → float v (a horizontal line at depth v, all distances)
                 polyline → list[(u, v)] (a trace drawn in slice_a's plan)
                 coincident / none → None
    piercings  — list[Piercing]: where each pick pierces slice_a.
    """
    locus_kind: str
    locus: object
    piercings: list


def slice_crossing(slice_a, slice_b, picks) -> SliceCrossing:
    """Generic "where does slice_b cross slice_a", in slice_a's coords.

    Dispatches the four slice pairings using each slice's embedding data
    (Section trace nodes, HorizontalSlice elevation) — the Slice protocol's
    point transforms alone don't give the locus, so this is a small per-pairing
    helper, not a general geometry engine. *picks* is the pool of observations
    to test (each sampler filters to the relevant section internally).
    """
    ka, kb = slice_a.kind, slice_b.kind

    # section × section — the existing ghost geometry, unchanged.
    if ka == "section" and kb == "section":
        result = find_section_intersection(slice_a, slice_b)
        if result is None:
            return SliceCrossing("none", None, [])
        s_a, s_b = result
        pierc: list[Piercing] = []
        for hp in picks:
            depth = sample_pick_at_distance(hp, s_b, slice_b.name)
            if depth is None:
                continue
            x, y, z = slice_a.to_world(s_a, depth)
            pierc.append(Piercing(hp, float(s_a), float(depth), x, y, z))
        return SliceCrossing("v_line", float(s_a), pierc)

    # section × horizontal — locus is a horizontal line at depth = -z0;
    # dots are where this section's picks cross z0.
    if ka == "section" and kb == "horizontal":
        target_depth = -float(slice_b.elevation)
        pierc = []
        for hp in picks:
            for c in sample_pick_at_elevation(hp, slice_b.elevation, slice_a):
                pierc.append(Piercing(hp, c.distance, c.depth, c.x, c.y, c.z))
        return SliceCrossing("h_line", target_depth, pierc)

    # horizontal × section — locus is the section's trace in plan; dots are
    # where the section's picks cross z0, in world (= plan) coords.
    if ka == "horizontal" and kb == "section":
        pierc = []
        for hp in picks:
            for c in sample_pick_at_elevation(hp, slice_a.elevation, slice_b):
                pierc.append(Piercing(hp, c.x, c.y, c.x, c.y, c.z))
        trace = [(float(x), float(y)) for x, y in slice_b._nodes]
        return SliceCrossing("polyline", trace, pierc)

    # horizontal × horizontal — parallel planes: coincident iff equal elevation.
    if abs(float(slice_a.elevation) - float(slice_b.elevation)) < 1e-9:
        return SliceCrossing("coincident", None, [])
    return SliceCrossing("none", None, [])
