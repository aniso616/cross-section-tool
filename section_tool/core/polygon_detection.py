"""Detect closed polygonal regions from a set of section-space lines.

Uses Shapely 2.x to build a planar arrangement and find all enclosed faces.
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import polygonize, unary_union


def detect_polygons(
    horizon_picks,          # list[HorizonPick]
    fault_picks,            # list[HorizonPick]
    reference_lines,        # list[ReferenceLine]
    section,                # Section
    section_name: str,
    min_area: float = 1.0,
    gap_tolerance: float = 50.0,
) -> list[Polygon]:
    """Return a list of Shapely Polygon objects representing detected closed regions.

    Parameters
    ----------
    horizon_picks, fault_picks
        Interpretation lines.
    reference_lines
        Construction guides.
    section
        The active section (for boundary dimensions and pick filtering).
    section_name
        Only picks tagged with this section (or global '') are used.
    min_area
        Polygons smaller than this (in section-space units squared) are ignored.
    gap_tolerance
        Near-miss tolerance: line endpoints within this distance are snapped together.
    """
    total  = section.total_length()
    xl, xr = 0.0, total

    lines: list[LineString] = []

    def _add_pick(hp) -> None:
        d_sec, z_sec = hp.picks_for_section(section_name)
        if len(d_sec) < 2:
            return
        # Sort by distance
        order = np.argsort(d_sec)
        d_sec, z_sec = d_sec[order], z_sec[order]

        # Extend to section left edge
        if d_sec[0] > xl + 1e-6:
            if len(d_sec) >= 2:
                slope = (z_sec[1] - z_sec[0]) / max(d_sec[1] - d_sec[0], 1e-9)
                z_left = z_sec[0] - slope * (d_sec[0] - xl)
            else:
                z_left = z_sec[0]
            d_sec = np.concatenate([[xl], d_sec])
            z_sec = np.concatenate([[z_left], z_sec])

        # Extend to section right edge
        if d_sec[-1] < xr - 1e-6:
            if len(d_sec) >= 2:
                slope = (z_sec[-1] - z_sec[-2]) / max(d_sec[-1] - d_sec[-2], 1e-9)
                z_right = z_sec[-1] + slope * (xr - d_sec[-1])
            else:
                z_right = z_sec[-1]
            d_sec = np.concatenate([d_sec, [xr]])
            z_sec = np.concatenate([z_sec, [z_right]])

        coords = list(zip(d_sec.tolist(), z_sec.tolist()))
        lines.append(LineString(coords))

    for hp in horizon_picks:
        _add_pick(hp)
    for fp in fault_picks:
        _add_pick(fp)
    for rl in reference_lines:
        if not rl.visible:
            continue
        if rl.kind == "horizontal":
            lines.append(LineString([(xl, rl.value), (xr, rl.value)]))
        elif rl.kind == "vertical":
            lines.append(LineString([(rl.value, -1e9), (rl.value, 1e9)]))
        elif rl.kind == "angled":
            ang = math.radians(rl.angle_deg)
            far = total * 2
            dx  = math.cos(ang) * far
            dy  = math.sin(ang) * far
            lines.append(LineString([
                (rl.anchor_x - dx, rl.anchor_y + dy),
                (rl.anchor_x + dx, rl.anchor_y - dy),
            ]))

    if not lines:
        return []

    # Vertical bounds from picks + margin
    all_depths = []
    for hp in list(horizon_picks) + list(fault_picks):
        d_sec, z_sec = hp.picks_for_section(section_name)
        if len(z_sec):
            all_depths.extend(z_sec.tolist())
    yt = 0.0
    yb = (max(all_depths) * 1.25 + 500.0) if all_depths else 5000.0
    if yb <= yt:
        yb = yt + 5000.0

    # Section boundary rectangle
    boundary = LineString([
        (xl, yt), (xr, yt), (xr, yb), (xl, yb), (xl, yt)
    ])
    lines.append(boundary)

    # Section bounding box for clipping results
    from shapely.geometry import box as _box
    section_box = _box(xl, yt, xr, yb)

    # Try unbuffered first; fall back to buffered if gap_tolerance > 0 and no results
    merged = unary_union(lines)
    polys_raw = list(polygonize(merged))

    if not polys_raw and gap_tolerance > 0:
        buffered = [ln.buffer(gap_tolerance / 2) for ln in lines]
        from shapely.ops import unary_union as _uu
        merged_buf = _uu(buffered)
        merged = merged_buf.boundary
        polys_raw = list(polygonize(merged))

    # Filter by minimum area, clip to section, and filter thin slivers
    section_area = total * yb
    area_threshold = max(min_area, section_area * 0.001)
    result = []
    for p in polys_raw:
        # Clip to section bounds
        try:
            p_clipped = p.intersection(section_box)
        except Exception:
            p_clipped = p
        if p_clipped.is_empty:
            continue
        # Handle MultiPolygon from clipping
        if p_clipped.geom_type == "MultiPolygon":
            parts = list(p_clipped.geoms)
        else:
            parts = [p_clipped]
        for part in parts:
            if part.area < area_threshold:
                continue
            # Check minimum bounding box dimension (filter thin slivers)
            try:
                mb = part.minimum_rotated_rectangle
                if mb is not None:
                    coords = list(mb.exterior.coords)
                    if len(coords) >= 4:
                        sides = [math.hypot(coords[i+1][0]-coords[i][0],
                                            coords[i+1][1]-coords[i][1])
                                 for i in range(3)]
                        if min(sides) < 20.0:
                            continue
            except Exception:
                pass
            result.append(part)
    return result
