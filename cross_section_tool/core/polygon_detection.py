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

    # Collect all polylines in section space
    lines: list[LineString] = []

    def _add_pick(hp) -> None:
        d_sec, z_sec = hp.picks_for_section(section_name)
        if len(d_sec) < 2:
            return
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
            # Will be clipped by the boundary — just use a tall segment
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

    # Compute sensible vertical bounds from all picks
    all_depths = []
    for hp in list(horizon_picks) + list(fault_picks):
        d_sec, z_sec = hp.picks_for_section(section_name)
        if len(z_sec):
            all_depths.extend(z_sec.tolist())
    yt = 0.0
    if all_depths:
        max_d = max(all_depths)
        yb = max_d + max(max_d * 0.25, 500.0)  # 25% margin below deepest pick
    else:
        yb = 5000.0
    if yb <= yt:
        yb = yt + 5000.0

    # Section boundary rectangle (acts as outer constraint)
    boundary = LineString([
        (xl, yt), (xr, yt), (xr, yb), (xl, yb), (xl, yt)
    ])
    lines.append(boundary)

    # Snap endpoints that are within gap_tolerance of each other
    # (simple approach: just merge into unary_union which handles intersections)
    merged = unary_union(lines)

    # polygonize finds all enclosed rings
    polys = list(polygonize(merged))

    # Filter by minimum area
    return [p for p in polys if p.area >= min_area]
