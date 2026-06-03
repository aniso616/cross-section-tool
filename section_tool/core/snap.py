"""Snap engine for construction drawing tools.

All coordinates are in section-space (distance_m, depth_m).
Pixel conversion is delegated to the ``to_screen`` callback so this
module remains free of any Qt/Matplotlib dependency and is fully unit-testable.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np

if TYPE_CHECKING:
    from section_tool.core.surfaces import HorizonPick


@dataclass
class SnapResult:
    """Nearest snap target found by :func:`find_snap`."""
    pt:         tuple[float, float]
    kind:       Literal["endpoint", "midpoint", "intersection", "edge"]
    source_cat: str | None     # "Horizons" | "Faults" | None for edge/topology
    source_idx: int | None
    source_seg: int | None     # index of first vertex of the matched segment


# ---------------------------------------------------------------------------
# Low-level geometry helpers
# ---------------------------------------------------------------------------

def _seg_intersect(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, dx: float, dy: float,
) -> tuple[float, float] | None:
    """Finite segment AB × finite segment CD intersection, or None."""
    dab_x, dab_y = bx - ax, by - ay
    dcd_x, dcd_y = dx - cx, dy - cy
    denom = dab_x * dcd_y - dab_y * dcd_x
    if abs(denom) < 1e-12:
        return None
    t = ((cx - ax) * dcd_y - (cy - ay) * dcd_x) / denom
    s = ((cx - ax) * dab_y - (cy - ay) * dab_x) / denom
    if not (0.0 <= t <= 1.0 and 0.0 <= s <= 1.0):
        return None
    return (ax + t * dab_x, ay + t * dab_y)


def _ray_seg_intersect(
    ox: float, oz: float, slope: float,
    seg_d0: float, seg_z0: float,
    seg_d1: float, seg_z1: float,
) -> tuple[float, float] | None:
    """Intersection of ray from (ox, oz) with given slope and a finite segment.

    The ray is unbounded (t can be any value).  The segment parameter s must
    be in [0, 1] for a valid hit.
    """
    dd = seg_d1 - seg_d0
    dz = seg_z1 - seg_z0
    denom = slope * dd - dz
    if abs(denom) < 1e-12:
        return None
    s = (seg_z0 - oz - slope * seg_d0 + slope * ox) / denom
    if not (0.0 <= s <= 1.0):
        return None
    t = seg_d0 - ox + s * dd
    return (ox + t, oz + slope * t)


# ---------------------------------------------------------------------------
# Public snap engine
# ---------------------------------------------------------------------------

def find_snap(
    cursor: tuple[float, float],
    picks_by_cat: dict[str, list],
    threshold_px: float,
    to_screen: Callable[[float, float], tuple[float, float]],
    *,
    section_edges: tuple[float, float] = (0.0, 0.0),
    topology_pts: list[tuple[float, float]] = (),
    sec_name: str = "",
) -> SnapResult | None:
    """Return the best snap target near *cursor* within *threshold_px*, or None.

    Priority (ties broken by pixel distance, lower = better):
    intersection = 0, endpoint = 1, midpoint = 2, edge = 3.

    Parameters
    ----------
    cursor:        (distance, depth) under the mouse in section space.
    picks_by_cat:  {"Horizons": [...], "Faults": [...]} pick lists.
    threshold_px:  Snap radius in screen pixels.
    to_screen:     Maps (distance, depth) → (canvas_x, canvas_y).
    section_edges: (start_distance, end_distance) for edge snap targets.
    topology_pts:  Pre-computed topology intersection points.
    sec_name:      Active section name for filtering picks.
    """
    cx, cy = cursor
    ecx, ecy = to_screen(cx, cy)

    PRIORITY: dict[str, int] = {
        "intersection": 0, "endpoint": 1, "midpoint": 2, "edge": 3,
    }
    best_dist  = threshold_px
    best_prio  = 999
    best: SnapResult | None = None

    def _try(
        px: float, py: float, kind: str,
        cat: str | None = None,
        idx: int | None = None,
        seg: int | None = None,
    ) -> None:
        nonlocal best_dist, best_prio, best
        spx, spy = to_screen(px, py)
        d = math.hypot(ecx - spx, ecy - spy)
        p = PRIORITY[kind]
        if d <= threshold_px and (d < best_dist or (d == best_dist and p < best_prio)):
            best_dist = d
            best_prio = p
            best = SnapResult(pt=(px, py), kind=kind,  # type: ignore[arg-type]
                              source_cat=cat, source_idx=idx, source_seg=seg)

    # ---- endpoints and midpoints ----------------------------------------
    for cat, picks_list in picks_by_cat.items():
        for oi, hp in enumerate(picks_list):
            si = hp.section_indices(sec_name)
            if len(si) == 0:
                continue
            d_sec = hp._distances[si]
            z_sec = hp._depths[si]
            n = len(d_sec)
            _try(float(d_sec[0]),  float(z_sec[0]),  "endpoint", cat, oi, 0)
            if n > 1:
                _try(float(d_sec[-1]), float(z_sec[-1]), "endpoint", cat, oi, n - 2)
            for i in range(n - 1):
                md = (float(d_sec[i]) + float(d_sec[i + 1])) / 2
                mz = (float(z_sec[i]) + float(z_sec[i + 1])) / 2
                _try(md, mz, "midpoint", cat, oi, i)

    # ---- pairwise segment–segment intersections -------------------------
    segs: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    for cat, picks_list in picks_by_cat.items():
        for oi, hp in enumerate(picks_list):
            si = hp.section_indices(sec_name)
            if len(si) >= 2:
                segs.append((cat, oi, hp._distances[si], hp._depths[si]))

    for i in range(len(segs)):
        cat_a, oi_a, da, za = segs[i]
        for j in range(i + 1, len(segs)):
            _, _, db, zb = segs[j]
            for si_i in range(len(da) - 1):
                for si_j in range(len(db) - 1):
                    p = _seg_intersect(
                        float(da[si_i]),     float(za[si_i]),
                        float(da[si_i + 1]), float(za[si_i + 1]),
                        float(db[si_j]),     float(zb[si_j]),
                        float(db[si_j + 1]), float(zb[si_j + 1]),
                    )
                    if p is not None:
                        _try(p[0], p[1], "intersection", cat_a, oi_a, si_i)

    # ---- pre-computed topology intersection points ----------------------
    for td, tz in topology_pts:
        _try(float(td), float(tz), "intersection")

    # ---- section edges (20 px zone, regardless of threshold) -----------
    start_d, end_d = section_edges
    edge_thresh = max(20.0, threshold_px)
    for edge_d in (start_d, end_d):
        spx, spy = to_screen(edge_d, cy)
        d = math.hypot(ecx - spx, ecy - spy)
        if d <= edge_thresh and (best is None or d < best_dist):
            best_dist = d
            best_prio = PRIORITY["edge"]
            best = SnapResult(pt=(edge_d, cy), kind="edge",
                              source_cat=None, source_idx=None, source_seg=None)

    return best


# ---------------------------------------------------------------------------
# Geometric operations on HorizonPick
# ---------------------------------------------------------------------------

def _replace_section_points(
    hp: HorizonPick,
    sec_name: str,
    new_d: np.ndarray,
    new_z: np.ndarray,
) -> HorizonPick:
    """Deep-copy *hp*, then replace its points on *sec_name* with *(new_d, new_z)*."""
    hp_new = copy.deepcopy(hp)
    sec_idxs = hp_new.section_indices(sec_name)
    for idx_r in sorted(sec_idxs, reverse=True):
        for attr in ("_distances", "_depths", "_section_names", "_slice_kinds",
                     "_confidence", "_quality", "_note", "_map_x", "_map_y"):
            setattr(hp_new, attr, np.delete(getattr(hp_new, attr), idx_r))
    for d, z in zip(new_d, new_z):
        hp_new.insert_pick(float(d), float(z), sec_name)
    return hp_new


def trim_pick_at_entity(
    hp: HorizonPick,
    cut_entity: HorizonPick,
    keep_side_x: float,
    sec_name: str,
) -> HorizonPick:
    """Trim *hp* to its first intersection with *cut_entity*.

    *keep_side_x* is the distance-coordinate of the original click — picks
    on that side of the intersection are retained.  The intersection becomes
    the exact terminal endpoint; there is no floating-point gap.

    Raises
    ------
    ValueError
        If either pick has fewer than 2 points on the section, or if the
        two lines do not intersect within their extents.
    """
    si_hp  = hp.section_indices(sec_name)
    si_cut = cut_entity.section_indices(sec_name)
    if len(si_hp) < 2:
        raise ValueError("trim_pick_at_entity: source pick needs ≥ 2 points on section")
    if len(si_cut) < 2:
        raise ValueError("trim_pick_at_entity: cut entity needs ≥ 2 points on section")

    d_hp  = hp._distances[si_hp]
    z_hp  = hp._depths[si_hp]
    d_cut = cut_entity._distances[si_cut]
    z_cut = cut_entity._depths[si_cut]

    ix: float | None = None
    iz: float | None = None
    for i in range(len(d_hp) - 1):
        for j in range(len(d_cut) - 1):
            p = _seg_intersect(
                float(d_hp[i]),     float(z_hp[i]),
                float(d_hp[i + 1]), float(z_hp[i + 1]),
                float(d_cut[j]),    float(z_cut[j]),
                float(d_cut[j + 1]), float(z_cut[j + 1]),
            )
            if p is not None:
                ix, iz = p
                break
        if ix is not None:
            break

    if ix is None:
        raise ValueError("trim_pick_at_entity: lines do not intersect")

    keep_left = keep_side_x <= ix
    mask  = d_hp <= ix if keep_left else d_hp >= ix
    if keep_left:
        d_new = np.append(d_hp[mask], ix)
        z_new = np.append(z_hp[mask], iz)
    else:
        d_new = np.concatenate([[ix], d_hp[mask]])
        z_new = np.concatenate([[iz], z_hp[mask]])

    return _replace_section_points(hp, sec_name, d_new, z_new)


def extend_pick_to_entity(
    hp: HorizonPick,
    endpoint: Literal["start", "end"],
    target: HorizonPick,
    sec_name: str,
) -> HorizonPick:
    """Extend *hp*'s terminal slope until it hits *target*.

    Returns a new HorizonPick with the intersection inserted as the new
    endpoint.  Does NOT mutate inputs.

    Raises
    ------
    ValueError
        If source or target has fewer than 2 points on the section, or if
        no intersection is found.
    """
    si_hp  = hp.section_indices(sec_name)
    si_tgt = target.section_indices(sec_name)
    if len(si_hp) < 2:
        raise ValueError("extend_pick_to_entity: source needs ≥ 2 points on section")
    if len(si_tgt) < 2:
        raise ValueError("extend_pick_to_entity: target needs ≥ 2 points on section")

    d_hp  = hp._distances[si_hp]
    z_hp  = hp._depths[si_hp]
    d_tgt = target._distances[si_tgt]
    z_tgt = target._depths[si_tgt]

    if endpoint == "start":
        ox, oz = float(d_hp[0]),  float(z_hp[0])
        slope  = (float(z_hp[1]) - oz) / max(float(d_hp[1]) - ox, 1e-9)
    else:
        ox, oz = float(d_hp[-1]), float(z_hp[-1])
        slope  = (oz - float(z_hp[-2])) / max(ox - float(d_hp[-2]), 1e-9)

    best_t = float("inf")
    ix: float | None = None
    iz: float | None = None
    for j in range(len(d_tgt) - 1):
        p = _ray_seg_intersect(
            ox, oz, slope,
            float(d_tgt[j]),     float(z_tgt[j]),
            float(d_tgt[j + 1]), float(z_tgt[j + 1]),
        )
        if p is None:
            continue
        t_abs = abs(p[0] - ox)
        if t_abs < best_t:
            best_t = t_abs
            ix, iz = p

    if ix is None:
        raise ValueError("extend_pick_to_entity: ray does not intersect target")

    hp_new = copy.deepcopy(hp)
    hp_new.insert_pick(ix, iz, sec_name)
    return hp_new
