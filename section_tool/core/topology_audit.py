"""Topology audit — detect common interpretation hygiene issues on a section.

Usage::

    from section_tool.core.topology_audit import audit_section
    issues = audit_section(section, project)
    for issue in issues:
        print(issue.severity, issue.description)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Literal


@dataclass
class AuditIssue:
    """One detected topology problem."""

    severity:     Literal["error", "warning", "info"]
    category:     str           # "Horizons" | "Faults" | "Polygons"
    entity_index: int           # index into the relevant list
    entity_name:  str
    description:  str
    position:     tuple[float, float] | None  # (distance, depth) for zoom-to
    auto_fixable: bool = False
    fix_action:   Callable | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _seg_intersect_self(
    d: list[float], z: list[float]
) -> list[tuple[float, float]]:
    """Return all self-intersection points of a polyline."""
    pts = []
    n = len(d) - 1
    for i in range(n):
        ax, ay = d[i], z[i]
        bx, by = d[i + 1], z[i + 1]
        dab_x, dab_y = bx - ax, by - ay
        for j in range(i + 2, n):   # skip adjacent
            cx, cy = d[j], z[j]
            ex, ey = d[j + 1], z[j + 1]
            dce_x, dce_y = ex - cx, ey - cy
            denom = dab_x * dce_y - dab_y * dce_x
            if abs(denom) < 1e-12:
                continue
            t = ((cx - ax) * dce_y - (cy - ay) * dce_x) / denom
            s = ((cx - ax) * dab_y - (cy - ay) * dab_x) / denom
            if 0.0 < t < 1.0 and 0.0 < s < 1.0:
                pts.append((ax + t * dab_x, ay + t * dab_y))
    return pts


def _deduplicate_picks(hp, sec_name: str, tol: float) -> "HorizonPick":
    """Return a copy of *hp* with consecutive duplicates on *sec_name* merged."""
    import copy
    import numpy as np
    from section_tool.core.surfaces import HorizonPick

    hp2 = copy.deepcopy(hp)
    idxs = list(hp2.section_indices(sec_name))
    if len(idxs) < 2:
        return hp2

    keep = [idxs[0]]
    for k in range(1, len(idxs)):
        prev = keep[-1]
        curr = idxs[k]
        dd = abs(float(hp2._distances[curr]) - float(hp2._distances[prev]))
        dz = abs(float(hp2._depths[curr]) - float(hp2._depths[prev]))
        if math.hypot(dd, dz) > tol:
            keep.append(curr)

    to_remove = sorted(set(idxs) - set(keep), reverse=True)
    for idx_r in to_remove:
        for attr in ("_distances", "_depths", "_section_names", "_slice_kinds",
                     "_confidence", "_quality", "_note", "_map_x", "_map_y"):
            import numpy as _np
            arr = getattr(hp2, attr)
            setattr(hp2, attr, _np.delete(arr, idx_r))
    return hp2


# ---------------------------------------------------------------------------
# Auditors
# ---------------------------------------------------------------------------

def _audit_pick_list(
    picks, category: str, sec_name: str, tol: float
) -> list[AuditIssue]:
    issues: list[AuditIssue] = []

    for oi, hp in enumerate(picks):
        idxs = hp.section_indices(sec_name)
        n = len(idxs)
        name = getattr(hp, "name", f"{category}[{oi}]") or f"{category}[{oi}]"

        if n == 0:
            continue

        if n < 2:
            issues.append(AuditIssue(
                severity="warning",
                category=category,
                entity_index=oi,
                entity_name=name,
                description=f"Only 1 node on this section — cannot form a line.",
                position=(float(hp._distances[idxs[0]]), float(hp._depths[idxs[0]])),
                auto_fixable=False,
            ))
            continue

        d_vals = [float(hp._distances[i]) for i in idxs]
        z_vals = [float(hp._depths[i])    for i in idxs]

        # ---- duplicate / near-duplicate points ----
        dupes: list[int] = []
        for k in range(n - 1):
            dd = abs(d_vals[k + 1] - d_vals[k])
            dz = abs(z_vals[k + 1] - z_vals[k])
            if math.hypot(dd, dz) <= tol:
                dupes.append(k)

        if dupes:
            pos = (d_vals[dupes[0]], z_vals[dupes[0]])

            def _make_fix(oi_=oi, cat_=category):
                def fix(state):
                    import copy
                    picks_ = (state.project.horizon_picks if cat_ == "Horizons"
                              else state.project.fault_picks)
                    hp_fixed = _deduplicate_picks(picks_[oi_], sec_name, tol)
                    if cat_ == "Horizons":
                        state.update_horizon_pick(oi_, hp_fixed)
                    else:
                        state.update_fault_pick(oi_, hp_fixed)
                return fix

            issues.append(AuditIssue(
                severity="warning",
                category=category,
                entity_index=oi,
                entity_name=name,
                description=f"{len(dupes)} near-duplicate node(s) within {tol:.1f} m.",
                position=pos,
                auto_fixable=True,
                fix_action=_make_fix(),
            ))

        # ---- tiny segments (< 5× tol) ----
        tiny: list[int] = []
        for k in range(n - 1):
            seg_len = math.hypot(d_vals[k + 1] - d_vals[k], z_vals[k + 1] - z_vals[k])
            if 0 < seg_len < 5 * tol and k not in dupes:
                tiny.append(k)

        if tiny:
            issues.append(AuditIssue(
                severity="info",
                category=category,
                entity_index=oi,
                entity_name=name,
                description=f"{len(tiny)} tiny segment(s) shorter than {5*tol:.1f} m.",
                position=(d_vals[tiny[0]], z_vals[tiny[0]]),
                auto_fixable=False,
            ))

        # ---- self-intersections ----
        crossings = _seg_intersect_self(d_vals, z_vals)
        if crossings:
            issues.append(AuditIssue(
                severity="error",
                category=category,
                entity_index=oi,
                entity_name=name,
                description=f"Self-intersecting polyline ({len(crossings)} crossing(s)).",
                position=crossings[0],
                auto_fixable=False,
            ))

    return issues


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def audit_section(section, project, tol: float = 1.0) -> list[AuditIssue]:
    """Audit all picks on *section* for topology issues.

    Parameters
    ----------
    section:
        The active :class:`~section_tool.core.section.Section`.
    project:
        The current :class:`~section_tool.io.project.Project`.
    tol:
        Duplicate-point tolerance in section units (default 1.0 m).

    Returns
    -------
    list[AuditIssue]
        Issues sorted by severity (error first), then entity name.
    """
    sec_name = section.name
    issues: list[AuditIssue] = []
    issues += _audit_pick_list(project.horizon_picks, "Horizons", sec_name, tol)
    issues += _audit_pick_list(project.fault_picks,   "Faults",   sec_name, tol)

    _severity_order = {"error": 0, "warning": 1, "info": 2}
    issues.sort(key=lambda i: (_severity_order[i.severity], i.entity_name))
    return issues
