"""Kinematic restoration engine — pure 2D geometric algorithms.

The functions here deform geometry: they receive ``(x, y)`` arrays (x =
along-section metres, y = depth metres, positive DOWN) plus scalar parameters and
return deformed ``(x, y)`` arrays.  No Section/Project/HorizonPick objects appear
inside the pure algorithms — just math — for maximum testability.

Algorithms (manual selection; automatic selection from construction metadata is a
later step):

* rigid_translation     — rigid-body translation (the plumbing baseline).
* flexural_slip_unfold  — layer-parallel slip; unfold to a datum, arc length
                          preserved (Dahlstrom 1969 / Suppe 1983).
* simple_shear          — inclined/vertical simple shear to a datum (extensional).
* fault_parallel_flow   — hangingwall translated parallel to the fault (thrusts).

A higher-level :func:`restore_snapshot` applies a chosen algorithm to a Step-3
interpretation snapshot, returning a NEW deformed snapshot (the original is never
touched) with the anchors-under-restoration rule applied.  SI internal.
"""
from __future__ import annotations

import copy

import numpy as np

KINEMATIC_ALGORITHMS = (
    "rigid_translation", "flexural_slip", "simple_shear", "fault_parallel_flow")

ALGORITHM_LABELS = {
    "none":                "None",
    "rigid_translation":   "Rigid translation",
    "flexural_slip":       "Flexural slip unfold",
    "simple_shear":        "Simple shear",
    "fault_parallel_flow": "Fault-parallel flow",
}


# ---------------------------------------------------------------------------
# Pure algorithms — (x, y) arrays in, (x, y) arrays out
# ---------------------------------------------------------------------------

def _as_xy(points) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("points must be an (N, 2) array of (x, y)")
    return p


def rigid_translation(points, dx: float, dy: float) -> np.ndarray:
    """Translate every point by ``(dx, dy)`` — rigid body, no rotation.

    The simplest case and the end-to-end plumbing check: trivial to hand-verify.
    """
    p = _as_xy(points)
    return p + np.array([float(dx), float(dy)])


def flexural_slip_unfold(points, pin_x: float, datum_y: float = 0.0) -> np.ndarray:
    """Unfold a folded layer to a horizontal *datum*, pinned at *pin_x*.

    Flexural (layer-parallel) slip preserves bed length, so the layer is laid out
    flat at ``y = datum_y`` with each node placed at its signed ARC-LENGTH distance
    from the pin: the pin node stays at ``x = pin_x`` and total length is conserved
    (Dahlstrom line-length balance).  *points* must be ordered by x.
    """
    p = _as_xy(points)
    if len(p) < 2:
        return np.column_stack([p[:, 0], np.full(len(p), float(datum_y))]) if len(p) else p
    seg = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])          # arc length from node 0
    s_pin = float(np.interp(pin_x, p[:, 0], s))          # arc length at the pin
    x_out = float(pin_x) + (s - s_pin)
    y_out = np.full(len(p), float(datum_y))
    return np.column_stack([x_out, y_out])


def simple_shear(points, shear_angle_deg: float, datum_y: float = 0.0) -> np.ndarray:
    """Restore points to a *datum* by inclined simple shear.

    Each point slides along the shear direction (``shear_angle_deg`` measured from
    VERTICAL; 0 = vertical shear) until it reaches ``y = datum_y``.  Vertical shear
    leaves x unchanged; an inclined shear shifts x by ``(datum_y − y)·tan(angle)``.
    The classic listric-fault hangingwall restoration to a regional datum.
    """
    p = _as_xy(points)
    theta = np.radians(float(shear_angle_deg))
    x_out = p[:, 0] + (float(datum_y) - p[:, 1]) * np.tan(theta)
    y_out = np.full(len(p), float(datum_y))
    return np.column_stack([x_out, y_out])


def fault_parallel_flow(points, fault_trace, slip: float) -> np.ndarray:
    """Translate the hangingwall *points* parallel to the fault by *slip* metres.

    *fault_trace* is the fault's ``(x, y)`` polyline; the slip direction is the
    fault's overall unit vector (first→last node).  For a planar fault this is the
    exact fault-parallel-flow result; for a curved fault it is the first-order
    (rigid-along-fault) approximation.  ``slip`` may be signed to choose direction.
    """
    p = _as_xy(points)
    ft = _as_xy(fault_trace)
    if len(ft) < 2:
        raise ValueError("fault_trace needs at least 2 nodes")
    d = ft[-1] - ft[0]
    norm = float(np.hypot(d[0], d[1]))
    if norm < 1e-12:
        raise ValueError("degenerate fault_trace (zero length)")
    u = d / norm
    return p + float(slip) * u


from dataclasses import dataclass


@dataclass
class AlgorithmProposal:
    """A restoration-algorithm proposal derived from an element's construction rule.

    Display-only defaults — the user always confirms (never silent/forced).
    """
    algorithm: str
    params: dict
    confidence: str          # "certain" (rule encodes the kinematic) | "suggested"
    source_kind: str         # the construction-rule kind it came from
    reason: str              # human-readable justification


# construction kind → (algorithm, confidence, reason). algorithm None = no inverse.
# Only listric_fault is "certain" — its rule explicitly records the fault geometry
# (the kinematic); the rest are the natural inverse with default params.
_RULE_INVERSE = {
    "parallel_to_bed":    ("flexural_slip", "suggested",
                           "a bed parallel to a reference bed restores by "
                           "layer-parallel (flexural) slip"),
    "kink_band":          ("flexural_slip", "suggested",
                           "kink folds are equal-area / constant bed length → "
                           "flexural slip unfold"),
    "dip_constrained":    ("simple_shear", "suggested",
                           "a constant-dip planar bed restores to the datum by "
                           "simple shear"),
    "listric_fault":      ("fault_parallel_flow", "certain",
                           "the listric fault geometry is recorded — the hangingwall "
                           "restores by fault-parallel flow along it"),
    "freehand":           (None, "none", "freehand geometry has no constraint to reverse"),
    "mirror_axial_trace": (None, "none", "mirror construction has no kinematic inverse here"),
}


def restore_by_construction_rule(entity, event=None) -> "AlgorithmProposal | None":
    """Propose a restoration algorithm for *entity* from its construction rule.

    The payoff of the construction-metadata architecture: the restoration rule
    reverses the construction rule. Reads ``entity.construction_rule.kind`` and
    returns an :class:`AlgorithmProposal` (algorithm + seeded params + confidence)
    or ``None`` when there is no applicable inverse (freehand / mirror / no rule).
    *event* carries the pin/datum context (sourced separately in the editor); it is
    accepted for symmetry and future seeding. Proposals are DEFAULTS — the caller
    presents them for the user to confirm or override.
    """
    rule = getattr(entity, "construction_rule", None)
    if rule is None:
        return None
    kind = getattr(rule, "kind", None)
    mapping = _RULE_INVERSE.get(kind)
    if mapping is None or mapping[0] is None:
        return None
    algorithm, confidence, reason = mapping
    params: dict = {}
    if algorithm == "simple_shear":
        params = {"shear_angle": 0.0}                    # vertical shear default
    elif algorithm == "fault_parallel_flow":
        params = {"fault_uuid": getattr(entity, "uuid", None)}   # this fault is the trace
    return AlgorithmProposal(algorithm=algorithm, params=params,
                             confidence=confidence, source_kind=kind, reason=reason)


def apply_algorithm(algorithm: str, points, params: dict, *,
                    fault_trace=None) -> np.ndarray:
    """Dispatch *algorithm* over *points* with *params* (and *fault_trace* when
    the algorithm needs one).  Returns deformed ``(x, y)``."""
    params = params or {}
    if algorithm == "rigid_translation":
        return rigid_translation(points, params.get("dx", 0.0), params.get("dy", 0.0))
    if algorithm == "flexural_slip":
        return flexural_slip_unfold(points, float(params.get("pin_x", 0.0)),
                                    float(params.get("datum_y", 0.0)))
    if algorithm == "simple_shear":
        return simple_shear(points, float(params.get("shear_angle", 0.0)),
                            float(params.get("datum_y", 0.0)))
    if algorithm == "fault_parallel_flow":
        if fault_trace is None:
            raise ValueError("fault_parallel_flow needs a fault_trace")
        return fault_parallel_flow(points, fault_trace, float(params.get("slip", 0.0)))
    raise ValueError(f"unknown kinematic algorithm: {algorithm!r}")


# ---------------------------------------------------------------------------
# Snapshot-level restoration (anchors-under-restoration applied here)
# ---------------------------------------------------------------------------

def _section_xy(pick, section_name):
    """(indices, (N,2) points) for *pick*'s nodes on *section_name*."""
    idxs = pick.section_indices(section_name)
    pts = np.column_stack([pick._distances[idxs], pick._depths[idxs]])
    return idxs, pts


def _reorder_point_arrays(pick) -> None:
    """Re-sort every per-point array by distance (deformation may reorder x)."""
    order = np.argsort(pick._distances, kind="stable")
    for arr in pick._POINT_ARRAYS:
        setattr(pick, arr, getattr(pick, arr)[order].copy())


def _deform_pick_inplace(pick, algorithm, params, section_name, fault_trace) -> None:
    """Deform *pick*'s nodes on *section_name* and apply anchors-under-restoration.

    The restored copy is in a NEW pre-deformation frame — it is no longer tied to
    the present seismic — so its TWT anchors are cleared and ``seismic_tied`` is
    dropped (``tie_kind`` then reads ``depth_native``).  *pick* is a snapshot copy;
    the original is untouched.
    """
    idxs, pts = _section_xy(pick, section_name)
    if len(pts):
        deformed = apply_algorithm(algorithm, pts, params, fault_trace=fault_trace)
        pick._distances[idxs] = deformed[:, 0]
        pick._depths[idxs] = deformed[:, 1]
    # anchors-under-restoration: the restored frame is depth-native, not seismic-tied
    if len(pick._twt_anchor):
        pick._twt_anchor[:] = np.nan
    pick.seismic_tied = False
    _reorder_point_arrays(pick)


def _ref_line_value(reference_lines, line_id):
    for rl in reference_lines or []:
        if getattr(rl, "uuid", None) == line_id:
            return float(getattr(rl, "value", 0.0))
    return None


def resolve_event_params(event, reference_lines) -> dict:
    """Effective params for *event*, resolving pin/datum from referenced lines.

    ``pin_line_id`` / ``datum_line_id`` (UUIDs) override the numeric ``pin_x`` /
    ``datum_y`` fallback when the line is found in *reference_lines* — so moving a
    pin/datum line updates every event that references it.  Numeric params stay the
    fallback for events without a line reference.
    """
    params = dict(getattr(event, "params", {}) or {})
    pid = getattr(event, "pin_line_id", None)
    if pid:
        v = _ref_line_value(reference_lines, pid)
        if v is not None:
            params["pin_x"] = v
    did = getattr(event, "datum_line_id", None)
    if did:
        v = _ref_line_value(reference_lines, did)
        if v is not None:
            params["datum_y"] = v
    return params


def _fault_trace(snapshot, fault_uuid, section_name):
    for fp in snapshot.faults:
        if getattr(fp, "uuid", None) == fault_uuid:
            _idxs, pts = _section_xy(fp, section_name)
            return pts
    return None


def restore_snapshot(snapshot, event, *, section_name: str, reference_lines=None):
    """Apply *event*'s algorithm to *snapshot*, returning a NEW deformed snapshot.

    The input snapshot (and therefore the original interpretation) is never
    mutated.  Every horizon / fault / polygon node on *section_name* is deformed by
    ``event.algorithm`` + its (resolved) params; the result carries
    ``restoration_frame = True`` and depth-native (anchor-cleared) picks — see
    :func:`_deform_pick_inplace`.  Pin/datum come from the referenced ReferenceLine
    (``pin_line_id`` / ``datum_line_id``) when set, else the numeric ``params``
    fallback; *reference_lines* defaults to the snapshot's captured lines but a
    caller can pass the LIVE lines so a moved pin updates the result immediately.
    Fault-parallel flow reads ``params['fault_uuid']``.
    """
    algorithm = getattr(event, "algorithm", "none")
    out = copy.deepcopy(snapshot)
    ref_lines = reference_lines if reference_lines is not None else out.reference_lines
    params = resolve_event_params(event, ref_lines)
    if algorithm in ("none", None):
        out.restoration_frame = True
        return out

    fault_trace = None
    if algorithm == "fault_parallel_flow":
        fault_trace = _fault_trace(out, params.get("fault_uuid"), section_name)

    for pick in list(out.horizons) + list(out.faults):
        _deform_pick_inplace(pick, algorithm, params, section_name, fault_trace)
    for poly in out.polygons:
        verts = apply_algorithm(algorithm, poly._vertices, params,
                                fault_trace=fault_trace)
        poly._vertices[:] = verts
        poly.free_points = poly._vertices
    out.restoration_frame = True
    return out
