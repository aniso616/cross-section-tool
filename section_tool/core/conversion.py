"""Conversion engine — the method ladder + the TWT-anchor operations (M2).

Pure and headless-testable.  Produces (a) depth-stretched traces and (b) TWT
anchors for seismic-tied horizons, through a :class:`VelocityModel`.  Both
directions are first-class and retained: twt→depth drives the display; depth→twt
recovers anchors at pick/edit time and stays available for QC / export / well
calibration.  The model is invertible and stays that way.
"""
from __future__ import annotations

import numpy as np

from section_tool.core.velocity_model import (
    VelocityModel, VelocityLayer, VelocityFunction)


# ---------------------------------------------------------------------------
# The method ladder — model builders
# ---------------------------------------------------------------------------

def build_bulk(v: float) -> VelocityModel:
    """Bulk velocity (z = V·t/2). Needs no picks — the bootstrap default."""
    return VelocityModel.bulk(v)


def build_average_vz(v0: float, k: float) -> VelocityModel:
    """Average V(z) = v0 + k·z. Needs no picks."""
    return VelocityModel.average_vz(v0, k)


def build_layered_from_formations(zone_tops, strat_column,
                                  default_v: float = 3000.0) -> VelocityModel:
    """Layered interval-velocity model from picked zone-bounding horizons.

    *zone_tops* — ordered ``[(top_twt_s, formation_name), ...]`` for the layer
    tops, from the TWT anchors of the picked zone-bounding horizons (the first
    top is typically the datum/seafloor at t=0).  Each layer's interval velocity
    seeds from that formation's ``matrix_velocity`` (m/s), tagged *assumed*.

    Interpretation-gated: raises if there are no zone tops — with no picked
    horizons the layered method is unavailable by design (use bulk / average).
    """
    if not zone_tops:
        raise ValueError(
            "layered-from-formations needs picked zone-bounding horizons; "
            "use bulk or average V(z) until horizons exist")
    layers: list[VelocityLayer] = []
    for top_twt, fmname in sorted(zone_tops, key=lambda zt: zt[0]):
        fm = strat_column.get_formation(fmname) if strat_column is not None else None
        v = float(fm.matrix_velocity) if fm is not None else float(default_v)
        layers.append(VelocityLayer(
            VelocityFunction("constant", v0=v), top_twt_s=float(top_twt),
            name=fmname, formation=fmname, provenance="assumed",
            method_label=f"interval {v:.0f} m/s ({fmname or 'zone'})"))
    return VelocityModel(layers=layers,
                         construction={"kind": "velocity_model",
                                       "parents": [], "params": {"method": "layered_from_formations"}})


def zone_tops_from_picks(picks, base_twt_s: float = 0.0):
    """Ordered ``[(top_twt_s, formation_name), ...]`` for layered-from-formations,
    read from the seismic-tied picks' TWT anchors (Prompt 1).

    Layer tops are the picks' anchors.  The interval ABOVE the shallowest pick —
    from *base_twt_s* (the datum on land, the seafloor in a marine setup) — takes
    that pick's ``formation_above``; each pick's anchor then starts a layer of its
    ``formation_below``.  Depth-native and unanchored picks are ignored; returns
    ``[]`` when there are no seismic-tied picks (layered method unavailable).
    """
    tied: list[tuple[float, object]] = []
    for hp in picks or []:
        if not getattr(hp, "seismic_tied", False):
            continue
        anch = getattr(hp, "_twt_anchor", None)
        if anch is None or len(anch) == 0:
            continue
        t = float(np.nanmedian(anch))
        if t != t:                      # all-NaN anchor
            continue
        tied.append((t, hp))
    if not tied:
        return []
    tied.sort(key=lambda e: e[0])
    tops: list[tuple[float, str]] = []
    if float(base_twt_s) < tied[0][0] - 1e-9:        # cap layer (datum/seafloor → H1)
        tops.append((float(base_twt_s),
                     getattr(tied[0][1], "formation_above", "") or ""))
    for t, hp in tied:
        tops.append((t, getattr(hp, "formation_below", "") or getattr(hp, "name", "")))
    return tops


# ---------------------------------------------------------------------------
# Seismic image: vertical stretch (twt → depth)
# ---------------------------------------------------------------------------

def stretch_trace_to_depth(amp, dt_s: float, model: VelocityModel,
                           z_max: float, dz: float, t0: float = 0.0):
    """Resample one TWT trace onto a depth axis.

    *amp* — amplitudes at samples ``t0 + i·dt_s`` (s).  Returns ``(z_axis,
    depth_amp)`` on ``[0, z_max]`` step ``dz``; each depth sample pulls the
    amplitude at its TWT (``model.depth_to_twt(z)``) by linear interpolation.
    """
    amp = np.asarray(amp, dtype=float)
    nt = len(amp)
    t_samples = t0 + np.arange(nt) * float(dt_s)
    z_axis = np.arange(0.0, float(z_max) + float(dz), float(dz))
    twt_at_z = np.array([model.depth_to_twt(float(z)) for z in z_axis])
    depth_amp = np.interp(twt_at_z, t_samples, amp, left=0.0, right=0.0)
    return z_axis, depth_amp


def stretch_image_to_depth(amp2d, dt_s: float, model: VelocityModel,
                           z_max: float, dz: float, t0: float = 0.0):
    """Stretch every trace of a (n_traces, n_samples) TWT image to depth.
    Returns ``(z_axis, depth_image)`` with depth_image (n_traces, n_depth).

    The depth→TWT mapping ``twt_at_z`` is trace-INDEPENDENT, so it (and its
    interpolation indices/weights) is computed once and applied to all traces in
    one vectorized blend — not rebuilt per trace.  This is what keeps a re-stretch
    off the O(n_traces · n_depth) Python path that froze navigation.
    """
    amp2d = np.asarray(amp2d, dtype=float)
    if amp2d.ndim != 2:
        amp2d = np.atleast_2d(amp2d)
    n_traces, nt = amp2d.shape
    t_samples = t0 + np.arange(nt) * float(dt_s)
    z_axis    = np.arange(0.0, float(z_max) + float(dz), float(dz))
    twt_at_z  = np.array([model.depth_to_twt(float(z)) for z in z_axis])  # once

    if nt < 2:
        return z_axis, np.zeros((n_traces, len(z_axis)))

    # Precompute linear-interp indices + weights once for the shared query grid,
    # then blend all traces at once.  Out-of-range (left/right) → 0, matching
    # np.interp(..., left=0.0, right=0.0) used by stretch_trace_to_depth.
    idx = np.clip(np.searchsorted(t_samples, twt_at_z, side="right") - 1, 0, nt - 2)
    w   = (twt_at_z - t_samples[idx]) / (t_samples[idx + 1] - t_samples[idx])
    lo  = amp2d[:, idx]
    hi  = amp2d[:, idx + 1]
    depth_image = lo * (1.0 - w) + hi * w
    oob = (twt_at_z < t_samples[0]) | (twt_at_z > t_samples[-1])
    if oob.any():
        depth_image[:, oob] = 0.0
    return z_axis, depth_image


def stretch_image_to_depth_lateral(amp2d, dt_s: float, lateral_model, distances,
                                   z_max: float | None = None,
                                   dz: float | None = None, t0: float = 0.0):
    """Stretch a (n_traces, n_samples) TWT image to depth with a LATERALLY varying
    velocity: each trace is converted through ``lateral_model.model_at(distance)``
    for its along-section distance.  All traces land on a common depth grid so the
    result is still a rectangular image.

    *distances* — along-section distance (m) per trace (len == n_traces).
    Returns ``(z_axis, depth_image)``.  Unlike the single-model stretch the
    depth→TWT map differs per trace, so this is O(n_traces · n_depth); it is meant
    to run once on Apply (the view memoizes it — navigation never re-runs it).
    """
    amp2d = np.asarray(amp2d, dtype=float)
    if amp2d.ndim != 2:
        amp2d = np.atleast_2d(amp2d)
    n_traces, nt = amp2d.shape
    distances = np.asarray(distances, dtype=float)
    if distances.shape[0] != n_traces:
        raise ValueError("distances must have one entry per trace")
    t_samples = t0 + np.arange(nt) * float(dt_s)
    t_max = float(t_samples[-1]) if nt else 0.0

    models = [lateral_model.model_at(float(d)) for d in distances]
    if z_max is None:
        z_max = max((m.twt_to_depth(t_max) for m in models), default=0.0)
    if not (z_max > 0):
        return np.array([0.0]), np.zeros((n_traces, 1))
    if dz is None:
        dz = z_max / max(nt - 1, 1)
    z_axis = np.arange(0.0, float(z_max) + float(dz), float(dz))

    depth_image = np.zeros((n_traces, len(z_axis)))
    if nt >= 2:
        for i, m in enumerate(models):
            twt_at_z = np.array([m.depth_to_twt(float(z)) for z in z_axis])
            depth_image[i] = np.interp(twt_at_z, t_samples, amp2d[i],
                                       left=0.0, right=0.0)
    return z_axis, depth_image


# ---------------------------------------------------------------------------
# TWT anchors — both directions are first-class
# ---------------------------------------------------------------------------

def recover_anchors(hp, model: VelocityModel) -> np.ndarray:
    """Per-node TWT anchors (s) recovered from the horizon's current depths,
    through the SAME model that drives the display: anchor = depth_to_twt(z).
    Exact for the displayed geometry, however crude the model."""
    return np.array([model.depth_to_twt(float(z)) for z in hp._depths], dtype=float)


def set_anchors(hp, model: VelocityModel) -> None:
    """Seismic-tie *hp*: store TWT anchors from its current depths and flag it
    tied.  Called at pick/edit time (the anchor changes only here)."""
    hp._twt_anchor = recover_anchors(hp, model)
    hp.seismic_tied = True


def derive_depths(hp, model: VelocityModel) -> np.ndarray:
    """Per-node depths (m) derived from the horizon's invariant TWT anchors:
    z = twt_to_depth(anchor)."""
    return np.array([model.twt_to_depth(float(t)) for t in hp._twt_anchor], dtype=float)


def apply_depths_from_anchors(hp, model: VelocityModel) -> None:
    """Re-derive a seismic-tied horizon's depths from its invariant anchors
    (velocity iteration: anchors are NOT rewritten, depth follows the model so
    the horizon stays glued to its reflector).  No-op for depth-native geometry."""
    if getattr(hp, "seismic_tied", False) and len(getattr(hp, "_twt_anchor", [])):
        hp._depths = derive_depths(hp, model)


# ---------------------------------------------------------------------------
# Velocity iteration / re-stretch — apply a model across a section's geometry
# ---------------------------------------------------------------------------

def restretch_picks(picks, model: VelocityModel) -> int:
    """Re-derive every SEISMIC-TIED pick's depths from its invariant TWT anchors
    through *model*; depth-native picks stay fixed.  This is the velocity-
    iteration step — and the re-stretch contract: tied geometry follows its TWT
    through the new model (staying glued to its reflector, moving with the
    seismic backdrop), while depth-native geometry (freehand surfaces,
    well-marker horizons, imported depth surfaces) does not move.  Returns the
    number of picks re-derived."""
    moved = 0
    for hp in picks:
        if getattr(hp, "seismic_tied", False) and len(getattr(hp, "_twt_anchor", [])):
            apply_depths_from_anchors(hp, model)
            moved += 1
    return moved


def restretch_project(project, model: VelocityModel) -> int:
    """Apply *model* across a project's horizons and faults (tied geometry
    re-derives from anchors; depth-native untouched).  Returns count re-derived.
    The seismic backdrop is re-stretched separately via stretch_image_to_depth."""
    return (restretch_picks(getattr(project, "horizon_picks", []), model)
            + restretch_picks(getattr(project, "fault_picks", []), model))
