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
    Returns ``(z_axis, depth_image)`` with depth_image (n_traces, n_depth)."""
    amp2d = np.asarray(amp2d, dtype=float)
    z_axis = None
    rows = []
    for tr in amp2d:
        z_axis, d = stretch_trace_to_depth(tr, dt_s, model, z_max, dz, t0)
        rows.append(d)
    return z_axis, np.array(rows)


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
