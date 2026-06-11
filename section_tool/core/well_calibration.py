"""Well calibration (M5) — optional promotion of assumed velocities to well-tied.

Opt-in and ADDITIVE: nothing in M1–M4 depends on this.  It runs only when wells
exist.  Given T-D control (depth/MD + two-way time, from formation tops + a
checkshot, or a sonic-derived TDR), it fits ``(v0, k)`` per layer with a ROBUST
estimator (Huber M-estimator / iteratively-reweighted least squares) so a single
bad marker can't drag the fit, then promotes the touched layers to
``well_calibrated`` (others stay ``assumed``/``interpolated``).  The retained
depth↔twt invertibility lets it report, per marker, both the depth residual and
the model-vs-measured TWT — the consistency diagnostic that only exists here.

scipy-only.  The sonic-log → TDR integration (welly) and rock-physics relations
(bruges) are the only place those libraries would enter; not bundled in the base.
SI internal: metres, seconds, m/s; ``k`` in s⁻¹.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from section_tool.core.velocity_model import (
    VelocityFunction, VelocityLayer, VelocityModel)

_V0_BOUNDS = (200.0, 7000.0)
_K_BOUNDS = (-0.5, 3.0)


@dataclass
class Marker:
    """A time-depth control point: a marker at depth *depth_m* (m) seen at
    two-way time *twt_s* (s).  Built from a well's formation tops + checkshot."""
    depth_m: float
    twt_s: float
    name: str = ""


def fit_v0k(depths, twts, *, v0_init: float = 1800.0, k_init: float = 0.3,
            robust: bool = True, f_scale: float = 0.02) -> tuple[float, float]:
    """Fit a ``v(z)=v0+k·z`` layer to (depth, twt) control by minimizing the TWT
    residual ``depth_to_twt(z) − t``.

    *robust* uses scipy's Huber loss (an M-estimator / IRLS), so an outlier is
    down-weighted instead of dragging the fit.  Returns ``(v0, k)`` in SI.
    Requires ≥ 2 markers (two free parameters).
    """
    z = np.asarray(depths, dtype=float)
    t = np.asarray(twts, dtype=float)
    if z.size < 2:
        raise ValueError("fit_v0k needs at least 2 markers to fit (v0, k)")
    from scipy.optimize import least_squares

    def resid(p):
        # Vectorized linear_v0k twt(z), domain-safe: the solver can explore
        # (v0, k) where 1 + k·z/v0 <= 0 (log domain error); penalize that region
        # instead of raising, so least_squares simply backs off.
        v0, k = float(p[0]), float(p[1])
        v0 = max(v0, 1.0)
        zc = np.clip(z, 0.0, None)
        if abs(k) < 1e-12:
            model = 2.0 * zc / v0
        else:
            arg = 1.0 + k * zc / v0
            model = np.where(arg > 1e-9,
                             (2.0 / k) * np.log(np.clip(arg, 1e-9, None)),
                             1.0e6)          # invalid region → large residual
        return model - t

    sol = least_squares(
        resid, [float(v0_init), float(k_init)],
        loss=("huber" if robust else "linear"), f_scale=f_scale,
        bounds=([_V0_BOUNDS[0], _K_BOUNDS[0]], [_V0_BOUNDS[1], _K_BOUNDS[1]]))
    return float(sol.x[0]), float(sol.x[1])


def marker_residuals(model: VelocityModel, markers) -> list[dict]:
    """Per-marker consistency diagnostic through *model* — both directions.

    depth_residual_m = model.twt_to_depth(measured_twt) − measured_depth
    twt_residual_s   = model.depth_to_twt(measured_depth) − measured_twt
    """
    out: list[dict] = []
    for m in markers:
        md = model.twt_to_depth(m.twt_s)
        mt = model.depth_to_twt(m.depth_m)
        out.append({
            "name": m.name,
            "depth_m": m.depth_m, "twt_s": m.twt_s,
            "model_depth_m": md, "depth_residual_m": md - m.depth_m,
            "model_twt_s": mt, "twt_residual_s": mt - m.twt_s,
        })
    return out


def calibrate_model(model: VelocityModel, markers, *, robust: bool = True) -> VelocityModel:
    """Return a NEW model with each layer that has ≥ 2 markers re-fit to the
    well control and promoted to ``well_calibrated``.

    Layer-top TWT boundaries are invariant (they come from seismic-tied picks);
    only the velocity law inside each layer is calibrated.  Layers with too few
    markers are copied unchanged (their provenance is preserved).  With no
    markers the model is returned effectively unchanged — the base path is
    unaffected when wells are absent.
    """
    if model.is_empty:
        return model
    tops = model._tops()                      # [(top_twt_s, top_depth_m), ...]
    n = len(model.layers)
    # Bin markers to the layer whose depth range contains them.
    by_layer: dict[int, list[Marker]] = {i: [] for i in range(n)}
    for m in markers:
        by_layer[model._layer_index_for(m.depth_m, axis=1)].append(m)

    new_layers: list[VelocityLayer] = []
    for i, layer in enumerate(model.layers):
        t_top, z_top = tops[i]
        ms = by_layer[i]
        if len(ms) >= 2:
            # Fit in LOCAL coordinates (measured from this layer's top).
            v0, k = fit_v0k([m.depth_m - z_top for m in ms],
                            [m.twt_s - t_top for m in ms], robust=robust,
                            v0_init=layer.function.v0,
                            k_init=layer.function.k or 0.3)
            new_layers.append(VelocityLayer(
                VelocityFunction("linear_v0k", v0=v0, k=k),
                top_twt_s=layer.top_twt_s, name=layer.name,
                formation=layer.formation, provenance="well_calibrated"))
        else:
            new_layers.append(VelocityLayer(
                VelocityFunction(layer.function.method, v0=layer.function.v0,
                                 k=layer.function.k),
                top_twt_s=layer.top_twt_s, name=layer.name,
                formation=layer.formation, provenance=layer.provenance,
                method_label=layer.method_label))
    return VelocityModel(
        layers=new_layers,
        construction={"kind": "velocity_model", "parents": [model.uuid],
                      "params": {"method": "well_calibrated",
                                 "n_markers": len(list(markers))}})


def resolve_marker_twt(depth_m: float, *, checkshot=None,
                       anchor_twt: float | None = None) -> tuple[float | None, str]:
    """The single, canonical source of a marker's TWT for well-tied calibration.

    Priority (Part A permanent fix — TWT is always an INDEPENDENT measurement,
    NEVER ``depth_to_twt(depth)`` off the current model):

    1. **Checkshot / TDR lookup** at the marker's depth (m, section frame) when a
       TDR covers it → an independent measured time.
    2. else the **matching picked horizon's ``twt_anchor``** (independent
       reflector tie) when supplied.
    3. else ``None`` — the marker has no independent TWT and must be EXCLUDED
       from the fit (the caller flags it).

    Returns ``(twt_s | None, source)`` where source ∈ {checkshot, horizon_anchor,
    excluded}.
    """
    if checkshot is not None:
        lo, hi = checkshot.depth_range()
        if lo - 1e-6 <= depth_m <= hi + 1e-6:
            return float(checkshot.twt_at_depth(depth_m)), "checkshot"
    if anchor_twt is not None and np.isfinite(anchor_twt):
        return float(anchor_twt), "horizon_anchor"
    return None, "excluded"


def _anchor_lookup(picks) -> dict:
    """name / formation_below → median ``twt_anchor`` for seismic-tied picks."""
    out: dict[str, float] = {}
    for hp in picks or []:
        if not getattr(hp, "seismic_tied", False):
            continue
        anch = getattr(hp, "_twt_anchor", None)
        if anch is None or len(anch) == 0:
            continue
        t = float(np.nanmedian(np.asarray(anch, dtype=float)))
        if t != t:                       # all-NaN
            continue
        for key in (getattr(hp, "name", ""), getattr(hp, "formation_below", "")):
            if key:
                out.setdefault(key, t)
    return out


def build_well_markers(well, picks=None, *, checkshot=None) -> tuple[list[Marker], dict]:
    """Markers for well-tied calibration, TWT sourced by :func:`resolve_marker_twt`.

    Marker depth is the formation top's MD brought to the section frame (TVDSS)
    through the well's deviation; TWT comes from the checkshot/TDR (or a matching
    horizon anchor), never the model.  Markers with no independent TWT are
    excluded and listed in the report.

    Returns ``(markers, report)``; report = ``{"used": [...], "excluded": [...]}``.
    """
    cs = checkshot if checkshot is not None else (
        well.primary_checkshot() if hasattr(well, "primary_checkshot") else None)
    anchors = _anchor_lookup(picks)
    markers: list[Marker] = []
    report: dict = {"used": [], "excluded": []}
    for name, md in sorted(well.formation_tops.items(), key=lambda kv: kv[1]):
        tvdss = float(well.deviation.tvd_at_md(float(md))) - float(well.kb)
        twt, source = resolve_marker_twt(
            tvdss, checkshot=cs, anchor_twt=anchors.get(name))
        if twt is None:
            report["excluded"].append({"name": name, "md": float(md)})
            continue
        markers.append(Marker(tvdss, twt, name))
        report["used"].append({"name": name, "twt_s": twt, "source": source})
    return markers, report


def well_td_control(well, checkshot) -> list[Marker]:
    """Build T-D markers from a well's formation tops + a *checkshot*.

    *checkshot* maps depth (m) → two-way time (s): either a callable ``z → t`` or
    a sequence of ``(depth_m, twt_s)`` pairs (linearly interpolated; clipped at
    the ends).  Formation-top MD is taken as TVD (v1 assumes near-vertical wells).
    """
    if callable(checkshot):
        to_twt = checkshot
    else:
        arr = np.asarray(sorted(checkshot, key=lambda p: p[0]), dtype=float)
        zs, ts = arr[:, 0], arr[:, 1]
        to_twt = lambda z: float(np.interp(z, zs, ts))
    return [Marker(depth_m=float(md), twt_s=float(to_twt(md)), name=name)
            for name, md in well.formation_tops.items()]
