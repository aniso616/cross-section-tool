"""Grounded velocity-model construction — the data-driven rungs of the ladder.

Two builders that turn well data into a layered :class:`VelocityModel` of the
SAME interval-velocity shape the rest of the stretch code already consumes:

``build_from_tdr``     — checkshot-tied: reproduce a TDR's (depth, twt) knots
                          with piecewise interval velocities (the top rung when a
                          checkshot exists).
``build_from_sonic``   — sonic V(z): integrate a sonic log to a TWT(z) curve,
                          optionally drift-corrected to a checkshot or to picked
                          horizon anchors.

Exposed on the entity as ``VelocityModel.from_tdr`` / ``.from_sonic`` (thin
delegators) so callers use one API; the heavy lifting lives here to keep the
entity class lean.  SI internal: depth m, TWT s, velocity m/s.

Datum: a TDR / log depth is brought into the **section frame (TVDSS, sea-level
referenced)** before building, always *through the well's deviation* (vertical
wells included), so a deviated well would be handled by the same path.
"""
from __future__ import annotations

import numpy as np

from section_tool.core.velocity_model import (
    VelocityModel, VelocityLayer, VelocityFunction)

# Physical slowness window for sonic despike, in µs/ft (≈ 1220–3810 m/s).
_SLOWNESS_MIN_US_FT = 80.0
_SLOWNESS_MAX_US_FT = 250.0
_US_FT_TO_S_PER_M = 1e-6 / 0.3048   # µs/ft → s/m
_US_M_TO_S_PER_M = 1e-6             # µs/m  → s/m
WATER_VELOCITY_MS = 1480.0


# ---------------------------------------------------------------------------
# Datum: bring a TDR's depth column into the section (TVDSS) frame via deviation
# ---------------------------------------------------------------------------

def depths_to_tvdss(depth_values, depth_reference: str, well) -> np.ndarray:
    """Convert a depth column in *depth_reference* to TVDSS (m), via the well.

    ``TVDSS``  → as-is (already sea-level referenced).
    ``MD``     → ``deviation.tvd_at_md(md) − kb`` (the path used for every well,
                 vertical or not).
    ``TVD_KB`` → ``tvd − kb``.
    """
    z = np.asarray(depth_values, dtype=float)
    ref = (depth_reference or "TVDSS").upper()
    if ref == "TVDSS":
        return z
    if ref == "TVD_KB":
        return z - float(well.kb)
    # MD (default): through the deviation survey, then subtract KB elevation.
    tvd_from_kb = np.array([well.deviation.tvd_at_md(float(m)) for m in z])
    return tvd_from_kb - float(well.kb)


# ---------------------------------------------------------------------------
# Shared: build cap + interval + extrapolation layers from (depth, twt) knots
# ---------------------------------------------------------------------------

def _interval_layers_from_knots(
    z, t, *, provenance: str, datum_twt_s: float,
    setting: str, seafloor_twt_s: float | None, water_v: float,
) -> tuple[list[VelocityLayer], dict]:
    """Layers reproducing (TVDSS *z*, TWT *t*) knots; shared by from_tdr/from_sonic.

    Between knots: constant interval velocity ``v = 2·Δz/Δt``.  CAP above the
    shallowest knot (marine water→seafloor then a fill landing on the first knot;
    onshore a single datum fill).  Extrapolate the last interval below the
    deepest knot.  Returns ``(layers, cap_params)``.
    """
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)
    order = np.argsort(t, kind="stable")
    z, t = z[order], t[order]
    if len(t) < 2:
        raise ValueError("need at least 2 knots")

    datum = float(datum_twt_s)
    z0, t0 = float(z[0]), float(t[0])
    layers: list[VelocityLayer] = []
    cap_params: dict = {"kind": "none"}

    if t0 > datum + 1e-9:
        if (setting == "marine" and seafloor_twt_s is not None
                and datum < float(seafloor_twt_s) < t0):
            sf = float(seafloor_twt_s)
            z_sf = water_v * (sf - datum) / 2.0
            layers.append(VelocityLayer(
                VelocityFunction("constant", v0=water_v),
                top_twt_s=datum, name="Water", provenance=provenance,
                method_label=f"water {water_v:.0f} m/s"))
            v_fill = max(2.0 * (z0 - z_sf) / (t0 - sf), 1.0) if t0 > sf else water_v
            layers.append(VelocityLayer(
                VelocityFunction("constant", v0=v_fill),
                top_twt_s=sf, name="Cap", provenance=provenance,
                method_label=f"cap {v_fill:.0f} m/s"))
            cap_params = {"kind": "marine_water_fill", "water_v": water_v,
                          "seafloor_twt_s": sf, "fill_v": v_fill}
        else:
            v_cap = max(2.0 * z0 / (t0 - datum), 1.0) if (t0 - datum) > 0 else 1500.0
            layers.append(VelocityLayer(
                VelocityFunction("constant", v0=v_cap),
                top_twt_s=datum, name="Cap", provenance=provenance,
                method_label=f"cap {v_cap:.0f} m/s"))
            cap_params = {"kind": "datum_fill", "v": v_cap}

    v_last = None
    for i in range(len(z) - 1):
        dt = t[i + 1] - t[i]
        v = max(2.0 * (z[i + 1] - z[i]) / dt, 1.0) if dt > 0 else 1500.0
        v_last = v
        layers.append(VelocityLayer(
            VelocityFunction("constant", v0=v),
            top_twt_s=float(t[i]), name=f"interval{i}", provenance=provenance,
            method_label=f"interval {v:.0f} m/s"))

    v_ext = v_last if v_last is not None else 1500.0
    layers.append(VelocityLayer(
        VelocityFunction("constant", v0=v_ext),
        top_twt_s=float(t[-1]), name="below", provenance=provenance,
        method_label=f"extrapolated {v_ext:.0f} m/s"))
    layers.sort(key=lambda l: l.top_twt_s)
    return layers, cap_params


# ---------------------------------------------------------------------------
# Step 1 — checkshot-tied: from a TDR
# ---------------------------------------------------------------------------

def build_from_tdr(
    well,
    tdr,
    *,
    setting: str = "marine",
    datum_twt_s: float = 0.0,
    seafloor_twt_s: float | None = None,
    water_v: float = WATER_VELOCITY_MS,
    provenance: str = "checkshot_tied",
) -> VelocityModel:
    """Layered model reproducing *tdr*'s (depth, twt) knots (see module docstring)."""
    z = depths_to_tvdss(tdr.depth_m, tdr.depth_reference, well)
    if len(np.asarray(z)) < 2:
        raise ValueError("from_tdr needs a TDR with at least 2 knots")
    layers, cap_params = _interval_layers_from_knots(
        z, tdr.twt_s, provenance=provenance, datum_twt_s=datum_twt_s,
        setting=setting, seafloor_twt_s=seafloor_twt_s, water_v=water_v)
    return VelocityModel(layers=layers, construction={
        "kind": "from_tdr",
        "parents": [getattr(well, "uuid", ""), tdr.uuid],
        "params": {
            "well_uuid": getattr(well, "uuid", ""),
            "tdr_uuid": tdr.uuid, "tdr_kind": tdr.kind,
            "extrapolation": "last_interval_velocity",
            "cap": cap_params,
        }})


# ---------------------------------------------------------------------------
# Step 2 — sonic V(z): integrate a sonic log, optionally drift-correct
# ---------------------------------------------------------------------------

def _sonic_candidates(well) -> list[str]:
    return [n for n in getattr(well, "log_names", [])
            if n.upper().startswith(("DT", "AC")) or "SONIC" in n.upper()]


def select_sonic_curve(well, curve: str | None = None) -> str:
    """Pick the sonic curve, preferring the corrected variant (DT:2 / *corr*)."""
    if curve is not None:
        return curve
    cands = _sonic_candidates(well)
    if not cands:
        raise ValueError("well has no sonic (DT) curve to integrate")
    for n in cands:
        up = n.upper()
        if "CORR" in up or up.endswith(":2"):
            return n
    return cands[0]


def slowness_to_s_per_m(values, units: str) -> tuple[np.ndarray, str]:
    """Convert a slowness column to s/m. Unit field is authoritative (µs/ft vs µs/m)."""
    arr = np.asarray(values, dtype=float)
    u = (units or "").lower().replace("µ", "u").replace(" ", "")
    if "ft" in u:
        return arr * _US_FT_TO_S_PER_M, "us/ft"
    if "/m" in u or u in ("us/m", "usec/m"):
        return arr * _US_M_TO_S_PER_M, "us/m"
    # Unlabelled → assume µs/ft (by far the most common log unit) and flag it.
    return arr * _US_FT_TO_S_PER_M, "us/ft?"


def integrate_sonic(well, curve_name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Integrate a sonic log to TWT(z): returns (tvdss_m, twt_rel_s, meta).

    ``twt_rel`` is two-way time relative to the log top (0 at the shallowest
    surviving sample).  Slowness is despiked (nulls + out-of-window samples
    dropped) and integrated over TVDSS (MD→TVD via the well's deviation).
    """
    log = well.get_log(curve_name)
    md = log.depths
    slow, unit_kind = slowness_to_s_per_m(log.values, log.units)

    finite = np.isfinite(slow) & np.isfinite(md)
    n_null = int((~finite).sum())
    md_c, slow_c = md[finite], slow[finite]

    lo = _SLOWNESS_MIN_US_FT * _US_FT_TO_S_PER_M
    hi = _SLOWNESS_MAX_US_FT * _US_FT_TO_S_PER_M
    inwin = (slow_c >= lo) & (slow_c <= hi)
    n_spike = int((~inwin).sum())
    md_c, slow_c = md_c[inwin], slow_c[inwin]
    if len(md_c) < 2:
        raise ValueError("sonic log too sparse after despiking")

    tvdss = np.array([well.deviation.tvd_at_md(float(m)) for m in md_c]) - float(well.kb)
    order = np.argsort(tvdss, kind="stable")
    tvdss, slow_c = tvdss[order], slow_c[order]
    # collapse any duplicate depths (keep first) so diffs are positive
    keep = np.concatenate([[True], np.diff(tvdss) > 1e-6])
    tvdss, slow_c = tvdss[keep], slow_c[keep]

    owt = np.concatenate([[0.0],
                          np.cumsum(0.5 * (slow_c[1:] + slow_c[:-1]) * np.diff(tvdss))])
    twt_rel = 2.0 * owt
    meta = {"curve": curve_name, "unit_kind": unit_kind,
            "n_null_dropped": n_null, "n_spike_dropped": n_spike,
            "n_used": int(len(tvdss)),
            "log_top_tvdss": float(tvdss[0]), "log_base_tvdss": float(tvdss[-1])}
    return tvdss, twt_rel, meta


def _strict_increasing(z, t):
    """Filter to strictly-increasing (z, t) — drops flats so layer math is valid."""
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)
    keep = [0]
    for i in range(1, len(z)):
        if z[i] > z[keep[-1]] + 1e-6 and t[i] > t[keep[-1]] + 1e-9:
            keep.append(i)
    keep = np.array(keep)
    return z[keep], t[keep]


def build_from_sonic(
    well,
    curve: str | None = None,
    drift_target: str = "none",
    *,
    checkshot=None,
    anchor_knots=None,
    setting: str = "marine",
    datum_twt_s: float = 0.0,
    seafloor_twt_s: float | None = None,
    water_v: float = WATER_VELOCITY_MS,
    knot_spacing_m: float = 100.0,
) -> VelocityModel:
    """Sonic V(z) model, optionally drift-corrected to a checkshot or anchors.

    *drift_target*: ``checkshot`` (warp to a checkshot TDR — provenance
    ``well_calibrated``), ``anchors`` (warp to (depth, twt) ``anchor_knots`` —
    ``sonic_derived``), or ``none`` (uncorrected — ``sonic_derived``).
    Knot depths used for the warp are honored exactly in the output model.
    """
    curve_name = select_sonic_curve(well, curve)
    tvdss, twt_rel, meta = integrate_sonic(well, curve_name)
    rel = lambda z: np.interp(z, tvdss, twt_rel)   # relative twt at depth

    # Model knot grid: a uniform depth grid plus any tie depths (honored exactly).
    grid = np.arange(tvdss[0], tvdss[-1] + knot_spacing_m, knot_spacing_m)
    drift_report = {"target": drift_target, "knots": []}

    if drift_target == "checkshot":
        cs = checkshot if checkshot is not None else well.primary_checkshot()
        if cs is None:
            raise ValueError("drift_target='checkshot' needs a checkshot TDR")
        zk = depths_to_tvdss(cs.depth_m, cs.depth_reference, well)
        obs = np.asarray(cs.twt_s, dtype=float)
        provenance = "well_calibrated"
    elif drift_target == "anchors":
        if not anchor_knots:
            raise ValueError("drift_target='anchors' needs anchor_knots [(depth,twt)]")
        ak = np.asarray(sorted(anchor_knots), dtype=float)
        zk, obs = ak[:, 0], ak[:, 1]
        provenance = "sonic_derived"
    else:  # none — anchor the log top to the cap (water) time from the datum
        zk, obs = np.array([]), np.array([])
        provenance = "sonic_derived"

    if len(zk):
        drift_k = obs - rel(zk)                    # includes the absolute offset
        order = np.argsort(zk); zk, drift_k = zk[order], drift_k[order]
        drift = lambda z: np.interp(z, zk, drift_k)  # flat-extrapolated at ends
        knot_z = np.unique(np.concatenate([grid, zk]))
        corrected = rel(knot_z) + drift(knot_z)
        for zi, oi in zip(zk, obs):
            drift_report["knots"].append({
                "depth_m": float(zi), "twt_obs_s": float(oi),
                "residual_pre_ms": float((oi - rel(zi)) * 1000.0),
                "residual_post_ms": float((oi - (rel(zi) + drift(zi))) * 1000.0)})
    else:
        # Uncorrected: shift so twt at the log top reflects the cap (water) time
        # from the datum (0 when the log starts at the datum, as for F02-01).
        t_top_abs = 2.0 * max(tvdss[0] - 0.0, 0.0) / water_v
        knot_z = grid
        corrected = t_top_abs + rel(knot_z)
        drift_report["params"] = "uncorrected"

    # Guard monotonic twt; the shared layer builder adds the CAP (datum→log top)
    # when the shallowest knot's twt is below the datum.
    corrected = np.maximum.accumulate(corrected)
    zk_m, tk_s = _strict_increasing(knot_z, corrected)
    if len(zk_m) < 2:
        raise ValueError("sonic integration produced too few usable knots")
    drift_report["max_residual_post_ms"] = max(
        (abs(k["residual_post_ms"]) for k in drift_report["knots"]), default=0.0)

    layers, cap_params = _interval_layers_from_knots(
        zk_m, tk_s, provenance=provenance, datum_twt_s=datum_twt_s,
        setting=setting, seafloor_twt_s=seafloor_twt_s, water_v=water_v)
    return VelocityModel(layers=layers, construction={
        "kind": "from_sonic",
        "parents": [getattr(well, "uuid", "")],
        "params": {
            "well_uuid": getattr(well, "uuid", ""),
            "curve": curve_name, "drift_target": drift_target,
            "sonic": meta, "drift": drift_report,
            "extrapolation": "last_interval_velocity", "cap": cap_params,
        }})
