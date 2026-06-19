"""Burial history at a section point — the restoration↔thermal seam.

Each :class:`RestorationEvent` carries a geological age and strips the younger
overburden; :mod:`core.decompaction` gives the surviving column's decompacted
depths at that time. Combined, they yield a tracked horizon's BURIAL DEPTH through
geological time at a chosen sample point — the input the transient thermal solver
needs (Step 3 consumes it). This replaces the old hardcoded burial proxy.

Mechanics: the snapshot's horizons sampled at the sample point form the column
(shallow→deep). Restoration removes elements youngest-first — exactly the order
decompaction strips layers — so at restoration step *k* the count of column
horizons in ``removed_ids_at_step(k)`` is how many youngest layers are stripped;
the tracked horizon's depth is then its decompacted depth in that surviving column.

Units (boundary): age in **Ma** (million years), depth in **metres** positive
down. SI internal; the thermal solver converts Ma→s and °C→K itself.

Provenance-as-data: every result carries a ``source`` label — "restoration
sequence …" or "user-specified" — so the interface never lies about its origin.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_PHI0 = 0.5        # surface porosity (shale-like) when no lithology is known
DEFAULT_C = 5.0e-4        # compaction coefficient (1/m)


@dataclass
class BurialHistory:
    """``(age_Ma, depth_m)`` burial curve at a sample point, oldest first."""
    points: list[tuple[float, float]] = field(default_factory=list)
    source: str = "user-specified"
    horizon_uuid: str = ""
    x_along_section: float = 0.0

    @property
    def ages_ma(self) -> np.ndarray:
        return np.array([p[0] for p in self.points], dtype=float)

    @property
    def depths_m(self) -> np.ndarray:
        return np.array([p[1] for p in self.points], dtype=float)


def _sample_depth(pick, section_name: str, x_along: float):
    """Depth (m) of *pick* at along-section distance *x_along*, or None."""
    try:
        d, z = pick.picks_for_section(section_name)
    except Exception:
        return None
    if len(d) == 0:
        return None
    if len(d) == 1:
        return float(z[0])
    return float(np.interp(x_along, d, z, left=z[0], right=z[-1]))


def manual_burial_history(pairs, *, source: str = "user-specified") -> BurialHistory:
    """A user-specified ``(age_Ma, depth_m)`` table → a :class:`BurialHistory`.

    Sorted oldest-first; non-finite / negative-depth rows dropped.
    """
    pts: list[tuple[float, float]] = []
    for a, d in pairs:
        try:
            af, df = float(a), float(d)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(af) and np.isfinite(df)) or df < 0.0:
            continue
        pts.append((af, df))
    pts.sort(key=lambda p: -p[0])
    return BurialHistory(points=pts, source=source)


def burial_history_from_restoration(
    sequence, horizon_uuid: str, x_along_section: float, *, snapshot,
    section_name: str | None = None, porosity_lookup=None,
    phi0: float = DEFAULT_PHI0, c: float = DEFAULT_C,
) -> BurialHistory:
    """Derive the tracked horizon's burial history from the restoration *sequence*.

    *snapshot* is the captured interpretation (Step 3); its horizons sampled at
    *x_along_section* build the stratigraphic column. *porosity_lookup(name)* may
    return ``(phi0, c)`` per layer (e.g. from the lithology library); otherwise the
    shale-like defaults are used. Returns ``(age_Ma, depth_m)`` pairs oldest-first,
    one per restoration event plus present day.
    """
    from section_tool.core.decompaction import burial_history as _decomp_bh

    sec = section_name or (snapshot.section.get("name", "") if snapshot else "")
    seq_name = getattr(sequence, "name", "") or "sequence"
    src = f"restoration sequence: {seq_name}"

    # Column of horizons at the sample point, shallow → deep.
    cols: list[tuple[str, str, float]] = []
    for hp in getattr(snapshot, "horizons", []):
        depth = _sample_depth(hp, sec, x_along_section)
        if depth is not None:
            cols.append((hp.uuid, getattr(hp, "name", ""), depth))
    cols.sort(key=lambda t: t[2])

    tracked = next((i for i, (u, _n, _d) in enumerate(cols) if u == horizon_uuid), None)
    if tracked is None:
        return BurialHistory(points=[], source=src, horizon_uuid=horizon_uuid,
                             x_along_section=x_along_section)

    # Layers between the surface (0) and each horizon (boundary i = horizon i depth).
    boundaries = [0.0] + [d for (_u, _n, d) in cols]
    layers = []
    for i in range(len(cols)):
        p0, cc = (porosity_lookup(cols[i][1]) if porosity_lookup else (phi0, c))
        layers.append({"name": cols[i][1], "z_top": boundaries[i],
                       "z_bottom": boundaries[i + 1], "phi0": p0, "c": cc})

    bh = _decomp_bh(layers)                       # (n+1 steps, n+1 boundaries)
    tracked_boundary = tracked + 1               # base of the tracked horizon's layer
    overburden_uuids = [cols[i][0] for i in range(tracked + 1)]  # tracked + shallower

    points: list[tuple[float, float]] = []
    # Events are stored youngest-first (index 0 applied first); iterate oldest first.
    for j in range(len(sequence.events) - 1, -1, -1):
        ev = sequence.events[j]
        removed = sequence.removed_ids_at_step(j + 1)         # events[0..j] applied
        n_strip = sum(1 for u in overburden_uuids if u in removed)
        if n_strip > tracked:                                 # tracked not yet deposited
            continue
        age = float(ev.age_ma) if ev.age_ma is not None else 0.0
        points.append((age, float(bh[n_strip, tracked_boundary])))
    points.append((0.0, float(bh[0, tracked_boundary])))      # present day

    # De-duplicate (same age) keeping the first, oldest first.
    seen, ordered = set(), []
    for a, d in sorted(points, key=lambda p: -p[0]):
        if a not in seen:
            seen.add(a)
            ordered.append((a, d))
    return BurialHistory(points=ordered, source=src, horizon_uuid=horizon_uuid,
                         x_along_section=x_along_section)
