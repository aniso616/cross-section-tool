"""Velocity model — layered interval velocities for time ↔ depth conversion.

Evolved from the original single-function model (kept as the per-layer kernel).

SI internal truth: depth in metres, two-way time in SECONDS, velocity in m/s,
gradient ``k`` in s⁻¹.  v(z) = v0 + k·z with z LOCAL to each layer top (so v0 is
the velocity AT the layer top).  Display / IO convert at the boundary; time is
shown in ms where it surfaces.  The unit lives on the function so the
silent-error surface is an unlabelled number, not a wrong one.

The model is a layer-cake: layers ordered top→bottom, each keyed to a formation /
horizon-bounded zone, each carrying a velocity function plus provenance
(``assumed`` by default — the honest well-free norm) and an identifiable method
label.  Conversion accumulates: the running depth/time at each layer top is the
next layer's boundary condition.

Schema-versioned.  ``from_dict`` detects the v1 single-function schema
(``{model_type, v0, k}``) and wraps it as one *assumed* layer.
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np

from section_tool.core.surfaces import new_entity_uuid

SCHEMA_VERSION = 2

Provenance = Literal[
    "assumed", "interpolated", "sonic_derived", "seismic_velocity",
    "well_calibrated", "checkshot_tied",
]
Method     = Literal["constant", "linear_v0k", "linear_vt"]

# Display labels describe the velocity's SOURCE / modification, not a certainty
# claim — every zone is an interpretation; "well-tied" means fit to well control,
# not ground truth.  Internal enum values are unchanged (persistence-stable).
#
# Grounded-ladder additions (Prompt 05): ``sonic_derived`` (velocity integrated
# from a sonic log) and ``checkshot_tied`` (calibrated to a checkshot/TDR).
# ``seismic_velocity`` is reserved for the Dix prompt (Prompt 08) — the value is
# defined for persistence stability, but no rung/UI claims it yet.
PROVENANCE_LABEL: dict[str, str] = {
    "assumed":          "regional default",
    "interpolated":     "interpolated",
    "sonic_derived":    "sonic-derived",
    "seismic_velocity": "seismic-velocity",
    "well_calibrated":  "well-tied",
    "checkshot_tied":   "checkshot-tied",
}
# Weakest-provenance-dominates the headline (min by rank).  The tie/derived forms
# all sit above 'interpolated'; checkshot is the hardest tie.
_PROV_RANK = {
    "assumed": 0, "interpolated": 1,
    "sonic_derived": 2, "seismic_velocity": 2, "well_calibrated": 2,
    "checkshot_tied": 3,
}


# ---------------------------------------------------------------------------
# Per-layer kernel (the original validated math)
# ---------------------------------------------------------------------------

class VelocityFunction:
    """A single layer's velocity law, converting between LOCAL depth (m, measured
    from the layer top) and LOCAL two-way time (s).  ``v0`` is the velocity at the
    layer top; ``k`` is the gradient in s⁻¹.

    linear_v0k:  v(z) = v0 + k·z
        depth→twt:  T = (2/k)·ln(1 + k·z/v0)
        twt→depth:  z = (v0/k)·(exp(k·T/2) − 1)
    constant (or k→0):  z = v0·T/2 ,  T = 2·z/v0
    """

    def __init__(self, method: Method = "constant", v0: float = 2000.0,
                 k: float = 0.0, units: str = "m/s") -> None:
        if units != "m/s":
            raise ValueError(f"VelocityFunction is SI-internal (m/s); got {units!r}")
        if method == "linear_vt":
            # Room reserved for a v(t) variant; not implemented yet — fail loud
            # rather than silently mis-convert.
            raise NotImplementedError("linear_vt velocity function not implemented")
        self.method: Method = method
        self.v0: float = float(v0)   # m/s, at the layer top
        self.k:  float = float(k)    # s⁻¹
        self.units: str = units

    def velocity_at(self, dz: float) -> float:
        """Interval velocity (m/s) at local depth *dz* (m) into the layer."""
        if self.method == "constant":
            return self.v0
        return self.v0 + self.k * dz

    def depth_to_twt(self, dz: float) -> float:
        """Two-way time (s) to traverse local depth *dz* (m) from the layer top."""
        if dz <= 0.0:
            return 0.0
        if self.method == "constant" or self.k == 0.0:
            return 2.0 * dz / self.v0
        return (2.0 / self.k) * math.log(1.0 + self.k * dz / self.v0)

    def twt_to_depth(self, dt: float) -> float:
        """Local depth (m) reached after local two-way time *dt* (s)."""
        if dt <= 0.0:
            return 0.0
        if self.method == "constant" or self.k == 0.0:
            return dt * self.v0 / 2.0
        return (self.v0 / self.k) * (math.exp(self.k * dt / 2.0) - 1.0)

    def to_dict(self) -> dict:
        return {"method": self.method, "v0": self.v0, "k": self.k, "units": self.units}

    @classmethod
    def from_dict(cls, d: dict) -> "VelocityFunction":
        return cls(method=d.get("method", "constant"),
                   v0=d.get("v0", 2000.0), k=d.get("k", 0.0),
                   units=d.get("units", "m/s"))

    def __repr__(self) -> str:
        return f"VelocityFunction({self.method!r}, v0={self.v0}, k={self.k})"


# ---------------------------------------------------------------------------
# A layer in the cake
# ---------------------------------------------------------------------------

class VelocityLayer:
    """One layer: a velocity function + its top boundary (TWT, s), plus identity
    and provenance for honest display."""

    def __init__(self, function: VelocityFunction, top_twt_s: float = 0.0,
                 name: str = "", formation: str = "",
                 provenance: Provenance = "assumed",
                 method_label: str = "") -> None:
        self.function = function
        self.top_twt_s = float(top_twt_s)
        self.name = str(name)
        self.formation = str(formation)        # framework key (formation/zone)
        self.provenance: Provenance = provenance
        self.method_label = method_label or self.default_method_label()

    def default_method_label(self) -> str:
        f = self.function
        if f.method == "constant":
            return f"bulk {f.v0:.0f} m/s"
        if f.method == "linear_v0k":
            return f"V(z) v0={f.v0:.0f} m/s, k={f.k:g} s⁻¹"
        return f.method

    def to_dict(self) -> dict:
        return {"function": self.function.to_dict(), "top_twt_s": self.top_twt_s,
                "name": self.name, "formation": self.formation,
                "provenance": self.provenance, "method_label": self.method_label}

    @classmethod
    def from_dict(cls, d: dict) -> "VelocityLayer":
        return cls(VelocityFunction.from_dict(d.get("function", {})),
                   top_twt_s=d.get("top_twt_s", 0.0), name=d.get("name", ""),
                   formation=d.get("formation", ""),
                   provenance=d.get("provenance", "assumed"),
                   method_label=d.get("method_label", ""))


# ---------------------------------------------------------------------------
# The layered model
# ---------------------------------------------------------------------------

class VelocityModel:
    """Ordered (top→bottom) layered velocity model for TWT ↔ depth conversion.

    First-class project entity: stable ``uuid`` + construction metadata
    (``{kind, parents, params}`` — the core/construction.py shape).  Well-free by
    default; every layer's provenance defaults to ``assumed``.
    """

    def __init__(self, layers: list[VelocityLayer] | None = None,
                 uuid: str | None = None, construction: dict | None = None) -> None:
        self.layers: list[VelocityLayer] = list(layers) if layers else []
        self.uuid: str = uuid or new_entity_uuid()
        # Construction metadata, mirroring core/construction.py: kind discriminant,
        # parent uuids, params. Reversible / auditable; feeds restoration later.
        self.construction: dict = construction or {
            "kind": "velocity_model", "parents": [], "params": {}}
        self.schema_version = SCHEMA_VERSION

    # ---- convenience constructors (the simple methods on the ladder) ----

    @classmethod
    def bulk(cls, v: float, **kw) -> "VelocityModel":
        """Bulk velocity: a single constant-velocity layer (z = V·t/2)."""
        return cls(layers=[VelocityLayer(VelocityFunction("constant", v0=v),
                                         method_label=f"bulk {v:.0f} m/s")], **kw)

    @classmethod
    def average_vz(cls, v0: float, k: float, **kw) -> "VelocityModel":
        """Average V(z): a single v0 + k·z layer."""
        return cls(layers=[VelocityLayer(VelocityFunction("linear_v0k", v0=v0, k=k))],
                   **kw)

    # ---- identity / honesty ----

    @property
    def is_empty(self) -> bool:
        return not self.layers

    @property
    def method_label(self) -> str:
        if not self.layers:
            return "unconverted"
        if len(self.layers) == 1:
            return self.layers[0].method_label
        if self.construction.get("params", {}).get("method") == "layered_from_formations":
            return "layered — formation matrix velocities"
        return f"layered ({len(self.layers)} interval velocities)"

    @property
    def provenance(self) -> Provenance:
        """Weakest layer provenance dominates the headline — a no-well model
        never reports more certainty than its least-certain layer."""
        if not self.layers:
            return "assumed"
        return min((l.provenance for l in self.layers), key=lambda p: _PROV_RANK[p])

    # ---- layer-cake accumulation ----

    def _tops(self) -> list[tuple[float, float]]:
        """(top_twt_s, top_depth_m) per layer, accumulated top-down.  The running
        depth at each top is the boundary condition carried into the next layer."""
        tops: list[tuple[float, float]] = []
        z = 0.0
        for i, layer in enumerate(self.layers):
            tops.append((layer.top_twt_s, z))
            if i + 1 < len(self.layers):
                dt = self.layers[i + 1].top_twt_s - layer.top_twt_s
                z += layer.function.twt_to_depth(max(dt, 0.0))
        return tops

    def _layer_index_for(self, value: float, axis: int) -> int:
        """Index of the layer whose top boundary is the last ≤ *value*. *axis*:
        0 = compare TWT tops, 1 = compare depth tops."""
        tops = self._tops()
        idx = 0
        for i in range(len(tops)):
            if value >= tops[i][axis]:
                idx = i
            else:
                break
        return idx

    def twt_to_depth(self, twt_s: float) -> float:
        """Depth (m) for two-way time *twt_s* (s) through the layer cake."""
        if not self.layers:
            raise ValueError("velocity model is empty (unconverted)")
        tops = self._tops()
        idx = self._layer_index_for(twt_s, axis=0)
        t_top, z_top = tops[idx]
        return z_top + self.layers[idx].function.twt_to_depth(twt_s - t_top)

    def depth_to_twt(self, z_m: float) -> float:
        """Two-way time (s) for depth *z_m* (m) through the layer cake."""
        if not self.layers:
            raise ValueError("velocity model is empty (unconverted)")
        tops = self._tops()
        idx = self._layer_index_for(z_m, axis=1)
        t_top, z_top = tops[idx]
        return t_top + self.layers[idx].function.depth_to_twt(z_m - z_top)

    # ---- persistence ----

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "uuid": self.uuid,
            "construction": self.construction,
            "layers": [l.to_dict() for l in self.layers],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VelocityModel":
        if not d:
            return cls()
        # v1 single-function schema → one assumed layer spanning the section.
        if "schema_version" not in d and "layers" not in d:
            method: Method = ("linear_v0k" if d.get("model_type") == "linear_v0k"
                              else "constant")
            fn = VelocityFunction(method=method, v0=d.get("v0", 2000.0),
                                  k=d.get("k", 0.0))
            return cls(layers=[VelocityLayer(fn, top_twt_s=0.0,
                                             name="(migrated v1)",
                                             provenance="assumed")])
        return cls(
            layers=[VelocityLayer.from_dict(x) for x in d.get("layers", [])],
            uuid=d.get("uuid"),
            construction=d.get("construction"),
        )

    # ---- grounded constructors (data-driven rungs; logic in grounded_velocity) ----

    @classmethod
    def from_tdr(cls, well, tdr, **kw) -> "VelocityModel":
        """Checkshot-tied: reproduce a TimeDepthRelation's knots with interval
        velocities. See :func:`grounded_velocity.build_from_tdr`."""
        from section_tool.core.grounded_velocity import build_from_tdr
        return build_from_tdr(well, tdr, **kw)

    @classmethod
    def from_sonic(cls, well, curve=None, drift_target="none", **kw) -> "VelocityModel":
        """Sonic V(z): integrate a sonic log, optionally drift-corrected.
        See :func:`grounded_velocity.build_from_sonic`."""
        from section_tool.core.grounded_velocity import build_from_sonic
        return build_from_sonic(well, curve=curve, drift_target=drift_target, **kw)

    def __repr__(self) -> str:
        return (f"VelocityModel(layers={len(self.layers)}, "
                f"method={self.method_label!r}, provenance={self.provenance!r})")
