"""Lateral velocity variation along a section (M4) — well-free pseudo-well control.

Velocity is pinned at along-section **control points** (distance in metres), each
carrying a full layered :class:`VelocityModel`.  Between controls the per-layer
parameters (layer-top TWT, ``v0``, ``k``) are interpolated linearly along the
section; **beyond** the outermost control they are CLIPPED to the edge value — no
wild values past the data (the convex-hull instinct).

A control is just ``(distance, model)``: it needs no well, so a purely *assumed*
lateral field is first-class — pseudo-well control with zero real wells.  Real
wells become control points later (M5) without changing this structure.

1-D along-section for v1 (no cross-line / true 3-D).  SI internal: metres,
seconds, m/s; ``k`` in s⁻¹.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from section_tool.core.surfaces import new_entity_uuid
from section_tool.core.velocity_model import (
    VelocityFunction, VelocityLayer, VelocityModel)

_TOL_M = 1e-6   # distance tolerance for "exactly at a control point"


@dataclass
class LateralControl:
    """A velocity model pinned at a distance along the section (metres)."""
    distance_m: float
    model: VelocityModel


class LateralVelocityModel:
    """Layered velocity that varies along the section between control points.

    All control models must share the same layer *structure* (same number of
    layers) — v1 assumes laterally continuous zones; only the velocities and
    layer-top times vary.  Differing structures raise (flag, don't silently
    guess).  Use :meth:`model_at` to get the effective 1-D model at any distance.
    """

    def __init__(self, controls, uuid: str | None = None,
                 construction: dict | None = None) -> None:
        pts = [c if isinstance(c, LateralControl)
               else LateralControl(float(c[0]), c[1]) for c in controls]
        if not pts:
            raise ValueError("LateralVelocityModel requires at least one control point")
        pts.sort(key=lambda c: c.distance_m)
        n = len(pts[0].model.layers)
        if n == 0:
            raise ValueError("control models must have at least one layer")
        for c in pts:
            if len(c.model.layers) != n:
                raise ValueError(
                    "lateral controls must share layer structure "
                    f"(expected {n} layers, got {len(c.model.layers)} at "
                    f"{c.distance_m} m)")
        self.controls: list[LateralControl] = pts
        self.n_layers = n
        self.uuid = uuid or new_entity_uuid()
        self.construction = construction or {
            "kind": "lateral_velocity_model", "parents": [], "params": {}}

    @property
    def distances(self) -> np.ndarray:
        return np.array([c.distance_m for c in self.controls], dtype=float)

    def model_at(self, distance_m: float) -> VelocityModel:
        """The effective layered model at *distance_m* along the section.

        Exact at a control point (returns that model verbatim, preserving its
        provenance/labels).  Linearly interpolated between controls; clipped to
        the edge model beyond the outermost control.
        """
        d = float(distance_m)
        for c in self.controls:
            if abs(d - c.distance_m) <= _TOL_M:
                return c.model                       # exact — verbatim
        if len(self.controls) == 1:
            return self.controls[0].model            # constant laterally

        xs = self.distances
        layers: list[VelocityLayer] = []
        for i in range(self.n_layers):
            col = [c.model.layers[i] for c in self.controls]
            # np.interp clips to the edge value beyond [xs[0], xs[-1]] — exactly
            # the "no wild extrapolation past the data" behaviour we want.
            top = float(np.interp(d, xs, [L.top_twt_s for L in col]))
            v0  = float(np.interp(d, xs, [L.function.v0 for L in col]))
            k   = float(np.interp(d, xs, [L.function.k for L in col]))
            # A layer that is linear in ANY control is linear here (constant is
            # just k=0), so a lateral gradient is never silently dropped.
            method = ("linear_v0k"
                      if any(L.function.method == "linear_v0k" for L in col)
                      else "constant")
            base = col[0]
            layers.append(VelocityLayer(
                VelocityFunction(method=method, v0=v0, k=k),
                top_twt_s=top, name=base.name, formation=base.formation,
                provenance="interpolated"))
        return VelocityModel(
            layers=layers,
            construction={"kind": "velocity_model", "parents": [self.uuid],
                          "params": {"interpolated_at_m": d}})

    # ---- conversion at a lateral position --------------------------------

    def twt_to_depth(self, twt_s: float, distance_m: float) -> float:
        return self.model_at(distance_m).twt_to_depth(twt_s)

    def depth_to_twt(self, z_m: float, distance_m: float) -> float:
        return self.model_at(distance_m).depth_to_twt(z_m)

    # ---- persistence -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "uuid": self.uuid,
            "construction": self.construction,
            "controls": [{"distance_m": c.distance_m, "model": c.model.to_dict()}
                         for c in self.controls],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LateralVelocityModel":
        controls = [LateralControl(float(c["distance_m"]),
                                   VelocityModel.from_dict(c["model"]))
                    for c in d.get("controls", [])]
        return cls(controls, uuid=d.get("uuid"), construction=d.get("construction"))

    def __repr__(self) -> str:
        return (f"LateralVelocityModel(controls={len(self.controls)}, "
                f"layers={self.n_layers})")
