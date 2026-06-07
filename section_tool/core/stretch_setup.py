"""StretchSetup — the depth-stretch tool's logic (headless, testable).

The geologist declares the depositional setting (marine / onshore) and the
approximate bounding surfaces (datum/SRD, seafloor, basement), picks a method
from the ladder, and this builds + applies a :class:`VelocityModel`.  The Qt
window (M3 part 2) is a thin shell over this; keeping the logic here makes the
"front door" testable without UI.

Marine adds a water + replacement-velocity top layer (sensible default, user
overridable).  The chosen method + setting are recorded in the model's
construction metadata, so the depth section can always state HOW it was made
(architectural rule 2 — no black-box conversion).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from section_tool.core.conversion import (
    build_layered_from_formations, restretch_project)
from section_tool.core.velocity_model import (
    VelocityModel, VelocityLayer, VelocityFunction)

Setting = Literal["marine", "onshore"]
Method  = Literal["bulk", "average_vz", "layered_from_formations"]

WATER_VELOCITY_MS        = 1480.0   # sensible default sea-water velocity
DEFAULT_REPLACEMENT_V_MS = 2000.0
DEFAULT_BULK_V_MS        = 2400.0


@dataclass
class StretchSetup:
    """Declared setup for a depth stretch. SI throughout: velocities m/s, k s⁻¹,
    bounding surfaces in TWT seconds."""

    setting: Setting = "onshore"
    datum_twt_s: float = 0.0                 # top of the model (SRD / datum)
    seafloor_twt_s: float | None = None      # marine: approximate seafloor (TWT)
    basement_twt_s: float | None = None      # approximate basement (scaffolds layers)
    method: Method = "bulk"
    bulk_v: float = DEFAULT_BULK_V_MS
    v0: float = 1800.0
    k: float = 0.6
    water_v: float = WATER_VELOCITY_MS
    replacement_v: float = DEFAULT_REPLACEMENT_V_MS

    # ------------------------------------------------------------------

    def method_available(self, has_zone_tops: bool) -> bool:
        """Interpretation gate: layered needs picked zone-bounding horizons;
        bulk / average always available (the no-pick bootstrap)."""
        if self.method == "layered_from_formations":
            return has_zone_tops
        return True

    def build_model(self, zone_tops=None, strat_column=None) -> VelocityModel:
        """Build a VelocityModel from this setup.

        *zone_tops* — ``[(top_twt_s, formation_name), ...]`` from picked
        zone-bounding horizons (required for the layered method).
        """
        layers: list[VelocityLayer] = []
        base_twt = float(self.datum_twt_s)

        # Marine: a water + replacement-velocity top layer from datum to seafloor.
        if self.setting == "marine" and self.seafloor_twt_s is not None:
            layers.append(VelocityLayer(
                VelocityFunction("constant", v0=self.water_v),
                top_twt_s=float(self.datum_twt_s), name="Water",
                provenance="assumed",
                method_label=f"water {self.water_v:.0f} m/s"))
            base_twt = float(self.seafloor_twt_s)

        if self.method == "bulk":
            layers.append(VelocityLayer(
                VelocityFunction("constant", v0=self.bulk_v),
                top_twt_s=base_twt, name="Bulk", provenance="assumed",
                method_label=f"bulk {self.bulk_v:.0f} m/s"))
        elif self.method == "average_vz":
            layers.append(VelocityLayer(
                VelocityFunction("linear_v0k", v0=self.v0, k=self.k),
                top_twt_s=base_twt, name="V(z)", provenance="assumed"))
        elif self.method == "layered_from_formations":
            if not zone_tops:
                raise ValueError(
                    "layered method needs picked zone-bounding horizons")
            # Keep only zone tops at/below the base of the marine water layer.
            sub = build_layered_from_formations(
                [(t, f) for (t, f) in zone_tops if t >= base_twt - 1e-9],
                strat_column)
            layers.extend(sub.layers)
        else:
            raise ValueError(f"unknown method {self.method!r}")

        layers.sort(key=lambda l: l.top_twt_s)
        return VelocityModel(layers=layers, construction={
            "kind": "velocity_model", "parents": [],
            "params": {
                "setting": self.setting, "method": self.method,
                "datum_twt_s": self.datum_twt_s,
                "seafloor_twt_s": self.seafloor_twt_s,
                "basement_twt_s": self.basement_twt_s,
            }})

    def apply(self, project, zone_tops=None) -> VelocityModel:
        """Build the model, install it on *project*, and re-stretch tied geometry
        (depth-native untouched). Returns the model.  This is "on apply, write the
        velocity model and produce the depth display" — minus the Qt render."""
        model = self.build_model(zone_tops, getattr(project, "strat_column", None))
        project.velocity_model = model
        restretch_project(project, model)
        return model
