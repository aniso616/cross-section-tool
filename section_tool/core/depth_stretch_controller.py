"""DepthStretchController — headless logic behind the recommendation-first panel.

Given the app state it answers, with no Qt:

* what data is loaded (``inventory()`` → :class:`DataInventory`),
* which rung is recommended (``recommended_rung()`` — deterministic, no scoring),
* the ordered rung specs to render (``rung_specs()`` — label, plain-language
  one-liner referencing the *actual* data, unlocked/locked + reason + import
  action),
* how to build each rung's model (``build_model``) reusing the existing
  construction paths, and apply it (``apply``) through the existing vectorized
  re-stretch,
* round-trip honesty (``applied_rung`` from construction metadata),
* the non-blocking upgrade signal (``upgrade_rung`` / ``keep_current``).

The panel is a thin renderer over this; the tests target this class directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from section_tool.core.data_inventory import (
    DataInventory, build_inventory, wells_in_corridor, RECOMMENDATION_ORDER)
from section_tool.core.stretch_setup import StretchSetup
from section_tool.core.velocity_model import VelocityModel, conversion_caption
from section_tool.core.well_calibration import build_well_markers, calibrate_model
from section_tool.core.conversion import zone_tops_from_picks, restretch_project

RUNG_LABELS = {
    "sonic_checkshot": "Sonic V(z) · checkshot-tied",
    "checkshot":       "Checkshot-tied",
    "sonic_anchors":   "Sonic V(z) · anchor-tied",
    "marker_tied":     "Marker-tied calibration",
    "layered":         "Layered from formations",
    "average_vz":      "Average V(z)",
    "bulk":            "Bulk velocity",
}

# A LOCKED rung is a door, not a dead end: map the missing-data reason → the
# Prompt-05 importer (action token, button label) that unlocks it. 'no tied
# horizons' is interpretation, not an import, so it offers no importer.
_IMPORT_FOR_REASON = {
    "no checkshot loaded":       ("checkshot", "Import checkshot…"),
    "no sonic log loaded":       ("las",       "Import sonic log…"),
    "no formation tops loaded":  ("markers",   "Import formation tops…"),
}

# Expected provenance label per rung (for the card tag — derived, not invented).
_RUNG_PROVENANCE = {
    "sonic_checkshot": "well-tied",
    "checkshot":       "checkshot-tied",
    "sonic_anchors":   "sonic-derived",
    "marker_tied":     "well-tied",
    "layered":         "regional default",
    "average_vz":      "regional default",
    "bulk":            "regional default",
}


@dataclass
class RungSpec:
    key: str
    label: str
    one_liner: str
    unlocked: bool
    provenance_label: str = ""
    recommended: bool = False
    reason: str = ""                       # why locked
    import_action: str | None = None       # importer token (locked rungs)
    import_label: str = ""


class DepthStretchController:
    def __init__(self, state) -> None:
        self._state = state

    # ---- context ----------------------------------------------------------

    @property
    def project(self):
        return self._state.project

    @property
    def section(self):
        return getattr(self._state, "active_section", None)

    def _picks(self):
        return getattr(self.project, "horizon_picks", [])

    def _corridor_wells(self):
        return wells_in_corridor(self.section, list(getattr(self.project, "wells", [])))

    def inventory(self) -> DataInventory:
        return build_inventory(self.section, self._corridor_wells(), self._picks())

    def recommended_rung(self) -> str:
        return self.inventory().recommended_rung()

    def inventory_chips(self) -> list[tuple[str, bool]]:
        """(text, present) chips for the inventory strip. Absent items read as
        'No … loaded' — the inventory speaks only to loaded project state."""
        from section_tool.core.data_inventory import detect_sonic_curve
        inv = self.inventory()
        chips: list[tuple[str, bool]] = []
        sw = self._sonic_well()
        chips.append((f"Sonic ({detect_sonic_curve(sw)}) · {sw.name}", True)
                     if sw is not None else ("No sonic loaded", False))
        cw = self._checkshot_well()
        cs = cw.primary_checkshot() if cw else None
        if cs is not None:
            from section_tool.io.tdr_io import tdr_shape_is_sonic
            if tdr_shape_is_sonic(cs.depth_m):
                # Stamped 'checkshot' but dense/regular like a sonic TDR — almost
                # certainly imported through the wrong door. Flag, don't trust.
                chips.append((f"Checkshot? · {cs.n_points} pts — verify kind", True))
            else:
                chips.append((f"Checkshot · {cs.n_points} pts", True))
        else:
            chips.append(("No checkshot loaded", False))
        tw = self._tops_well()
        chips.append((f"{len(tw.formation_tops)} tops · {tw.name}", True)
                     if tw is not None else ("No tops loaded", False))
        n = inv.n_tied_horizons
        chips.append((f"{n} tied horizons" if n else "No tied horizons", n > 0))
        return chips

    # ---- rung specs (what the panel renders) ------------------------------

    def rung_specs(self) -> list[RungSpec]:
        inv = self.inventory()
        unlocked = inv.unlocked_rungs()
        rec = inv.recommended_rung()
        specs: list[RungSpec] = []
        for key in RECOMMENDATION_ORDER:
            is_unlocked = key in unlocked
            spec = RungSpec(
                key=key, label=RUNG_LABELS[key],
                one_liner=self._one_liner(key, inv),
                unlocked=is_unlocked,
                provenance_label=_RUNG_PROVENANCE.get(key, ""),
                recommended=(key == rec),
            )
            if not is_unlocked:
                spec.reason = self._lock_reason(key, inv)
                act = _IMPORT_FOR_REASON.get(spec.reason)
                if act:
                    spec.import_action, spec.import_label = act
            specs.append(spec)
        return specs

    # ---- one-liners + lock reasons (reference the real data) --------------

    def _sonic_well(self):
        for w in self._corridor_wells():
            from section_tool.core.data_inventory import detect_sonic_curve
            if detect_sonic_curve(w):
                return w
        return None

    def _checkshot_well(self):
        for w in self._corridor_wells():
            if w.tdrs_of_kind("checkshot") or w.tdrs_of_kind("sonic_integrated"):
                return w
        return None

    def _tops_well(self):
        for w in self._corridor_wells():
            if getattr(w, "formation_tops", {}):
                return w
        return None

    def _one_liner(self, key: str, inv: DataInventory) -> str:
        from section_tool.core.data_inventory import detect_sonic_curve
        if key == "bulk":
            return "Single constant velocity — the no-data default, always available."
        if key == "average_vz":
            return "v = v₀ + k·z — a smooth bootstrap; needs no picks."
        if key == "layered":
            return f"Interval velocities from {inv.n_tied_horizons} picked zone tops."
        if key == "marker_tied":
            w = self._tops_well()
            n = len(getattr(w, "formation_tops", {})) if w else 0
            nm = w.name if w else "a well"
            return f"Calibrate to {nm}'s {n} formation tops at the tied reflectors."
        if key == "checkshot":
            w = self._checkshot_well()
            cs = w.primary_checkshot() if w else None
            if cs is not None:
                return f"Reproduces {w.name}'s {cs.n_points}-point checkshot."
            return "Reproduce a well's checkshot time–depth pairs."
        if key == "sonic_anchors":
            w = self._sonic_well()
            cv = detect_sonic_curve(w) if w else None
            nm = w.name if w else "a well"
            return (f"Integrates {nm}'s sonic ({cv}), tied to "
                    f"{inv.n_tied_horizons} picked reflectors.")
        if key == "sonic_checkshot":
            w = self._sonic_well() or self._checkshot_well()
            cv = detect_sonic_curve(w) if w else None
            cs = w.primary_checkshot() if w else None
            n = cs.n_points if cs is not None else 0
            nm = w.name if w else "a well"
            return (f"Integrates {nm}'s sonic ({cv}), drift-corrected to the "
                    f"{n}-point checkshot.")
        return ""

    def _lock_reason(self, key: str, inv: DataInventory) -> str:
        if key in ("checkshot",):
            return "no checkshot loaded"
        if key == "sonic_checkshot":
            if not inv.any_sonic:
                return "no sonic log loaded"
            return "no checkshot loaded"
        if key == "sonic_anchors":
            if not inv.any_sonic:
                return "no sonic log loaded"
            return "no tied horizons"
        if key == "marker_tied":
            if not inv.any_tops:
                return "no formation tops loaded"
            return "no tied horizons"
        if key == "layered":
            return "no tied horizons"
        return ""

    # ---- model construction per rung --------------------------------------

    def _base_setup(self, method: str, knobs: dict) -> StretchSetup:
        return StretchSetup(
            setting=knobs.get("setting", "onshore"), method=method,
            datum_twt_s=knobs.get("datum_twt_s", 0.0),
            seafloor_twt_s=knobs.get("seafloor_twt_s", 0.4),
            basement_twt_s=knobs.get("basement_twt_s", 3.0),
            bulk_v=knobs.get("bulk_v", 2400.0),
            v0=knobs.get("v0", 1800.0), k=knobs.get("k", 0.6),
            water_v=knobs.get("water_v", 1480.0))

    def _zone_tops(self, knobs: dict):
        marine = knobs.get("setting", "onshore") == "marine"
        base = (knobs.get("seafloor_twt_s", 0.4) if marine
                else knobs.get("datum_twt_s", 0.0))
        return zone_tops_from_picks(self._picks(), base)

    def build_model(self, rung: str, **knobs) -> VelocityModel:
        strat = getattr(self.project, "strat_column", None)
        if rung == "bulk":
            return self._base_setup("bulk", knobs).build_model()
        if rung == "average_vz":
            return self._base_setup("average_vz", knobs).build_model()
        if rung == "layered":
            return self._base_setup("layered_from_formations", knobs).build_model(
                self._zone_tops(knobs), strat)
        if rung == "marker_tied":
            zt = self._zone_tops(knobs)
            base = self._base_setup(
                "layered_from_formations" if zt else "average_vz", knobs
            ).build_model(zt, strat)
            w = self._tops_well()
            if w is None:
                return base
            markers, _report = build_well_markers(w, self._picks())
            return calibrate_model(base, markers) if len(markers) >= 2 else base
        if rung == "checkshot":
            w = self._checkshot_well()
            cs = w.primary_checkshot() if w else None
            if cs is None:
                raise ValueError("checkshot rung needs a checkshot TDR")
            return VelocityModel.from_tdr(w, cs, setting=knobs.get("setting", "onshore"),
                                          datum_twt_s=knobs.get("datum_twt_s", 0.0),
                                          seafloor_twt_s=knobs.get("seafloor_twt_s"))
        if rung == "sonic_checkshot":
            w = self._sonic_well()
            return VelocityModel.from_sonic(
                w, curve=knobs.get("curve"), drift_target="checkshot",
                setting=knobs.get("setting", "onshore"),
                datum_twt_s=knobs.get("datum_twt_s", 0.0))
        if rung == "sonic_anchors":
            w = self._sonic_well()
            knots = self._anchor_knots(w)
            return VelocityModel.from_sonic(
                w, curve=knobs.get("curve"), drift_target="anchors",
                anchor_knots=knots, setting=knobs.get("setting", "onshore"),
                datum_twt_s=knobs.get("datum_twt_s", 0.0))
        raise ValueError(f"unknown rung {rung!r}")

    def _anchor_knots(self, well):
        markers, _ = build_well_markers(well, self._picks(), checkshot=None)
        return [(m.depth_m, m.twt_s) for m in markers]

    # ---- apply + round-trip ----------------------------------------------

    def apply(self, rung: str, **knobs) -> VelocityModel:
        model = self.build_model(rung, **knobs)
        # Record which recommendation was current at apply time (a deliberate
        # choice acknowledges it), so the upgrade tag does not nag.
        model.construction.setdefault("params", {})["ack_rung"] = self.recommended_rung()
        self.project.lateral_velocity_model = None
        self.project.velocity_model = model
        restretch_project(self.project, model)
        return model

    @staticmethod
    def rung_of_model(model) -> str | None:
        """Round-trip: which rung built *model*, read from construction metadata."""
        if model is None or getattr(model, "is_empty", True):
            return None
        c = getattr(model, "construction", {}) or {}
        kind = c.get("kind")
        params = c.get("params", {}) or {}
        if kind == "from_tdr":
            return "checkshot"
        if kind == "from_sonic":
            return {"checkshot": "sonic_checkshot",
                    "anchors": "sonic_anchors"}.get(params.get("drift_target"),
                                                    "sonic_anchors")
        method = params.get("method")
        return {"bulk": "bulk", "average_vz": "average_vz",
                "layered_from_formations": "layered",
                "well_calibrated": "marker_tied"}.get(method, "bulk")

    def applied_rung(self) -> str | None:
        return self.rung_of_model(getattr(self.project, "velocity_model", None))

    def caption(self, model=None) -> str | None:
        return conversion_caption(model if model is not None
                                  else getattr(self.project, "velocity_model", None))

    # ---- non-blocking upgrade signal -------------------------------------

    def upgrade_rung(self) -> str | None:
        """A more-grounded rung than the applied one is now available (e.g. a
        checkshot got imported) AND has not been acknowledged.  Never restretches;
        the panel surfaces this as a quiet tag only."""
        applied = self.applied_rung()
        if applied is None:
            return None
        rec = self.recommended_rung()
        order = RECOMMENDATION_ORDER
        if order.index(rec) >= order.index(applied):
            return None                       # nothing more grounded
        model = getattr(self.project, "velocity_model", None)
        ack = (getattr(model, "construction", {}) or {}).get("params", {}).get("ack_rung")
        return None if ack == rec else rec

    def keep_current(self) -> None:
        """Record a deliberate keep: acknowledge the current recommendation so the
        upgrade tag clears (keeping is an act, not silence)."""
        model = getattr(self.project, "velocity_model", None)
        if model is not None:
            model.construction.setdefault("params", {})["ack_rung"] = self.recommended_rung()

    # ---- residual summary (shown, never auto-demotes) ---------------------

    def residual_summary(self, model=None) -> str | None:
        model = model if model is not None else getattr(self.project, "velocity_model", None)
        if model is None or getattr(model, "is_empty", True):
            return None
        c = getattr(model, "construction", {}) or {}
        params = c.get("params", {}) or {}
        if c.get("kind") == "from_sonic":
            drift = params.get("drift", {})
            knots = drift.get("knots", [])
            if knots:
                mx = max(abs(k["residual_post_ms"]) for k in knots)
                return (f"drift-corrected to {len(knots)} knots · "
                        f"max residual {mx:.1f} ms")
            return "uncorrected sonic integration"
        if c.get("kind") == "from_tdr":
            return "reproduces the checkshot knots exactly"
        if params.get("method") == "well_calibrated":
            w = self._tops_well()
            if w is not None:
                from section_tool.core.well_calibration import marker_residuals
                markers, _ = build_well_markers(w, self._picks())
                if markers:
                    res = marker_residuals(model, markers)
                    mx = max(abs(r["twt_residual_s"]) * 1000 for r in res)
                    return f"calibrated to {len(markers)} markers · max Δtwt {mx:.1f} ms"
        return None
