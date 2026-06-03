"""Construction rules — per-element metadata describing HOW an interpreted
element was geometrically constructed.

Each rule is a small dataclass with a ``kind`` discriminant that drives
serialisation.  The rule is stored as JSON on each horizon, fault, or
polygon, and is the basis for kinematic restoration.

Rule hierarchy
--------------
FreehandRule
    User drew it by hand — no geometric constraint.
ParallelToBedRule
    Constructed parallel (or at fixed offset) to a reference bed.
DipConstrainedRule
    Dip is locked to a measured value; only endpoint moves freely.
KinkBandRule
    Layer is carried rigidly by kink-band migration (equal-area).
ListricFaultRule
    Fault geometry follows a listric curve to a given detachment depth.
MirrorAcrossAxialTraceRule
    Layer geometry is mirrored about a fold axial trace.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Rule dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FreehandRule:
    """No geometric constraint — user drew the element by hand."""
    kind: Literal["freehand"] = field(default="freehand", init=False)


@dataclass
class ParallelToBedRule:
    """Element is geometrically parallel to a reference bed.

    Parameters
    ----------
    reference_name:
        Display label — name of the HorizonPick this element was constructed
        parallel to. Denormalised; may go stale if the bed is renamed.
    reference_uuid:
        Stable identity of the reference bed (rename-safe link). This is the
        authoritative reference; ``reference_name`` is a human-readable label.
    offset_m:
        Constant vertical offset from the reference bed, in metres
        (positive = deeper).
    """
    reference_name: str
    offset_m: float = 0.0
    reference_uuid: str = ""
    kind: Literal["parallel_to_bed"] = field(default="parallel_to_bed", init=False)


@dataclass
class DipConstrainedRule:
    """Dip is locked to a measured value.

    Parameters
    ----------
    dip_deg:
        Dip angle in degrees (0 = horizontal, 90 = vertical).
    dip_direction_deg:
        Azimuth of dip direction in degrees from north (0–360).
    measurement_source:
        Identifier of the measurement this dip came from (e.g. a well
        or outcrop sample name).
    """
    dip_deg: float
    dip_direction_deg: float = 0.0
    measurement_source: str = ""
    kind: Literal["dip_constrained"] = field(default="dip_constrained", init=False)


@dataclass
class KinkBandRule:
    """Layer is carried rigidly by kink-band migration.

    The axial surface bisects the dip change between adjacent bed dips.
    Equal-area (constant bed length) is assumed.

    Parameters
    ----------
    axial_surface_dip_deg:
        Dip of the kink-band axial surface (degrees from horizontal).
    fore_dip_deg:
        Dip of the forelimb.
    back_dip_deg:
        Dip of the backlimb (commonly ~0 for a simple drape fold).
    """
    axial_surface_dip_deg: float
    fore_dip_deg: float = 0.0
    back_dip_deg: float = 0.0
    kind: Literal["kink_band"] = field(default="kink_band", init=False)


@dataclass
class ListricFaultRule:
    """Fault follows a listric (concave-upward) curve.

    Parameters
    ----------
    detachment_depth_m:
        Depth of the basal detachment horizon (m, positive downward).
    ramp_dip_deg:
        Dip of the upper straight ramp segment (degrees).
    hangingwall_reference:
        Display label — name of the hangingwall cutoff pick used to define
        displacement. Denormalised; may go stale if that pick is renamed.
    hangingwall_uuid:
        Stable identity of the hangingwall cutoff pick (rename-safe link).
    """
    detachment_depth_m: float
    ramp_dip_deg: float = 30.0
    hangingwall_reference: str = ""
    hangingwall_uuid: str = ""
    kind: Literal["listric_fault"] = field(default="listric_fault", init=False)


@dataclass
class MirrorAcrossAxialTraceRule:
    """Layer is mirrored about a fold axial trace.

    Parameters
    ----------
    axial_trace_name:
        Name of the reference line or horizon representing the axial trace.
    mirror_side:
        Which limb is the 'copy': ``'left'`` or ``'right'``.
    """
    axial_trace_name: str
    mirror_side: Literal["left", "right"] = "right"
    kind: Literal["mirror_axial_trace"] = field(default="mirror_axial_trace", init=False)


# Union type for type hints
ConstructionRule = (
    FreehandRule
    | ParallelToBedRule
    | DipConstrainedRule
    | KinkBandRule
    | ListricFaultRule
    | MirrorAcrossAxialTraceRule
)

# ---------------------------------------------------------------------------
# Registry — maps kind string → dataclass
# ---------------------------------------------------------------------------

RULE_REGISTRY: dict[str, type] = {
    "freehand":           FreehandRule,
    "parallel_to_bed":    ParallelToBedRule,
    "dip_constrained":    DipConstrainedRule,
    "kink_band":          KinkBandRule,
    "listric_fault":      ListricFaultRule,
    "mirror_axial_trace": MirrorAcrossAxialTraceRule,
}


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def serialize_rule(rule: ConstructionRule | None) -> str | None:
    """Serialise a construction rule to a JSON string.

    Returns ``None`` when *rule* is ``None``.
    """
    if rule is None:
        return None
    d = asdict(rule)
    return json.dumps(d)


def deserialize_rule(json_str: str | None) -> ConstructionRule | None:
    """Deserialise a JSON string back to a :class:`ConstructionRule`.

    Returns ``None`` when *json_str* is ``None`` or empty.
    Raises :class:`ValueError` for unknown rule kinds.
    """
    if not json_str:
        return None
    d: dict[str, Any] = json.loads(json_str)
    kind = d.get("kind")
    cls = RULE_REGISTRY.get(kind)
    if cls is None:
        raise ValueError(f"Unknown construction rule kind: {kind!r}")
    # Strip the 'kind' field — it is a default in the dataclass
    kwargs = {k: v for k, v in d.items() if k != "kind"}
    return cls(**kwargs)
