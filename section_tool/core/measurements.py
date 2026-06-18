"""Thermochronometric / thermal point measurements on a well — observed data.

These are the OBSERVED constraints a thermal history is tested against: vitrinite
reflectance, fission-track and (U-Th)/He ages, bottom-hole temperatures, etc. They
are project data (stored on the well, survive save/reopen), distinct from the
modelled values the thermal solver predicts.

Units (display boundary): depth in **metres**; each type carries its natural unit —
°C (temperatures), Ma (ages), %Ro (reflectance), µm (track length). The thermal
solver converts °C→K internally; measurements are stored and shown in °C.
"""
from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass, field

# type → (label, default unit, plausible (min, max), min_exclusive)
MEASUREMENT_TYPES: dict[str, tuple] = {
    "vitrinite_ro": ("Vitrinite reflectance",        "%Ro", (0.2, 4.0),     False),
    "aft_age":      ("Apatite fission-track age",     "Ma",  (0.0, 4600.0),  True),
    "aft_length":   ("AFT mean track length",         "µm",  (0.0, 20.0),    True),
    "ahe_age":      ("Apatite (U-Th)/He age",         "Ma",  (0.0, 4600.0),  True),
    "zhe_age":      ("Zircon (U-Th)/He age",          "Ma",  (0.0, 4600.0),  True),
    "bht":          ("Bottom-hole temperature",       "°C",  (-20.0, 400.0), False),
    "dst_temp":     ("DST / formation temperature",   "°C",  (-20.0, 400.0), False),
}

MEASUREMENT_TYPE_ORDER = (
    "vitrinite_ro", "aft_age", "aft_length", "ahe_age", "zhe_age", "bht", "dst_temp")


def measurement_label(mtype: str) -> str:
    spec = MEASUREMENT_TYPES.get(mtype)
    return spec[0] if spec else mtype


def default_units(mtype: str) -> str:
    spec = MEASUREMENT_TYPES.get(mtype)
    return spec[1] if spec else ""


def validate_measurement(mtype: str, value) -> str | None:
    """Return ``None`` if ``(mtype, value)`` is physically plausible, else a clear
    human-readable reason. Callers reject (don't silently clamp) on a reason."""
    spec = MEASUREMENT_TYPES.get(mtype)
    if spec is None:
        return f"unknown measurement type {mtype!r}"
    label, unit, (lo, hi), min_excl = spec
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "value must be a number"
    if not (v == v):                                  # NaN
        return "value must be a number"
    if min_excl and v <= lo:
        return f"{label} must be > {lo:g} {unit}"
    if v < lo or v > hi:
        return f"{label} out of plausible range [{lo:g}, {hi:g}] {unit}"
    return None


@dataclass
class Measurement:
    """One observed datum at a depth on a well. ``uuid`` is the sample identity
    (rename-safe); ``well_uuid`` links it to its well."""
    depth_m: float
    measurement_type: str
    value: float
    uncertainty: float | None = None
    units: str = ""
    source: str = ""                                  # lab / publication / provenance
    notes: str = ""
    sample_id: str = ""
    well_uuid: str = ""
    uuid: str = field(default_factory=lambda: str(_uuid.uuid4()))

    @classmethod
    def from_db_row(cls, row: dict, well_uuid: str = "") -> "Measurement":
        return cls(
            depth_m=float(row.get("depth_md", 0.0) or 0.0),
            measurement_type=row.get("kind", "") or "",
            value=float(row.get("value", 0.0) or 0.0),
            uncertainty=(None if row.get("uncertainty") is None
                         else float(row.get("uncertainty"))),
            units=row.get("units") or "",
            source=row.get("lab") or "",
            notes=row.get("note") or "",
            sample_id=row.get("sample_id") or "",
            uuid=row.get("uuid") or str(_uuid.uuid4()),
            well_uuid=well_uuid,
        )


def parse_measurements_csv(text: str, measurement_type: str,
                           units: str = "") -> "tuple[list[Measurement], list[str]]":
    """Parse a two-column ``depth_m, value`` CSV into :class:`Measurement` objects.

    A header row (non-numeric first line) is skipped. Blank / ``#``-comment lines
    are ignored. Tabs are accepted as separators. Returns ``(measurements,
    errors)``; rows failing plausibility are reported in *errors* and skipped.
    Raises :class:`ValueError` (clear message) when nothing valid could be parsed —
    the wrong-format case.
    """
    out: list[Measurement] = []
    errors: list[str] = []
    unit = units or default_units(measurement_type)
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.replace("\t", ",").split(",")]
        if len(parts) < 2:
            errors.append(f"line {i}: expected 'depth_m, value', got {line!r}")
            continue
        try:
            depth = float(parts[0])
            value = float(parts[1])
        except ValueError:
            if i == 1:
                continue                              # header row — skip silently
            errors.append(f"line {i}: non-numeric '{line}'")
            continue
        reason = validate_measurement(measurement_type, value)
        if reason:
            errors.append(f"line {i}: {reason}")
            continue
        out.append(Measurement(depth_m=depth, measurement_type=measurement_type,
                               value=value, units=unit))
    if not out:
        raise ValueError(
            "No valid measurements parsed. Expected two columns 'depth_m, value'."
            + (f" ({errors[0]})" if errors else ""))
    return out, errors
