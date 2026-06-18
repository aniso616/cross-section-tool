"""Interpretation snapshot — a faithful, isolated, non-destructive copy of a
section's full interpretation, so a copy can be deformed without touching the
original.  This is the precondition for kinematic restoration (duplicate the
interpretation, deform the copy, compare to the original).

No geometry is altered here — this module only duplicates and restores.

UUID policy — **preserved UUIDs in an isolated namespace** (option b of the build
order).  A snapshot is a standalone bundle that is NEVER merged into the live
project's collections, so the duplicated entities' UUIDs cannot collide with the
originals.  Preserving the UUIDs (rather than reassigning them) is the faithful
choice for three reasons:

* round-trip reproduces the original exactly — no reassignment that could drift;
* construction-rule ``{*_uuid}`` references stay valid inside the bundle (a
  ``parallel_to_bed`` copy still references the snapshot of its reference bed);
* restored↔original pairing — the comparison every later restoration step needs —
  is a plain UUID equality.

Heavy/shared data (seismic volumes, the VelocityModel) is referenced by id, never
copied.  SI internal, depth-canonical — unchanged by the snapshot.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

SNAPSHOT_SCHEMA_VERSION = 1


@dataclass
class InterpretationSnapshot:
    """A complete, isolated, deep-copied duplicate of a section's interpretation.

    Geometry-bearing entities (horizons, faults-as-HorizonPicks, polygons,
    reference lines) are deep copies — independent point arrays and construction
    rules, no aliasing with the originals.  ``velocity_model_id`` /
    ``seismic_ref_names`` are references only.
    """
    section: dict                                          # Section.snapshot()
    horizons: list = field(default_factory=list)
    faults: list = field(default_factory=list)
    polygons: list = field(default_factory=list)
    reference_lines: list = field(default_factory=list)    # project-global, kept for fidelity
    velocity_model_id: str | None = None                   # reference, not the object
    seismic_ref_names: list = field(default_factory=list)  # references, not the data
    schema_version: int = SNAPSHOT_SCHEMA_VERSION
    # True once a kinematic algorithm has deformed this bundle (Step 6): the
    # entities are in a pre-deformation frame, depth-native (anchors cleared), and
    # must never be mistaken for the live seismic-tied interpretation.
    restoration_frame: bool = False


def _on_section(pick, section_name: str) -> bool:
    """Whether *pick* (a HorizonPick) has any node on *section_name*."""
    if not section_name:
        return True
    try:
        return pick.n_picks_for_section(section_name) >= 1
    except Exception:
        return True


def snapshot_interpretation(section, project) -> InterpretationSnapshot:
    """Capture *section*'s full interpretation from *project* as an isolated,
    deep-copied bundle (see :class:`InterpretationSnapshot` for the UUID policy).

    Horizons and faults are included when they appear on *section*; polygons when
    tagged to it (or untagged).  Reference lines are project-global and captured
    whole for fidelity.  The VelocityModel and seismic volumes are referenced by
    id, not copied.
    """
    sec_name = getattr(section, "name", "")
    horizons = [copy.deepcopy(hp) for hp in getattr(project, "horizon_picks", [])
                if _on_section(hp, sec_name)]
    faults = [copy.deepcopy(fp) for fp in getattr(project, "fault_picks", [])
              if _on_section(fp, sec_name)]
    polygons = [copy.deepcopy(p) for p in getattr(project, "polygons", [])
                if getattr(p, "section_name", "") in ("", sec_name)]
    ref_lines = [copy.deepcopy(rl) for rl in getattr(project, "reference_lines", [])]

    vm = getattr(project, "velocity_model", None)
    vm_id = getattr(vm, "uuid", None) if vm is not None else None
    seis = [getattr(r, "name", None) for r in getattr(project, "seismic_refs", [])]

    return InterpretationSnapshot(
        section=section.snapshot(),
        horizons=horizons,
        faults=faults,
        polygons=polygons,
        reference_lines=ref_lines,
        velocity_model_id=vm_id,
        seismic_ref_names=[s for s in seis if s],
    )


def restore_from_snapshot(snapshot: InterpretationSnapshot) -> dict:
    """Reconstruct the interpretation state from *snapshot* as fresh, independent
    deep copies.

    Returns ``{"section": Section, "horizons": [...], "faults": [...],
    "polygons": [...], "reference_lines": [...], "velocity_model_id": ...,
    "seismic_ref_names": [...]}``.  The returned entities are independent of BOTH
    the snapshot and the originals, so restoring twice yields independent states
    and deforming a restored copy can never reach back into the snapshot.
    """
    from section_tool.core.section import Section
    sec_snap = snapshot.section
    section = Section(sec_snap["nodes"], name=sec_snap.get("name", ""))
    section.load_snapshot(sec_snap)
    return {
        "section": section,
        "horizons": [copy.deepcopy(hp) for hp in snapshot.horizons],
        "faults": [copy.deepcopy(fp) for fp in snapshot.faults],
        "polygons": [copy.deepcopy(p) for p in snapshot.polygons],
        "reference_lines": [copy.deepcopy(rl) for rl in snapshot.reference_lines],
        "velocity_model_id": snapshot.velocity_model_id,
        "seismic_ref_names": list(snapshot.seismic_ref_names),
    }
