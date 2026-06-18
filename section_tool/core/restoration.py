"""Kinematic restoration data model.

A :class:`RestorationSequence` is a project-level ordered list of
:class:`RestorationEvent` objects that describes how the section is
progressively un-deformed, oldest event last.

Each event records:
  * which geological elements to remove at that restoration step
  * optional decompaction parameters for the elements being restored

This module is pure data — no Qt, no GUI.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, ClassVar


# ---------------------------------------------------------------------------
# RestorationEvent
# ---------------------------------------------------------------------------

@dataclass
class RestorationEvent:
    """A single restoration step — remove a set of elements and decompact.

    Parameters
    ----------
    event_id:
        Unique integer identifier.
    name:
        Human-readable label, e.g. "Remove Oligocene package".
    age_ma:
        Geological age at this restoration step (Ma before present),
        or ``None`` when unknown.
    description:
        Free-text notes on what this step represents.
    remove_element_ids:
        Stable **UUIDs** of HorizonPick / fault / polygon objects removed at
        this restoration step.  UUID-keyed so a rename can't silently break the
        reference (the Step-1 fix; was a list of names).
    remove_element_names:
        Legacy / unresolved element names.  A v1 (name-keyed) project loads its
        names here; :func:`migrate_names_to_ids` resolves them to
        ``remove_element_ids`` against the live entities.  Names that no longer
        resolve stay here (flagged, not dropped) so the event's intent survives.
    decompact_params:
        Per-formation decompaction overrides.  Maps formation name →
        ``{"phi0": float, "c": float}`` dict.  Falls back to the
        project lithology library when absent.
    """
    event_id: int
    name: str
    age_ma: float | None = None
    description: str = ""
    remove_element_ids: list[str] = field(default_factory=list)
    remove_element_names: list[str] = field(default_factory=list)
    decompact_params: dict[str, dict[str, float]] = field(default_factory=dict)
    # Kinematic restoration (Step 6): which geometric algorithm undeforms this
    # step, and its parameters (pin_x / datum_y / dx / dy / shear_angle / slip /
    # fault_uuid). "none" = a remove-only event (no deformation).
    algorithm: str = "none"
    params: dict = field(default_factory=dict)
    # Optional pin / datum sourced from a named ReferenceLine (by UUID, rename-safe).
    # When set, the engine resolves pin_x / datum_y from the line at deform time, so
    # moving the line updates every event that references it. ``None`` → use the
    # numeric ``params`` (pin_x / datum_y) fallback.
    pin_line_id: str | None = None
    datum_line_id: str | None = None

    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id":             self.event_id,
            "name":                 self.name,
            "age_ma":               self.age_ma,
            "description":          self.description,
            "remove_element_ids":   list(self.remove_element_ids),
            "remove_element_names": list(self.remove_element_names),
            "decompact_params": {k: dict(v) for k, v in self.decompact_params.items()},
            "algorithm":            self.algorithm,
            "params":               dict(self.params),
            "pin_line_id":          self.pin_line_id,
            "datum_line_id":        self.datum_line_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RestorationEvent:
        # v2 carries remove_element_ids; v1 carried remove_elements (names) — load
        # those into remove_element_names, pending migration to UUIDs on load.
        legacy_names = list(d.get("remove_elements", []))
        return cls(
            event_id=int(d["event_id"]),
            name=str(d["name"]),
            age_ma=d.get("age_ma"),
            description=str(d.get("description", "")),
            remove_element_ids=list(d.get("remove_element_ids", [])),
            remove_element_names=list(d.get("remove_element_names", [])) + legacy_names,
            decompact_params={
                k: dict(v) for k, v in d.get("decompact_params", {}).items()
            },
            algorithm=str(d.get("algorithm", "none")),
            params=dict(d.get("params", {})),
            pin_line_id=d.get("pin_line_id"),
            datum_line_id=d.get("datum_line_id"),
        )


# ---------------------------------------------------------------------------
# RestorationSequence
# ---------------------------------------------------------------------------

@dataclass
class RestorationSequence:
    """Ordered list of restoration events for a project.

    Events are stored youngest-first (index 0 = first step to apply
    during forward restoration, i.e. youngest stratigraphy removed first).

    Parameters
    ----------
    events:
        Ordered list of :class:`RestorationEvent`.
    current_step:
        Index of the currently-displayed restoration step
        (0 = present day, 1 = first event applied, etc.).
    """
    events: list[RestorationEvent] = field(default_factory=list)
    current_step: int = 0

    # JSON schema version: 1 = name-keyed removal (legacy), 2 = UUID-keyed.
    SCHEMA_VERSION: ClassVar[int] = 2

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.events)

    def event_by_id(self, event_id: int) -> RestorationEvent | None:
        for ev in self.events:
            if ev.event_id == event_id:
                return ev
        return None

    def add_event(self, event: RestorationEvent) -> None:
        """Append a new event; auto-assign id if not set."""
        if any(e.event_id == event.event_id for e in self.events):
            raise ValueError(f"Duplicate event_id {event.event_id}")
        self.events.append(event)

    def remove_event(self, event_id: int) -> bool:
        before = len(self.events)
        self.events = [e for e in self.events if e.event_id != event_id]
        return len(self.events) < before

    def move_event_up(self, event_id: int) -> bool:
        for i, ev in enumerate(self.events):
            if ev.event_id == event_id and i > 0:
                self.events[i - 1], self.events[i] = self.events[i], self.events[i - 1]
                return True
        return False

    def move_event_down(self, event_id: int) -> bool:
        for i, ev in enumerate(self.events):
            if ev.event_id == event_id and i < len(self.events) - 1:
                self.events[i], self.events[i + 1] = self.events[i + 1], self.events[i]
                return True
        return False

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps({
            "schema_version": self.SCHEMA_VERSION,
            "current_step": self.current_step,
            "events": [e.to_dict() for e in self.events],
        })

    @classmethod
    def from_json(cls, json_str: str) -> RestorationSequence:
        d = json.loads(json_str)
        seq = cls(current_step=int(d.get("current_step", 0)))
        for ed in d.get("events", []):
            seq.events.append(RestorationEvent.from_dict(ed))
        return seq

    # ------------------------------------------------------------------
    # Removal resolution (UUID-keyed)
    # ------------------------------------------------------------------

    def removed_ids_at_step(self, step: int) -> set[str]:
        """UUIDs of elements removed by restoration *step*.

        step=0 means present day (nothing removed).  For step k, elements removed
        by events 0…k-1 are hidden.  Only resolved UUIDs are returned; unresolved
        legacy names (``remove_element_names``) are not — they hide nothing until
        migrated.
        """
        removed: set[str] = set()
        for ev in self.events[:step]:
            removed.update(ev.remove_element_ids)
        return removed


def _entity_collections(project) -> "list[list]":
    """The three removable interpretation collections on *project*."""
    return [
        getattr(project, "horizon_picks", []),
        getattr(project, "fault_picks", []),
        getattr(project, "polygons", []),
    ]


def resolve_removed_entities(sequence: RestorationSequence, step: int,
                             project) -> list:
    """The live entities hidden at restoration *step*, matched by UUID.

    The single resolution helper: collects the removed UUIDs for *step* and
    returns the matching HorizonPick / fault / polygon objects from *project*.
    Type-agnostic — UUIDs are unique across collections, so an id resolves to at
    most one entity regardless of its kind.
    """
    removed = sequence.removed_ids_at_step(step)
    if not removed:
        return []
    return [obj for coll in _entity_collections(project) for obj in coll
            if getattr(obj, "uuid", None) in removed]


def migrate_names_to_ids(sequence: RestorationSequence, project) -> list[str]:
    """Resolve any legacy name-keyed removals to UUIDs, in place.

    For each event, every name in ``remove_element_names`` is looked up by
    ``name`` across the project's horizons / faults / polygons; a match moves its
    UUID into ``remove_element_ids`` and drops the name.  A name that no longer
    resolves (renamed / deleted) is **kept** in ``remove_element_names`` (the
    event's intent is preserved, not silently dropped) and returned to the caller
    to flag.  Idempotent: a fully UUID-keyed sequence is unchanged.

    Returns the sorted, de-duplicated list of names that could not be resolved.
    """
    by_name: dict[str, str] = {}
    for coll in _entity_collections(project):
        for obj in coll:
            nm = getattr(obj, "name", "")
            uid = getattr(obj, "uuid", None)
            if nm and uid and nm not in by_name:
                by_name[nm] = uid

    unresolved: set[str] = set()
    for ev in sequence.events:
        still_unresolved: list[str] = []
        for nm in ev.remove_element_names:
            uid = by_name.get(nm)
            if uid is not None:
                if uid not in ev.remove_element_ids:
                    ev.remove_element_ids.append(uid)
            else:
                still_unresolved.append(nm)
                unresolved.add(nm)
        ev.remove_element_names = still_unresolved
    return sorted(unresolved)


def restore_remove_layer(
    sequence: RestorationSequence,
    step: int,
    horizon_picks: list,
    fault_picks: list,
    polygons: list,
) -> tuple[list, list, list]:
    """Return filtered copies of interpretation lists for restoration *step*.

    This is a decompaction stub — it only handles element removal (UUID-keyed).
    Full 2D geometric restoration (bed-length balancing, fault-parallel flow,
    etc.) will be added in a later session.

    Parameters
    ----------
    sequence:
        The project restoration sequence.
    step:
        Restoration step index (0 = present day).
    horizon_picks, fault_picks, polygons:
        Lists of all interpreted elements.

    Returns
    -------
    (horizon_picks, fault_picks, polygons) filtered to the visible set.
    """
    if step == 0 or not sequence.events:
        return horizon_picks, fault_picks, polygons

    removed = sequence.removed_ids_at_step(step)

    def keep(obj):
        return getattr(obj, "uuid", None) not in removed

    return (
        [h for h in horizon_picks if keep(h)],
        [f for f in fault_picks   if keep(f)],
        [p for p in polygons      if keep(p)],
    )
