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
from typing import Any


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
    remove_elements:
        Names of HorizonPick / fault / polygon objects that are removed
        (set to absent) at this restoration step.
    decompact_params:
        Per-formation decompaction overrides.  Maps formation name →
        ``{"phi0": float, "c": float}`` dict.  Falls back to the
        project lithology library when absent.
    """
    event_id: int
    name: str
    age_ma: float | None = None
    description: str = ""
    remove_elements: list[str] = field(default_factory=list)
    decompact_params: dict[str, dict[str, float]] = field(default_factory=dict)

    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id":        self.event_id,
            "name":            self.name,
            "age_ma":          self.age_ma,
            "description":     self.description,
            "remove_elements": list(self.remove_elements),
            "decompact_params": {k: dict(v) for k, v in self.decompact_params.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RestorationEvent:
        return cls(
            event_id=int(d["event_id"]),
            name=str(d["name"]),
            age_ma=d.get("age_ma"),
            description=str(d.get("description", "")),
            remove_elements=list(d.get("remove_elements", [])),
            decompact_params={
                k: dict(v) for k, v in d.get("decompact_params", {}).items()
            },
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
    # Restoration stub
    # ------------------------------------------------------------------

    def elements_visible_at_step(self, step: int) -> set[str]:
        """Names of elements present at restoration *step*.

        step=0 means present day (all elements visible).  For step k,
        elements removed by events 0…k-1 are hidden.
        """
        removed: set[str] = set()
        for ev in self.events[:step]:
            removed.update(ev.remove_elements)
        return removed


def restore_remove_layer(
    sequence: RestorationSequence,
    step: int,
    horizon_picks: list,
    fault_picks: list,
    polygons: list,
) -> tuple[list, list, list]:
    """Return filtered copies of interpretation lists for restoration *step*.

    This is a decompaction stub — it only handles element removal.
    Full 2D geometric restoration (bed-length balancing, fault-parallel
    flow, etc.) will be added in a later session.

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

    removed = sequence.elements_visible_at_step(step)

    def keep(obj):
        return getattr(obj, "name", "") not in removed

    return (
        [h for h in horizon_picks if keep(h)],
        [f for f in fault_picks   if keep(f)],
        [p for p in polygons      if keep(p)],
    )
