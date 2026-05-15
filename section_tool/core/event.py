"""Phase 1 — Geological event sequencing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

_EVENT_TYPES = (
    "deposition", "erosion", "fault_activation", "fault_cessation",
    "intrusion", "folding",
)

_next_id: int = 1


def _new_id() -> int:
    global _next_id
    _id = _next_id
    _next_id += 1
    return _id


@dataclass
class Event:
    name: str
    event_type: str = "deposition"
    age_ma: Optional[float] = None
    related_objects: list[str] = field(default_factory=list)
    event_id: int = field(default_factory=_new_id)

    def to_dict(self) -> dict:
        return {
            "event_id":       self.event_id,
            "name":           self.name,
            "event_type":     self.event_type,
            "age_ma":         self.age_ma,
            "related_objects": list(self.related_objects),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        obj = cls(
            name=d["name"],
            event_type=d.get("event_type", "deposition"),
            age_ma=d.get("age_ma"),
            related_objects=list(d.get("related_objects", [])),
            event_id=d.get("event_id", _new_id()),
        )
        return obj


class EventSequence:
    """Ordered list of Events, oldest-first (index 0 = oldest)."""

    def __init__(self) -> None:
        self._events: list[Event] = []

    def add_event(self, event: Event, position: int = -1) -> None:
        if position < 0 or position >= len(self._events):
            self._events.append(event)
        else:
            self._events.insert(position, event)

    def remove_event(self, event_id: int) -> None:
        self._events = [e for e in self._events if e.event_id != event_id]

    def reorder(self, event_id: int, new_position: int) -> None:
        idx = self._index_of(event_id)
        if idx is None:
            return
        ev = self._events.pop(idx)
        new_position = max(0, min(new_position, len(self._events)))
        self._events.insert(new_position, ev)

    def get_events_for_object(self, name: str) -> list[Event]:
        return [e for e in self._events if name in e.related_objects]

    def get_event(self, event_id: int) -> Optional[Event]:
        for e in self._events:
            if e.event_id == event_id:
                return e
        return None

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def to_list(self) -> list[dict]:
        return [e.to_dict() for e in self._events]

    @classmethod
    def from_list(cls, data: list[dict]) -> "EventSequence":
        seq = cls()
        for d in data:
            seq.add_event(Event.from_dict(d))
        return seq

    def _index_of(self, event_id: int) -> Optional[int]:
        for i, e in enumerate(self._events):
            if e.event_id == event_id:
                return i
        return None
