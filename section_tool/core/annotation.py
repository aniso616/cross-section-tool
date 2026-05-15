"""Phase 6 — Annotation objects for section view."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Annotation:
    text: str
    position: tuple[float, float]             # (distance, depth) in section space
    section_name: str = ""
    font_size: int = 10
    rotation_degrees: float = 0.0
    color: tuple[int, int, int] = (0, 0, 0)
    anchor_point: Optional[tuple[float, float]] = None  # leader line target

    def to_dict(self) -> dict:
        return {
            "text":              self.text,
            "position":          list(self.position),
            "section_name":      self.section_name,
            "font_size":         self.font_size,
            "rotation_degrees":  self.rotation_degrees,
            "color":             list(self.color),
            "anchor_point":      list(self.anchor_point) if self.anchor_point else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Annotation":
        ap = d.get("anchor_point")
        return cls(
            text=d["text"],
            position=tuple(d["position"]),
            section_name=d.get("section_name", ""),
            font_size=d.get("font_size", 10),
            rotation_degrees=d.get("rotation_degrees", 0.0),
            color=tuple(d.get("color", [0, 0, 0])),
            anchor_point=tuple(ap) if ap else None,
        )
