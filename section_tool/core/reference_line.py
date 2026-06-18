from __future__ import annotations

import uuid as _uuid
from typing import Literal


class ReferenceLine:
    """Construction geometry — not a geological interpretation.

    kind="horizontal"  → infinite horizontal line at depth *value*
    kind="vertical"    → infinite vertical line at section distance *value*
    kind="angled"      → line through (*anchor_x*, *anchor_y*) at *angle_deg*
                         degrees measured clockwise from horizontal
    """

    def __init__(
        self,
        kind: Literal["horizontal", "vertical", "angled"],
        value: float = 0.0,
        name: str = "",
        visible: bool = True,
        color: str = "#999999",
        angle_deg: float = 0.0,
        anchor_x: float = 0.0,
        anchor_y: float = 0.0,
        map_x: float | None = None,
        map_y: float | None = None,
        uuid: str | None = None,
        restoration_role: str | None = None,
    ) -> None:
        self.kind: Literal["horizontal", "vertical", "angled"] = kind
        self.value: float = float(value)
        self.name = name
        self.visible = bool(visible)
        self.color = color
        # Stable identity (UUID4) — rename-safe link for the restoration role,
        # mirroring HorizonPick / SectionPolygon. Generated if not restored.
        self.uuid: str = uuid if uuid else str(_uuid.uuid4())
        # Restoration role: None (plain construction line) | "pin" | "datum".
        # A 'pin' (vertical line) fixes an x; a 'datum' (horizontal line) fixes a
        # depth — events reference them by UUID so a rename can't break the link.
        self.restoration_role: str | None = restoration_role
        # angled-only fields
        self.angle_deg: float = float(angle_deg)
        self.anchor_x: float = float(anchor_x)
        self.anchor_y: float = float(anchor_y)
        # map-space source of truth for vertical reference lines
        self.map_x: float | None = map_x
        self.map_y: float | None = map_y

    def __repr__(self) -> str:
        if self.kind == "angled":
            return (
                f"ReferenceLine(kind='angled', anchor=({self.anchor_x}, {self.anchor_y}), "
                f"angle={self.angle_deg}°, name={self.name!r})"
            )
        return (
            f"ReferenceLine(kind={self.kind!r}, value={self.value}, name={self.name!r})"
        )
