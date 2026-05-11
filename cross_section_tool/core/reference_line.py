from __future__ import annotations

from typing import Literal


class ReferenceLine:
    """Construction geometry used for alignment — not a geological interpretation.

    kind="horizontal"  → infinite horizontal line at depth *value*
    kind="vertical"    → infinite vertical line at section distance *value*
    """

    def __init__(
        self,
        kind: Literal["horizontal", "vertical"],
        value: float,
        name: str = "",
        visible: bool = True,
        color: str = "#999999",
    ) -> None:
        self.kind: Literal["horizontal", "vertical"] = kind
        self.value: float = float(value)
        self.name = name
        self.visible = bool(visible)
        self.color = color

    def __repr__(self) -> str:
        return (
            f"ReferenceLine(kind={self.kind!r}, value={self.value}, name={self.name!r})"
        )
