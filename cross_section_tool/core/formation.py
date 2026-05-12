"""Formation and StratigraphicColumn data model for Phase 5."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Formation:
    """A geological formation record in the stratigraphic column.

    Fields are intentionally optional so that users can fill in what they know.
    """

    name: str
    age_ma: Optional[float] = None           # depositional age in Ma
    lithology: str = ""                       # "sandstone", "shale", …
    color: tuple[int, int, int] = (150, 180, 220)  # RGB 0–255
    density_kg_m3: Optional[float] = None    # kg/m³
    porosity_initial: Optional[float] = None # surface porosity (Athy's law)
    compaction_coefficient: Optional[float] = None  # 1/m
    thermal_conductivity: Optional[float] = None    # W/(m·K)

    def to_dict(self) -> dict:
        return {
            "name":                   self.name,
            "age_ma":                 self.age_ma,
            "lithology":              self.lithology,
            "color":                  list(self.color),
            "density_kg_m3":          self.density_kg_m3,
            "porosity_initial":       self.porosity_initial,
            "compaction_coefficient": self.compaction_coefficient,
            "thermal_conductivity":   self.thermal_conductivity,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Formation":
        color = d.get("color", [150, 180, 220])
        return cls(
            name=d["name"],
            age_ma=d.get("age_ma"),
            lithology=d.get("lithology", ""),
            color=tuple(color),
            density_kg_m3=d.get("density_kg_m3"),
            porosity_initial=d.get("porosity_initial"),
            compaction_coefficient=d.get("compaction_coefficient"),
            thermal_conductivity=d.get("thermal_conductivity"),
        )


class StratigraphicColumn:
    """Ordered list of formations — youngest (index 0) at top, oldest at bottom.

    Polygons reference formations by name via :attr:`~SectionPolygon.formation`.
    """

    def __init__(self) -> None:
        self._formations: list[Formation] = []

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_formation(self, formation: Formation, position: int = -1) -> None:
        """Insert *formation* at *position* (default: append at bottom)."""
        if position < 0 or position >= len(self._formations):
            self._formations.append(formation)
        else:
            self._formations.insert(position, formation)

    def remove_formation(self, name: str) -> None:
        self._formations = [f for f in self._formations if f.name != name]

    def reorder(self, name: str, new_position: int) -> None:
        idx = self._index_of(name)
        if idx is None:
            return
        f = self._formations.pop(idx)
        new_position = max(0, min(new_position, len(self._formations)))
        self._formations.insert(new_position, f)

    def get_formation(self, name: str) -> Formation | None:
        for f in self._formations:
            if f.name == name:
                return f
        return None

    def is_above(self, name_a: str, name_b: str) -> bool:
        """Return True if formation A is stratigraphically above B."""
        ia = self._index_of(name_a)
        ib = self._index_of(name_b)
        if ia is None or ib is None:
            raise KeyError(f"Formation not found: {name_a!r} or {name_b!r}")
        return ia < ib   # lower index = younger = shallower = above

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def formations(self) -> list[Formation]:
        return list(self._formations)

    def __len__(self) -> int:
        return len(self._formations)

    def __repr__(self) -> str:
        names = [f.name for f in self._formations]
        return f"StratigraphicColumn({names!r})"

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def to_list(self) -> list[dict]:
        return [f.to_dict() for f in self._formations]

    @classmethod
    def from_list(cls, data: list[dict]) -> "StratigraphicColumn":
        col = cls()
        for d in data:
            col.add_formation(Formation.from_dict(d))
        return col

    # ------------------------------------------------------------------

    def _index_of(self, name: str) -> int | None:
        for i, f in enumerate(self._formations):
            if f.name == name:
                return i
        return None
