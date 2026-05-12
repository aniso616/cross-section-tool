"""Formation, StratigraphicColumn, and lithology defaults (Phase C)."""
from __future__ import annotations

import math
from typing import Optional


# ---------------------------------------------------------------------------
# Lithology default physical properties
# (standard petroleum-geoscience values)
# ---------------------------------------------------------------------------

LITHOLOGY_DEFAULTS: dict[str, dict] = {
    "sandstone": {
        "porosity_surface": 0.40, "compaction_coeff": 0.00027,
        "grain_density": 2650.0, "matrix_thermal_conductivity": 3.0,
        "radiogenic_heat_production": 1.2, "specific_heat_capacity": 830.0,
        "matrix_velocity": 5500.0,
    },
    "shale": {
        "porosity_surface": 0.63, "compaction_coeff": 0.00051,
        "grain_density": 2720.0, "matrix_thermal_conductivity": 1.5,
        "radiogenic_heat_production": 2.5, "specific_heat_capacity": 1000.0,
        "matrix_velocity": 3500.0,
    },
    "limestone": {
        "porosity_surface": 0.40, "compaction_coeff": 0.00040,
        "grain_density": 2710.0, "matrix_thermal_conductivity": 2.5,
        "radiogenic_heat_production": 0.8, "specific_heat_capacity": 900.0,
        "matrix_velocity": 5200.0,
    },
    "dolomite": {
        "porosity_surface": 0.35, "compaction_coeff": 0.00038,
        "grain_density": 2850.0, "matrix_thermal_conductivity": 3.5,
        "radiogenic_heat_production": 0.6, "specific_heat_capacity": 880.0,
        "matrix_velocity": 5700.0,
    },
    "salt": {
        "porosity_surface": 0.01, "compaction_coeff": 0.0,
        "grain_density": 2170.0, "matrix_thermal_conductivity": 6.0,
        "radiogenic_heat_production": 0.1, "specific_heat_capacity": 840.0,
        "matrix_velocity": 4500.0,
    },
    "basement": {
        "porosity_surface": 0.02, "compaction_coeff": 0.0,
        "grain_density": 2750.0, "matrix_thermal_conductivity": 2.8,
        "radiogenic_heat_production": 2.0, "specific_heat_capacity": 900.0,
        "matrix_velocity": 6000.0,
    },
    "siltstone": {
        "porosity_surface": 0.50, "compaction_coeff": 0.00045,
        "grain_density": 2680.0, "matrix_thermal_conductivity": 2.0,
        "radiogenic_heat_production": 1.8, "specific_heat_capacity": 950.0,
        "matrix_velocity": 4200.0,
    },
    "conglomerate": {
        "porosity_surface": 0.30, "compaction_coeff": 0.00020,
        "grain_density": 2650.0, "matrix_thermal_conductivity": 2.8,
        "radiogenic_heat_production": 1.0, "specific_heat_capacity": 840.0,
        "matrix_velocity": 5000.0,
    },
    "coal": {
        "porosity_surface": 0.10, "compaction_coeff": 0.00020,
        "grain_density": 1500.0, "matrix_thermal_conductivity": 0.3,
        "radiogenic_heat_production": 0.2, "specific_heat_capacity": 1300.0,
        "matrix_velocity": 2400.0,
    },
    "volcanic": {
        "porosity_surface": 0.10, "compaction_coeff": 0.00010,
        "grain_density": 2900.0, "matrix_thermal_conductivity": 2.0,
        "radiogenic_heat_production": 1.5, "specific_heat_capacity": 850.0,
        "matrix_velocity": 5500.0,
    },
}


class Formation:
    """Geological formation with full physical property set.

    All physical properties are optional so users can fill in what they know.
    Use :meth:`populate_from_lithology` to auto-fill standard values.
    """

    def __init__(
        self,
        name: str,
        # Identity / age
        age_top_ma: Optional[float] = None,
        age_base_ma: Optional[float] = None,
        # Display
        color: tuple = (200, 200, 220),
        opacity: float = 0.6,
        lithology_pattern: str = "none",
        pattern_scale: float = 1.0,
        # Lithology classification
        primary_lithology: str = "shale",
        sand_fraction: float = 0.0,
        shale_fraction: float = 1.0,
        carbonate_fraction: float = 0.0,
        # Compaction (Athy's law: phi = phi0 * exp(-c * z))
        porosity_surface: float = 0.50,
        compaction_coeff: float = 0.00050,
        grain_density: float = 2720.0,
        # Thermal
        matrix_thermal_conductivity: float = 2.5,
        radiogenic_heat_production: float = 1.0,
        specific_heat_capacity: float = 1000.0,
        # Velocity
        matrix_velocity: float = 4500.0,
        # Depositional
        environment: Optional[str] = None,
    ) -> None:
        self.name = name
        self.age_top_ma  = age_top_ma
        self.age_base_ma = age_base_ma

        self.color   = tuple(color)
        self.opacity = float(opacity)
        self.lithology_pattern = lithology_pattern
        self.pattern_scale     = float(pattern_scale)

        self.primary_lithology   = primary_lithology
        self.sand_fraction       = float(sand_fraction)
        self.shale_fraction      = float(shale_fraction)
        self.carbonate_fraction  = float(carbonate_fraction)

        self.porosity_surface   = float(porosity_surface)
        self.compaction_coeff   = float(compaction_coeff)
        self.grain_density      = float(grain_density)

        self.matrix_thermal_conductivity = float(matrix_thermal_conductivity)
        self.radiogenic_heat_production  = float(radiogenic_heat_production)
        self.specific_heat_capacity      = float(specific_heat_capacity)

        self.matrix_velocity = float(matrix_velocity)
        self.environment     = environment

    # ------------------------------------------------------------------
    # Auto-population from lithology
    # ------------------------------------------------------------------

    def populate_from_lithology(self, lithology: Optional[str] = None) -> None:
        """Overwrite physical properties with defaults for *lithology*.

        Leaves display properties (color, pattern, fractions) unchanged.
        Pass *lithology* to update :attr:`primary_lithology` at the same time.
        """
        lit = lithology or self.primary_lithology
        if lithology:
            self.primary_lithology = lit
        defaults = LITHOLOGY_DEFAULTS.get(lit, {})
        for attr, val in defaults.items():
            setattr(self, attr, val)

    # ------------------------------------------------------------------
    # Computation methods
    # ------------------------------------------------------------------

    def porosity_at_depth(self, z: float) -> float:
        """Athy's law: phi(z) = phi0 * exp(-c * z)."""
        return self.porosity_surface * math.exp(-self.compaction_coeff * max(z, 0.0))

    def bulk_density_at_depth(
        self, z: float, fluid_density: float = 1000.0
    ) -> float:
        """Bulk density at depth *z* (kg/m³).

        rho_bulk = rho_grain * (1 - phi) + rho_fluid * phi
        """
        phi = self.porosity_at_depth(z)
        return self.grain_density * (1.0 - phi) + fluid_density * phi

    def decompacted_thickness(
        self,
        current_thickness: float,
        current_depth: float,
        target_depth: float = 0.0,
    ) -> float:
        """Thickness the unit would have if its top were at *target_depth*.

        Uses the integral of the porosity function over the layer thickness.
        Returns a positive float.
        """
        if self.compaction_coeff <= 0.0:
            return current_thickness   # incompressible (salt, basement)

        c  = self.compaction_coeff
        p0 = self.porosity_surface
        z1 = float(current_depth)
        z2 = z1 + float(current_thickness)

        # Solid thickness = integral of (1 - phi) dz from z1 to z2
        #   = (z2 - z1) - (p0/c) * (exp(-c*z1) - exp(-c*z2))
        solid = (z2 - z1) - (p0 / c) * (
            math.exp(-c * z1) - math.exp(-c * z2)
        )

        # Decompact: find H such that integral of (1-phi) from target to target+H = solid
        # (1-phi0)*H + (p0/c)*(1 - exp(-c*H)) = solid   [at surface target=0]
        # Solve numerically with Newton iteration
        zr = float(target_depth)
        H = solid / max(1.0 - p0, 0.01)   # initial guess
        for _ in range(50):
            f  = (H
                  - (p0 / c) * (math.exp(-c * zr) - math.exp(-c * (zr + H)))
                  - solid)
            df = 1.0 - p0 * math.exp(-c * (zr + H))
            if abs(df) < 1e-20:
                break
            H -= f / df
            if H < 0:
                H = max(solid, 0.01)
        return max(H, 0.0)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "age_top_ma": self.age_top_ma,
            "age_base_ma": self.age_base_ma,
            "color": list(self.color),
            "opacity": self.opacity,
            "lithology_pattern": self.lithology_pattern,
            "pattern_scale": self.pattern_scale,
            "primary_lithology": self.primary_lithology,
            "sand_fraction": self.sand_fraction,
            "shale_fraction": self.shale_fraction,
            "carbonate_fraction": self.carbonate_fraction,
            "porosity_surface": self.porosity_surface,
            "compaction_coeff": self.compaction_coeff,
            "grain_density": self.grain_density,
            "matrix_thermal_conductivity": self.matrix_thermal_conductivity,
            "radiogenic_heat_production": self.radiogenic_heat_production,
            "specific_heat_capacity": self.specific_heat_capacity,
            "matrix_velocity": self.matrix_velocity,
            "environment": self.environment,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Formation":
        color = tuple(d.get("color", [200, 200, 220]))
        return cls(
            name=d["name"],
            age_top_ma=d.get("age_top_ma"),
            age_base_ma=d.get("age_base_ma"),
            color=color,
            opacity=d.get("opacity", 0.6),
            lithology_pattern=d.get("lithology_pattern", "none"),
            pattern_scale=d.get("pattern_scale", 1.0),
            primary_lithology=d.get("primary_lithology", "shale"),
            sand_fraction=d.get("sand_fraction", 0.0),
            shale_fraction=d.get("shale_fraction", 1.0),
            carbonate_fraction=d.get("carbonate_fraction", 0.0),
            porosity_surface=d.get("porosity_surface", 0.50),
            compaction_coeff=d.get("compaction_coeff", 0.00050),
            grain_density=d.get("grain_density", 2720.0),
            matrix_thermal_conductivity=d.get("matrix_thermal_conductivity", 2.5),
            radiogenic_heat_production=d.get("radiogenic_heat_production", 1.0),
            specific_heat_capacity=d.get("specific_heat_capacity", 1000.0),
            matrix_velocity=d.get("matrix_velocity", 4500.0),
            environment=d.get("environment"),
        )

    def __repr__(self) -> str:
        return (
            f"Formation(name={self.name!r}, lithology={self.primary_lithology!r}, "
            f"phi0={self.porosity_surface:.2f}, c={self.compaction_coeff:.5f})"
        )


# ---------------------------------------------------------------------------
# StratigraphicColumn
# ---------------------------------------------------------------------------

class StratigraphicColumn:
    """Ordered list of formations — youngest (index 0) at top, oldest at bottom."""

    def __init__(self) -> None:
        self._formations: list[Formation] = []

    def add_formation(self, formation: Formation, position: int = -1) -> None:
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

    def get_formation(self, name: str) -> Optional[Formation]:
        for f in self._formations:
            if f.name == name:
                return f
        return None

    def is_above(self, name_a: str, name_b: str) -> bool:
        ia = self._index_of(name_a)
        ib = self._index_of(name_b)
        if ia is None or ib is None:
            raise KeyError(f"Formation not found: {name_a!r} or {name_b!r}")
        return ia < ib

    @property
    def formations(self) -> list[Formation]:
        return list(self._formations)

    def __len__(self) -> int:
        return len(self._formations)

    def __repr__(self) -> str:
        return f"StratigraphicColumn({[f.name for f in self._formations]!r})"

    def to_list(self) -> list[dict]:
        return [f.to_dict() for f in self._formations]

    @classmethod
    def from_list(cls, data: list[dict]) -> "StratigraphicColumn":
        col = cls()
        for d in data:
            col.add_formation(Formation.from_dict(d))
        return col

    def _index_of(self, name: str) -> Optional[int]:
        for i, f in enumerate(self._formations):
            if f.name == name:
                return i
        return None
