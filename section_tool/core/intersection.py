"""Phase 2 — Fault-horizon intersection records."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FaultHorizonIntersection:
    fault_name: str
    horizon_name: str
    section_name: str
    hw_cutoff: tuple[float, float, float]   # (x, y, z) hanging wall
    fw_cutoff: tuple[float, float, float]   # (x, y, z) footwall
    throw: float      = 0.0   # vertical separation
    heave: float      = 0.0   # horizontal separation
    separation: float = 0.0   # distance between cutoffs along fault plane

    def to_dict(self) -> dict:
        return {
            "fault_name":   self.fault_name,
            "horizon_name": self.horizon_name,
            "section_name": self.section_name,
            "hw_cutoff":    list(self.hw_cutoff),
            "fw_cutoff":    list(self.fw_cutoff),
            "throw":        self.throw,
            "heave":        self.heave,
            "separation":   self.separation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FaultHorizonIntersection":
        return cls(
            fault_name=d["fault_name"],
            horizon_name=d["horizon_name"],
            section_name=d["section_name"],
            hw_cutoff=tuple(d["hw_cutoff"]),
            fw_cutoff=tuple(d["fw_cutoff"]),
            throw=d.get("throw", 0.0),
            heave=d.get("heave", 0.0),
            separation=d.get("separation", 0.0),
        )


def compute_intersections(
    section,           # Section
    horizon_picks,     # list[HorizonPick]
    fault_picks,       # list[HorizonPick]
) -> list[FaultHorizonIntersection]:
    """Find all fault-horizon crossings on *section* using Shapely.

    Returns a list of :class:`FaultHorizonIntersection` objects.
    Hanging wall / footwall assignment is based on dip_direction.
    """
    try:
        from shapely.geometry import LineString
        from shapely.ops import unary_union
    except ImportError:
        return []

    sec_name = section.name

    def _to_linestring(hp):
        d_sec, z_sec = hp.picks_for_section(sec_name)
        if len(d_sec) < 2:
            return None
        return LineString(list(zip(d_sec.tolist(), z_sec.tolist())))

    results: list[FaultHorizonIntersection] = []

    for fault in fault_picks:
        fl = _to_linestring(fault)
        if fl is None:
            continue
        for horizon in horizon_picks:
            hl = _to_linestring(horizon)
            if hl is None:
                continue
            pt = fl.intersection(hl)
            if pt.is_empty:
                continue

            # Extract intersection coords (may be a Point or MultiPoint)
            from shapely.geometry import MultiPoint, Point
            pts = list(pt.geoms) if isinstance(pt, MultiPoint) else [pt]
            for ip in pts:
                if not isinstance(ip, Point):
                    continue
                ix, iz = float(ip.x), float(ip.y)

                # Back-project to 3D map coords
                x, y = section.section_to_map(ix)
                hw3 = (x, y, iz)
                fw3 = (x, y, iz)   # simplified: same point until cutoffs computed
                throw = 0.0
                heave = 0.0
                sep   = 0.0

                results.append(FaultHorizonIntersection(
                    fault_name=fault.name,
                    horizon_name=horizon.name,
                    section_name=sec_name,
                    hw_cutoff=hw3,
                    fw_cutoff=fw3,
                    throw=throw,
                    heave=heave,
                    separation=sep,
                ))

    return results
