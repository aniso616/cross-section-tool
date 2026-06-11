"""Data inventory — a read-side service answering "which conversion rungs does
the data in this project actually unlock?"

Pure and side-effect free.  It reports only what is **loaded in the project**
(wells, their logs/TDRs/tops, seismic-tied horizon picks) — never what might
exist on disk.  Prompt 07's panel consumes it; nothing here touches the UI.

``unlocked_rungs()`` is a deterministic mapping from the inventory to the set of
available method-ladder rungs:

==========================  ============================================
condition                    rung(s) unlocked
==========================  ============================================
always                       ``bulk``, ``average``, ``layered``
a depth↔twt tie              ``checkshot``
sonic log + a tie            ``sonic_checkshot``
sonic log + seismic anchors  ``sonic_anchors``
tops + seismic anchors       ``marker_tied``
==========================  ============================================

A "depth↔twt tie" is a checkshot **or** a sonic-integrated TDR — both give an
independent, usable time-depth relationship.  ``velocity_functions_present`` is
reported but maps to no rung yet (Dix lands in Prompt 08).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Mnemonic hints that mark a log as a (compressional) sonic / slowness curve.
# F3 F02-01 carries DT:1 / DT:2; other vendors use DTC, DTCO, AC, SONIC.
def detect_sonic_curve(well) -> str | None:
    """Return the name of the well's sonic log curve, or None if absent."""
    for name in getattr(well, "log_names", []):
        up = name.upper()
        if up.startswith("DT") or up.startswith("AC") or "SONIC" in up:
            return name
    return None


def _is_tied(hp) -> bool:
    """A horizon pick carries a usable TWT anchor (seismic-tied)."""
    if not getattr(hp, "seismic_tied", False):
        return False
    anch = getattr(hp, "_twt_anchor", None)
    if anch is None:
        return False
    arr = np.asarray(anch, dtype=float)
    return bool(arr.size) and bool(np.any(np.isfinite(arr)))


@dataclass(frozen=True)
class WellData:
    """Per-well data availability (loaded state only)."""
    well_name: str
    well_uuid: str
    has_sonic: bool
    sonic_curve: str | None
    has_checkshot: bool
    has_sonic_tdr: bool
    n_tops: int

    @property
    def has_td_tie(self) -> bool:
        """A usable depth↔twt tie: a checkshot or a sonic-integrated TDR."""
        return self.has_checkshot or self.has_sonic_tdr


@dataclass(frozen=True)
class DataInventory:
    """What a section's corridor wells + the project's tied horizons make available."""
    section_name: str
    wells: tuple[WellData, ...] = ()
    n_tied_horizons: int = 0
    velocity_functions_present: bool = False

    # ---- derived corridor-level capabilities ----

    @property
    def has_anchors(self) -> bool:
        return self.n_tied_horizons > 0

    @property
    def any_sonic(self) -> bool:
        return any(w.has_sonic for w in self.wells)

    @property
    def any_td_tie(self) -> bool:
        return any(w.has_td_tie for w in self.wells)

    @property
    def any_sonic_with_tie(self) -> bool:
        """A single well carrying BOTH a sonic log and a tie (sonic→checkshot)."""
        return any(w.has_sonic and w.has_td_tie for w in self.wells)

    @property
    def any_tops(self) -> bool:
        return any(w.n_tops > 0 for w in self.wells)

    def unlocked_rungs(self) -> frozenset[str]:
        """Deterministic inventory → available rungs (a SET, supersets allowed)."""
        rungs = set(ALWAYS_RUNGS)                    # bulk + average_vz, always
        # 'layered' needs zone tops, i.e. seismic-tied picks (build_layered_from_
        # formations raises without them) — so it is gated on anchors, not free.
        if self.has_anchors:
            rungs.add("layered")
        if self.any_td_tie:
            rungs.add("checkshot")
        if self.any_sonic_with_tie:
            rungs.add("sonic_checkshot")
        if self.any_sonic and self.has_anchors:
            rungs.add("sonic_anchors")
        if self.any_tops and self.has_anchors:
            rungs.add("marker_tied")
        # velocity_functions_present → Dix: reserved, not mapped yet (Prompt 08).
        return frozenset(rungs)

    def recommended_rung(self) -> str:
        """The single best available rung — highest in RECOMMENDATION_ORDER that
        is unlocked. Deterministic; always returns at least ``bulk``."""
        unlocked = self.unlocked_rungs()
        for rung in RECOMMENDATION_ORDER:
            if rung in unlocked:
                return rung
        return "bulk"


# Methods that need no data — the floor of the ladder, always available.
# ('layered' is NOT here: it needs zone tops / seismic anchors — see unlocked_rungs.)
ALWAYS_RUNGS = frozenset({"bulk", "average_vz"})

# Composed precedence, top (most grounded) → bottom (least). recommended_rung()
# returns the highest unlocked entry. The grounded rungs sit above the
# interpretation/bootstrap rungs.
RECOMMENDATION_ORDER: tuple[str, ...] = (
    "sonic_checkshot",   # sonic log + a tie (drift-corrected) — best
    "checkshot",         # a depth↔twt tie alone
    "sonic_anchors",     # sonic log + seismic anchors
    "marker_tied",       # tops + seismic anchors
    "layered",           # interval velocities from picked zone tops
    "bulk",              # single constant velocity — the honest no-knowledge default
    "average_vz",        # v0 + k·z bootstrap (a user choice; never auto-recommended)
)


def build_well_data(well) -> WellData:
    """Inspect a single :class:`~section_tool.core.wells.Well`'s loaded data."""
    curve = detect_sonic_curve(well)
    return WellData(
        well_name=getattr(well, "name", ""),
        well_uuid=getattr(well, "uuid", ""),
        has_sonic=curve is not None,
        sonic_curve=curve,
        has_checkshot=bool(well.tdrs_of_kind("checkshot")),
        has_sonic_tdr=bool(well.tdrs_of_kind("sonic_integrated")),
        n_tops=len(getattr(well, "formation_tops", {})),
    )


def build_inventory(
    section,
    wells,
    horizon_picks,
    *,
    velocity_functions_present: bool = False,
) -> DataInventory:
    """Assemble a :class:`DataInventory` from loaded project objects.

    *wells* should already be the corridor wells for *section* (use
    :func:`wells_in_corridor` to filter).  *horizon_picks* is the project's pick
    list; tied picks (carrying a TWT anchor) are counted toward ``n_tied_horizons``.
    """
    wds = tuple(build_well_data(w) for w in wells)
    n_tied = sum(1 for hp in horizon_picks if _is_tied(hp))
    return DataInventory(
        section_name=getattr(section, "name", "") if section is not None else "",
        wells=wds,
        n_tied_horizons=n_tied,
        velocity_functions_present=velocity_functions_present,
    )


def wells_in_corridor(section, wells, corridor_m: float = 2000.0) -> list:
    """Wells whose collar projects within ``±corridor_m`` of *section* (perp offset).

    A thin geometric filter so the inventory speaks to the section at hand. Wells
    that fail to project (no section / error) are skipped.
    """
    if section is None:
        return list(wells)
    out = []
    for w in wells:
        try:
            _dist, perp = w.project_to_section(section)
        except Exception:
            continue
        if abs(perp) <= corridor_m:
            out.append(w)
    return out
