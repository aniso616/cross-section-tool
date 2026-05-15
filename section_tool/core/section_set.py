"""Serial section sets — ordered collections of parallel / sub-parallel sections.

A SectionSet groups sections for stratigraphic correlation.  It provides:
  * ordered membership (sort_index within the set)
  * adjacency queries (prev/next section for correlation panels)
  * ghost-pick projection: horizon picks from an adjacent section are
    geometrically projected onto the current section so interpreters can
    see correlation constraints without leaving their section view.

Ghost-pick geometry
-------------------
Given a pick at *distance_along_adjacent* on section A at depth *z*:

1. Convert to map coordinates:  (x, y) = A.section_to_map(distance_along_adjacent)
2. Project (x, y) onto section B: (dist_B, perp_B) = B.project_point(x, y)
3. If |perp_B| <= max_projection_dist AND 0 <= dist_B <= B.total_length():
   include (dist_B, z) in the ghost picks.

Depth is preserved (not re-projected) — ghost picks sit at the same
true depth as the original, just repositioned along the receiving section.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SectionSetMember:
    """One section within a SectionSet."""
    section_id:   int
    section_name: str
    sort_index:   int


@dataclass
class SectionSet:
    """An ordered collection of sections for serial correlation.

    Parameters
    ----------
    id:
        Database row ID (0 for unsaved sets).
    name:
        Display name.
    description:
        Optional free-text description.
    sort_order_field:
        How the ``sort_index`` values were assigned (e.g. ``'distance'``,
        ``'easting'``, ``'manual'``).  Informational only.
    members:
        Ordered list of :class:`SectionSetMember` records.
    """
    id:               int
    name:             str
    description:      str = ""
    sort_order_field: str = "distance"
    members:          list[SectionSetMember] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Membership helpers
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self.members)

    def section_ids(self) -> list[int]:
        """Return section IDs in sort_index order."""
        return [m.section_id for m in self._sorted_members()]

    def section_names(self) -> list[str]:
        """Return section names in sort_index order."""
        return [m.section_name for m in self._sorted_members()]

    def get_member(self, section_id: int) -> Optional[SectionSetMember]:
        for m in self.members:
            if m.section_id == section_id:
                return m
        return None

    def adjacent(
        self, section_id: int
    ) -> tuple[Optional[SectionSetMember], Optional[SectionSetMember]]:
        """Return (previous, next) :class:`SectionSetMember` for *section_id*.

        Returns ``None`` for prev/next when at the edges of the set, or
        ``(None, None)`` if *section_id* is not a member.
        """
        ordered = self._sorted_members()
        idx = next((i for i, m in enumerate(ordered)
                    if m.section_id == section_id), None)
        if idx is None:
            return None, None
        prev_m = ordered[idx - 1] if idx > 0 else None
        next_m = ordered[idx + 1] if idx < len(ordered) - 1 else None
        return prev_m, next_m

    def _sorted_members(self) -> list[SectionSetMember]:
        return sorted(self.members, key=lambda m: m.sort_index)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_db_dict(cls, d: dict) -> "SectionSet":
        """Construct from the dict returned by :meth:`ProjectDatabase.get_section_set`."""
        members = [
            SectionSetMember(
                section_id=m["section_id"],
                section_name=m["section_name"],
                sort_index=m["sort_index"],
            )
            for m in d.get("members", [])
        ]
        return cls(
            id=d["id"],
            name=d["name"],
            description=d.get("description", ""),
            sort_order_field=d.get("sort_order_field", "distance"),
            members=members,
        )


# ---------------------------------------------------------------------------
# Ghost-pick projection
# ---------------------------------------------------------------------------

@dataclass
class GhostPick:
    """A horizon pick from an adjacent section projected onto the current one.

    Attributes
    ----------
    distance:
        Projected distance along the *current* section (metres).
    depth:
        Original depth of the pick (preserved from the source section).
    source_section_name:
        Name of the section the pick originated from.
    perpendicular_offset:
        Signed perpendicular distance from the current section to the
        source map point (positive = left of travel direction).
    original_distance:
        Distance along the *source* section.
    """
    distance:             float
    depth:                float
    source_section_name:  str
    perpendicular_offset: float
    original_distance:    float


def get_ghost_picks(
    current_section,
    adjacent_section,
    horizon_pick,
    max_projection_dist: float = 5_000.0,
) -> list[GhostPick]:
    """Project horizon picks from *adjacent_section* onto *current_section*.

    Parameters
    ----------
    current_section:
        :class:`~section_tool.core.section.Section` — the section being viewed.
    adjacent_section:
        :class:`~section_tool.core.section.Section` — the neighbouring section
        whose picks should be ghosted onto *current_section*.
    horizon_pick:
        :class:`~section_tool.core.surfaces.HorizonPick` — the horizon pick
        object (may contain picks from multiple sections).
    max_projection_dist:
        Maximum perpendicular distance (same units as the CRS, typically
        metres) from *current_section* within which a source pick is
        included.  Picks further away are excluded.

    Returns
    -------
    list[GhostPick]
        Ghost picks sorted by projected distance along *current_section*.
        May be empty if no picks project within *max_projection_dist*.
    """
    adj_name = adjacent_section.name
    src_dists, src_depths = horizon_pick.picks_for_section(adj_name)
    if len(src_dists) == 0:
        return []

    total_len = current_section.total_length()
    result: list[GhostPick] = []

    for d_src, z in zip(src_dists, src_depths):
        # 1. Map coordinates of this pick on the adjacent section
        x, y = adjacent_section.section_to_map(float(d_src))
        # 2. Project onto current section (unclamped)
        d_proj, perp = current_section.project_point(x, y)
        # 3. Filter by perpendicular distance and section extent
        if abs(perp) > max_projection_dist:
            continue
        if d_proj < 0.0 or d_proj > total_len:
            continue
        result.append(GhostPick(
            distance=float(d_proj),
            depth=float(z),
            source_section_name=adj_name,
            perpendicular_offset=float(perp),
            original_distance=float(d_src),
        ))

    result.sort(key=lambda g: g.distance)
    return result


def get_all_ghost_picks(
    current_section,
    set_members_sections: list,
    horizon_pick,
    max_projection_dist: float = 5_000.0,
) -> dict[str, list[GhostPick]]:
    """Get ghost picks from all sections in a set (excluding the current one).

    Parameters
    ----------
    current_section:
        The section being viewed.
    set_members_sections:
        List of Section objects for all members of the set.  The current
        section is automatically excluded.
    horizon_pick:
        The horizon whose picks should be ghosted.
    max_projection_dist:
        Perpendicular cut-off distance.

    Returns
    -------
    dict[str, list[GhostPick]]
        Mapping of ``adjacent_section.name → ghost_picks`` for every
        neighbouring section that contributes at least one ghost pick.
    """
    result: dict[str, list[GhostPick]] = {}
    cur_name = current_section.name
    for sec in set_members_sections:
        if sec.name == cur_name:
            continue
        ghosts = get_ghost_picks(current_section, sec, horizon_pick,
                                 max_projection_dist)
        if ghosts:
            result[sec.name] = ghosts
    return result
