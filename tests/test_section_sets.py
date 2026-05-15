"""Tests for serial section sets — database schema, CRUD, adjacency, ghost picks.

All tests are headless (no Qt).  Database tests use real SQLite in a temp dir.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from section_tool.core.section import Section
from section_tool.core.section_set import (
    GhostPick,
    SectionSet,
    SectionSetMember,
    get_all_ghost_picks,
    get_ghost_picks,
)
from section_tool.core.surfaces import HorizonPick
from section_tool.io.database import ProjectDatabase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path) -> ProjectDatabase:
    return ProjectDatabase(tmp_path / "test.db")


def _add_test_sections(db: ProjectDatabase) -> list[int]:
    """Insert 4 EW sections spaced 1 km apart; return their IDs."""
    ids = []
    for i in range(4):
        sec = Section([(0.0, float(i * 1000)), (10_000.0, float(i * 1000))],
                      name=f"EW{i}")
        db.upsert_section(sec)
        row = db.conn.execute(
            "SELECT id FROM sections WHERE name=?", (f"EW{i}",)
        ).fetchone()
        ids.append(row["id"])
    return ids


def _ew_section(y: float = 0.0, name: str = "EW") -> Section:
    return Section([(0.0, y), (10_000.0, y)], name=name)


def _horizon(section_name: str, n: int = 5, base_depth: float = 1000.0) -> HorizonPick:
    d = np.linspace(0, 10_000, n)
    z = np.linspace(base_depth, base_depth + 200, n)
    return HorizonPick(d, z, name="TopSand",
                       section_names=[section_name] * n, color="#0000ff")


# ---------------------------------------------------------------------------
# 1. Schema — tables exist and constraints work
# ---------------------------------------------------------------------------

class TestSchema:

    def test_section_sets_table_exists(self, db):
        rows = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='section_sets'"
        ).fetchall()
        assert len(rows) == 1

    def test_section_set_members_table_exists(self, db):
        rows = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='section_set_members'"
        ).fetchall()
        assert len(rows) == 1

    def test_unique_set_name_constraint(self, db):
        db.add_section_set("DipLines")
        with pytest.raises(Exception):
            db.add_section_set("DipLines")

    def test_unique_member_constraint(self, db):
        """A section can appear at most once in a given set."""
        sec_ids = _add_test_sections(db)
        sid = db.add_section_set("Set1")
        db.add_section_to_set(sid, sec_ids[0], 0)
        with pytest.raises(Exception):
            db.add_section_to_set(sid, sec_ids[0], 1)

    def test_cascade_delete_removes_members(self, db):
        sec_ids = _add_test_sections(db)
        sid = db.add_section_set("Set1")
        db.add_section_to_set(sid, sec_ids[0], 0)
        db.add_section_to_set(sid, sec_ids[1], 1)
        db.delete_section_set(sid)
        remaining = db.conn.execute(
            "SELECT COUNT(*) FROM section_set_members WHERE set_id=?", (sid,)
        ).fetchone()[0]
        assert remaining == 0

    def test_section_delete_cascades_to_memberships(self, db):
        """Deleting a section removes it from all sets automatically."""
        sec_ids = _add_test_sections(db)
        sid = db.add_section_set("Set1")
        db.add_section_to_set(sid, sec_ids[0], 0)
        db.add_section_to_set(sid, sec_ids[1], 1)
        # Delete section EW0
        db.delete_section("EW0")
        remaining = db.conn.execute(
            "SELECT COUNT(*) FROM section_set_members WHERE set_id=?", (sid,)
        ).fetchone()[0]
        assert remaining == 1   # only EW1 remains


# ---------------------------------------------------------------------------
# 2. add_section_set / delete_section_set
# ---------------------------------------------------------------------------

class TestAddDeleteSectionSet:

    def test_add_returns_id(self, db):
        sid = db.add_section_set("DipLines")
        assert isinstance(sid, int) and sid > 0

    def test_add_stores_name(self, db):
        sid = db.add_section_set("Strike Lines")
        result = db.get_section_set(sid)
        assert result["name"] == "Strike Lines"

    def test_add_stores_description(self, db):
        sid = db.add_section_set("Set A", description="Basin traverse")
        result = db.get_section_set(sid)
        assert result["description"] == "Basin traverse"

    def test_add_stores_sort_order_field(self, db):
        sid = db.add_section_set("Set A", sort_order_field="easting")
        result = db.get_section_set(sid)
        assert result["sort_order_field"] == "easting"

    def test_default_sort_order_field(self, db):
        sid = db.add_section_set("Set A")
        result = db.get_section_set(sid)
        assert result["sort_order_field"] == "distance"

    def test_delete_removes_set(self, db):
        sid = db.add_section_set("Temp")
        db.delete_section_set(sid)
        assert db.get_section_set(sid) is None

    def test_get_nonexistent_returns_none(self, db):
        assert db.get_section_set(9999) is None


# ---------------------------------------------------------------------------
# 3. add_section_to_set / remove_section_from_set
# ---------------------------------------------------------------------------

class TestMembership:

    def test_add_section_to_set(self, db):
        sec_ids = _add_test_sections(db)
        sid = db.add_section_set("Set1")
        db.add_section_to_set(sid, sec_ids[0], 0)
        result = db.get_section_set(sid)
        assert len(result["members"]) == 1

    def test_member_has_correct_section_name(self, db):
        sec_ids = _add_test_sections(db)
        sid = db.add_section_set("Set1")
        db.add_section_to_set(sid, sec_ids[0], 0)
        result = db.get_section_set(sid)
        assert result["members"][0]["section_name"] == "EW0"

    def test_member_has_correct_sort_index(self, db):
        sec_ids = _add_test_sections(db)
        sid = db.add_section_set("Set1")
        db.add_section_to_set(sid, sec_ids[2], 7)
        result = db.get_section_set(sid)
        assert result["members"][0]["sort_index"] == 7

    def test_multiple_members_ordered(self, db):
        sec_ids = _add_test_sections(db)
        sid = db.add_section_set("Set1")
        db.add_section_to_set(sid, sec_ids[2], 2)
        db.add_section_to_set(sid, sec_ids[0], 0)
        db.add_section_to_set(sid, sec_ids[1], 1)
        result = db.get_section_set(sid)
        names = [m["section_name"] for m in result["members"]]
        assert names == ["EW0", "EW1", "EW2"]

    def test_remove_section_from_set(self, db):
        sec_ids = _add_test_sections(db)
        sid = db.add_section_set("Set1")
        db.add_section_to_set(sid, sec_ids[0], 0)
        db.add_section_to_set(sid, sec_ids[1], 1)
        db.remove_section_from_set(sid, sec_ids[0])
        result = db.get_section_set(sid)
        assert len(result["members"]) == 1
        assert result["members"][0]["section_name"] == "EW1"

    def test_remove_nonexistent_member_noop(self, db):
        sec_ids = _add_test_sections(db)
        sid = db.add_section_set("Set1")
        db.remove_section_from_set(sid, sec_ids[0])   # never added — must not crash


# ---------------------------------------------------------------------------
# 4. get_all_section_sets
# ---------------------------------------------------------------------------

class TestGetAllSectionSets:

    def test_empty_returns_empty_list(self, db):
        assert db.get_all_section_sets() == []

    def test_returns_all_sets(self, db):
        db.add_section_set("A")
        db.add_section_set("B")
        db.add_section_set("C")
        result = db.get_all_section_sets()
        assert len(result) == 3

    def test_each_set_includes_members(self, db):
        sec_ids = _add_test_sections(db)
        sid = db.add_section_set("Set1")
        db.add_section_to_set(sid, sec_ids[0], 0)
        db.add_section_to_set(sid, sec_ids[1], 1)
        all_sets = db.get_all_section_sets()
        assert len(all_sets[0]["members"]) == 2


# ---------------------------------------------------------------------------
# 5. reorder_set_member
# ---------------------------------------------------------------------------

class TestReorderSetMember:

    def test_reorder_changes_sort_index(self, db):
        sec_ids = _add_test_sections(db)
        sid = db.add_section_set("Set1")
        db.add_section_to_set(sid, sec_ids[0], 0)
        db.reorder_set_member(sid, sec_ids[0], 5)
        result = db.get_section_set(sid)
        assert result["members"][0]["sort_index"] == 5

    def test_reorder_changes_display_order(self, db):
        sec_ids = _add_test_sections(db)
        sid = db.add_section_set("Set1")
        db.add_section_to_set(sid, sec_ids[0], 0)
        db.add_section_to_set(sid, sec_ids[1], 1)
        db.add_section_to_set(sid, sec_ids[2], 2)
        # Swap first and last
        db.reorder_set_member(sid, sec_ids[0], 10)
        db.reorder_set_member(sid, sec_ids[2], 0)
        result = db.get_section_set(sid)
        names = [m["section_name"] for m in result["members"]]
        assert names[0] == "EW2"
        assert names[-1] == "EW0"


# ---------------------------------------------------------------------------
# 6. get_adjacent_sections
# ---------------------------------------------------------------------------

class TestGetAdjacentSections:

    def _setup_linear_set(self, db):
        sec_ids = _add_test_sections(db)
        sid = db.add_section_set("Linear")
        for i, sec_id in enumerate(sec_ids):
            db.add_section_to_set(sid, sec_id, i)
        return sid, sec_ids

    def test_first_section_has_no_prev(self, db):
        sid, sec_ids = self._setup_linear_set(db)
        prev, nxt = db.get_adjacent_sections(sid, sec_ids[0])
        assert prev is None
        assert nxt is not None

    def test_last_section_has_no_next(self, db):
        sid, sec_ids = self._setup_linear_set(db)
        prev, nxt = db.get_adjacent_sections(sid, sec_ids[-1])
        assert nxt is None
        assert prev is not None

    def test_middle_section_has_both_neighbours(self, db):
        sid, sec_ids = self._setup_linear_set(db)
        prev, nxt = db.get_adjacent_sections(sid, sec_ids[1])
        assert prev is not None and nxt is not None
        assert prev["section_name"] == "EW0"
        assert nxt["section_name"] == "EW2"

    def test_nonmember_returns_none_none(self, db):
        sid, sec_ids = self._setup_linear_set(db)
        prev, nxt = db.get_adjacent_sections(sid, 9999)
        assert prev is None
        assert nxt is None

    def test_single_member_has_no_neighbours(self, db):
        sec_ids = _add_test_sections(db)
        sid = db.add_section_set("Solo")
        db.add_section_to_set(sid, sec_ids[0], 0)
        prev, nxt = db.get_adjacent_sections(sid, sec_ids[0])
        assert prev is None
        assert nxt is None

    def test_neighbours_reference_correct_sections(self, db):
        sid, sec_ids = self._setup_linear_set(db)
        prev, nxt = db.get_adjacent_sections(sid, sec_ids[2])
        assert prev["section_id"] == sec_ids[1]
        assert nxt["section_id"] == sec_ids[3]


# ---------------------------------------------------------------------------
# 7. update_section_set
# ---------------------------------------------------------------------------

class TestUpdateSectionSet:

    def test_update_name(self, db):
        sid = db.add_section_set("Old Name")
        db.update_section_set(sid, name="New Name")
        assert db.get_section_set(sid)["name"] == "New Name"

    def test_update_description(self, db):
        sid = db.add_section_set("Set1")
        db.update_section_set(sid, description="Updated desc")
        assert db.get_section_set(sid)["description"] == "Updated desc"

    def test_update_sort_order_field(self, db):
        sid = db.add_section_set("Set1")
        db.update_section_set(sid, sort_order_field="easting")
        assert db.get_section_set(sid)["sort_order_field"] == "easting"

    def test_update_nothing_is_noop(self, db):
        sid = db.add_section_set("Set1", description="orig")
        db.update_section_set(sid)   # no kwargs — must not crash
        assert db.get_section_set(sid)["description"] == "orig"


# ---------------------------------------------------------------------------
# 8. SectionSet core model
# ---------------------------------------------------------------------------

class TestSectionSetModel:

    def _make_set(self) -> SectionSet:
        return SectionSet(
            id=1, name="DipLines",
            members=[
                SectionSetMember(section_id=10, section_name="EW0", sort_index=0),
                SectionSetMember(section_id=11, section_name="EW1", sort_index=1),
                SectionSetMember(section_id=12, section_name="EW2", sort_index=2),
            ]
        )

    def test_size(self):
        ss = self._make_set()
        assert ss.size == 3

    def test_section_names_ordered(self):
        ss = self._make_set()
        assert ss.section_names() == ["EW0", "EW1", "EW2"]

    def test_section_ids_ordered(self):
        ss = self._make_set()
        assert ss.section_ids() == [10, 11, 12]

    def test_get_member_found(self):
        ss = self._make_set()
        m = ss.get_member(11)
        assert m is not None
        assert m.section_name == "EW1"

    def test_get_member_not_found(self):
        ss = self._make_set()
        assert ss.get_member(99) is None

    def test_adjacent_middle(self):
        ss = self._make_set()
        prev, nxt = ss.adjacent(11)
        assert prev.section_id == 10
        assert nxt.section_id == 12

    def test_adjacent_first_no_prev(self):
        ss = self._make_set()
        prev, nxt = ss.adjacent(10)
        assert prev is None
        assert nxt.section_id == 11

    def test_adjacent_last_no_next(self):
        ss = self._make_set()
        prev, nxt = ss.adjacent(12)
        assert prev.section_id == 11
        assert nxt is None

    def test_adjacent_nonmember(self):
        ss = self._make_set()
        prev, nxt = ss.adjacent(99)
        assert prev is None
        assert nxt is None

    def test_from_db_dict(self):
        d = {
            "id": 5, "name": "Strike",
            "description": "Basin", "sort_order_field": "northing",
            "members": [
                {"section_id": 1, "section_name": "S1", "sort_index": 0},
                {"section_id": 2, "section_name": "S2", "sort_index": 1},
            ]
        }
        ss = SectionSet.from_db_dict(d)
        assert ss.id == 5
        assert ss.name == "Strike"
        assert ss.sort_order_field == "northing"
        assert len(ss.members) == 2
        assert ss.members[0].section_name == "S1"


# ---------------------------------------------------------------------------
# 9. get_ghost_picks — geometry
# ---------------------------------------------------------------------------

class TestGetGhostPicks:
    """Two parallel EW sections 1 km apart; picks from section A ghosted on B."""

    @pytest.fixture
    def sections(self):
        s_a = _ew_section(y=0.0,    name="A")   # y=0
        s_b = _ew_section(y=1000.0, name="B")   # 1 km north
        return s_a, s_b

    def test_pick_projects_at_correct_distance(self, sections):
        s_a, s_b = sections
        hp = _horizon("A", n=1)   # single pick at d=0
        hp._distances[0] = 5000.0
        hp._depths[0] = 1200.0
        ghosts = get_ghost_picks(s_b, s_a, hp, max_projection_dist=2000.0)
        assert len(ghosts) == 1
        # Parallel sections: ghost distance on B == original distance on A
        assert ghosts[0].distance == pytest.approx(5000.0, abs=0.1)

    def test_depth_preserved(self, sections):
        s_a, s_b = sections
        hp = _horizon("A", n=1)
        hp._depths[0] = 1500.0
        hp._distances[0] = 3000.0
        ghosts = get_ghost_picks(s_b, s_a, hp, max_projection_dist=2000.0)
        assert ghosts[0].depth == pytest.approx(1500.0)

    def test_pick_outside_max_dist_excluded(self, sections):
        s_a, s_b = sections   # 1 km apart
        hp = _horizon("A", n=1)
        ghosts = get_ghost_picks(s_b, s_a, hp, max_projection_dist=500.0)
        assert len(ghosts) == 0

    def test_pick_within_max_dist_included(self, sections):
        s_a, s_b = sections   # 1 km apart
        hp = _horizon("A", n=3)
        ghosts = get_ghost_picks(s_b, s_a, hp, max_projection_dist=1500.0)
        assert len(ghosts) == 3

    def test_ghost_sorted_by_distance(self, sections):
        s_a, s_b = sections
        hp = HorizonPick(
            np.array([7000.0, 2000.0, 5000.0]),
            np.array([1100.0, 1000.0, 1050.0]),
            name="TopSand",
            section_names=["A", "A", "A"],
        )
        ghosts = get_ghost_picks(s_b, s_a, hp, max_projection_dist=2000.0)
        dists = [g.distance for g in ghosts]
        assert dists == sorted(dists)

    def test_no_picks_on_adjacent_section_returns_empty(self, sections):
        s_a, s_b = sections
        # Horizon has picks only on "C", not on "A"
        hp = _horizon("C", n=3)
        ghosts = get_ghost_picks(s_b, s_a, hp, max_projection_dist=5000.0)
        assert ghosts == []

    def test_ghost_pick_source_name(self, sections):
        s_a, s_b = sections
        hp = _horizon("A", n=1)
        ghosts = get_ghost_picks(s_b, s_a, hp, max_projection_dist=2000.0)
        assert ghosts[0].source_section_name == "A"

    def test_ghost_pick_perpendicular_offset(self, sections):
        s_a, s_b = sections   # B is 1000 m north of A
        hp = _horizon("A", n=1)
        hp._distances[0] = 5000.0
        ghosts = get_ghost_picks(s_b, s_a, hp, max_projection_dist=2000.0)
        # A is south of B — perp offset should be negative (right of B's travel)
        assert ghosts[0].perpendicular_offset == pytest.approx(-1000.0, abs=0.1)

    def test_ghost_pick_original_distance_preserved(self, sections):
        s_a, s_b = sections
        hp = _horizon("A", n=1)
        hp._distances[0] = 4321.0
        ghosts = get_ghost_picks(s_b, s_a, hp, max_projection_dist=2000.0)
        assert ghosts[0].original_distance == pytest.approx(4321.0, abs=0.1)

    def test_pick_before_section_start_excluded(self):
        """Pick that projects before x=0 on receiving section is excluded."""
        s_a = Section([(2000.0, 0.0), (8000.0, 0.0)], name="A")   # short section
        s_b = Section([(0.0, 1000.0), (10000.0, 1000.0)], name="B")
        hp = HorizonPick(
            np.array([0.0]),   # distance=0 on A → map (2000, 0)
            np.array([1000.0]),
            name="H", section_names=["A"],
        )
        ghosts = get_ghost_picks(s_b, s_a, hp, max_projection_dist=2000.0)
        # (2000, 0) projects to d=2000 on B — well within [0, 10000]
        assert len(ghosts) == 1
        assert ghosts[0].distance == pytest.approx(2000.0, abs=0.1)

    def test_pick_past_section_end_excluded(self):
        """Pick projecting past total_length is excluded."""
        s_a = _ew_section(y=0.0, name="A")     # 0–10000
        s_b = Section([(0.0, 1000.0), (5000.0, 1000.0)], name="B")  # 0–5000
        hp = HorizonPick(
            np.array([8000.0]),
            np.array([1000.0]),
            name="H", section_names=["A"],
        )
        ghosts = get_ghost_picks(s_b, s_a, hp, max_projection_dist=2000.0)
        # (8000, 0) projects to d=8000 on B, but B only goes to 5000 → excluded
        assert len(ghosts) == 0

    def test_default_max_dist_is_5000(self, sections):
        """Default max_projection_dist=5000 includes picks 1 km away."""
        s_a, s_b = sections   # 1 km apart
        hp = _horizon("A", n=3)
        ghosts = get_ghost_picks(s_b, s_a, hp)  # no max_projection_dist kwarg
        assert len(ghosts) == 3


# ---------------------------------------------------------------------------
# 10. get_all_ghost_picks
# ---------------------------------------------------------------------------

class TestGetAllGhostPicks:

    def test_excludes_current_section(self):
        s_curr = _ew_section(y=0.0, name="Current")
        s_a    = _ew_section(y=1000.0, name="A")
        hp = HorizonPick(
            np.array([5000.0]),
            np.array([1000.0]),
            name="H",
            section_names=["Current"],  # pick only on current
        )
        result = get_all_ghost_picks(s_curr, [s_curr, s_a], hp, max_projection_dist=5000.0)
        # No ghost picks from A (no picks on A)
        assert "Current" not in result

    def test_gathers_from_multiple_adjacents(self):
        s_curr = _ew_section(y=0.0,    name="C")
        s_a    = _ew_section(y=500.0,  name="A")
        s_b    = _ew_section(y=-500.0, name="B")
        hp_a = _horizon("A", n=2)
        hp_b = _horizon("B", n=2)
        # Combine into one HorizonPick with picks on both A and B
        combined = HorizonPick(
            np.concatenate([hp_a._distances, hp_b._distances]),
            np.concatenate([hp_a._depths,    hp_b._depths]),
            name="TopSand",
            section_names=["A", "A", "B", "B"],
        )
        result = get_all_ghost_picks(s_curr, [s_curr, s_a, s_b], combined,
                                     max_projection_dist=1000.0)
        assert "A" in result
        assert "B" in result

    def test_empty_when_all_out_of_range(self):
        s_curr = _ew_section(y=0.0,       name="C")
        s_far  = _ew_section(y=100_000.0, name="Far")   # 100 km away
        hp = _horizon("Far", n=3)
        result = get_all_ghost_picks(s_curr, [s_curr, s_far], hp,
                                     max_projection_dist=5000.0)
        assert result == {}


# ---------------------------------------------------------------------------
# 11. Integration — full workflow
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_full_workflow(self, db):
        """Create set → add sections → query adjacency → get ghost picks."""
        # Insert sections
        sec_ids = _add_test_sections(db)  # EW0-EW3 at y=0,1000,2000,3000

        # Create a dip-line set in northward order
        sid = db.add_section_set("Dip Lines", description="Basin dip direction")
        for i, sec_id in enumerate(sec_ids):
            db.add_section_to_set(sid, sec_id, i)

        # Verify membership
        ss_dict = db.get_section_set(sid)
        assert len(ss_dict["members"]) == 4

        # Build SectionSet model
        ss = SectionSet.from_db_dict(ss_dict)
        assert ss.section_names() == ["EW0", "EW1", "EW2", "EW3"]

        # Adjacency for EW1
        prev, nxt = db.get_adjacent_sections(sid, sec_ids[1])
        assert prev["section_name"] == "EW0"
        assert nxt["section_name"] == "EW2"

        # Ghost picks: EW0 picks appear on EW1 (1 km away)
        s_ew0 = Section([(0.0, 0.0),    (10_000.0, 0.0)],    name="EW0")
        s_ew1 = Section([(0.0, 1000.0), (10_000.0, 1000.0)], name="EW1")
        hp = _horizon("EW0", n=5, base_depth=1000.0)
        ghosts = get_ghost_picks(s_ew1, s_ew0, hp, max_projection_dist=1500.0)
        assert len(ghosts) == 5
        # All ghost picks preserve their depth
        src_depths = sorted(hp._depths.tolist())
        ghost_depths = sorted(g.depth for g in ghosts)
        assert src_depths == pytest.approx(ghost_depths, abs=0.01)

    def test_reorder_then_adjacency(self, db):
        """After reordering, adjacency should reflect the new order."""
        sec_ids = _add_test_sections(db)
        sid = db.add_section_set("Set1")
        for i, sec_id in enumerate(sec_ids):
            db.add_section_to_set(sid, sec_id, i)

        # Move EW3 to position 0 (before EW0)
        db.reorder_set_member(sid, sec_ids[3], -1)
        prev, nxt = db.get_adjacent_sections(sid, sec_ids[3])
        assert prev is None  # now first
        assert nxt["section_name"] == "EW0"
