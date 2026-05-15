"""Comprehensive tests for the Command pattern undo/redo system.

All tests are headless (no Qt): they use SimpleState — a minimal in-memory
state object that implements the same mutation interface as AppState without
any Qt/Signal machinery.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from section_tool.core.commands import (
    Command,
    CommandStack,
    cmd_add_fault_pick,
    cmd_add_horizon_pick,
    cmd_add_section,
    cmd_add_well,
    cmd_remove_fault_pick,
    cmd_remove_horizon_pick,
    cmd_remove_section,
    cmd_remove_well,
    cmd_update_fault_pick,
    cmd_update_horizon_pick,
    cmd_update_section,
)
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick
from section_tool.core.wells import Well


# ---------------------------------------------------------------------------
# Minimal in-memory state (no Qt)
# ---------------------------------------------------------------------------

class _SimpleProject:
    def __init__(self):
        self.horizon_picks: list = []
        self.fault_picks: list = []
        self.sections: list = []
        self.wells: list = []


class SimpleState:
    """Minimal AppState stand-in for testing command factories headlessly."""

    def __init__(self):
        self.project = _SimpleProject()
        self._stack = CommandStack()

    def execute_command(self, command: Command) -> None:
        self._stack.execute(command)

    def undo(self) -> str | None:
        return self._stack.undo()

    def redo(self) -> str | None:
        return self._stack.redo()

    # Horizon picks
    def add_horizon_pick(self, pick):
        self.project.horizon_picks.append(pick)

    def remove_horizon_pick(self, pick):
        self.project.horizon_picks.remove(pick)

    def update_horizon_pick(self, idx: int, pick):
        self.project.horizon_picks[idx] = pick

    # Fault picks
    def add_fault_pick(self, pick):
        self.project.fault_picks.append(pick)

    def remove_fault_pick(self, pick):
        self.project.fault_picks.remove(pick)

    def update_fault_pick(self, idx: int, pick):
        self.project.fault_picks[idx] = pick

    # Sections
    def add_section(self, section):
        self.project.sections.append(section)

    def remove_section(self, section):
        self.project.sections.remove(section)

    def update_section(self, idx: int, section):
        self.project.sections[idx] = section

    # Wells
    def add_well(self, well):
        self.project.wells.append(well)

    def remove_well(self, well):
        self.project.wells.remove(well)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _horizon(name="TopSand", n=3, section="S1") -> HorizonPick:
    d = np.linspace(0, 10_000, n)
    z = np.linspace(1000, 1200, n)
    return HorizonPick(d, z, name=name,
                       section_names=[section] * n, color="#0000ff")


def _section(name="EW") -> Section:
    return Section([(0.0, 0.0), (10_000.0, 0.0)], name=name)


def _well(name="W1") -> Well:
    return Well(name=name, x=5_000.0, y=0.0, kb=20.0)


@pytest.fixture
def state() -> SimpleState:
    return SimpleState()


# ---------------------------------------------------------------------------
# 1. CommandStack basics
# ---------------------------------------------------------------------------

class TestCommandStack:

    def test_empty_stack_cannot_undo(self):
        stack = CommandStack()
        assert not stack.can_undo()
        assert stack.undo() is None

    def test_empty_stack_cannot_redo(self):
        stack = CommandStack()
        assert not stack.can_redo()
        assert stack.redo() is None

    def test_execute_makes_can_undo_true(self):
        stack = CommandStack()
        cmd = Command("op", do=lambda: None, undo=lambda: None)
        stack.execute(cmd)
        assert stack.can_undo()

    def test_execute_clears_redo_stack(self):
        stack = CommandStack()
        calls = []
        cmd1 = Command("c1", do=lambda: calls.append("do1"), undo=lambda: calls.append("u1"))
        cmd2 = Command("c2", do=lambda: calls.append("do2"), undo=lambda: calls.append("u2"))
        stack.execute(cmd1)
        stack.undo()
        assert stack.can_redo()
        stack.execute(cmd2)
        assert not stack.can_redo()

    def test_undo_returns_description(self):
        stack = CommandStack()
        cmd = Command("Add horizon", do=lambda: None, undo=lambda: None)
        stack.execute(cmd)
        assert stack.undo() == "Add horizon"

    def test_redo_returns_description(self):
        stack = CommandStack()
        cmd = Command("Delete fault", do=lambda: None, undo=lambda: None)
        stack.execute(cmd)
        stack.undo()
        assert stack.redo() == "Delete fault"

    def test_undo_calls_undo_callable(self):
        calls = []
        cmd = Command("op", do=lambda: None, undo=lambda: calls.append("undo"))
        stack = CommandStack()
        stack.execute(cmd)
        stack.undo()
        assert calls == ["undo"]

    def test_redo_calls_do_callable(self):
        calls = []
        cmd = Command("op", do=lambda: calls.append("do"), undo=lambda: None)
        stack = CommandStack()
        stack.execute(cmd)   # first "do"
        calls.clear()
        stack.undo()
        stack.redo()         # second "do"
        assert calls == ["do"]

    def test_undo_description_property(self):
        stack = CommandStack()
        assert stack.undo_description is None
        cmd = Command("Horizon A", do=lambda: None, undo=lambda: None)
        stack.execute(cmd)
        assert stack.undo_description == "Horizon A"

    def test_redo_description_property(self):
        stack = CommandStack()
        cmd = Command("Horizon A", do=lambda: None, undo=lambda: None)
        stack.execute(cmd)
        stack.undo()
        assert stack.redo_description == "Horizon A"

    def test_push_does_not_call_do(self):
        calls = []
        cmd = Command("op", do=lambda: calls.append("do"), undo=lambda: None)
        stack = CommandStack()
        stack.push(cmd)   # backward-compat: already-applied
        assert calls == []
        assert stack.can_undo()

    def test_clear_empties_both_stacks(self):
        stack = CommandStack()
        cmd = Command("op", do=lambda: None, undo=lambda: None)
        stack.execute(cmd)
        stack.undo()
        stack.clear()
        assert not stack.can_undo()
        assert not stack.can_redo()

    def test_stack_size_properties(self):
        stack = CommandStack()
        assert stack.undo_stack_size == 0
        assert stack.redo_stack_size == 0
        cmd = Command("op", do=lambda: None, undo=lambda: None)
        stack.execute(cmd)
        assert stack.undo_stack_size == 1
        stack.undo()
        assert stack.undo_stack_size == 0
        assert stack.redo_stack_size == 1


# ---------------------------------------------------------------------------
# 2. Depth limit
# ---------------------------------------------------------------------------

class TestDepthLimit:

    def test_oldest_command_dropped_when_limit_exceeded(self):
        stack = CommandStack(max_depth=5)
        calls = []
        for i in range(6):
            cmd = Command(f"cmd{i}", do=lambda i=i: calls.append(f"do{i}"),
                          undo=lambda i=i: calls.append(f"undo{i}"))
            stack.execute(cmd)
        assert stack.undo_stack_size == 5
        # Undo all: should execute undo5, undo4, undo3, undo2, undo1 (not undo0)
        calls.clear()
        for _ in range(5):
            stack.undo()
        assert "undo0" not in calls

    def test_depth_100_limit(self):
        stack = CommandStack(max_depth=100)
        for i in range(101):
            stack.execute(Command(f"cmd{i}", do=lambda: None, undo=lambda: None))
        assert stack.undo_stack_size == 100

    def test_redo_cleared_on_new_action(self):
        stack = CommandStack()
        cmd = Command("op", do=lambda: None, undo=lambda: None)
        stack.execute(cmd)
        stack.undo()
        assert stack.can_redo()
        stack.execute(Command("new", do=lambda: None, undo=lambda: None))
        assert not stack.can_redo()
        assert stack.redo_stack_size == 0


# ---------------------------------------------------------------------------
# 3. cmd_add_horizon_pick: add → undo → redo cycle
# ---------------------------------------------------------------------------

class TestCmdAddHorizonPick:

    def test_add_inserts_pick(self, state):
        hp = _horizon()
        state.execute_command(cmd_add_horizon_pick(state, hp))
        assert len(state.project.horizon_picks) == 1

    def test_add_undo_removes_pick(self, state):
        hp = _horizon()
        state.execute_command(cmd_add_horizon_pick(state, hp))
        state.undo()
        assert len(state.project.horizon_picks) == 0

    def test_add_undo_redo_restores_pick(self, state):
        hp = _horizon(name="TopSand")
        state.execute_command(cmd_add_horizon_pick(state, hp))
        state.undo()
        state.redo()
        assert len(state.project.horizon_picks) == 1
        assert state.project.horizon_picks[0].name == "TopSand"

    def test_multiple_undo_redo_cycles(self, state):
        hp = _horizon()
        state.execute_command(cmd_add_horizon_pick(state, hp))
        for _ in range(3):
            state.undo()
            assert len(state.project.horizon_picks) == 0
            state.redo()
            assert len(state.project.horizon_picks) == 1

    def test_added_pick_is_deep_copy(self, state):
        """Mutating original after execute must not affect stored pick."""
        hp = _horizon()
        state.execute_command(cmd_add_horizon_pick(state, hp))
        hp._depths[0] = 9999.0
        stored = state.project.horizon_picks[0]
        assert stored._depths[0] != 9999.0

    def test_description_contains_name(self, state):
        hp = _horizon(name="BaseSand")
        cmd = cmd_add_horizon_pick(state, hp)
        assert "BaseSand" in cmd.description


# ---------------------------------------------------------------------------
# 4. cmd_remove_horizon_pick: delete → undo (restore) → redo (delete again)
# ---------------------------------------------------------------------------

class TestCmdRemoveHorizonPick:

    def test_remove_deletes_pick(self, state):
        hp = _horizon()
        state.add_horizon_pick(hp)
        state.execute_command(cmd_remove_horizon_pick(state, hp))
        assert len(state.project.horizon_picks) == 0

    def test_undo_restores_pick_with_all_points(self, state):
        hp = _horizon(n=5, name="TopSand")
        state.add_horizon_pick(hp)
        state.execute_command(cmd_remove_horizon_pick(state, hp))
        state.undo()
        assert len(state.project.horizon_picks) == 1
        restored = state.project.horizon_picks[0]
        assert restored.name == "TopSand"
        assert restored.n_picks == 5

    def test_undo_preserves_depth_values(self, state):
        hp = _horizon(n=3)
        original_depths = hp._depths.copy()
        state.add_horizon_pick(hp)
        state.execute_command(cmd_remove_horizon_pick(state, hp))
        state.undo()
        np.testing.assert_array_almost_equal(
            state.project.horizon_picks[0]._depths, original_depths
        )

    def test_redo_removes_restored_pick(self, state):
        hp = _horizon()
        state.add_horizon_pick(hp)
        state.execute_command(cmd_remove_horizon_pick(state, hp))
        state.undo()
        state.redo()
        assert len(state.project.horizon_picks) == 0

    def test_multiple_undo_redo_cycles(self, state):
        hp = _horizon()
        state.add_horizon_pick(hp)
        state.execute_command(cmd_remove_horizon_pick(state, hp))
        for _ in range(3):
            state.undo()
            assert len(state.project.horizon_picks) == 1
            state.redo()
            assert len(state.project.horizon_picks) == 0


# ---------------------------------------------------------------------------
# 5. cmd_update_horizon_pick: move pick point
# ---------------------------------------------------------------------------

class TestCmdUpdateHorizonPick:

    def test_update_changes_pick(self, state):
        hp = _horizon(n=3)
        state.add_horizon_pick(hp)
        hp_new = copy.deepcopy(hp)
        hp_new.move_pick(1, 4000.0, 1500.0)
        state.execute_command(cmd_update_horizon_pick(state, 0, hp_new))
        current = state.project.horizon_picks[0]
        assert any(abs(d - 4000.0) < 0.1 for d in current._distances)

    def test_undo_restores_original_positions(self, state):
        hp = _horizon(n=3)
        state.add_horizon_pick(hp)
        original_distances = hp._distances.copy()
        hp_new = copy.deepcopy(hp)
        hp_new.move_pick(1, 4000.0, 1500.0)
        state.execute_command(cmd_update_horizon_pick(state, 0, hp_new))
        state.undo()
        np.testing.assert_array_almost_equal(
            state.project.horizon_picks[0]._distances, original_distances
        )

    def test_redo_re_applies_move(self, state):
        hp = _horizon(n=3)
        state.add_horizon_pick(hp)
        hp_new = copy.deepcopy(hp)
        hp_new.move_pick(1, 4000.0, 1500.0)
        state.execute_command(cmd_update_horizon_pick(state, 0, hp_new))
        state.undo()
        state.redo()
        current = state.project.horizon_picks[0]
        assert any(abs(d - 4000.0) < 0.1 for d in current._distances)

    def test_captures_old_state_at_factory_time(self, state):
        """old_pick should be captured at factory creation, not execution."""
        hp = _horizon(n=3)
        state.add_horizon_pick(hp)
        hp_new = copy.deepcopy(hp)
        hp_new.move_pick(0, 500.0, 1100.0)
        original_d0 = hp._distances[0]
        cmd = cmd_update_horizon_pick(state, 0, hp_new)
        # Mutate hp AFTER factory creation (should not affect undo)
        hp._depths[0] = 9999.0
        state.execute_command(cmd)
        state.undo()
        # Original distances (not the mutated version) should be restored
        assert abs(state.project.horizon_picks[0]._distances[0] - original_d0) < 0.1

    def test_insert_point_via_update(self, state):
        """Using update_horizon_pick to cover insert-point operation."""
        hp = _horizon(n=2)
        state.add_horizon_pick(hp)
        hp_after = copy.deepcopy(hp)
        hp_after.insert_pick(3333.0, 1100.0, "S1")
        state.execute_command(cmd_update_horizon_pick(state, 0, hp_after))
        assert state.project.horizon_picks[0].n_picks == 3
        state.undo()
        assert state.project.horizon_picks[0].n_picks == 2
        state.redo()
        assert state.project.horizon_picks[0].n_picks == 3

    def test_delete_point_via_update(self, state):
        """Using update_horizon_pick to cover delete-point operation."""
        hp = _horizon(n=4)
        state.add_horizon_pick(hp)
        hp_after = copy.deepcopy(hp)
        hp_after.delete_pick(2)
        state.execute_command(cmd_update_horizon_pick(state, 0, hp_after))
        assert state.project.horizon_picks[0].n_picks == 3
        state.undo()
        assert state.project.horizon_picks[0].n_picks == 4


# ---------------------------------------------------------------------------
# 6. cmd_add_fault_pick / cmd_remove_fault_pick
# ---------------------------------------------------------------------------

class TestFaultPickCommands:

    def test_add_fault_pick_undo_redo(self, state):
        fp = _horizon(name="F1")
        state.execute_command(cmd_add_fault_pick(state, fp))
        assert len(state.project.fault_picks) == 1
        state.undo()
        assert len(state.project.fault_picks) == 0
        state.redo()
        assert len(state.project.fault_picks) == 1

    def test_remove_fault_pick_undo_restores(self, state):
        fp = _horizon(name="MainFault", n=4)
        state.add_fault_pick(fp)
        state.execute_command(cmd_remove_fault_pick(state, fp))
        assert len(state.project.fault_picks) == 0
        state.undo()
        assert state.project.fault_picks[0].name == "MainFault"
        assert state.project.fault_picks[0].n_picks == 4

    def test_update_fault_pick_undo(self, state):
        fp = _horizon(name="F1", n=3)
        state.add_fault_pick(fp)
        fp_new = copy.deepcopy(fp)
        fp_new.move_pick(0, 100.0, 200.0)
        original_d0 = fp._distances[0]
        state.execute_command(cmd_update_fault_pick(state, 0, fp_new))
        state.undo()
        assert abs(state.project.fault_picks[0]._distances[0] - original_d0) < 0.1


# ---------------------------------------------------------------------------
# 7. cmd_add_section / cmd_remove_section / cmd_update_section
# ---------------------------------------------------------------------------

class TestSectionCommands:

    def test_add_section_undo_redo(self, state):
        sec = _section("EW")
        state.execute_command(cmd_add_section(state, sec))
        assert len(state.project.sections) == 1
        state.undo()
        assert len(state.project.sections) == 0
        state.redo()
        assert len(state.project.sections) == 1
        assert state.project.sections[0].name == "EW"

    def test_remove_section_undo_restores(self, state):
        sec = _section("NS")
        state.add_section(sec)
        state.execute_command(cmd_remove_section(state, sec))
        assert len(state.project.sections) == 0
        state.undo()
        assert state.project.sections[0].name == "NS"

    def test_update_section_undo(self, state):
        sec = _section("EW")
        state.add_section(sec)
        sec_new = Section([(0.0, 0.0), (5_000.0, 0.0)], name="EW")  # shorter
        state.execute_command(cmd_update_section(state, 0, sec_new))
        assert abs(state.project.sections[0].total_length() - 5_000.0) < 0.1
        state.undo()
        assert abs(state.project.sections[0].total_length() - 10_000.0) < 0.1


# ---------------------------------------------------------------------------
# 8. cmd_add_well / cmd_remove_well
# ---------------------------------------------------------------------------

class TestWellCommands:

    def test_add_well_undo_redo(self, state):
        w = _well("F02-01")
        state.execute_command(cmd_add_well(state, w))
        assert len(state.project.wells) == 1
        state.undo()
        assert len(state.project.wells) == 0
        state.redo()
        assert state.project.wells[0].name == "F02-01"

    def test_remove_well_undo_restores_with_tops(self, state):
        w = _well("F03-01")
        w.add_formation_top("TopSand", 980.0)
        w.add_formation_top("BaseSand", 1120.0)
        state.add_well(w)
        state.execute_command(cmd_remove_well(state, w))
        assert len(state.project.wells) == 0
        state.undo()
        assert len(state.project.wells) == 1
        restored = state.project.wells[0]
        assert restored.name == "F03-01"
        assert "TopSand" in restored.formation_tops
        assert restored.formation_tops["TopSand"] == pytest.approx(980.0)


# ---------------------------------------------------------------------------
# 9. Multiple undo/redo sequences
# ---------------------------------------------------------------------------

class TestMultipleUndoRedo:

    def test_sequential_operations_undo_in_lifo_order(self, state):
        h1 = _horizon(name="H1")
        h2 = _horizon(name="H2")
        h3 = _horizon(name="H3")
        state.execute_command(cmd_add_horizon_pick(state, h1))
        state.execute_command(cmd_add_horizon_pick(state, h2))
        state.execute_command(cmd_add_horizon_pick(state, h3))
        assert len(state.project.horizon_picks) == 3

        # Undo removes in LIFO order: H3, H2, H1
        desc3 = state.undo()
        assert "H3" in desc3
        assert len(state.project.horizon_picks) == 2
        desc2 = state.undo()
        assert "H2" in desc2
        desc1 = state.undo()
        assert "H1" in desc1
        assert len(state.project.horizon_picks) == 0

    def test_redo_after_partial_undo(self, state):
        h1 = _horizon(name="H1")
        h2 = _horizon(name="H2")
        state.execute_command(cmd_add_horizon_pick(state, h1))
        state.execute_command(cmd_add_horizon_pick(state, h2))
        state.undo()   # removes H2
        state.undo()   # removes H1
        state.redo()   # re-adds H1
        state.redo()   # re-adds H2
        names = [p.name for p in state.project.horizon_picks]
        assert "H1" in names
        assert "H2" in names

    def test_new_action_after_undo_clears_redo(self, state):
        h1 = _horizon(name="H1")
        h2 = _horizon(name="H2")
        h3 = _horizon(name="H3")
        state.execute_command(cmd_add_horizon_pick(state, h1))
        state.execute_command(cmd_add_horizon_pick(state, h2))
        state.undo()   # H2 on redo stack
        assert state._stack.can_redo()
        state.execute_command(cmd_add_horizon_pick(state, h3))
        assert not state._stack.can_redo()
        names = [p.name for p in state.project.horizon_picks]
        assert "H1" in names
        assert "H3" in names
        assert "H2" not in names

    def test_interleaved_add_and_remove(self, state):
        hp = _horizon(name="TopSand")
        state.execute_command(cmd_add_horizon_pick(state, hp))
        state.execute_command(
            cmd_remove_horizon_pick(state, state.project.horizon_picks[0])
        )
        assert len(state.project.horizon_picks) == 0
        state.undo()   # restore TopSand
        assert len(state.project.horizon_picks) == 1
        state.undo()   # undo the add → empty
        assert len(state.project.horizon_picks) == 0


# ---------------------------------------------------------------------------
# 10. can_undo / can_redo state tracking
# ---------------------------------------------------------------------------

class TestCanUndoRedo:

    def test_can_undo_false_on_empty_stack(self, state):
        assert not state._stack.can_undo()

    def test_can_undo_true_after_execute(self, state):
        state.execute_command(cmd_add_horizon_pick(state, _horizon()))
        assert state._stack.can_undo()

    def test_can_undo_false_after_full_undo(self, state):
        state.execute_command(cmd_add_horizon_pick(state, _horizon()))
        state.undo()
        assert not state._stack.can_undo()

    def test_can_redo_false_initially(self, state):
        assert not state._stack.can_redo()

    def test_can_redo_true_after_undo(self, state):
        state.execute_command(cmd_add_horizon_pick(state, _horizon()))
        state.undo()
        assert state._stack.can_redo()

    def test_can_redo_false_after_new_action(self, state):
        h1 = _horizon(name="H1")
        h2 = _horizon(name="H2")
        state.execute_command(cmd_add_horizon_pick(state, h1))
        state.undo()
        state.execute_command(cmd_add_horizon_pick(state, h2))
        assert not state._stack.can_redo()

    def test_can_redo_false_after_full_redo(self, state):
        state.execute_command(cmd_add_horizon_pick(state, _horizon()))
        state.undo()
        state.redo()
        assert not state._stack.can_redo()


# ---------------------------------------------------------------------------
# 11. Stack depth limit (101 commands, oldest dropped)
# ---------------------------------------------------------------------------

class TestDepthLimitIntegration:

    def test_101_commands_keeps_only_100(self, state):
        """The CommandStack default max_depth=100 drops the oldest on overflow."""
        for i in range(101):
            hp = _horizon(name=f"H{i}", n=1)
            state.execute_command(cmd_add_horizon_pick(state, hp))
        assert state._stack.undo_stack_size == 100

    def test_oldest_command_is_dropped(self, state):
        """After 101 commands, undoing 100 times should NOT undo cmd0."""
        for i in range(101):
            hp = _horizon(name=f"H{i}", n=1)
            state.execute_command(cmd_add_horizon_pick(state, hp))
        # There are 101 picks in the list but only 100 in the undo stack
        assert len(state.project.horizon_picks) == 101
        for _ in range(100):
            state.undo()
        # One pick (H0, the oldest) should remain because its command was dropped
        assert len(state.project.horizon_picks) == 1
        assert state.project.horizon_picks[0].name == "H0"

    def test_custom_depth_limit(self):
        stack = CommandStack(max_depth=5)
        for i in range(10):
            stack.execute(Command(f"cmd{i}", do=lambda: None, undo=lambda: None))
        assert stack.undo_stack_size == 5


# ---------------------------------------------------------------------------
# 12. record_command backward compatibility (push pattern)
# ---------------------------------------------------------------------------

class TestRecordCommandCompat:

    def test_push_registers_for_undo(self):
        """push() records without calling do()."""
        stack = CommandStack()
        applied = []
        cmd = Command("op", do=lambda: applied.append("do"),
                      undo=lambda: applied.append("undo"))
        # Simulate "already applied" pattern
        applied.append("pre")
        stack.push(cmd)
        assert applied == ["pre"]     # do() was NOT called
        stack.undo()
        assert applied == ["pre", "undo"]

    def test_redo_after_push_calls_do(self):
        """After push + undo, redo calls do()."""
        calls = []
        stack = CommandStack()
        cmd = Command("op", do=lambda: calls.append("do"),
                      undo=lambda: calls.append("undo"))
        stack.push(cmd)
        stack.undo()
        stack.redo()
        assert calls == ["undo", "do"]


# ---------------------------------------------------------------------------
# 13. Command.data field
# ---------------------------------------------------------------------------

class TestCommandData:

    def test_data_field_is_optional(self):
        cmd = Command("op", do=lambda: None, undo=lambda: None)
        assert cmd.data is None

    def test_data_field_stores_payload(self):
        payload = {"horizon": "TopSand", "section": "EW", "depth": 1200.0}
        cmd = Command("Add pick", do=lambda: None, undo=lambda: None, data=payload)
        assert cmd.data["depth"] == 1200.0
        assert cmd.data["horizon"] == "TopSand"
