"""Command pattern: Command dataclass, CommandStack, and command factories.

No Qt dependencies — safe to import and test in headless environments.

Two execution patterns are supported:

Execute pattern (preferred for new code)::

    cmd = cmd_add_horizon_pick(state, pick)
    state.execute_command(cmd)   # calls cmd.do() then pushes

Record pattern (backward compat for code that already applied the op)::

    state.add_horizon_pick(pick)           # mutation already done
    state.record_command(description,       # just push for undo/redo
                         undo=lambda: state.remove_horizon_pick(pick))
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

@dataclass
class Command:
    """An undoable/redoable operation.

    ``do``   — callable that applies (or re-applies) the operation.
    ``undo`` — callable that reverses the operation.
    ``data`` — optional payload stored for debugging / logging only.
    """
    description: str
    do: Callable[[], None]
    undo: Callable[[], None]
    data: Any = None


class CommandStack:
    """LIFO undo/redo stack with configurable depth limit.

    Parameters
    ----------
    max_depth:
        Maximum number of commands kept on the undo stack.  Oldest commands
        are dropped when the limit is exceeded.
    """

    def __init__(self, max_depth: int = 100) -> None:
        self._undo_stack: list[Command] = []
        self._redo_stack: list[Command] = []
        self._max_depth = max_depth

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def execute(self, command: Command) -> None:
        """Call ``command.do()``, push onto undo stack, clear redo stack."""
        command.do()
        self._push(command)

    def push(self, command: Command) -> None:
        """Push *command* without calling ``do()`` (already-applied pattern)."""
        self._push(command)

    def _push(self, command: Command) -> None:
        self._undo_stack.append(command)
        if len(self._undo_stack) > self._max_depth:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self) -> Optional[str]:
        """Undo the last command. Returns description or None if stack empty."""
        if not self._undo_stack:
            return None
        cmd = self._undo_stack.pop()
        cmd.undo()
        self._redo_stack.append(cmd)
        return cmd.description

    def redo(self) -> Optional[str]:
        """Re-apply the last undone command. Returns description or None."""
        if not self._redo_stack:
            return None
        cmd = self._redo_stack.pop()
        cmd.do()
        self._undo_stack.append(cmd)
        return cmd.description

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    @property
    def undo_description(self) -> Optional[str]:
        return self._undo_stack[-1].description if self._undo_stack else None

    @property
    def redo_description(self) -> Optional[str]:
        return self._redo_stack[-1].description if self._redo_stack else None

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def undo_stack_size(self) -> int:
        return len(self._undo_stack)

    @property
    def redo_stack_size(self) -> int:
        return len(self._redo_stack)


# ---------------------------------------------------------------------------
# Command factories — horizon picks
# ---------------------------------------------------------------------------

def cmd_add_horizon_pick(state, pick) -> Command:
    """Add a new HorizonPick to the project.

    On undo, removes the pick by searching for the most recently added instance
    with the same name (identity-based removal can break when interleaved
    remove/undo cycles substitute a different-identity copy into the list).
    """
    saved = copy.deepcopy(pick)
    live: list = [None]   # mutable cell: tracks the currently-live instance

    def do() -> None:
        added = copy.deepcopy(saved)
        state.add_horizon_pick(added)
        live[0] = added

    def undo() -> None:
        # Prefer identity match; fall back to name search for robustness.
        target = live[0]
        picks = state.project.horizon_picks
        if target is not None and target in picks:
            state.remove_horizon_pick(target)
        else:
            for p in reversed(picks):
                if p.name == saved.name:
                    state.remove_horizon_pick(p)
                    break
        live[0] = None

    return Command(f"Add horizon {saved.name}", do, undo)


def cmd_remove_horizon_pick(state, pick) -> Command:
    """Remove a HorizonPick from the project (fully restorable via undo).

    On undo, a deep copy of the original pick is added back.  On redo the
    most-recently-restored copy is removed.
    """
    saved = copy.deepcopy(pick)
    live: list = [pick]   # starts as the original; updated by undo

    def do() -> None:
        state.remove_horizon_pick(live[0])

    def undo() -> None:
        restored = copy.deepcopy(saved)
        state.add_horizon_pick(restored)
        live[0] = restored   # redo will remove this exact copy

    return Command(f"Delete horizon {saved.name}", do, undo)


def cmd_update_horizon_pick(state, idx: int, new_pick, old_pick=None) -> Command:
    """Replace the horizon pick at *idx* (before/after snapshot pattern).

    Covers all point-level edits: insert pick point, move pick point, delete
    pick point.  The caller constructs the desired new state of the pick and
    passes it in; this factory records both states for bidirectional undo/redo.

    Parameters
    ----------
    idx:
        Index in ``state.project.horizon_picks`` to update.
    new_pick:
        Desired new state of the pick (deep-copied inside).
    old_pick:
        Previous state (deep-copied inside).  If *None*, captured from the
        current project at factory-creation time.
    """
    if old_pick is None:
        old_pick = copy.deepcopy(state.project.horizon_picks[idx])
    saved_old = copy.deepcopy(old_pick)
    saved_new = copy.deepcopy(new_pick)

    def do() -> None:
        state.update_horizon_pick(idx, copy.deepcopy(saved_new))

    def undo() -> None:
        state.update_horizon_pick(idx, copy.deepcopy(saved_old))

    return Command(f"Update {saved_new.name}", do, undo)


# ---------------------------------------------------------------------------
# Command factories — fault picks
# ---------------------------------------------------------------------------

def cmd_add_fault_pick(state, pick) -> Command:
    """Add a new fault pick (HorizonPick) to the project."""
    saved = copy.deepcopy(pick)
    live: list = [None]

    def do() -> None:
        added = copy.deepcopy(saved)
        state.add_fault_pick(added)
        live[0] = added

    def undo() -> None:
        target = live[0]
        picks = state.project.fault_picks
        if target is not None and target in picks:
            state.remove_fault_pick(target)
        else:
            for p in reversed(picks):
                if p.name == saved.name:
                    state.remove_fault_pick(p)
                    break
        live[0] = None

    return Command(f"Add fault {saved.name}", do, undo)


def cmd_remove_fault_pick(state, pick) -> Command:
    """Remove a fault pick from the project (restorable via undo)."""
    saved = copy.deepcopy(pick)
    live: list = [pick]

    def do() -> None:
        state.remove_fault_pick(live[0])

    def undo() -> None:
        restored = copy.deepcopy(saved)
        state.add_fault_pick(restored)
        live[0] = restored

    return Command(f"Delete fault {saved.name}", do, undo)


def cmd_update_fault_pick(state, idx: int, new_pick, old_pick=None) -> Command:
    """Replace the fault pick at *idx* (before/after snapshot pattern)."""
    if old_pick is None:
        old_pick = copy.deepcopy(state.project.fault_picks[idx])
    saved_old = copy.deepcopy(old_pick)
    saved_new = copy.deepcopy(new_pick)

    def do() -> None:
        state.update_fault_pick(idx, copy.deepcopy(saved_new))

    def undo() -> None:
        state.update_fault_pick(idx, copy.deepcopy(saved_old))

    return Command(f"Update fault {saved_new.name}", do, undo)


# ---------------------------------------------------------------------------
# Command factories — sections
# ---------------------------------------------------------------------------

def cmd_add_section(state, section) -> Command:
    """Add a Section to the project."""
    saved = copy.deepcopy(section)
    live: list = [None]

    def do() -> None:
        added = copy.deepcopy(saved)
        state.add_section(added)
        live[0] = added

    def undo() -> None:
        target = live[0]
        sections = state.project.sections
        if target is not None and target in sections:
            state.remove_section(target)
        else:
            for s in reversed(sections):
                if s.name == saved.name:
                    state.remove_section(s)
                    break
        live[0] = None

    return Command(f"Add section {saved.name}", do, undo)


def cmd_remove_section(state, section) -> Command:
    """Remove a Section (restorable via undo)."""
    saved = copy.deepcopy(section)
    live: list = [section]

    def do() -> None:
        state.remove_section(live[0])

    def undo() -> None:
        restored = copy.deepcopy(saved)
        state.add_section(restored)
        live[0] = restored

    return Command(f"Delete section {saved.name}", do, undo)


def cmd_update_section(state, idx: int, new_section, old_section=None) -> Command:
    """Replace the section at *idx* (before/after snapshot)."""
    if old_section is None:
        old_section = copy.deepcopy(state.project.sections[idx])
    saved_old = copy.deepcopy(old_section)
    saved_new = copy.deepcopy(new_section)

    def do() -> None:
        state.update_section(idx, copy.deepcopy(saved_new))

    def undo() -> None:
        state.update_section(idx, copy.deepcopy(saved_old))

    return Command(f"Update section {saved_new.name}", do, undo)


# ---------------------------------------------------------------------------
# Command factories — wells
# ---------------------------------------------------------------------------

def cmd_add_well(state, well) -> Command:
    """Add a Well to the project."""
    saved = copy.deepcopy(well)
    live: list = [None]

    def do() -> None:
        added = copy.deepcopy(saved)
        state.add_well(added)
        live[0] = added

    def undo() -> None:
        target = live[0]
        wells = state.project.wells
        if target is not None and target in wells:
            state.remove_well(target)
        else:
            for w in reversed(wells):
                if w.name == saved.name:
                    state.remove_well(w)
                    break
        live[0] = None

    return Command(f"Add well {saved.name}", do, undo)


def cmd_remove_well(state, well) -> Command:
    """Remove a Well (restorable via undo)."""
    saved = copy.deepcopy(well)
    live: list = [well]

    def do() -> None:
        state.remove_well(live[0])

    def undo() -> None:
        restored = copy.deepcopy(saved)
        state.add_well(restored)
        live[0] = restored

    return Command(f"Delete well {saved.name}", do, undo)
