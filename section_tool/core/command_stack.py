"""Phase 7 — Undo/redo command stack."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Command:
    description: str
    undo: Callable
    redo: Callable


class CommandStack:
    """LIFO undo/redo stack (max_depth entries).

    Usage pattern — record AFTER the operation has already been applied::

        state.do_something(args)          # mutates AppState
        state.record_command(
            "Add horizon pick",
            undo=lambda: state.undo_something(args),
            redo=lambda: state.do_something(args),
        )
    """

    def __init__(self, max_depth: int = 50) -> None:
        self._undo: list[Command] = []
        self._redo: list[Command] = []
        self._max  = max_depth

    # ------------------------------------------------------------------

    def push(self, command: Command) -> None:
        """Record *command*; clears the redo stack."""
        self._undo.append(command)
        if len(self._undo) > self._max:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self) -> str | None:
        """Undo the last command. Returns its description or None."""
        if not self._undo:
            return None
        cmd = self._undo.pop()
        cmd.undo()
        self._redo.append(cmd)
        return cmd.description

    def redo(self) -> str | None:
        """Re-apply the last undone command. Returns its description or None."""
        if not self._redo:
            return None
        cmd = self._redo.pop()
        cmd.redo()
        self._undo.append(cmd)
        return cmd.description

    # ------------------------------------------------------------------

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_description(self) -> str | None:
        return self._undo[-1].description if self._undo else None

    @property
    def redo_description(self) -> str | None:
        return self._redo[-1].description if self._redo else None

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()
