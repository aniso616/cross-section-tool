"""Restoration Step 8: end-to-end workflow, provenance, and the docstring sweep."""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from section_tool.core.restoration import RestorationSequence


_RESTORATION_FILES = [
    "views/balance_check_dialog.py",
    "views/restoration_stack_dialog.py",
    "views/restoration_panel.py",
    "core/restoration.py",
    "core/restoration_snapshot.py",
    "core/kinematics.py",
    "core/balance.py",
]


def test_no_tools_arrow_in_restoration_files():
    """The restoration items live under Model now — no stale 'Tools →' vestige."""
    root = pathlib.Path(__file__).resolve().parents[1] / "section_tool"
    for rel in _RESTORATION_FILES:
        text = (root / rel).read_text(encoding="utf-8")
        assert "Tools →" not in text, f"stale 'Tools →' in {rel}"


def test_sequence_persists_full_event_state_round_trip():
    """Save/reopen fidelity at the data layer: algorithm + params + pin/datum line
    refs survive a sequence JSON round-trip (what the project_meta store persists)."""
    from section_tool.core.restoration import RestorationEvent
    seq = RestorationSequence(events=[RestorationEvent(
        1, "Unfold", age_ma=100.0, remove_element_ids=["u1"],
        algorithm="flexural_slip", params={"pin_x": 0.0, "datum_y": 0.0},
        pin_line_id="pin-uuid", datum_line_id="datum-uuid")], current_step=1)
    seq2 = RestorationSequence.from_json(seq.to_json())
    ev = seq2.events[0]
    assert ev.algorithm == "flexural_slip" and ev.params["pin_x"] == 0.0
    assert ev.pin_line_id == "pin-uuid" and ev.datum_line_id == "datum-uuid"
    assert ev.remove_element_ids == ["u1"] and ev.age_ma == 100.0
    assert seq2.current_step == 1
