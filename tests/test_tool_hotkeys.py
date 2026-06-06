"""TOOL_HOTKEYS is the single source of truth: tooltips and shortcut bindings
read the same map, so a tooltip can never drift from its real binding."""
from __future__ import annotations

from PySide6.QtCore import Qt

from section_tool.views.tool_palette import (
    TOOL_HOTKEYS, compose_tooltip, _TOOL_DEFS)
from section_tool.interaction.tool_manager import TOOL_KEYS
from section_tool.app import SectionMainWindow


def test_tooltip_uses_real_hotkey_not_stale():
    # horizon's hardcoded "(P)" is replaced by the real binding "(H)".
    out = compose_tooltip("horizon_pick", "Horizon Pick  (P)\nClick to add picks.")
    assert "(H)" in out
    assert "(P)" not in out
    assert "Click to add picks." in out          # body preserved


def test_tooltip_label_only_when_no_hotkey():
    # pan has no binding → first line carries no parens (no stale "(H)").
    first = compose_tooltip("pan", "Pan  (H)\nLeft-drag to pan.").split("\n")[0]
    assert "(" not in first


def test_every_keyed_tool_tooltip_matches_binding():
    defs = {t[0]: t[3] for t in _TOOL_DEFS if isinstance(t, tuple)}
    for tid, key in TOOL_HOTKEYS.items():
        if tid in defs:
            assert f"({key})" in compose_tooltip(tid, defs[tid]), tid


def test_toolmgr_bindings_reconcile_with_hotkeys():
    """Letters routed through the ToolManager must resolve — via TOOL_KEYS and
    _NEW_TO_OLD — back to the same palette tool_id TOOL_HOTKEYS names. This locks
    the two binding registries to the single source."""
    n2o = SectionMainWindow._NEW_TO_OLD
    for tid in ("select", "node_edit", "horizon_pick",
                "fault_pick", "polygon", "measure"):
        qk = getattr(Qt.Key, f"Key_{TOOL_HOTKEYS[tid]}")
        assert qk in TOOL_KEYS, f"{tid}: Key_{TOOL_HOTKEYS[tid]} missing from TOOL_KEYS"
        assert n2o[TOOL_KEYS[qk]] == tid
