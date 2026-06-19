"""Thermal Step 2: burial history from the restoration sequence (the seam),
decompaction between steps, manual fallback, and provenance labelling."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from section_tool.core.burial import (
    burial_history_from_restoration, manual_burial_history)
from section_tool.core.surfaces import HorizonPick
from section_tool.core.restoration import RestorationEvent, RestorationSequence


def _flat_horizon(name, depth):
    return HorizonPick([0.0, 1000.0], [depth, depth], name=name,
                       section_names=["L1", "L1"])


def _snapshot(horizons):
    return SimpleNamespace(horizons=list(horizons), faults=[], polygons=[],
                           section={"name": "L1"})


def _setup():
    h0 = _flat_horizon("Top", 500.0)        # youngest / shallowest
    h1 = _flat_horizon("Mid", 1000.0)
    h2 = _flat_horizon("Base", 1500.0)      # oldest / deepest — the tracked horizon
    snap = _snapshot([h2, h0, h1])          # unsorted on purpose
    seq = RestorationSequence(name="Test") if hasattr(RestorationSequence, "name") \
        else RestorationSequence()
    seq.add_event(RestorationEvent(1, "remove Top", age_ma=10.0,
                                   remove_element_ids=[h0.uuid]))   # youngest
    seq.add_event(RestorationEvent(2, "remove Mid", age_ma=20.0,
                                   remove_element_ids=[h1.uuid]))   # older
    return seq, h0, h1, h2, snap


def test_burial_pairs_oldest_first_with_event_ages():
    seq, h0, h1, h2, snap = _setup()
    bh = burial_history_from_restoration(seq, h2.uuid, 500.0, snapshot=snap)
    ages = [a for a, _ in bh.points]
    assert ages == [20.0, 10.0, 0.0]                     # oldest first, present last
    depths = [d for _, d in bh.points]
    assert depths[2] == pytest.approx(1500.0)            # present-day burial depth
    assert depths[0] < depths[1] < depths[2]             # older = shallower (overburden stripped)


def test_decompaction_thickens_stripped_column():
    """At the oldest step only the deepest layer survives; decompacting it back to
    the surface makes it THICKER than its present compacted thickness (500 m)."""
    seq, h0, h1, h2, snap = _setup()
    bh = burial_history_from_restoration(seq, h2.uuid, 500.0, snapshot=snap)
    oldest_depth = bh.points[0][1]                       # age 20, only "Base" layer left
    assert oldest_depth > 500.0                          # decompaction thickened it
    assert oldest_depth < 1500.0


def test_unknown_horizon_returns_empty():
    seq, h0, h1, h2, snap = _setup()
    bh = burial_history_from_restoration(seq, "no-such-uuid", 500.0, snapshot=snap)
    assert bh.points == []


def test_provenance_label_restoration_vs_manual():
    seq, h0, h1, h2, snap = _setup()
    bh = burial_history_from_restoration(seq, h2.uuid, 500.0, snapshot=snap)
    assert bh.source.startswith("restoration sequence")
    man = manual_burial_history([(30.0, 800.0), (0.0, 1500.0)])
    assert man.source == "user-specified"


def test_manual_sorts_oldest_first_and_drops_bad_rows():
    man = manual_burial_history([(0.0, 1500.0), (30.0, 800.0),
                                 (10.0, -5.0), ("x", 1.0)])
    assert man.points == [(30.0, 800.0), (0.0, 1500.0)]  # sorted, bad rows dropped


def test_hardcoded_burial_proxy_is_gone():
    """The fabricated burial (depths × [0.5, 0.75, 1.0]) must no longer exist."""
    import pathlib
    dlg = (pathlib.Path(__file__).resolve().parents[1]
           / "section_tool" / "views" / "thermal_modeling_dialog.py")
    text = dlg.read_text(encoding="utf-8")
    assert "depths * 0.5" not in text and "depths * 0.75" not in text
