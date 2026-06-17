"""Tests for construction rule serialisation round-trips and restoration sequence.

All headless — no Qt, no GUI.
"""
from __future__ import annotations

import json

import pytest

from section_tool.core.construction import (
    RULE_REGISTRY,
    DipConstrainedRule,
    FreehandRule,
    KinkBandRule,
    ListricFaultRule,
    MirrorAcrossAxialTraceRule,
    ParallelToBedRule,
    deserialize_rule,
    serialize_rule,
)
from section_tool.core.restoration import (
    RestorationEvent,
    RestorationSequence,
    restore_remove_layer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pick(name: str):
    """Minimal HorizonPick stub (avoids importing numpy/Qt-heavy module)."""
    class _Stub:
        pass
    s = _Stub()
    s.name = name
    s.uuid = name          # use the name as a stable id for these stubs
    s.construction_rule = None
    return s


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

def test_rule_registry_complete():
    expected = {
        "freehand", "parallel_to_bed", "dip_constrained",
        "kink_band", "listric_fault", "mirror_axial_trace",
    }
    assert set(RULE_REGISTRY.keys()) == expected


# ---------------------------------------------------------------------------
# serialize_rule
# ---------------------------------------------------------------------------

def test_serialize_none_returns_none():
    assert serialize_rule(None) is None


def test_serialize_freehand():
    rule = FreehandRule()
    js = serialize_rule(rule)
    d = json.loads(js)
    assert d["kind"] == "freehand"


def test_serialize_parallel_to_bed():
    rule = ParallelToBedRule(reference_name="TopChallenges", offset_m=50.0)
    js = serialize_rule(rule)
    d = json.loads(js)
    assert d["kind"] == "parallel_to_bed"
    assert d["reference_name"] == "TopChallenges"
    assert d["offset_m"] == pytest.approx(50.0)


def test_serialize_dip_constrained():
    rule = DipConstrainedRule(dip_deg=30.0, dip_direction_deg=270.0, measurement_source="W-1")
    js = serialize_rule(rule)
    d = json.loads(js)
    assert d["kind"] == "dip_constrained"
    assert d["dip_deg"] == pytest.approx(30.0)
    assert d["measurement_source"] == "W-1"


def test_serialize_kink_band():
    rule = KinkBandRule(axial_surface_dip_deg=60.0, fore_dip_deg=45.0, back_dip_deg=5.0)
    js = serialize_rule(rule)
    d = json.loads(js)
    assert d["kind"] == "kink_band"
    assert d["axial_surface_dip_deg"] == pytest.approx(60.0)


def test_serialize_listric_fault():
    rule = ListricFaultRule(detachment_depth_m=8000.0, ramp_dip_deg=25.0)
    js = serialize_rule(rule)
    d = json.loads(js)
    assert d["kind"] == "listric_fault"
    assert d["detachment_depth_m"] == pytest.approx(8000.0)


def test_serialize_mirror_axial_trace():
    rule = MirrorAcrossAxialTraceRule(axial_trace_name="Axis_1", mirror_side="left")
    js = serialize_rule(rule)
    d = json.loads(js)
    assert d["kind"] == "mirror_axial_trace"
    assert d["mirror_side"] == "left"


# ---------------------------------------------------------------------------
# deserialize_rule round-trips
# ---------------------------------------------------------------------------

_ALL_RULES = [
    FreehandRule(),
    ParallelToBedRule(reference_name="Ref", offset_m=10.0),
    DipConstrainedRule(dip_deg=15.0, dip_direction_deg=90.0),
    KinkBandRule(axial_surface_dip_deg=55.0),
    ListricFaultRule(detachment_depth_m=5000.0),
    MirrorAcrossAxialTraceRule(axial_trace_name="fold_axis"),
]


@pytest.mark.parametrize("rule", _ALL_RULES, ids=lambda r: r.kind)
def test_round_trip(rule):
    """Serialise then deserialise; kind must be preserved."""
    js = serialize_rule(rule)
    result = deserialize_rule(js)
    assert result.kind == rule.kind


def test_deserialize_none_returns_none():
    assert deserialize_rule(None) is None


def test_deserialize_empty_string_returns_none():
    assert deserialize_rule("") is None


def test_deserialize_unknown_kind_raises():
    js = json.dumps({"kind": "invented_rule"})
    with pytest.raises(ValueError, match="Unknown construction rule kind"):
        deserialize_rule(js)


def test_round_trip_parallel_preserves_fields():
    rule = ParallelToBedRule(reference_name="TopJurassic", offset_m=-25.0)
    result = deserialize_rule(serialize_rule(rule))
    assert isinstance(result, ParallelToBedRule)
    assert result.reference_name == "TopJurassic"
    assert result.offset_m == pytest.approx(-25.0)


def test_round_trip_listric_preserves_fields():
    rule = ListricFaultRule(detachment_depth_m=12000.0, ramp_dip_deg=40.0,
                            hangingwall_reference="HW-1")
    result = deserialize_rule(serialize_rule(rule))
    assert isinstance(result, ListricFaultRule)
    assert result.detachment_depth_m == pytest.approx(12000.0)
    assert result.hangingwall_reference == "HW-1"


# ---------------------------------------------------------------------------
# RestorationEvent round-trip
# ---------------------------------------------------------------------------

def test_restoration_event_round_trip():
    ev = RestorationEvent(
        event_id=1,
        name="Remove Oligocene",
        age_ma=34.0,
        description="Erase post-Eocene package",
        remove_element_ids=["uuid-top", "uuid-base"],
        decompact_params={"Shale": {"phi0": 0.63, "c": 0.00051}},
    )
    d = ev.to_dict()
    ev2 = RestorationEvent.from_dict(d)
    assert ev2.event_id == 1
    assert ev2.name == "Remove Oligocene"
    assert ev2.age_ma == pytest.approx(34.0)
    assert ev2.remove_element_ids == ["uuid-top", "uuid-base"]
    assert ev2.decompact_params["Shale"]["phi0"] == pytest.approx(0.63)


def test_restoration_event_none_age():
    ev = RestorationEvent(event_id=5, name="Unknown age", age_ma=None)
    ev2 = RestorationEvent.from_dict(ev.to_dict())
    assert ev2.age_ma is None


# ---------------------------------------------------------------------------
# RestorationSequence
# ---------------------------------------------------------------------------

def _make_seq() -> RestorationSequence:
    seq = RestorationSequence()
    seq.add_event(RestorationEvent(1, "Step A", age_ma=30.0))
    seq.add_event(RestorationEvent(2, "Step B", age_ma=50.0))
    seq.add_event(RestorationEvent(3, "Step C", age_ma=65.0))
    return seq


def test_sequence_len():
    seq = _make_seq()
    assert len(seq) == 3


def test_sequence_event_by_id():
    seq = _make_seq()
    ev = seq.event_by_id(2)
    assert ev is not None
    assert ev.name == "Step B"


def test_sequence_event_by_id_missing():
    seq = _make_seq()
    assert seq.event_by_id(99) is None


def test_sequence_add_duplicate_raises():
    seq = _make_seq()
    with pytest.raises(ValueError, match="Duplicate event_id"):
        seq.add_event(RestorationEvent(1, "Duplicate"))


def test_sequence_remove_event():
    seq = _make_seq()
    removed = seq.remove_event(2)
    assert removed is True
    assert len(seq) == 2
    assert seq.event_by_id(2) is None


def test_sequence_remove_missing_returns_false():
    seq = _make_seq()
    assert seq.remove_event(999) is False


def test_sequence_move_up():
    seq = _make_seq()
    seq.move_event_up(2)
    assert seq.events[0].event_id == 2
    assert seq.events[1].event_id == 1


def test_sequence_move_down():
    seq = _make_seq()
    seq.move_event_down(2)
    assert seq.events[1].event_id == 3
    assert seq.events[2].event_id == 2


def test_sequence_json_round_trip():
    seq = _make_seq()
    seq.current_step = 2
    js = seq.to_json()
    seq2 = RestorationSequence.from_json(js)
    assert len(seq2) == 3
    assert seq2.current_step == 2
    assert seq2.events[1].name == "Step B"


def test_sequence_empty_json_round_trip():
    seq = RestorationSequence()
    seq2 = RestorationSequence.from_json(seq.to_json())
    assert len(seq2) == 0
    assert seq2.current_step == 0


def test_elements_visible_step_0():
    seq = _make_seq()
    seq.events[0].remove_element_ids = ["TopA"]
    seq.events[1].remove_element_ids = ["TopB"]
    visible = seq.removed_ids_at_step(0)
    assert visible == set()


def test_elements_visible_step_1():
    seq = _make_seq()
    seq.events[0].remove_element_ids = ["TopA"]
    seq.events[1].remove_element_ids = ["TopB"]
    removed = seq.removed_ids_at_step(1)
    assert "TopA" in removed
    assert "TopB" not in removed


def test_elements_visible_step_2():
    seq = _make_seq()
    seq.events[0].remove_element_ids = ["TopA"]
    seq.events[1].remove_element_ids = ["TopB"]
    removed = seq.removed_ids_at_step(2)
    assert "TopA" in removed
    assert "TopB" in removed


# ---------------------------------------------------------------------------
# restore_remove_layer stub
# ---------------------------------------------------------------------------

def test_restore_step_0_unchanged():
    seq = _make_seq()
    h = [_make_pick("H1"), _make_pick("H2")]
    f = [_make_pick("F1")]
    p = [_make_pick("P1")]
    h2, f2, p2 = restore_remove_layer(seq, 0, h, f, p)
    assert h2 is h
    assert f2 is f
    assert p2 is p


def test_restore_step_1_removes_element():
    seq = _make_seq()
    seq.events[0].remove_element_ids = ["H2"]
    h = [_make_pick("H1"), _make_pick("H2")]
    f = [_make_pick("F1")]
    p = []
    h2, f2, p2 = restore_remove_layer(seq, 1, h, f, p)
    assert len(h2) == 1
    assert h2[0].name == "H1"
    assert len(f2) == 1


def test_restore_step_removes_fault():
    seq = _make_seq()
    seq.events[0].remove_element_ids = ["F1"]
    h = [_make_pick("H1")]
    f = [_make_pick("F1"), _make_pick("F2")]
    p = []
    h2, f2, p2 = restore_remove_layer(seq, 1, h, f, p)
    assert len(f2) == 1
    assert f2[0].name == "F2"


def test_restore_empty_sequence_unchanged():
    seq = RestorationSequence()
    h = [_make_pick("H1")]
    h2, f2, p2 = restore_remove_layer(seq, 5, h, [], [])
    assert h2 is h
