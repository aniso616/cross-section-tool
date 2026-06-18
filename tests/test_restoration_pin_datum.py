"""Step 5b Part B: ReferenceLine restoration role + UUID, RestorationEvent
pin/datum line references, and the engine resolving them at deformation time."""
from __future__ import annotations

import numpy as np
import pytest

from section_tool.core.reference_line import ReferenceLine
from section_tool.core import kinematics as K
from section_tool.core.restoration import RestorationEvent, RestorationSequence
from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick
from section_tool.core.restoration_snapshot import snapshot_interpretation


def test_reference_line_uuid_and_default_role():
    rl = ReferenceLine("vertical", value=400.0, name="P1")
    assert isinstance(rl.uuid, str) and len(rl.uuid) >= 32
    assert rl.restoration_role is None                         # backward-compat default
    rl2 = ReferenceLine("horizontal", value=0.0, restoration_role="datum", uuid="fix")
    assert rl2.uuid == "fix" and rl2.restoration_role == "datum"
    assert ReferenceLine("vertical").uuid != ReferenceLine("vertical").uuid  # distinct


def test_event_pin_datum_line_round_trip_and_legacy():
    seq = RestorationSequence(events=[RestorationEvent(
        1, "e", algorithm="flexural_slip", pin_line_id="pin-1", datum_line_id="dat-1")])
    seq2 = RestorationSequence.from_json(seq.to_json())
    assert seq2.events[0].pin_line_id == "pin-1"
    assert seq2.events[0].datum_line_id == "dat-1"
    # legacy events (no fields) load as None — backward compatible
    legacy = RestorationSequence.from_json('{"events":[{"event_id":1,"name":"x"}]}')
    assert legacy.events[0].pin_line_id is None and legacy.events[0].datum_line_id is None


def test_resolve_params_line_overrides_numeric_else_fallback():
    pin = ReferenceLine("vertical", value=250.0, restoration_role="pin")
    datum = ReferenceLine("horizontal", value=0.0, restoration_role="datum")
    ev = RestorationEvent(1, "e", algorithm="flexural_slip",
                          params={"pin_x": 999.0, "datum_y": 999.0},
                          pin_line_id=pin.uuid, datum_line_id=datum.uuid)
    p = K.resolve_event_params(ev, [pin, datum])
    assert p["pin_x"] == 250.0 and p["datum_y"] == 0.0        # line overrides numeric

    ev2 = RestorationEvent(1, "e", params={"pin_x": 123.0, "datum_y": 5.0})
    p2 = K.resolve_event_params(ev2, [pin, datum])
    assert p2["pin_x"] == 123.0 and p2["datum_y"] == 5.0      # numeric fallback intact


def test_uuid_invariance_after_rename():
    pin = ReferenceLine("vertical", value=300.0, name="Pin A", restoration_role="pin")
    ev = RestorationEvent(1, "e", algorithm="flexural_slip", pin_line_id=pin.uuid)
    pin.name = "Renamed Pin"                                  # rename
    assert K.resolve_event_params(ev, [pin])["pin_x"] == 300.0   # still resolves by uuid


def test_engine_resolves_pin_line_live_at_deform_time():
    state = AppState()
    state.add_section(Section([(0.0, 0.0), (1000.0, 0.0)], name="L1", crs_epsg=32631))
    state.set_active_section(state.project.sections[0])
    state.project.horizon_picks.append(
        HorizonPick([0.0, 500.0, 1000.0], [100.0, 150.0, 200.0], name="Top",
                    section_names=["L1", "L1", "L1"]))
    pin = ReferenceLine("vertical", value=0.0, name="Pin", restoration_role="pin")
    datum = ReferenceLine("horizontal", value=0.0, name="Datum", restoration_role="datum")
    state.project.reference_lines.extend([pin, datum])
    snap = snapshot_interpretation(state.active_section, state.project)

    ev = RestorationEvent(1, "e", algorithm="flexural_slip",
                          pin_line_id=pin.uuid, datum_line_id=datum.uuid)
    out = K.restore_snapshot(snap, ev, section_name="L1",
                             reference_lines=state.project.reference_lines)
    d, z = out.horizons[0].picks_for_section("L1")
    assert np.allclose(z, 0.0)                                # unfolded to datum y=0
    assert d[0] == pytest.approx(0.0)                         # pinned at x=0

    pin.value = 500.0                                         # move the pin live
    out2 = K.restore_snapshot(snap, ev, section_name="L1",
                              reference_lines=state.project.reference_lines)
    d2, _ = out2.horizons[0].picks_for_section("L1")
    assert d2[1] == pytest.approx(500.0)                      # pin node now at x=500
