"""Kinematic restoration engine (Step 6): pure algorithms against hand-calculated
fixtures, anchors-under-restoration, non-destructiveness, and balance integration."""
from __future__ import annotations

import numpy as np
import pytest

from section_tool.core import kinematics as K
from section_tool.core import balance as B
from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick
from section_tool.core.polygons import SectionPolygon
from section_tool.core.restoration import RestorationEvent
from section_tool.core.restoration_snapshot import snapshot_interpretation


# ── pure algorithms ─────────────────────────────────────────────────────────

def test_rigid_translation():
    pts = [(0.0, 0.0), (100.0, 50.0)]
    out = K.rigid_translation(pts, 10.0, -5.0)
    assert np.allclose(out, [(10.0, -5.0), (110.0, 45.0)])


def test_flexural_slip_preserves_arc_length_and_flattens():
    # a sinusoidal fold; unfolding flattens it to the datum, arc length preserved
    x = np.linspace(0.0, 1000.0, 101)
    y = 50.0 * np.sin(x / 100.0)
    fold = np.column_stack([x, y])
    out = K.flexural_slip_unfold(fold, pin_x=0.0, datum_y=0.0)
    assert np.allclose(out[:, 1], 0.0)                          # flat at datum
    assert B.horizon_line_length(out) == pytest.approx(
        B.horizon_line_length(fold), rel=1e-9)                  # length conserved
    assert out[0, 0] == pytest.approx(0.0)                      # pin stays at pin_x


def test_flexural_slip_flat_layer_is_shifted_to_datum():
    flat = [(0.0, 500.0), (1000.0, 500.0)]                      # horizontal at depth 500
    out = K.flexural_slip_unfold(flat, pin_x=0.0, datum_y=0.0)
    assert np.allclose(out, [(0.0, 0.0), (1000.0, 0.0)])        # length 1000 preserved


def test_simple_shear_vertical_and_inclined():
    dip = [(0.0, 0.0), (100.0, 100.0)]
    vert = K.simple_shear(dip, shear_angle_deg=0.0, datum_y=0.0)
    assert np.allclose(vert, [(0.0, 0.0), (100.0, 0.0)])        # x unchanged, y→datum
    inc = K.simple_shear(dip, shear_angle_deg=45.0, datum_y=0.0)
    assert np.allclose(inc, [(0.0, 0.0), (0.0, 0.0)])           # x += (0−y)·tan45 = −y


def test_fault_parallel_flow_planar():
    fault = [(0.0, 0.0), (100.0, 100.0)]                        # 45° planar fault
    pts = [(50.0, 50.0)]
    out = K.fault_parallel_flow(pts, fault, slip=np.hypot(100.0, 100.0))
    assert np.allclose(out, [(150.0, 150.0)])                   # +slip along the fault


def test_apply_algorithm_dispatch_and_unknown():
    assert np.allclose(K.apply_algorithm("rigid_translation", [(0, 0)], {"dx": 5}),
                       [(5.0, 0.0)])
    with pytest.raises(ValueError):
        K.apply_algorithm("nope", [(0, 0)], {})


# ── snapshot-level restoration ──────────────────────────────────────────────

def _anchored_state():
    state = AppState()
    state.add_section(Section([(0.0, 0.0), (1000.0, 0.0)], name="L1", crs_epsg=32631))
    state.set_active_section(state.project.sections[0])
    hp = HorizonPick([0.0, 1000.0], [200.0, 300.0], name="Top",
                     section_names=["L1", "L1"],
                     twt_anchor=[0.20, 0.30], seismic_tied=True)
    state.project.horizon_picks.append(hp)
    poly = SectionPolygon([(0, 0), (100, 0), (100, 50)], name="Block",
                          section_name="L1")
    state.project.polygons.append(poly)
    return state, hp, poly


def _event(algorithm, **params):
    return RestorationEvent(1, "e", algorithm=algorithm, params=params)


def test_restore_marks_frame_and_translates_geometry():
    state, hp, poly = _anchored_state()
    snap = snapshot_interpretation(state.active_section, state.project)
    out = K.restore_snapshot(snap, _event("rigid_translation", dx=100.0, dy=10.0),
                             section_name="L1")
    assert out.restoration_frame is True
    d, z = out.horizons[0].picks_for_section("L1")
    assert np.allclose(d, [100.0, 1100.0]) and np.allclose(z, [210.0, 310.0])
    assert np.allclose(out.polygons[0].vertices, [(100, 10), (200, 10), (200, 60)])


def test_anchors_under_restoration():
    """The Part-A seam regression: restored copy is depth-native (anchors cleared,
    seismic_tied dropped); the ORIGINAL's anchors are untouched. Holds for every
    algorithm."""
    for algo, params in (("rigid_translation", {"dx": 50.0}),
                         ("flexural_slip", {"pin_x": 0.0, "datum_y": 0.0}),
                         ("simple_shear", {"shear_angle": 30.0, "datum_y": 0.0})):
        state, hp, poly = _anchored_state()
        snap = snapshot_interpretation(state.active_section, state.project)
        out = K.restore_snapshot(snap, _event(algo, **params), section_name="L1")
        # original untouched
        assert np.allclose(hp._twt_anchor, [0.20, 0.30]) and hp.seismic_tied is True
        # restored copy is depth-native
        rp = out.horizons[0]
        assert np.all(np.isnan(rp._twt_anchor)) and rp.seismic_tied is False
        assert rp.tie_kind == "depth_native"


def test_restore_is_non_destructive():
    state, hp, poly = _anchored_state()
    orig = hp.depths.copy()
    snap = snapshot_interpretation(state.active_section, state.project)
    out = K.restore_snapshot(snap, _event("rigid_translation", dx=500.0),
                             section_name="L1")
    out.horizons[0]._depths[:] = -9.0                          # deform the copy hard
    assert np.allclose(hp.depths, orig)                        # original byte-unchanged
    # the snapshot the engine received is also untouched
    assert np.allclose(snap.horizons[0].depths, orig)


def test_fault_parallel_flow_through_snapshot():
    state, hp, poly = _anchored_state()
    fault = HorizonPick([0.0, 100.0], [0.0, 100.0], name="F", fault_type="thrust",
                        section_names=["L1", "L1"])
    state.project.fault_picks.append(fault)
    snap = snapshot_interpretation(state.active_section, state.project)
    out = K.restore_snapshot(
        snap, _event("fault_parallel_flow", slip=np.hypot(100.0, 100.0),
                     fault_uuid=fault.uuid), section_name="L1")
    d, z = out.horizons[0].picks_for_section("L1")
    assert np.allclose(d, [100.0, 1100.0]) and np.allclose(z, [300.0, 400.0])


def test_balanced_section_restores_within_tolerance():
    """A flexural-slip unfold conserves bed length, so deformed vs restored
    line-length balance is ~0 — measured by Step 4's balance module."""
    state, hp, poly = _anchored_state()
    snap = snapshot_interpretation(state.active_section, state.project)
    out = K.restore_snapshot(snap, _event("flexural_slip", pin_x=0.0, datum_y=0.0),
                             section_name="L1")
    d0, z0 = hp.picks_for_section("L1")
    d1, z1 = out.horizons[0].picks_for_section("L1")
    bal = B.line_length_balance({hp.uuid: np.column_stack([d0, z0])},
                                {hp.uuid: np.column_stack([d1, z1])})
    assert bal[0].discrepancy < 1e-6                           # length conserved


def test_event_algorithm_params_round_trip():
    from section_tool.core.restoration import RestorationSequence
    seq = RestorationSequence(events=[_event("simple_shear", shear_angle=30.0,
                                             datum_y=0.0)])
    seq2 = RestorationSequence.from_json(seq.to_json())
    assert seq2.events[0].algorithm == "simple_shear"
    assert seq2.events[0].params["shear_angle"] == 30.0
