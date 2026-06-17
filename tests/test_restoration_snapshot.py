"""Interpretation snapshot (Restoration Step 3): faithful, non-destructive
duplicate of a section's interpretation, and a round-trip that reproduces it
field-for-field.  Fidelity is the theme — a lossy snapshot corrupts everything
built on it."""
from __future__ import annotations

import numpy as np

from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick
from section_tool.core.polygons import SectionPolygon
from section_tool.core.construction import (
    ParallelToBedRule, ListricFaultRule, serialize_rule)
from section_tool.core.reference_line import ReferenceLine
from section_tool.core.restoration_snapshot import (
    snapshot_interpretation, restore_from_snapshot, InterpretationSnapshot)


_SCALARS = ("uuid", "name", "z_units", "color", "line_width", "line_style",
            "contact_type", "formation_above", "formation_below", "age_ma",
            "confidence", "event_id", "fault_type", "dip_direction",
            "sense_of_slip", "displacement", "age_activation_ma",
            "age_cessation_ma", "seismic_tied")


def _assert_pick_equal(a: HorizonPick, b: HorizonPick) -> None:
    for fld in _SCALARS:
        assert getattr(a, fld) == getattr(b, fld), f"pick field {fld} differs"
    for arr in HorizonPick._POINT_ARRAYS:                # every per-point array
        va, vb = getattr(a, arr), getattr(b, arr)
        if va.dtype.kind == "f":
            assert np.allclose(va, vb, equal_nan=True), f"{arr} differs"
        else:
            assert np.array_equal(va, vb), f"{arr} differs"
    assert serialize_rule(a.construction_rule) == serialize_rule(b.construction_rule)


def _assert_poly_equal(a: SectionPolygon, b: SectionPolygon) -> None:
    for fld in ("uuid", "name", "formation", "section_name", "fill_color",
                "fill_alpha", "edge_color", "edge_width", "visible"):
        assert getattr(a, fld) == getattr(b, fld), f"polygon field {fld} differs"
    assert np.allclose(a.vertices, b.vertices)
    assert serialize_rule(a.construction_rule) == serialize_rule(b.construction_rule)


def _rich_state():
    """A section with a horizon (anchored + construction rule), a fault, a polygon
    (construction rule), and a reference line."""
    state = AppState()
    state.add_section(Section([(0.0, 0.0), (1000.0, 0.0)], name="L1", crs_epsg=32631))
    state.set_active_section(state.project.sections[0])

    hp = HorizonPick([0.0, 500.0, 1000.0], [100.0, 150.0, 200.0], name="Top Chalk",
                     section_names=["L1", "L1", "L1"],
                     twt_anchor=[0.10, 0.15, np.nan], seismic_tied=True,
                     formation_above="A", formation_below="B")
    hp.construction_rule = ParallelToBedRule(reference_name="Datum",
                                             offset_m=25.0, reference_uuid="ref-123")
    state.project.horizon_picks.append(hp)

    fp = HorizonPick([0.0, 1000.0], [300.0, 600.0], name="F1",
                     section_names=["L1", "L1"], fault_type="normal",
                     dip_direction="left", displacement=120.0)
    fp.construction_rule = ListricFaultRule(detachment_depth_m=5000.0,
                                            hangingwall_uuid="hw-9")
    state.project.fault_picks.append(fp)

    poly = SectionPolygon([[0, 0], [1000, 0], [1000, 500]], name="Block",
                          formation="B", section_name="L1")
    poly.construction_rule = ParallelToBedRule(reference_name="Top Chalk",
                                               reference_uuid=hp.uuid)
    state.project.polygons.append(poly)

    state.project.reference_lines.append(ReferenceLine("horizontal", value=0.0,
                                                       name="Sea level"))
    return state, hp, fp, poly


# ── round-trip fidelity ─────────────────────────────────────────────────────

def test_round_trip_is_field_for_field_identical():
    state, hp, fp, poly = _rich_state()
    snap = snapshot_interpretation(state.active_section, state.project)
    out = restore_from_snapshot(snap)

    assert len(out["horizons"]) == 1 and len(out["faults"]) == 1
    assert len(out["polygons"]) == 1 and len(out["reference_lines"]) == 1
    _assert_pick_equal(hp, out["horizons"][0])           # horizon, all fields
    _assert_pick_equal(fp, out["faults"][0])             # fault-as-HorizonPick
    _assert_poly_equal(poly, out["polygons"][0])
    # section line + metadata
    sec = out["section"]
    assert sec.name == "L1" and sec.crs_epsg == 32631
    assert np.allclose(sec.nodes, state.active_section.nodes)


def test_uuids_preserved_and_construction_refs_intact():
    state, hp, fp, poly = _rich_state()
    out = restore_from_snapshot(snapshot_interpretation(state.active_section,
                                                        state.project))
    # preserved-UUID policy: restored uuid == original uuid (pairing is equality)
    assert out["horizons"][0].uuid == hp.uuid
    assert out["faults"][0].uuid == fp.uuid
    assert out["polygons"][0].uuid == poly.uuid
    # construction-rule UUID references travel intact
    assert out["polygons"][0].construction_rule.reference_uuid == hp.uuid
    assert out["faults"][0].construction_rule.hangingwall_uuid == "hw-9"


def test_anchor_survives_snapshot_intact():
    state, hp, fp, poly = _rich_state()
    out = restore_from_snapshot(snapshot_interpretation(state.active_section,
                                                        state.project))
    a, b = hp._twt_anchor, out["horizons"][0]._twt_anchor
    assert np.allclose(a, b, equal_nan=True)             # NaN positions preserved
    assert out["horizons"][0].seismic_tied is True


# ── non-destructive (no aliasing) ───────────────────────────────────────────

def test_mutating_the_copy_leaves_the_original_unchanged():
    state, hp, fp, poly = _rich_state()
    orig_depths = hp.depths.copy()
    orig_offset = hp.construction_rule.offset_m
    orig_verts = poly.vertices.copy()

    out = restore_from_snapshot(snapshot_interpretation(state.active_section,
                                                        state.project))
    # deform the COPY in place
    out["horizons"][0]._depths[:] = 999.0
    out["horizons"][0].construction_rule.offset_m = -1.0
    out["polygons"][0]._vertices[:] = 0.0

    assert np.allclose(hp.depths, orig_depths)           # original geometry untouched
    assert hp.construction_rule.offset_m == orig_offset  # original rule untouched
    assert np.allclose(poly.vertices, orig_verts)


def test_restoring_twice_yields_independent_states():
    state, hp, fp, poly = _rich_state()
    snap = snapshot_interpretation(state.active_section, state.project)
    a = restore_from_snapshot(snap)
    b = restore_from_snapshot(snap)
    a["horizons"][0]._depths[:] = 123.0
    assert not np.allclose(a["horizons"][0].depths, b["horizons"][0].depths)
    assert a["horizons"][0] is not b["horizons"][0]


# ── references, not copies ──────────────────────────────────────────────────

def test_heavy_data_referenced_by_id_not_copied():
    state, hp, fp, poly = _rich_state()
    snap = snapshot_interpretation(state.active_section, state.project)
    # velocity model carried as an id reference (or None), never the object
    assert not hasattr(snap.velocity_model_id, "layers")
    assert isinstance(snap.seismic_ref_names, list)


# ── edge cases ──────────────────────────────────────────────────────────────

def test_empty_section_snapshots_cleanly():
    state = AppState()
    state.add_section(Section([(0.0, 0.0), (1000.0, 0.0)], name="L1", crs_epsg=32631))
    state.set_active_section(state.project.sections[0])
    snap = snapshot_interpretation(state.active_section, state.project)
    out = restore_from_snapshot(snap)
    assert out["horizons"] == [] and out["faults"] == [] and out["polygons"] == []
    assert out["section"].name == "L1"
    assert snap.schema_version == InterpretationSnapshot(section={}).schema_version
