"""Step 7: restore_by_construction_rule — automatic restoration-algorithm
proposals from each element's construction rule, with confidence levels."""
from __future__ import annotations

from section_tool.core.kinematics import restore_by_construction_rule
from section_tool.core.surfaces import HorizonPick
from section_tool.core.polygons import SectionPolygon
from section_tool.core.construction import (
    FreehandRule, ParallelToBedRule, DipConstrainedRule, KinkBandRule,
    ListricFaultRule, MirrorAcrossAxialTraceRule)


def _hp(rule=None):
    hp = HorizonPick([0.0, 1000.0], [100.0, 200.0], name="H")
    hp.construction_rule = rule
    return hp


def test_no_rule_returns_none():
    assert restore_by_construction_rule(_hp(None)) is None


def test_parallel_to_bed_to_flexural_slip():
    p = restore_by_construction_rule(_hp(ParallelToBedRule(reference_name="R")))
    assert p.algorithm == "flexural_slip" and p.confidence == "suggested"
    assert p.source_kind == "parallel_to_bed"


def test_kink_band_to_flexural_slip():
    p = restore_by_construction_rule(_hp(KinkBandRule(axial_surface_dip_deg=30.0)))
    assert p.algorithm == "flexural_slip" and p.confidence == "suggested"


def test_dip_constrained_to_simple_shear_with_seeded_params():
    p = restore_by_construction_rule(_hp(DipConstrainedRule(dip_deg=20.0)))
    assert p.algorithm == "simple_shear" and p.confidence == "suggested"
    assert p.params == {"shear_angle": 0.0}


def test_listric_fault_is_certain_and_seeds_fault_uuid():
    hp = _hp(ListricFaultRule(detachment_depth_m=5000.0))
    p = restore_by_construction_rule(hp)
    assert p.algorithm == "fault_parallel_flow" and p.confidence == "certain"
    assert p.params["fault_uuid"] == hp.uuid              # this fault is the trace


def test_freehand_and_mirror_have_no_inverse():
    assert restore_by_construction_rule(_hp(FreehandRule())) is None
    assert restore_by_construction_rule(
        _hp(MirrorAcrossAxialTraceRule(axial_trace_name="A"))) is None


def test_proposal_works_on_polygon():
    poly = SectionPolygon([(0, 0), (100, 0), (100, 50)], name="P")
    poly.construction_rule = ParallelToBedRule(reference_name="R")
    assert restore_by_construction_rule(poly).algorithm == "flexural_slip"


def test_certain_reserved_for_explicit_kinematic():
    # the natural-inverse (inferred-default) kinds are never "certain"
    for rule in (ParallelToBedRule(reference_name="R"),
                 KinkBandRule(axial_surface_dip_deg=10.0),
                 DipConstrainedRule(dip_deg=5.0)):
        assert restore_by_construction_rule(_hp(rule)).confidence == "suggested"
