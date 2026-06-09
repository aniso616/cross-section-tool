"""M4 — lateral velocity variation along a section (well-free pseudo-well control)."""
from __future__ import annotations

import numpy as np
import pytest

from section_tool.core.velocity_model import VelocityModel
from section_tool.core.lateral_velocity import LateralVelocityModel, LateralControl


def _bulk(v):
    return VelocityModel.bulk(v)


def test_exact_at_control_points():
    lvm = LateralVelocityModel([(0.0, _bulk(2000.0)), (1000.0, _bulk(3000.0))])
    # model_at returns the control model verbatim at a control distance
    assert lvm.model_at(0.0).layers[0].function.v0 == pytest.approx(2000.0)
    assert lvm.model_at(1000.0).layers[0].function.v0 == pytest.approx(3000.0)
    # conversion matches the control model exactly
    assert lvm.twt_to_depth(1.0, 0.0) == pytest.approx(_bulk(2000.0).twt_to_depth(1.0))
    assert lvm.twt_to_depth(1.0, 1000.0) == pytest.approx(_bulk(3000.0).twt_to_depth(1.0))


def test_interpolated_between_is_bounded_and_monotone():
    lvm = LateralVelocityModel([(0.0, _bulk(2000.0)), (1000.0, _bulk(3000.0))])
    v_mid = lvm.model_at(500.0).layers[0].function.v0
    assert 2000.0 < v_mid < 3000.0                      # bounded by neighbours
    assert v_mid == pytest.approx(2500.0)               # linear midpoint
    # depth at a fixed TWT increases monotonically with velocity across distance
    depths = [lvm.twt_to_depth(1.0, d) for d in (0.0, 250.0, 500.0, 750.0, 1000.0)]
    assert all(b > a for a, b in zip(depths, depths[1:]))


def test_extrapolation_is_clipped_not_wild():
    lvm = LateralVelocityModel([(0.0, _bulk(2000.0)), (1000.0, _bulk(3000.0))])
    # Beyond the outermost control, parameters clip to the edge value.
    assert lvm.model_at(-500.0).layers[0].function.v0 == pytest.approx(2000.0)
    assert lvm.model_at(5000.0).layers[0].function.v0 == pytest.approx(3000.0)
    assert lvm.twt_to_depth(1.0, 9999.0) == pytest.approx(_bulk(3000.0).twt_to_depth(1.0))


def test_works_with_purely_assumed_control():
    a = VelocityModel.average_vz(1800.0, 0.5)   # provenance defaults to assumed
    b = VelocityModel.average_vz(2100.0, 0.4)
    assert a.provenance == "assumed" and b.provenance == "assumed"
    lvm = LateralVelocityModel([(0.0, a), (2000.0, b)])
    mid = lvm.model_at(1000.0)
    z = mid.twt_to_depth(1.2)
    assert np.isfinite(z) and z > 0
    assert mid.provenance == "interpolated"             # derived between controls


def test_single_control_is_constant_laterally():
    lvm = LateralVelocityModel([(500.0, _bulk(2400.0))])
    for d in (-100.0, 0.0, 500.0, 9000.0):
        assert lvm.model_at(d).layers[0].function.v0 == pytest.approx(2400.0)


def test_structure_mismatch_raises():
    one = VelocityModel.bulk(2000.0)
    two = VelocityModel(layers=list(VelocityModel.bulk(2000.0).layers) +
                        list(VelocityModel.bulk(3000.0).layers))
    with pytest.raises(ValueError):
        LateralVelocityModel([(0.0, one), (1000.0, two)])


def test_empty_controls_raises():
    with pytest.raises(ValueError):
        LateralVelocityModel([])


def test_stretch_image_to_depth_lateral_varies_with_distance():
    """A reflector at fixed TWT lands deeper where the velocity is higher — the
    point of lateral variation in the image stretch."""
    from section_tool.core.conversion import stretch_image_to_depth_lateral
    lvm = LateralVelocityModel([(0.0, _bulk(2000.0)), (1000.0, _bulk(4000.0))])
    distances = np.array([0.0, 1000.0])             # two traces, the two extremes
    dt_s = 0.5
    n_samples = 5                                    # t = 0,0.5,1.0,1.5,2.0 s
    amp = np.zeros((n_samples, 2))
    amp[2, :] = 1.0                                  # bright reflector at t = 1.0 s
    z_axis, dimg = stretch_image_to_depth_lateral(amp.T, dt_s, lvm, distances)
    dimg = dimg                                      # (n_traces, n_depth)
    peak0 = z_axis[np.argmax(dimg[0])]
    peak1 = z_axis[np.argmax(dimg[1])]
    assert peak0 == pytest.approx(1000.0, abs=z_axis[1] - z_axis[0])   # 2000 m/s
    assert peak1 == pytest.approx(2000.0, abs=z_axis[1] - z_axis[0])   # 4000 m/s
    assert peak1 > peak0


def test_round_trip_to_from_dict():
    lvm = LateralVelocityModel([(0.0, _bulk(2000.0)),
                                (1500.0, VelocityModel.average_vz(1900.0, 0.6))])
    back = LateralVelocityModel.from_dict(lvm.to_dict())
    assert len(back.controls) == 2
    assert back.model_at(750.0).layers[0].function.v0 == pytest.approx(
        lvm.model_at(750.0).layers[0].function.v0)


def _well_tied(v):
    from section_tool.core.velocity_model import VelocityLayer, VelocityFunction
    return VelocityModel(layers=[VelocityLayer(
        VelocityFunction("constant", v0=v), provenance="well_calibrated")])


def test_provenance_headline_reads_weakest_across_section():
    # single well-tied control → well-tied (one control governs, no interpolation)
    assert LateralVelocityModel([(0.0, _well_tied(2500.0))]).provenance == "well_calibrated"
    # two well-tied controls → interpolated between them (weakest on the section)
    two_tied = LateralVelocityModel([(0.0, _well_tied(2500.0)), (1000.0, _well_tied(3000.0))])
    assert two_tied.provenance == "interpolated"
    # any assumed/regional control → assumed dominates
    mixed = LateralVelocityModel([(0.0, _well_tied(2500.0)), (1000.0, _bulk(3000.0))])
    assert mixed.provenance == "assumed"


def test_glue_holds_under_laterally_varying_model():
    """A seismic-tied horizon's depth varies laterally per the local model, while
    its TWT anchors stay invariant (pick-once, refine-forever — laterally)."""
    from section_tool.core.surfaces import HorizonPick
    from section_tool.core.conversion import set_anchors, restretch_project
    from types import SimpleNamespace
    # nodes at d=0 and d=1000, same depth → same anchor under bulk 2000 (0.5 s)
    hp = HorizonPick(np.array([0.0, 1000.0]), np.array([1000.0, 1000.0]), name="H")
    set_anchors(hp, _bulk(2000.0))
    anchors = hp._twt_anchor.copy()                       # both 1.0 s
    proj = SimpleNamespace(horizon_picks=[hp], fault_picks=[])
    lvm = LateralVelocityModel([(0.0, _bulk(2000.0)), (1000.0, _bulk(4000.0))])
    restretch_project(proj, lvm)
    assert np.allclose(hp._twt_anchor, anchors)           # anchors invariant
    # node 0 (2000 m/s) and node 1 (4000 m/s) at the same TWT land at different depths
    assert hp._depths[0] == pytest.approx(lvm.model_at(0.0).twt_to_depth(anchors[0]))
    assert hp._depths[1] == pytest.approx(lvm.model_at(1000.0).twt_to_depth(anchors[1]))
    assert hp._depths[1] > hp._depths[0] * 1.5            # deeper where faster
