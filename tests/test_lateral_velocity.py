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


def test_round_trip_to_from_dict():
    lvm = LateralVelocityModel([(0.0, _bulk(2000.0)),
                                (1500.0, VelocityModel.average_vz(1900.0, 0.6))])
    back = LateralVelocityModel.from_dict(lvm.to_dict())
    assert len(back.controls) == 2
    assert back.model_at(750.0).layers[0].function.v0 == pytest.approx(
        lvm.model_at(750.0).layers[0].function.v0)
