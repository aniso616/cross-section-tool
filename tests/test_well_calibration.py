"""M5 — well calibration: robust (v0,k) fit, residual reporting, opt-in promotion."""
from __future__ import annotations

import numpy as np
import pytest

from section_tool.core.velocity_model import VelocityModel, VelocityFunction
from section_tool.core.well_calibration import (
    Marker, fit_v0k, marker_residuals, calibrate_model, well_td_control)


def _synthetic_markers(v0, k, depths):
    """(depth, twt) pairs lying exactly on a v0+k*z model."""
    fn = VelocityFunction("linear_v0k", v0=v0, k=k)
    return [Marker(z, fn.depth_to_twt(z), name=f"m{i}")
            for i, z in enumerate(depths)]


def test_recovers_known_v0_k():
    true_v0, true_k = 1850.0, 0.55
    ms = _synthetic_markers(true_v0, true_k, [200, 600, 1100, 1700, 2400])
    v0, k = fit_v0k([m.depth_m for m in ms], [m.twt_s for m in ms])
    assert v0 == pytest.approx(true_v0, rel=0.02)
    assert k == pytest.approx(true_k, rel=0.05)


def test_irls_rejects_injected_outlier():
    true_v0, true_k = 1850.0, 0.55
    ms = _synthetic_markers(true_v0, true_k, [200, 600, 1100, 1700, 2400])
    depths = [m.depth_m for m in ms]
    twts = [m.twt_s for m in ms]
    twts[2] += 0.6              # gross outlier on one marker (+600 ms)

    v0_r, k_r = fit_v0k(depths, twts, robust=True)
    v0_n, k_n = fit_v0k(depths, twts, robust=False)
    # Robust fit stays near the truth; the plain fit is dragged off by the outlier.
    assert v0_r == pytest.approx(true_v0, rel=0.06)
    assert abs(v0_r - true_v0) < abs(v0_n - true_v0)


def test_marker_residuals_zero_on_model_and_signed_off_model():
    model = VelocityModel.average_vz(2000.0, 0.0)   # constant 2000 m/s → z = 1000*t
    on = marker_residuals(model, [Marker(1000.0, 1.0, "x")])[0]
    assert on["depth_residual_m"] == pytest.approx(0.0, abs=1e-6)
    assert on["twt_residual_s"] == pytest.approx(0.0, abs=1e-9)
    # A marker 100 m deeper than the model predicts → +100 m depth residual.
    off = marker_residuals(model, [Marker(900.0, 1.0, "y")])[0]
    assert off["depth_residual_m"] == pytest.approx(100.0, abs=1e-6)


def test_calibrate_promotes_layer_and_fits():
    base = VelocityModel.average_vz(1500.0, 0.2)        # assumed, wrong velocity
    assert base.provenance == "assumed"
    ms = _synthetic_markers(1850.0, 0.55, [300, 800, 1400, 2100])
    cal = calibrate_model(base, ms)
    assert cal.layers[0].provenance == "well_calibrated"
    assert cal.provenance == "well_calibrated"
    assert cal.layers[0].function.v0 == pytest.approx(1850.0, rel=0.03)


def test_base_path_unaffected_when_no_markers():
    base = VelocityModel.average_vz(1800.0, 0.5)
    cal = calibrate_model(base, [])
    # Nothing promoted; velocity law and provenance preserved.
    assert cal.provenance == "assumed"
    assert cal.layers[0].function.v0 == pytest.approx(1800.0)
    assert cal.layers[0].function.k == pytest.approx(0.5)


def test_layer_with_too_few_markers_stays_assumed():
    base = VelocityModel.average_vz(1800.0, 0.5)
    cal = calibrate_model(base, [Marker(1000.0, 1.0, "only-one")])
    assert cal.layers[0].provenance == "assumed"        # < 2 markers → not fit


def test_well_td_control_from_tops_and_checkshot():
    from section_tool.core.wells import Well
    w = Well("W1", 0.0, 0.0)
    w.add_formation_top("TopA", 500.0)
    w.add_formation_top("TopB", 1500.0)
    # checkshot as (depth, twt) pairs: 2000 m/s → t = z/1000
    ms = well_td_control(w, [(0.0, 0.0), (2000.0, 2.0)])
    by_name = {m.name: m for m in ms}
    assert by_name["TopA"].twt_s == pytest.approx(0.5)
    assert by_name["TopB"].twt_s == pytest.approx(1.5)


def test_calibrate_promotes_only_marker_bearing_zones():
    """Per-zone promotion: a layered model's shallow zone (with >=2 markers) is
    promoted to well-tied; the deep zone (no markers) stays assumed."""
    from section_tool.core.velocity_model import VelocityLayer, VelocityFunction
    m = VelocityModel(layers=[
        VelocityLayer(VelocityFunction("constant", v0=2500.0), top_twt_s=0.0, name="A"),
        VelocityLayer(VelocityFunction("constant", v0=4000.0), top_twt_s=1.0, name="B")])
    # Zone A spans depth 0..1250 m (2500·1.0/2); both markers land in A.
    cal = calibrate_model(m, [Marker(300.0, 0.24, "m1"), Marker(800.0, 0.64, "m2")])
    assert cal.layers[0].provenance == "well_calibrated"
    assert cal.layers[1].provenance == "assumed"
    # headline is the weakest — still "assumed" until every zone is tied
    assert cal.provenance == "assumed"
