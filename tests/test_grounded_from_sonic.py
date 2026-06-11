"""VelocityModel.from_sonic — units, integration, drift correction, CAP."""
from __future__ import annotations

import os

import numpy as np
import pytest

from section_tool.core.wells import Well, LogCurve
from section_tool.core.tdr import TimeDepthRelation
from section_tool.core.velocity_model import VelocityModel
from section_tool.core.grounded_velocity import (
    slowness_to_s_per_m, integrate_sonic, select_sonic_curve,
    _US_FT_TO_S_PER_M)

# Constant slowness giving v = 2000 m/s (so TWT = 2z/2000 = z/1000).
_SLOW_2000_US_FT = (1.0 / 2000.0) / _US_FT_TO_S_PER_M   # ≈ 152.4 µs/ft
_SLOW_2000_US_M = (1.0 / 2000.0) / 1e-6                 # 500 µs/m


def _sonic_well(units="us/ft", slow=_SLOW_2000_US_FT, kb=0.0, top_md=0.0,
                td=3000.0, name="DT"):
    w = Well("W", 0.0, 0.0, kb=kb, td=td)
    md = np.arange(top_md, td + 1.0, 5.0)
    w.add_log(LogCurve(name, units, md, np.full_like(md, slow)))
    return w


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

class TestUnits:
    def test_slowness_us_ft(self):
        s, kind = slowness_to_s_per_m([_SLOW_2000_US_FT], "us/ft")
        assert kind == "us/ft"
        assert float(s[0]) == pytest.approx(1.0 / 2000.0, rel=1e-6)

    def test_slowness_us_m(self):
        s, kind = slowness_to_s_per_m([_SLOW_2000_US_M], "us/m")
        assert kind == "us/m"
        assert float(s[0]) == pytest.approx(1.0 / 2000.0, rel=1e-6)

    def test_us_ft_and_us_m_give_same_model(self):
        m_ft = VelocityModel.from_sonic(_sonic_well("us/ft", _SLOW_2000_US_FT),
                                        drift_target="none")
        m_m = VelocityModel.from_sonic(_sonic_well("us/m", _SLOW_2000_US_M),
                                       drift_target="none")
        for z in (500.0, 1500.0, 2500.0):
            assert m_ft.depth_to_twt(z) == pytest.approx(m_m.depth_to_twt(z), abs=1e-4)

    def test_integration_matches_constant_velocity(self):
        m = VelocityModel.from_sonic(_sonic_well(), drift_target="none")
        # v = 2000 → twt = z/1000
        assert m.depth_to_twt(2000.0) == pytest.approx(2.0, abs=2e-3)
        assert m.twt_to_depth(1.0) == pytest.approx(1000.0, rel=2e-3)

    def test_despike_drops_nulls_and_spikes(self):
        w = _sonic_well()
        md = np.arange(0.0, 3001.0, 5.0)
        vals = np.full_like(md, _SLOW_2000_US_FT)
        vals[10] = np.nan          # null
        vals[20] = 5000.0          # spike (way outside window)
        w.add_log(LogCurve("DT", "us/ft", md, vals))
        _z, _t, meta = integrate_sonic(w, "DT")
        assert meta["n_null_dropped"] >= 1
        assert meta["n_spike_dropped"] >= 1


# ---------------------------------------------------------------------------
# Curve selection
# ---------------------------------------------------------------------------

class TestCurveSelection:
    def test_prefers_corrected(self):
        w = Well("W", 0.0, 0.0, td=1000.0)
        md = np.arange(0.0, 1001.0, 5.0)
        w.add_log(LogCurve("DT:1", "us/ft", md, np.full_like(md, 150.0)))
        w.add_log(LogCurve("DT:2", "us/ft", md, np.full_like(md, 150.0)))
        assert select_sonic_curve(w) == "DT:2"

    def test_explicit_override(self):
        w = _sonic_well(name="DT:1")
        assert select_sonic_curve(w, "DT:1") == "DT:1"


# ---------------------------------------------------------------------------
# Drift correction
# ---------------------------------------------------------------------------

class TestDrift:
    def _well_and_checkshot(self):
        w = _sonic_well()  # integrated v=2000 → twt = z/1000
        # Checkshot reads slightly slower (twt < integrated): mild drift.
        cs = TimeDepthRelation([0.0, 1000.0, 2000.0, 3000.0],
                               [0.0, 0.95, 1.90, 2.85],
                               kind="checkshot", depth_reference="TVDSS")
        w.add_tdr(cs)
        return w, cs

    def test_checkshot_drift_residuals_shrink_and_honor_knots(self):
        w, cs = self._well_and_checkshot()
        m = VelocityModel.from_sonic(w, drift_target="checkshot")
        report = m.construction["params"]["drift"]
        assert report["target"] == "checkshot"
        # Post-correction residual at every knot is ~0 (knots honored exactly).
        assert report["max_residual_post_ms"] < 1.0
        for k in report["knots"]:
            assert abs(k["residual_post_ms"]) < abs(k["residual_pre_ms"]) + 1e-9
        # The model reproduces the checkshot at its knots.
        assert m.depth_to_twt(2000.0) == pytest.approx(1.90, abs=2e-3)

    def test_checkshot_drift_provenance_well_tied(self):
        w, _ = self._well_and_checkshot()
        m = VelocityModel.from_sonic(w, drift_target="checkshot")
        assert m.provenance == "well_calibrated"

    def test_anchors_drift(self):
        w = _sonic_well()
        knots = [(1000.0, 0.95), (2000.0, 1.90)]
        m = VelocityModel.from_sonic(w, drift_target="anchors", anchor_knots=knots)
        assert m.provenance == "sonic_derived"
        assert m.depth_to_twt(2000.0) == pytest.approx(1.90, abs=3e-3)

    def test_none_uncorrected_provenance_and_note(self):
        m = VelocityModel.from_sonic(_sonic_well(), drift_target="none")
        assert m.provenance == "sonic_derived"
        assert m.construction["params"]["drift"].get("params") == "uncorrected"

    def test_checkshot_target_requires_a_checkshot(self):
        with pytest.raises(ValueError):
            VelocityModel.from_sonic(_sonic_well(), drift_target="checkshot")


# ---------------------------------------------------------------------------
# CAP above the log top
# ---------------------------------------------------------------------------

class TestCap:
    def test_cap_present_when_log_starts_below_datum(self):
        # Log starts at MD 300 (kb 0 → TVDSS 300); CAP must fill datum→log top.
        w = _sonic_well(top_md=300.0)
        m = VelocityModel.from_sonic(w, drift_target="none", setting="onshore")
        assert any(l.name in ("Cap", "Water") for l in m.layers)
        assert m.twt_to_depth(0.0) == pytest.approx(0.0, abs=1e-6)
        assert m.construction["params"]["cap"]["kind"] != "none"


# ---------------------------------------------------------------------------
# F3 sanity: real DT integrates to within the agreement band of the checkshot
# ---------------------------------------------------------------------------

_REAL_LAS = r"J:\data\F3_Demo_2023\Rawdata\Well_data\F02-01_logs.las"
_REAL_TD = r"J:\data\F3_Demo_2023\Rawdata\Well_data\F02-01_TD.txt"


@pytest.mark.skipif(not (os.path.exists(_REAL_LAS) and os.path.exists(_REAL_TD)),
                    reason="F3 data drive not present")
def test_real_f02_sonic_agrees_with_checkshot():
    from section_tool.io.las import read_las
    from section_tool.io.tdr_io import load_checkshot
    well = read_las(_REAL_LAS)
    cs = load_checkshot(_REAL_TD, well)
    well.add_tdr(cs)

    # Uncorrected integration should already be in the right ballpark vs the
    # checkshot (no gross unit/datum error) — loose band.
    m_none = VelocityModel.from_sonic(well, drift_target="none")
    for zt in (523.6, 1665.0):   # TVDSS depths of MFS11, CKGR-ish
        assert abs(m_none.depth_to_twt(zt) - float(cs.twt_at_depth(zt))) < 0.12

    # Drift-corrected to the checkshot: residuals collapse at the knots.
    m_cs = VelocityModel.from_sonic(well, drift_target="checkshot")
    assert m_cs.construction["params"]["drift"]["max_residual_post_ms"] < 5.0
    assert m_cs.provenance == "well_calibrated"
