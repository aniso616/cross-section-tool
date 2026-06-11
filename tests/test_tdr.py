"""TimeDepthRelation entity — interp, monotonic validation, round-trip, adapter."""
from __future__ import annotations

import numpy as np
import pytest

from section_tool.core.tdr import (
    TimeDepthRelation, seconds_from, metres_from)
from section_tool.core.zdomain import ZDomain


def _tdr():
    # 2000 m/s constant → twt = 2*z/2000 = z/1000
    z = [0.0, 500.0, 1000.0, 2000.0]
    t = [0.0, 0.5, 1.0, 2.0]
    return TimeDepthRelation(z, t, kind="checkshot", depth_reference="TVDSS",
                             source="x_TD.txt")


# ---------------------------------------------------------------------------
# Construction + interpolation
# ---------------------------------------------------------------------------

class TestInterp:
    def test_twt_at_depth_exact_and_interpolated(self):
        td = _tdr()
        assert td.twt_at_depth(500.0) == pytest.approx(0.5)
        assert float(td.twt_at_depth(750.0)) == pytest.approx(0.75)

    def test_depth_at_twt_inverse(self):
        td = _tdr()
        assert float(td.depth_at_twt(1.0)) == pytest.approx(1000.0)
        assert float(td.depth_at_twt(1.5)) == pytest.approx(1500.0)

    def test_roundtrip_depth_twt_depth(self):
        td = _tdr()
        for z in (123.0, 888.0, 1750.0):
            t = td.twt_at_depth(z)
            assert float(td.depth_at_twt(t)) == pytest.approx(z, abs=1e-6)

    def test_clamps_outside_range(self):
        td = _tdr()
        assert float(td.twt_at_depth(-100.0)) == pytest.approx(0.0)
        assert float(td.twt_at_depth(9999.0)) == pytest.approx(2.0)

    def test_array_input(self):
        td = _tdr()
        out = td.twt_at_depth([0.0, 1000.0, 2000.0])
        assert np.allclose(out, [0.0, 1.0, 2.0])

    def test_ranges_and_count(self):
        td = _tdr()
        assert td.n_points == 4
        assert td.depth_range() == (0.0, 2000.0)
        assert td.twt_range() == (0.0, 2.0)


# ---------------------------------------------------------------------------
# Monotonic validation
# ---------------------------------------------------------------------------

class TestMonotonicValidation:
    def test_rejects_non_increasing_depth(self):
        with pytest.raises(ValueError, match="depth"):
            TimeDepthRelation([0.0, 500.0, 500.0], [0.0, 0.5, 0.6])

    def test_rejects_non_increasing_twt(self):
        with pytest.raises(ValueError, match="twt"):
            TimeDepthRelation([0.0, 500.0, 1000.0], [0.0, 0.6, 0.5])

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="NaN"):
            TimeDepthRelation([0.0, 500.0, np.nan], [0.0, 0.5, 1.0])

    def test_rejects_too_few_points(self):
        with pytest.raises(ValueError, match="at least 2"):
            TimeDepthRelation([0.0], [0.0])

    def test_from_pairs_sorts_then_validates(self):
        # Unsorted but otherwise valid → from_pairs sorts and accepts.
        td = TimeDepthRelation.from_pairs([1000.0, 0.0, 500.0], [1.0, 0.0, 0.5])
        assert td.depth_range() == (0.0, 1000.0)
        assert float(td.twt_at_depth(500.0)) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_to_from_dict_roundtrip(self):
        td = _tdr()
        td2 = TimeDepthRelation.from_dict(td.to_dict())
        assert td2.kind == "checkshot"
        assert td2.depth_reference == "TVDSS"
        assert td2.source == "x_TD.txt"
        assert td2.uuid == td.uuid
        assert np.allclose(td2.depth_m, td.depth_m)
        assert np.allclose(td2.twt_s, td.twt_s)

    def test_to_dict_has_schema_version(self):
        assert _tdr().to_dict()["schema_version"] == 1

    def test_default_construction_metadata(self):
        d = _tdr().to_dict()["construction"]
        assert d["kind"] == "time_depth_relation"
        assert d["parents"] == [] and d["params"] == {}


# ---------------------------------------------------------------------------
# ZDomain thin adapter (IO-boundary unit conversion)
# ---------------------------------------------------------------------------

class TestAdapter:
    def test_seconds_from_ms(self):
        assert np.allclose(seconds_from([544.0, 1670.0], ZDomain.TWT_MS),
                           [0.544, 1.670])

    def test_seconds_from_s_identity(self):
        assert np.allclose(seconds_from([0.544, 1.670], ZDomain.TWT_S),
                           [0.544, 1.670])

    def test_seconds_from_rejects_depth_domain(self):
        with pytest.raises(ValueError):
            seconds_from([1.0], ZDomain.DEPTH_M)

    def test_metres_from_ft(self):
        assert np.allclose(metres_from([100.0], ZDomain.DEPTH_FT), [30.48])

    def test_metres_from_m_identity(self):
        assert np.allclose(metres_from([553.6], ZDomain.DEPTH_M), [553.6])


# ---------------------------------------------------------------------------
# Well integration
# ---------------------------------------------------------------------------

class TestWellIntegration:
    def test_add_tdr_stamps_well_uuid(self):
        from section_tool.core.wells import Well
        w = Well("F02-01", 0.0, 0.0, kb=30.0)
        td = _tdr()
        w.add_tdr(td)
        assert td.well_uuid == w.uuid
        assert len(w.tdrs) == 1
        assert w.primary_checkshot() is td
        assert w.tdrs_of_kind("sonic_integrated") == []

    def test_well_has_stable_uuid(self):
        from section_tool.core.wells import Well
        w = Well("W", 0.0, 0.0)
        assert isinstance(w.uuid, str) and len(w.uuid) > 0
        w2 = Well("W2", 0.0, 0.0, uuid="fixed-uuid")
        assert w2.uuid == "fixed-uuid"
