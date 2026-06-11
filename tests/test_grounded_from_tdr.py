"""VelocityModel.from_tdr — reproduce checkshot knots, datum, CAP, extrapolation."""
from __future__ import annotations

import numpy as np
import pytest

from section_tool.core.wells import Well
from section_tool.core.tdr import TimeDepthRelation
from section_tool.core.velocity_model import VelocityModel


def _well():
    return Well("F02-01", 606554.0, 6080126.0, kb=30.0, td=3500.0)


def _checkshot_tvdss():
    # F02-01-like: TVDSS knots starting at the datum (0, 0).
    z = [0.0, 523.6, 1665.0, 3120.0]
    t = [0.0, 0.544, 1.67, 3.234]
    return TimeDepthRelation(z, t, kind="checkshot", depth_reference="TVDSS",
                             source="F02-01_TD.txt")


class TestReproduction:
    def test_reproduces_knots_both_directions(self):
        tdr = _checkshot_tvdss()
        m = VelocityModel.from_tdr(_well(), tdr, setting="onshore")
        for z, t in zip(tdr.depth_m, tdr.twt_s):
            assert m.depth_to_twt(float(z)) == pytest.approx(float(t), abs=1e-4)
            assert m.twt_to_depth(float(t)) == pytest.approx(float(z), abs=1e-3)

    def test_interval_velocity_between_first_two_knots(self):
        tdr = _checkshot_tvdss()
        m = VelocityModel.from_tdr(_well(), tdr, setting="onshore")
        # v = 2·Δz/Δt = 2·523.6/0.544 ≈ 1925 m/s
        v = 2.0 * 523.6 / 0.544
        # twt_to_depth halfway through the first interval should match the law
        z_mid = m.twt_to_depth(0.272)
        assert z_mid == pytest.approx(v * 0.272 / 2.0, rel=1e-3)

    def test_provenance_is_checkshot_tied(self):
        m = VelocityModel.from_tdr(_well(), _checkshot_tvdss(), setting="onshore")
        assert m.provenance == "checkshot_tied"

    def test_construction_metadata(self):
        tdr = _checkshot_tvdss()
        w = _well()
        m = VelocityModel.from_tdr(w, tdr, setting="onshore")
        c = m.construction
        assert c["kind"] == "from_tdr"
        assert c["params"]["well_uuid"] == w.uuid
        assert c["params"]["tdr_uuid"] == tdr.uuid
        assert c["params"]["extrapolation"] == "last_interval_velocity"


class TestDatum:
    def test_md_referenced_tdr_goes_through_deviation(self):
        """An MD-referenced TDR is converted to TVDSS via the well (MD − KB for a
        vertical well), so the model lands on TVDSS depths."""
        w = _well()  # kb 30
        md = [30.0, 553.6, 1695.0, 3150.0]      # MD from KB
        t = [0.0, 0.544, 1.67, 3.234]
        tdr = TimeDepthRelation(md, t, kind="checkshot", depth_reference="MD")
        m = VelocityModel.from_tdr(w, tdr, setting="onshore")
        # MD 3150 → TVDSS 3120 at 3.234 s
        assert m.twt_to_depth(3.234) == pytest.approx(3120.0, abs=1e-2)
        assert m.depth_to_twt(0.0) == pytest.approx(0.0, abs=1e-6)


class TestCapAndExtrapolation:
    def test_cap_layer_when_shallowest_knot_below_datum(self):
        # Knots start at twt 0.5 s (not the datum) → a CAP layer fills 0 → 0.5 s.
        tdr = TimeDepthRelation([500.0, 1500.0], [0.5, 1.5],
                                kind="checkshot", depth_reference="TVDSS")
        m = VelocityModel.from_tdr(_well(), tdr, setting="onshore")
        assert any(l.name == "Cap" for l in m.layers)
        # Still reproduces the first knot exactly.
        assert m.twt_to_depth(0.5) == pytest.approx(500.0, abs=1e-3)

    def test_marine_water_cap(self):
        tdr = TimeDepthRelation([500.0, 1500.0], [0.5, 1.5],
                                kind="checkshot", depth_reference="TVDSS")
        m = VelocityModel.from_tdr(_well(), tdr, setting="marine",
                                   seafloor_twt_s=0.1, water_v=1480.0)
        names = [l.name for l in m.layers]
        assert "Water" in names
        assert m.construction["params"]["cap"]["kind"] == "marine_water_fill"
        assert m.twt_to_depth(0.5) == pytest.approx(500.0, abs=1e-3)

    def test_extrapolation_below_deepest_knot_uses_last_interval(self):
        tdr = _checkshot_tvdss()
        m = VelocityModel.from_tdr(_well(), tdr, setting="onshore")
        v_last = 2.0 * (3120.0 - 1665.0) / (3.234 - 1.67)
        # 0.5 s below the deepest knot
        z_deep = m.twt_to_depth(3.234 + 0.5)
        z_expect = 3120.0 + v_last * 0.5 / 2.0
        assert z_deep == pytest.approx(z_expect, rel=1e-3)
