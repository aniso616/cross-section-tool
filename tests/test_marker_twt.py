"""Unified marker-TWT source rule (Part A permanent fix): priority + non-circular."""
from __future__ import annotations

import numpy as np
import pytest

from section_tool.core.wells import Well
from section_tool.core.tdr import TimeDepthRelation
from section_tool.core.surfaces import HorizonPick
from section_tool.core.velocity_model import VelocityModel
from section_tool.core.well_calibration import (
    resolve_marker_twt, build_well_markers, calibrate_model, marker_residuals)


def _checkshot():
    return TimeDepthRelation([0.0, 1000.0, 2000.0, 3000.0],
                             [0.0, 1.0, 2.0, 3.0],
                             kind="checkshot", depth_reference="TVDSS")


# ---------------------------------------------------------------------------
# Priority rule
# ---------------------------------------------------------------------------

class TestPriority:
    def test_checkshot_wins(self):
        twt, src = resolve_marker_twt(1000.0, checkshot=_checkshot(),
                                      anchor_twt=0.7)
        assert src == "checkshot"
        assert twt == pytest.approx(1.0)

    def test_anchor_when_no_checkshot(self):
        twt, src = resolve_marker_twt(1000.0, checkshot=None, anchor_twt=0.85)
        assert src == "horizon_anchor"
        assert twt == pytest.approx(0.85)

    def test_anchor_when_checkshot_out_of_range(self):
        # Depth beyond the checkshot's range → fall through to the anchor.
        twt, src = resolve_marker_twt(9999.0, checkshot=_checkshot(),
                                      anchor_twt=0.85)
        assert src == "horizon_anchor"

    def test_excluded_when_neither(self):
        twt, src = resolve_marker_twt(1000.0, checkshot=None, anchor_twt=None)
        assert twt is None and src == "excluded"

    def test_never_model_derived(self):
        # Structural guarantee: the helper takes no velocity model, so it cannot
        # compute depth_to_twt(depth). Its only TWT sources are checkshot + anchor.
        import inspect
        from section_tool.core.well_calibration import resolve_marker_twt
        params = set(inspect.signature(resolve_marker_twt).parameters)
        assert "model" not in params
        assert params == {"depth_m", "checkshot", "anchor_twt"}
        # And the executable body never references a model conversion.
        body = inspect.getsource(resolve_marker_twt).split('"""')[-1]
        assert "depth_to_twt" not in body


# ---------------------------------------------------------------------------
# build_well_markers — sourcing + exclusion report
# ---------------------------------------------------------------------------

class TestBuildWellMarkers:
    def _well(self):
        w = Well("F02-01", 0.0, 0.0, kb=0.0, td=3500.0)  # kb 0 → TVDSS = MD
        w.add_formation_top("A", 1000.0)
        w.add_formation_top("B", 2000.0)
        w.add_formation_top("Deep", 3400.0)  # beyond checkshot → needs anchor/excluded
        return w

    def test_uses_checkshot_for_in_range_markers(self):
        w = self._well()
        w.add_tdr(_checkshot())
        markers, report = build_well_markers(w)
        used = {u["name"]: u for u in report["used"]}
        assert used["A"]["source"] == "checkshot"
        assert used["A"]["twt_s"] == pytest.approx(1.0)
        # 'Deep' is beyond the checkshot range and has no anchor → excluded.
        assert any(e["name"] == "Deep" for e in report["excluded"])
        assert all(m.name != "Deep" for m in markers)

    def test_anchor_fallback_by_formation_match(self):
        w = self._well()  # no checkshot
        pick = HorizonPick([0.0, 100.0], [3400.0, 3400.0], name="Deep")
        pick.seismic_tied = True
        pick._twt_anchor = np.array([2.9, 2.9])
        markers, report = build_well_markers(w, picks=[pick])
        used = {u["name"]: u for u in report["used"]}
        assert used["Deep"]["source"] == "horizon_anchor"
        assert used["Deep"]["twt_s"] == pytest.approx(2.9)
        # A and B have neither checkshot nor anchor → excluded.
        assert {e["name"] for e in report["excluded"]} == {"A", "B"}


# ---------------------------------------------------------------------------
# Non-circular (Part A) routed through the helper
# ---------------------------------------------------------------------------

def test_helper_sourced_markers_are_non_circular():
    """Markers built from a checkshot carry independent TWT, so against a wrong
    bootstrap the residuals are non-trivial and calibration moves toward truth —
    the same guarantee as the Part A test, now via the unified helper."""
    truth_v0, truth_k = 2000.0, 0.6
    # Truth TDR → independent checkshot times.
    from section_tool.core.velocity_model import VelocityFunction
    fn = VelocityFunction("linear_v0k", v0=truth_v0, k=truth_k)
    depths = np.array([300.0, 800.0, 1400.0, 2100.0])
    twts = np.array([fn.depth_to_twt(float(z)) for z in depths])
    cs = TimeDepthRelation(depths, twts, kind="checkshot", depth_reference="TVDSS")

    w = Well("W", 0.0, 0.0, kb=0.0, td=2500.0)
    for i, z in enumerate(depths):
        w.add_formation_top(f"m{i}", float(z))
    w.add_tdr(cs)

    markers, report = build_well_markers(w)
    assert len(markers) == 4 and not report["excluded"]

    bootstrap = VelocityModel.average_vz(1500.0, 0.1)   # deliberately wrong
    pre = max(abs(r["twt_residual_s"]) for r in marker_residuals(bootstrap, markers))
    assert pre > 0.02, "independent markers must misfit a wrong model (not circular)"

    cal = calibrate_model(bootstrap, markers)
    assert cal.layers[0].function.v0 == pytest.approx(truth_v0, rel=0.05)
    post = max(abs(r["twt_residual_s"]) for r in marker_residuals(cal, markers))
    assert post < pre / 5
