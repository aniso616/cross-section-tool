"""DepthStretchController — inventory permutations, recommendation, build, round-trip,
upgrade tag, caption single source."""
from __future__ import annotations

import numpy as np
import pytest

from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick
from section_tool.core.wells import Well, LogCurve
from section_tool.core.tdr import TimeDepthRelation
from section_tool.core.conversion import set_anchors, build_bulk
from section_tool.core.velocity_model import conversion_caption
from section_tool.core.depth_stretch_controller import DepthStretchController
from section_tool.core.grounded_velocity import _US_FT_TO_S_PER_M


def _empty_state():
    st = AppState()
    st.add_section(Section([(0, 0), (3000, 0)], name="L1"))
    return st


def _tied_horizon(name, depth, formation_below):
    hp = HorizonPick(np.array([0.0, 1500.0]), np.array([depth, depth]),
                     name=name, section_names=np.array(["L1", "L1"], dtype=object))
    hp.formation_below = formation_below
    set_anchors(hp, build_bulk(2000.0))
    return hp


def _sonic_well(name="F02-01", kb=0.0):
    w = Well(name, 0.0, 0.0, kb=kb, td=3200.0)
    md = np.arange(0.0, 3001.0, 5.0)
    slow = (1.0 / 2000.0) / _US_FT_TO_S_PER_M
    w.add_log(LogCurve("DT:2", "us/ft", md, np.full_like(md, slow)))
    return w


def _checkshot():
    return TimeDepthRelation([0.0, 1000.0, 2000.0, 3000.0],
                             [0.0, 0.95, 1.90, 2.85],
                             kind="checkshot", depth_reference="TVDSS")


# ---------------------------------------------------------------------------
# Inventory permutations → recommendation
# ---------------------------------------------------------------------------

class TestRecommendationPermutations:
    def test_empty_recommends_bulk(self):
        c = DepthStretchController(_empty_state())
        assert c.recommended_rung() == "bulk"
        rungs = {s.key: s for s in c.rung_specs()}
        # Everything grounded is locked; bulk + average_vz unlocked.
        assert rungs["bulk"].unlocked and rungs["average_vz"].unlocked
        assert not rungs["checkshot"].unlocked
        assert not rungs["layered"].unlocked
        assert rungs["bulk"].recommended

    def test_checkshot_only_recommends_checkshot(self):
        st = _empty_state()
        w = _sonic_well()
        w._logs.clear()                 # strip sonic → checkshot-only well
        w.add_tdr(_checkshot())
        st.add_well(w)
        c = DepthStretchController(st)
        assert c.recommended_rung() == "checkshot"
        assert {s.key for s in c.rung_specs() if s.unlocked} >= {"checkshot", "bulk"}

    def test_sonic_only_no_tie_recommends_bulk(self):
        st = _empty_state()
        st.add_well(_sonic_well())       # sonic, but no checkshot, no anchors
        c = DepthStretchController(st)
        # Sonic alone with no tie/anchors unlocks nothing grounded → bulk.
        assert c.recommended_rung() == "bulk"

    def test_full_f3_recommends_sonic_checkshot(self):
        st = _empty_state()
        w = _sonic_well()
        w.add_tdr(_checkshot())
        w.add_formation_top("FS8", 800.0)
        st.add_well(w)
        st.project.horizon_picks.append(_tied_horizon("H1", 800.0, "FS8"))
        c = DepthStretchController(st)
        assert c.recommended_rung() == "sonic_checkshot"
        recs = [s for s in c.rung_specs() if s.recommended]
        assert len(recs) == 1 and recs[0].key == "sonic_checkshot"
        assert "sonic" in recs[0].one_liner.lower() and "F02-01" in recs[0].one_liner


# ---------------------------------------------------------------------------
# Locked-rung doors: reason + import action
# ---------------------------------------------------------------------------

class TestLockedDoors:
    def test_locked_checkshot_offers_import(self):
        c = DepthStretchController(_empty_state())
        spec = next(s for s in c.rung_specs() if s.key == "checkshot")
        assert not spec.unlocked
        assert spec.reason == "no checkshot loaded"
        assert spec.import_action == "checkshot"
        assert "Import" in spec.import_label

    def test_locked_marker_tied_reason_progression(self):
        st = _empty_state()
        c = DepthStretchController(st)
        spec = next(s for s in c.rung_specs() if s.key == "marker_tied")
        assert spec.reason == "no formation tops loaded"
        assert spec.import_action == "markers"


# ---------------------------------------------------------------------------
# Build + Apply round-trip
# ---------------------------------------------------------------------------

class TestApplyRoundTrip:
    def _full_state(self):
        st = _empty_state()
        w = _sonic_well()
        w.add_tdr(_checkshot())
        st.add_well(w)
        st.project.horizon_picks.append(_tied_horizon("H1", 800.0, "FS8"))
        return st

    @pytest.mark.parametrize("rung", ["bulk", "average_vz", "checkshot",
                                      "sonic_checkshot"])
    def test_apply_then_applied_rung_round_trips(self, rung):
        c = DepthStretchController(self._full_state())
        model = c.apply(rung)
        assert not model.is_empty
        assert c.applied_rung() == rung

    def test_apply_checkshot_reproduces_knots(self):
        c = DepthStretchController(self._full_state())
        c.apply("checkshot")
        m = c.project.velocity_model
        assert m.twt_to_depth(1.90) == pytest.approx(2000.0, abs=2.0)

    def test_caption_is_single_source(self):
        c = DepthStretchController(self._full_state())
        m = c.apply("checkshot")
        assert c.caption() == conversion_caption(m)
        assert "checkshot-tied" in c.caption()


# ---------------------------------------------------------------------------
# Upgrade tag: non-blocking, never restretches
# ---------------------------------------------------------------------------

class TestUpgradeTag:
    def test_no_upgrade_when_applied_is_recommended(self):
        st = _empty_state()
        w = _sonic_well(); w.add_tdr(_checkshot()); st.add_well(w)
        st.project.horizon_picks.append(_tied_horizon("H1", 800.0, "FS8"))
        c = DepthStretchController(st)
        c.apply(c.recommended_rung())
        assert c.upgrade_rung() is None

    def test_upgrade_appears_when_better_data_arrives(self):
        st = _empty_state()
        c = DepthStretchController(st)
        c.apply("bulk")                       # empty project, bulk applied
        assert c.upgrade_rung() is None
        # A checkshot well is imported later → checkshot now beats bulk.
        depths_model = c.project.velocity_model
        w = _sonic_well(); w._logs.clear(); w.add_tdr(_checkshot())
        st.add_well(w)
        assert c.upgrade_rung() == "checkshot"
        # Tag never mutates the applied model / never restretches.
        assert c.project.velocity_model is depths_model

    def test_keep_current_clears_tag(self):
        st = _empty_state()
        c = DepthStretchController(st)
        c.apply("bulk")
        w = _sonic_well(); w._logs.clear(); w.add_tdr(_checkshot())
        st.add_well(w)
        assert c.upgrade_rung() == "checkshot"
        c.keep_current()
        assert c.upgrade_rung() is None       # acknowledged → no nag
