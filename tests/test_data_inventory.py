"""Data inventory service — exhaustive rung truth table + builders."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from section_tool.core.data_inventory import (
    DataInventory, WellData, build_well_data, build_inventory,
    detect_sonic_curve, wells_in_corridor, ALWAYS_RUNGS)
from section_tool.core.wells import Well
from section_tool.core.tdr import TimeDepthRelation
from section_tool.core.surfaces import HorizonPick


def _wd(*, sonic=False, checkshot=False, sonic_tdr=False, tops=0):
    return WellData(
        well_name="W", well_uuid="u",
        has_sonic=sonic, sonic_curve=("DT:1" if sonic else None),
        has_checkshot=checkshot, has_sonic_tdr=sonic_tdr, n_tops=tops)


def _expected(sonic, checkshot, sonic_tdr, has_tops, anchors) -> set[str]:
    tie = checkshot or sonic_tdr
    rungs = set(ALWAYS_RUNGS)
    if tie:
        rungs.add("checkshot")
    if sonic and tie:
        rungs.add("sonic_checkshot")
    if sonic and anchors:
        rungs.add("sonic_anchors")
    if has_tops and anchors:
        rungs.add("marker_tied")
    return rungs


class TestRungTruthTable:
    @pytest.mark.parametrize(
        "sonic,checkshot,sonic_tdr,has_tops,anchors",
        list(itertools.product([False, True], repeat=5)),
    )
    def test_single_well_all_32_combos(self, sonic, checkshot, sonic_tdr,
                                       has_tops, anchors):
        inv = DataInventory(
            section_name="S",
            wells=(_wd(sonic=sonic, checkshot=checkshot, sonic_tdr=sonic_tdr,
                       tops=(3 if has_tops else 0)),),
            n_tied_horizons=(2 if anchors else 0),
        )
        assert set(inv.unlocked_rungs()) == _expected(
            sonic, checkshot, sonic_tdr, has_tops, anchors)

    def test_always_rungs_with_no_wells(self):
        inv = DataInventory(section_name="S")
        assert set(inv.unlocked_rungs()) == set(ALWAYS_RUNGS)

    def test_sonic_tdr_counts_as_tie(self):
        inv = DataInventory("S", wells=(_wd(sonic_tdr=True),))
        assert "checkshot" in inv.unlocked_rungs()


class TestMultiWellSemantics:
    def test_sonic_checkshot_needs_same_well_both(self):
        # Sonic in one well, checkshot in another → NO sonic_checkshot, but
        # checkshot IS available, and sonic_anchors if anchors present.
        inv = DataInventory(
            section_name="S",
            wells=(_wd(sonic=True), _wd(checkshot=True)),
            n_tied_horizons=1,
        )
        rungs = inv.unlocked_rungs()
        assert "checkshot" in rungs
        assert "sonic_anchors" in rungs
        assert "sonic_checkshot" not in rungs

    def test_one_well_with_both_unlocks_sonic_checkshot(self):
        inv = DataInventory("S", wells=(_wd(sonic=True, checkshot=True),))
        assert "sonic_checkshot" in inv.unlocked_rungs()

    def test_marker_tied_needs_tops_and_anchors(self):
        assert "marker_tied" not in DataInventory(
            "S", wells=(_wd(tops=5),)).unlocked_rungs()
        assert "marker_tied" in DataInventory(
            "S", wells=(_wd(tops=5),), n_tied_horizons=1).unlocked_rungs()


# ---------------------------------------------------------------------------
# Builders against real objects
# ---------------------------------------------------------------------------

def _checkshot(well):
    td = TimeDepthRelation([0.0, 1000.0, 2000.0], [0.0, 1.0, 2.0],
                           kind="checkshot")
    well.add_tdr(td)


def _tied_pick(name="H1"):
    hp = HorizonPick([0.0, 100.0], [200.0, 210.0], name=name)
    hp.seismic_tied = True
    hp._twt_anchor = np.array([0.18, 0.19])
    return hp


class TestBuilders:
    def test_detect_sonic_curve(self):
        from section_tool.core.wells import LogCurve
        w = Well("W", 0.0, 0.0)
        assert detect_sonic_curve(w) is None
        w.add_log(LogCurve("DT:1", "us/ft", [0, 1], [100, 110]))
        assert detect_sonic_curve(w) == "DT:1"

    def test_build_well_data_reflects_loaded(self):
        from section_tool.core.wells import LogCurve
        w = Well("F02-01", 0.0, 0.0, kb=30.0)
        w.add_log(LogCurve("DT:1", "us/ft", [0, 1], [100, 110]))
        _checkshot(w)
        w.add_formation_top("MFS11", 553.6)
        wd = build_well_data(w)
        assert wd.has_sonic and wd.sonic_curve == "DT:1"
        assert wd.has_checkshot and not wd.has_sonic_tdr
        assert wd.n_tops == 1
        assert wd.well_uuid == w.uuid

    def test_build_inventory_counts_tied_horizons(self):
        w = Well("F02-01", 0.0, 0.0)
        _checkshot(w)
        picks = [_tied_pick("H1"), _tied_pick("H2"),
                 HorizonPick([0.0, 1.0], [1.0, 2.0], name="untied")]
        inv = build_inventory(None, [w], picks)
        assert inv.n_tied_horizons == 2
        assert "checkshot" in inv.unlocked_rungs()

    def test_velocity_functions_present_reported_not_mapped(self):
        inv = build_inventory(None, [], [], velocity_functions_present=True)
        assert inv.velocity_functions_present is True
        assert "dix" not in inv.unlocked_rungs()
        assert set(inv.unlocked_rungs()) == set(ALWAYS_RUNGS)


class TestCorridor:
    def test_wells_in_corridor_none_section_passthrough(self):
        w = Well("W", 0.0, 0.0)
        assert wells_in_corridor(None, [w]) == [w]
