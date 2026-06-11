"""TDR + well-uuid persistence: upsert_well saves, get_all_wells reloads."""
from __future__ import annotations

import pytest

from section_tool.io.database import ProjectDatabase
from section_tool.core.wells import Well
from section_tool.core.tdr import TimeDepthRelation


@pytest.fixture
def tmp_db(tmp_path):
    db = ProjectDatabase(str(tmp_path / "test.sqlite"))
    yield db
    db.close()


def _well_with_tdr():
    w = Well("F02-01", 606554.0, 6080126.0, kb=30.0, td=3200.0)
    w.add_tdr(TimeDepthRelation(
        [0.0, 1000.0, 3120.0], [0.0, 1.0, 3.234],
        kind="checkshot", depth_reference="TVDSS", source="F02-01_TD.txt"))
    w.add_tdr(TimeDepthRelation(
        [0.0, 1500.0], [0.0, 1.5],
        kind="sonic_integrated", depth_reference="TVDSS"))
    w.add_formation_top("MFS11", 553.6)
    return w


def test_well_uuid_round_trips(tmp_db):
    w = _well_with_tdr()
    tmp_db.upsert_well(w)
    wells = tmp_db.get_all_wells()
    assert wells[0]["uuid"] == w.uuid


def test_tdrs_round_trip(tmp_db):
    w = _well_with_tdr()
    tmp_db.upsert_well(w)
    rows = tmp_db.get_all_wells()[0]["tdrs"]
    assert len(rows) == 2
    kinds = {r["kind"] for r in rows}
    assert kinds == {"checkshot", "sonic_integrated"}

    # Reconstruct and verify the data survived.
    cs = next(r for r in rows if r["kind"] == "checkshot")
    import json
    tdr = TimeDepthRelation.from_dict(json.loads(cs["data_json"]))
    assert tdr.depth_reference == "TVDSS"
    assert tdr.source == "F02-01_TD.txt"
    assert float(tdr.twt_at_depth(3120.0)) == pytest.approx(3.234)
    assert tdr.well_uuid == w.uuid


def test_reupsert_replaces_tdrs_not_duplicates(tmp_db):
    w = _well_with_tdr()
    tmp_db.upsert_well(w)
    tmp_db.upsert_well(w)
    assert len(tmp_db.get_all_wells()[0]["tdrs"]) == 2


def test_full_state_reload_via_app_state(tmp_path):
    """End-to-end: save a project with a TDR-bearing well, reopen, TDRs present."""
    from section_tool.app_state import AppState
    from section_tool.core.section import Section

    folder = str(tmp_path / "Proj")
    st = AppState()
    st.add_section(Section([(0, 0), (1000, 0)], name="L1"))
    w = _well_with_tdr()
    st.add_well(w)
    st.save_project_as(folder)

    st2 = AppState()
    st2.open_project(folder)
    well = st2.project.wells[0]
    assert well.uuid == w.uuid
    assert len(well.tdrs) == 2
    cs = well.primary_checkshot()
    assert cs is not None
    assert cs.depth_reference == "TVDSS"
    assert float(cs.twt_at_depth(3120.0)) == pytest.approx(3.234)
