"""Comprehensive schema tests for the updated ProjectDatabase.

Covers:
- All tables exist after init
- Lithology library seeded with 15 standard entries
- Formation property inheritance (formation override > lithology > default)
- Measurements round-trip
- Well status/purpose fields
- CASCADE deletes
- well_sections projection metadata
- depth/elevation conversion helpers
- Full round-trip: write → close → reopen → verify
"""
from __future__ import annotations

import os
import sqlite3

import pytest

from section_tool.io.database import (
    ProjectDatabase,
    _LITHOLOGY_DEFAULTS,
    _PROPERTY_DEFAULTS,
    depth_to_elevation,
    elevation_to_depth,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    d = ProjectDatabase(str(tmp_path / "schema_test.sqlite"))
    yield d
    d.close()


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "roundtrip.sqlite")


# ---------------------------------------------------------------------------
# 1. All expected tables exist
# ---------------------------------------------------------------------------

EXPECTED_TABLES = {
    "project_meta", "sections", "horizons", "horizon_picks",
    "faults", "fault_picks", "wells", "well_tops", "well_logs",
    "lithologies", "formations", "polygons", "reference_lines",
    "seismic", "annotations", "velocity_model",
    "measurements", "well_sections",
}


def test_all_tables_exist(db):
    rows = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    tables = {r["name"] for r in rows}
    missing = EXPECTED_TABLES - tables
    assert not missing, f"Missing tables: {missing}"


# ---------------------------------------------------------------------------
# 2. Lithology library seeded
# ---------------------------------------------------------------------------

def test_lithology_library_seeded(db):
    rows = db.get_all_lithologies()
    assert len(rows) == len(_LITHOLOGY_DEFAULTS)


def test_lithology_names_present(db):
    names = {r["name"] for r in db.get_all_lithologies()}
    for lith in _LITHOLOGY_DEFAULTS:
        assert lith["name"] in names, f"Missing: {lith['name']}"


def test_sandstone_properties(db):
    sand = db.get_lithology("Sandstone")
    assert sand is not None
    assert sand["porosity_surface"] == pytest.approx(0.49)
    assert sand["compaction_coeff"] == pytest.approx(0.00027)
    assert sand["grain_density"] == pytest.approx(2650)
    assert sand["matrix_thermal_conductivity"] == pytest.approx(3.0)


def test_salt_zero_compaction(db):
    salt = db.get_lithology("Salt/Halite")
    assert salt is not None
    assert salt["compaction_coeff"] == pytest.approx(0.0)
    assert salt["matrix_thermal_conductivity"] == pytest.approx(6.0)


def test_lithology_not_reseeded_on_reopen(tmp_path):
    path = str(tmp_path / "seed_test.sqlite")
    d1 = ProjectDatabase(path)
    # Add a custom lithology
    d1.add_lithology("MyCustom", porosity_surface=0.15)
    d1.close()
    # Reopen — should NOT reseed (already 15 rows)
    d2 = ProjectDatabase(path)
    rows = d2.get_all_lithologies()
    names = {r["name"] for r in rows}
    assert "MyCustom" in names  # custom still present
    assert len(rows) == len(_LITHOLOGY_DEFAULTS) + 1
    d2.close()


def test_add_and_delete_custom_lithology(db):
    lid = db.add_lithology(
        "TestRock",
        porosity_surface=0.20,
        grain_density=2600.0,
        matrix_thermal_conductivity=2.2,
    )
    assert lid > 0
    row = db.get_lithology("TestRock")
    assert row is not None
    assert row["porosity_surface"] == pytest.approx(0.20)
    db.delete_lithology("TestRock")
    assert db.get_lithology("TestRock") is None


# ---------------------------------------------------------------------------
# 3. Formation property inheritance
# ---------------------------------------------------------------------------

def _insert_section(db):
    class _Sec:
        name = "S1"; nodes = __import__("numpy").array([[0,0],[1,0]])
        depth_domain = "md"; depth_units = "m"
        vertical_exaggeration = 1.0; crs_epsg = 32631
    db.upsert_section(_Sec())
    return db.conn.execute("SELECT id FROM sections WHERE name='S1'").fetchone()["id"]


def test_property_level1_formation_override(db):
    """Formation has an explicit value → use it."""
    lith = db.get_lithology("Shale")
    db.conn.execute(
        """INSERT INTO formations(name, primary_lithology_id, porosity_surface)
           VALUES(?,?,?)""",
        ("MyShale", lith["id"], 0.75)   # override: 0.75, shale default: 0.63
    )
    db.commit()
    fid = db.conn.execute(
        "SELECT id FROM formations WHERE name='MyShale'"
    ).fetchone()["id"]
    val = db.get_formation_property(fid, "porosity_surface")
    assert val == pytest.approx(0.75)


def test_property_level2_lithology_fallback(db):
    """Formation has NULL for the property → inherit from lithology."""
    lith = db.get_lithology("Sandstone")
    db.conn.execute(
        """INSERT INTO formations(name, primary_lithology_id)
           VALUES(?,?)""",
        ("MySandstone", lith["id"])
    )
    db.commit()
    fid = db.conn.execute(
        "SELECT id FROM formations WHERE name='MySandstone'"
    ).fetchone()["id"]
    val = db.get_formation_property(fid, "porosity_surface")
    assert val == pytest.approx(0.49)   # from Sandstone lithology


def test_property_level3_hardcoded_default(db):
    """Formation has no lithology ref → use hardcoded default."""
    db.conn.execute(
        "INSERT INTO formations(name) VALUES(?)", ("Bare",)
    )
    db.commit()
    fid = db.conn.execute(
        "SELECT id FROM formations WHERE name='Bare'"
    ).fetchone()["id"]
    val = db.get_formation_property(fid, "porosity_surface")
    assert val == pytest.approx(_PROPERTY_DEFAULTS["porosity_surface"])


def test_property_unknown_formation_returns_default(db):
    val = db.get_formation_property(99999, "grain_density")
    assert val == pytest.approx(_PROPERTY_DEFAULTS["grain_density"])


def test_formation_thermal_conductivity_inheritance(db):
    lith = db.get_lithology("Limestone")
    db.conn.execute(
        "INSERT INTO formations(name, primary_lithology_id) VALUES(?,?)",
        ("MyLimestone", lith["id"])
    )
    db.commit()
    fid = db.conn.execute(
        "SELECT id FROM formations WHERE name='MyLimestone'"
    ).fetchone()["id"]
    val = db.get_formation_property(fid, "matrix_thermal_conductivity")
    assert val == pytest.approx(2.5)    # from Limestone lithology


def test_formation_uuid_and_rank(db):
    import uuid
    uid = str(uuid.uuid4())
    db.conn.execute(
        "INSERT INTO formations(name, uuid, rank) VALUES(?,?,?)",
        ("GroupA", uid, "group")
    )
    db.commit()
    row = dict(db.conn.execute(
        "SELECT * FROM formations WHERE name='GroupA'"
    ).fetchone())
    assert row["uuid"] == uid
    assert row["rank"] == "group"


def test_formation_parent_hierarchy(db):
    db.conn.execute(
        "INSERT INTO formations(name, rank) VALUES(?,?)", ("TopGroup", "group")
    )
    db.commit()
    pid = db.conn.execute(
        "SELECT id FROM formations WHERE name='TopGroup'"
    ).fetchone()["id"]
    db.conn.execute(
        "INSERT INTO formations(name, rank, parent_id) VALUES(?,?,?)",
        ("SubFormation", "formation", pid)
    )
    db.commit()
    child = dict(db.conn.execute(
        "SELECT * FROM formations WHERE name='SubFormation'"
    ).fetchone())
    assert child["parent_id"] == pid


# ---------------------------------------------------------------------------
# 4. Measurements
# ---------------------------------------------------------------------------

class _FakeWell:
    name = "W1"; uwi = ""; x = 0.0; y = 0.0; kb = 0.0; uwi = ""
    original_x = None; original_y = None; original_crs_epsg = None
    log_names = []; _formation_tops = {}

    @property
    def formation_tops(self): return self._formation_tops

    class deviation:
        max_tvd = 3000.0


def _insert_well(db, name="W1") -> int:
    w = _FakeWell()
    w.name = name
    db.upsert_well(w)
    return db.conn.execute(
        "SELECT id FROM wells WHERE name=?", (name,)
    ).fetchone()["id"]


def test_add_and_get_measurement(db):
    wid = _insert_well(db)
    mid = db.add_measurement(
        wid, kind="vitrinite_ro", depth_md=1500.0, value=0.8,
        uncertainty=0.05, units="%Ro", sample_id="S-001", lab="LabX",
        method="random reflectance", note="good sample"
    )
    assert mid > 0
    rows = db.get_measurements(wid)
    assert len(rows) == 1
    m = rows[0]
    assert m["kind"] == "vitrinite_ro"
    assert m["depth_md"] == pytest.approx(1500.0)
    assert m["value"] == pytest.approx(0.8)
    assert m["uncertainty"] == pytest.approx(0.05)
    assert m["units"] == "%Ro"
    assert m["sample_id"] == "S-001"
    assert m["lab"] == "LabX"
    assert m["note"] == "good sample"


def test_measurements_filter_by_kind(db):
    wid = _insert_well(db)
    db.add_measurement(wid, kind="bht", depth_md=1000.0, value=85.0, units="°C")
    db.add_measurement(wid, kind="vitrinite_ro", depth_md=1500.0, value=0.7)
    db.add_measurement(wid, kind="bht", depth_md=2000.0, value=110.0, units="°C")
    bhts = db.get_measurements(wid, kind="bht")
    assert len(bhts) == 2
    ro = db.get_measurements(wid, kind="vitrinite_ro")
    assert len(ro) == 1


def test_measurement_all_kinds_accepted(db):
    wid = _insert_well(db)
    kinds = ["vitrinite_ro", "aft_age", "aft_length", "ahe_age", "zhe_age",
             "bht", "dst_temp", "fluid_inclusion", "clumped_isotope", "cai"]
    for i, k in enumerate(kinds):
        db.add_measurement(wid, kind=k, depth_md=float(i * 100), value=float(i))
    rows = db.get_measurements(wid)
    assert len(rows) == len(kinds)


def test_delete_measurement(db):
    wid = _insert_well(db)
    mid = db.add_measurement(wid, kind="bht", depth_md=1000.0, value=90.0)
    db.delete_measurement(mid)
    assert db.get_measurements(wid) == []


def test_measurements_sorted_by_depth(db):
    wid = _insert_well(db)
    for d in [3000, 500, 1500]:
        db.add_measurement(wid, kind="bht", depth_md=float(d), value=0.0)
    rows = db.get_measurements(wid)
    depths = [r["depth_md"] for r in rows]
    assert depths == sorted(depths)


# ---------------------------------------------------------------------------
# 5. Well status and purpose fields
# ---------------------------------------------------------------------------

def test_well_default_status_and_purpose(db):
    wid = _insert_well(db)
    row = dict(db.conn.execute("SELECT * FROM wells WHERE id=?", (wid,)).fetchone())
    assert row["status"] == "actual"
    assert row["purpose"] == "exploration"


def test_well_custom_status_and_purpose(db):
    class _PlannedWell(_FakeWell):
        name = "Planned-1"
        status = "planned"
        purpose = "production"

    db.upsert_well(_PlannedWell())
    wid = db.conn.execute(
        "SELECT id FROM wells WHERE name='Planned-1'"
    ).fetchone()["id"]
    row = dict(db.conn.execute("SELECT * FROM wells WHERE id=?", (wid,)).fetchone())
    assert row["status"] == "planned"
    assert row["purpose"] == "production"


def test_well_update_preserves_status(db):
    class _W(_FakeWell):
        name = "UpdateMe"; status = "hypothetical"; purpose = "geothermal"
    db.upsert_well(_W())
    _W.status = "actual"; _W.purpose = "injection"
    db.upsert_well(_W())
    wid = db.conn.execute(
        "SELECT id FROM wells WHERE name='UpdateMe'"
    ).fetchone()["id"]
    row = dict(db.conn.execute("SELECT * FROM wells WHERE id=?", (wid,)).fetchone())
    assert row["status"] == "actual"
    assert row["purpose"] == "injection"


# ---------------------------------------------------------------------------
# 6. CASCADE deletes
# ---------------------------------------------------------------------------

def test_cascade_delete_well_removes_tops(db):
    wid = _insert_well(db)
    db.conn.execute(
        "INSERT INTO well_tops(well_id, formation_name, md) VALUES(?,?,?)",
        (wid, "TopA", 1000.0)
    )
    db.commit()
    db.delete_well("W1")
    rows = db.conn.execute(
        "SELECT * FROM well_tops WHERE well_id=?", (wid,)
    ).fetchall()
    assert len(rows) == 0


def test_cascade_delete_well_removes_measurements(db):
    wid = _insert_well(db)
    db.add_measurement(wid, kind="bht", depth_md=1000.0, value=90.0)
    db.add_measurement(wid, kind="vitrinite_ro", depth_md=1500.0, value=0.7)
    db.delete_well("W1")
    rows = db.conn.execute(
        "SELECT * FROM measurements WHERE well_id=?", (wid,)
    ).fetchall()
    assert len(rows) == 0


def test_cascade_delete_well_removes_well_sections(db):
    wid = _insert_well(db)
    sid = _insert_section(db)
    db.upsert_well_section(wid, sid, 5000.0, 10.0)
    db.delete_well("W1")
    rows = db.conn.execute(
        "SELECT * FROM well_sections WHERE well_id=?", (wid,)
    ).fetchall()
    assert len(rows) == 0


def _insert_section(db) -> int:
    import numpy as np
    class _Sec:
        name = "S1"; nodes = np.array([[0,0],[10000,0]])
        depth_domain = "md"; depth_units = "m"
        vertical_exaggeration = 1.0; crs_epsg = 32631
    db.upsert_section(_Sec())
    return db.conn.execute("SELECT id FROM sections WHERE name='S1'").fetchone()["id"]


# ---------------------------------------------------------------------------
# 7. Well-section projection metadata
# ---------------------------------------------------------------------------

def test_upsert_and_get_well_section(db):
    wid = _insert_well(db)
    sid = _insert_section(db)
    rid = db.upsert_well_section(wid, sid, 5000.0, -25.3,
                                 nearest_segment=0, display_mode="near",
                                 projection_tolerance=1500.0)
    assert rid > 0
    rows = db.get_well_sections(wid)
    assert len(rows) == 1
    r = rows[0]
    assert r["distance_along"] == pytest.approx(5000.0)
    assert r["perpendicular_offset"] == pytest.approx(-25.3)
    assert r["nearest_segment"] == 0
    assert r["display_mode"] == "near"
    assert r["projection_tolerance"] == pytest.approx(1500.0)


def test_well_section_update(db):
    wid = _insert_well(db)
    sid = _insert_section(db)
    db.upsert_well_section(wid, sid, 5000.0, 0.0)
    db.upsert_well_section(wid, sid, 6000.0, 100.0)  # update
    rows = db.get_well_sections(wid)
    assert len(rows) == 1   # no duplicate
    assert rows[0]["distance_along"] == pytest.approx(6000.0)


def test_well_section_delete(db):
    wid = _insert_well(db)
    sid = _insert_section(db)
    db.upsert_well_section(wid, sid, 5000.0, 0.0)
    db.delete_well_section(wid, sid)
    assert db.get_well_sections(wid) == []


# ---------------------------------------------------------------------------
# 8. depth/elevation helpers
# ---------------------------------------------------------------------------

def test_depth_to_elevation():
    assert depth_to_elevation(1000.0) == pytest.approx(-1000.0)
    assert depth_to_elevation(0.0) == pytest.approx(0.0)
    assert depth_to_elevation(-100.0) == pytest.approx(100.0)


def test_elevation_to_depth():
    assert elevation_to_depth(-1000.0) == pytest.approx(1000.0)
    assert elevation_to_depth(0.0) == pytest.approx(0.0)
    assert elevation_to_depth(50.0) == pytest.approx(-50.0)


def test_round_trip_conversion():
    for d in [0.0, 500.0, 3150.0, -100.0]:
        assert elevation_to_depth(depth_to_elevation(d)) == pytest.approx(d)


def test_elevation_column_exists_in_picks(db):
    cols_h = {r[1] for r in db.conn.execute(
        "PRAGMA table_info(horizon_picks)"
    ).fetchall()}
    assert "elevation" in cols_h
    cols_f = {r[1] for r in db.conn.execute(
        "PRAGMA table_info(fault_picks)"
    ).fetchall()}
    assert "elevation" in cols_f


# ---------------------------------------------------------------------------
# 9. Round-trip: write → close → reopen → verify
# ---------------------------------------------------------------------------

def test_full_round_trip(db_path):
    # ---- Write ----
    d1 = ProjectDatabase(db_path)
    d1.set_project_settings("RoundTrip", 32631, "m", "md", 0.0, 5000.0)

    # Custom lithology
    d1.add_lithology("TestLith", porosity_surface=0.33, grain_density=2700.0)

    # Formation referencing custom lithology
    lith = d1.get_lithology("TestLith")
    d1.conn.execute(
        "INSERT INTO formations(name, primary_lithology_id) VALUES(?,?)",
        ("TestForm", lith["id"])
    )
    d1.commit()
    fid = d1.conn.execute(
        "SELECT id FROM formations WHERE name='TestForm'"
    ).fetchone()["id"]

    # Well
    class _W(_FakeWell):
        name = "RoundW"; status = "planned"; purpose = "geothermal"
    d1.upsert_well(_W())
    wid = d1.conn.execute(
        "SELECT id FROM wells WHERE name='RoundW'"
    ).fetchone()["id"]

    # Measurement
    d1.add_measurement(wid, kind="bht", depth_md=2000.0, value=120.0,
                       units="°C", lab="ThermoLab")
    d1.close()

    # ---- Reopen ----
    d2 = ProjectDatabase(db_path)

    assert d2.get_meta("name") == "RoundTrip"
    assert d2.get_meta("crs_epsg") == "32631"

    # Lithology survived
    l2 = d2.get_lithology("TestLith")
    assert l2 is not None
    assert l2["porosity_surface"] == pytest.approx(0.33)

    # Formation property chain
    val = d2.get_formation_property(fid, "porosity_surface")
    assert val == pytest.approx(0.33)   # from TestLith

    # Well
    row = dict(d2.conn.execute(
        "SELECT * FROM wells WHERE name='RoundW'"
    ).fetchone())
    assert row["status"] == "planned"
    assert row["purpose"] == "geothermal"

    # Measurement
    meas = d2.get_measurements(wid)
    assert len(meas) == 1
    assert meas[0]["value"] == pytest.approx(120.0)
    assert meas[0]["lab"] == "ThermoLab"

    d2.close()
