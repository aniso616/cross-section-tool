"""Thermal Step 1: measurement model + validation + CSV import, and the save/
reopen round-trip that proves measurements are real project data."""
from __future__ import annotations

import pytest

from section_tool.core.measurements import (
    Measurement, validate_measurement, parse_measurements_csv, default_units)
from section_tool.app_state import AppState
from section_tool.core.wells import Well


# ── validation ──────────────────────────────────────────────────────────────

def test_validate_plausibility_ranges():
    assert validate_measurement("vitrinite_ro", 0.7) is None
    assert validate_measurement("vitrinite_ro", 5.0) is not None     # > 4.0 %Ro
    assert validate_measurement("vitrinite_ro", 0.1) is not None     # < 0.2 %Ro
    assert validate_measurement("aft_age", 0.0) is not None          # must be > 0 Ma
    assert validate_measurement("aft_age", 50.0) is None
    assert validate_measurement("bht", 90.0) is None
    assert validate_measurement("bht", 5000.0) is not None           # out of °C range
    assert validate_measurement("nope", 1.0) is not None             # unknown type
    assert validate_measurement("aft_age", "x") is not None          # non-numeric


# ── CSV import ──────────────────────────────────────────────────────────────

def test_csv_parses_two_columns_header_skipped():
    out, errors = parse_measurements_csv("depth_m, ro\n1000, 0.5\n2000, 0.8\n",
                                         "vitrinite_ro")
    assert [m.depth_m for m in out] == [1000.0, 2000.0]
    assert [m.value for m in out] == [0.5, 0.8]
    assert out[0].units == default_units("vitrinite_ro") and errors == []


def test_csv_out_of_range_rows_skipped_with_errors():
    out, errors = parse_measurements_csv("500, 0.6\n1000, 9.0\n", "vitrinite_ro")
    assert len(out) == 1 and len(errors) == 1 and "range" in errors[0]


def test_csv_wrong_format_raises_clear_error():
    with pytest.raises(ValueError, match="depth_m, value"):
        parse_measurements_csv("not a csv\njust prose here\n", "vitrinite_ro")


def test_from_db_row_maps_columns():
    row = {"depth_md": 1500.0, "kind": "aft_age", "value": 80.0, "uncertainty": 5.0,
           "units": "Ma", "lab": "UTChron", "note": "n", "sample_id": "S1", "uuid": "u1"}
    m = Measurement.from_db_row(row, well_uuid="w1")
    assert m.depth_m == 1500.0 and m.measurement_type == "aft_age"
    assert m.source == "UTChron" and m.notes == "n"
    assert m.uuid == "u1" and m.well_uuid == "w1"


# ── save / reopen round-trip (real project data) ────────────────────────────

def test_measurements_survive_save_reopen(tmp_path):
    folder = str(tmp_path / "proj")
    src = AppState()
    src.new_project(name="M", crs_epsg=32631, folder_path=folder)
    well = Well("W1", 100.0, 200.0, td=3000.0)
    well.add_measurement(Measurement(depth_m=1500.0, measurement_type="aft_age",
                                     value=80.0, uncertainty=5.0, units="Ma",
                                     source="lab"))
    well.add_measurement(Measurement(depth_m=1000.0, measurement_type="vitrinite_ro",
                                     value=0.6, units="%Ro"))
    src.add_well(well)
    src.save_project()

    dst = AppState()
    dst.open_project(folder)
    wells = dst.project.wells
    assert len(wells) == 1
    ms = sorted(wells[0].measurements, key=lambda m: m.depth_m)
    assert len(ms) == 2
    assert ms[0].measurement_type == "vitrinite_ro" and ms[0].value == pytest.approx(0.6)
    assert ms[1].measurement_type == "aft_age"
    assert ms[1].uncertainty == pytest.approx(5.0) and ms[1].units == "Ma"
    assert ms[1].well_uuid == wells[0].uuid            # stamped to the well on load
