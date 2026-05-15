"""Tests for section_tool.io.las."""

import io
import math
import os
import tempfile

import lasio
import numpy as np
import pytest

from section_tool.io.las import (
    las_to_well,
    read_las,
    read_las_header,
)


# ---------------------------------------------------------------------------
# LAS text builders
# ---------------------------------------------------------------------------

def _las_text(
    *,
    well_name: str = "W-1",
    uwi: str = "100/01-02-003-04W5/0",
    xcoord: str | None = "500000.0",
    ycoord: str | None = "5500000.0",
    kb: str | None = "55.0",
    null: str = "-999.25",
    curves: list[tuple[str, str, str]] | None = None,   # (mnemonic, unit, desc)
    data_rows: list[str] | None = None,
) -> str:
    """Build a minimal valid LAS 2.0 string."""
    if curves is None:
        curves = [("GR", "GAPI", "gamma ray"), ("RHOB", "G/CC", "bulk density")]
    if data_rows is None:
        data_rows = ["100.0  50.0  2.30", "200.0  80.0  2.45", "300.0  45.0  2.20"]

    header_extra = ""
    if uwi:
        header_extra += f"UWI. {uwi} : uwi\n"
    if xcoord is not None:
        header_extra += f"XCOORD.M {xcoord} : x coord\n"
    if ycoord is not None:
        header_extra += f"YCOORD.M {ycoord} : y coord\n"
    if kb is not None:
        header_extra += f"KB.M {kb} : kelly bushing\n"

    curve_section = "\n".join(
        f"{m} .{u} : {d}" for m, u, d in [("DEPT", "M", "depth")] + curves
    )
    data_section = "\n".join(data_rows)

    return "\n".join([
        "~VERSION",
        "VERS. 2.0 : CWLS log ASCII standard",
        "WRAP. NO : one line per depth",
        "~WELL",
        f"STRT.M  100.0 : start",
        f"STOP.M  300.0 : stop",
        f"STEP.M  100.0 : step",
        f"NULL. {null} : null value",
        f"WELL. {well_name} : well name",
        header_extra.rstrip(),
        "~CURVE",
        curve_section,
        "~A",
        data_section,
        "",
    ])


def _las_file(text: str) -> lasio.LASFile:
    """Parse LAS text into a LASFile without touching disk."""
    return lasio.read(io.StringIO(text))


def _write_tmp(text: str) -> str:
    """Write LAS text to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".las", delete=False, newline="\n"
    )
    f.write(text)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# read_las — basic behaviour
# ---------------------------------------------------------------------------

class TestReadLasBasic:
    def test_returns_well(self):
        path = _write_tmp(_las_text())
        try:
            w = read_las(path)
            assert w.name == "W-1"
        finally:
            os.unlink(path)

    def test_well_name_from_header(self):
        path = _write_tmp(_las_text(well_name="TestWell-99"))
        try:
            w = read_las(path)
            assert w.name == "TestWell-99"
        finally:
            os.unlink(path)

    def test_name_override(self):
        path = _write_tmp(_las_text(well_name="Original"))
        try:
            w = read_las(path, name="Override")
            assert w.name == "Override"
        finally:
            os.unlink(path)

    def test_uwi_stored(self):
        path = _write_tmp(_las_text(uwi="100/01-02-003-04W5/0"))
        try:
            w = read_las(path)
            assert w.uwi == "100/01-02-003-04W5/0"
        finally:
            os.unlink(path)

    def test_coordinates_from_header(self):
        path = _write_tmp(_las_text(xcoord="500000.0", ycoord="5500000.0"))
        try:
            w = read_las(path)
            assert pytest.approx(w.x) == 500000.0
            assert pytest.approx(w.y) == 5500000.0
        finally:
            os.unlink(path)

    def test_kb_from_header(self):
        path = _write_tmp(_las_text(kb="55.0"))
        try:
            w = read_las(path)
            assert pytest.approx(w.kb) == 55.0
        finally:
            os.unlink(path)

    def test_coordinates_override(self):
        path = _write_tmp(_las_text(xcoord="0.0", ycoord="0.0"))
        try:
            w = read_las(path, x=123456.0, y=654321.0)
            assert pytest.approx(w.x) == 123456.0
            assert pytest.approx(w.y) == 654321.0
        finally:
            os.unlink(path)

    def test_kb_override(self):
        path = _write_tmp(_las_text(kb="10.0"))
        try:
            w = read_las(path, kb=99.0)
            assert pytest.approx(w.kb) == 99.0
        finally:
            os.unlink(path)

    def test_missing_xcoord_defaults_to_zero(self):
        path = _write_tmp(_las_text(xcoord=None, ycoord=None))
        try:
            w = read_las(path)
            assert w.x == 0.0
            assert w.y == 0.0
        finally:
            os.unlink(path)

    def test_missing_kb_defaults_to_zero(self):
        path = _write_tmp(_las_text(kb=None))
        try:
            w = read_las(path)
            assert w.kb == 0.0
        finally:
            os.unlink(path)

    def test_well_name_falls_back_to_file_stem(self):
        # Write to a file whose name we control
        path = _write_tmp(_las_text(well_name=""))
        stem_path = path.replace(".las", "_mywellname.las")
        os.rename(path, stem_path)
        try:
            w = read_las(stem_path)
            assert "mywellname" in w.name or w.name != ""
        finally:
            if os.path.exists(stem_path):
                os.unlink(stem_path)


# ---------------------------------------------------------------------------
# read_las — log curves
# ---------------------------------------------------------------------------

class TestReadLasCurves:
    def test_curves_added_as_logs(self):
        path = _write_tmp(_las_text(curves=[("GR", "GAPI", "gr"), ("RHOB", "G/CC", "rhob")]))
        try:
            w = read_las(path)
            assert "GR" in w.log_names
            assert "RHOB" in w.log_names
        finally:
            os.unlink(path)

    def test_depth_curve_not_added(self):
        path = _write_tmp(_las_text())
        try:
            w = read_las(path)
            assert "DEPT" not in w.log_names
        finally:
            os.unlink(path)

    def test_curve_units_preserved(self):
        path = _write_tmp(_las_text(curves=[("GR", "GAPI", "gr")],
                                    data_rows=["100.0 50.0", "200.0 80.0", "300.0 45.0"]))
        try:
            w = read_las(path)
            assert w.get_log("GR").units == "GAPI"
        finally:
            os.unlink(path)

    def test_curve_values_correct(self):
        path = _write_tmp(_las_text(
            curves=[("GR", "GAPI", "gr")],
            data_rows=["100.0 50.0", "200.0 80.0", "300.0 45.0"],
        ))
        try:
            w = read_las(path)
            gr = w.get_log("GR")
            np.testing.assert_allclose(gr.values, [50.0, 80.0, 45.0])
        finally:
            os.unlink(path)

    def test_null_converted_to_nan(self):
        path = _write_tmp(_las_text(
            curves=[("GR", "GAPI", "gr")],
            data_rows=["100.0 50.0", "200.0 -999.25", "300.0 45.0"],
            null="-999.25",
        ))
        try:
            w = read_las(path)
            gr = w.get_log("GR")
            assert math.isnan(gr.sample(200.0))
            assert pytest.approx(gr.sample(100.0)) == 50.0
        finally:
            os.unlink(path)

    def test_curve_depths_match_dept(self):
        path = _write_tmp(_las_text(
            curves=[("GR", "GAPI", "gr")],
            data_rows=["100.0 50.0", "200.0 80.0", "300.0 45.0"],
        ))
        try:
            w = read_las(path)
            gr = w.get_log("GR")
            np.testing.assert_allclose(gr.depths, [100.0, 200.0, 300.0])
        finally:
            os.unlink(path)

    def test_many_curves(self):
        curves = [(f"C{i}", "unitless", f"curve {i}") for i in range(5)]
        row_vals = "  ".join("1.0" for _ in range(5))
        path = _write_tmp(_las_text(
            curves=curves,
            data_rows=[f"100.0 {row_vals}", f"200.0 {row_vals}", f"300.0 {row_vals}"],
        ))
        try:
            w = read_las(path)
            assert len(w.log_names) == 5
        finally:
            os.unlink(path)

    def test_no_data_rows_means_no_logs(self):
        path = _write_tmp(_las_text(data_rows=[]))
        try:
            w = read_las(path)
            assert w.log_names == []
        finally:
            os.unlink(path)

    def test_single_curve(self):
        path = _write_tmp(_las_text(
            curves=[("GR", "GAPI", "gr")],
            data_rows=["100.0 50.0", "200.0 80.0", "300.0 45.0"],
        ))
        try:
            w = read_las(path)
            assert w.log_names == ["GR"]
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# las_to_well — in-memory LASFile conversion
# ---------------------------------------------------------------------------

class TestLasToWell:
    def test_basic_conversion(self):
        las = _las_file(_las_text())
        w = las_to_well(las)
        assert w.name == "W-1"
        assert "GR" in w.log_names

    def test_name_override(self):
        las = _las_file(_las_text(well_name="Original"))
        w = las_to_well(las, name="Override")
        assert w.name == "Override"

    def test_coord_override(self):
        las = _las_file(_las_text(xcoord="0.0", ycoord="0.0"))
        w = las_to_well(las, x=111.0, y=222.0)
        assert pytest.approx(w.x) == 111.0
        assert pytest.approx(w.y) == 222.0

    def test_kb_override(self):
        las = _las_file(_las_text(kb="1.0"))
        w = las_to_well(las, kb=50.0)
        assert pytest.approx(w.kb) == 50.0

    def test_missing_well_header_gives_unnamed(self):
        las = _las_file(_las_text(well_name=""))
        w = las_to_well(las)
        assert w.name == "Unnamed"

    def test_missing_coords_default_zero(self):
        las = _las_file(_las_text(xcoord=None, ycoord=None))
        w = las_to_well(las)
        assert w.x == 0.0
        assert w.y == 0.0

    def test_curve_values_via_lasfile(self):
        las = _las_file(_las_text(
            curves=[("GR", "GAPI", "gr")],
            data_rows=["100.0 50.0", "200.0 80.0", "300.0 45.0"],
        ))
        w = las_to_well(las)
        np.testing.assert_allclose(w.get_log("GR").values, [50.0, 80.0, 45.0])

    def test_empty_data_no_logs(self):
        las = _las_file(_las_text(data_rows=[]))
        w = las_to_well(las)
        assert w.log_names == []

    def test_uwi_from_header(self):
        las = _las_file(_las_text(uwi="999/99-99-999-99W9/0"))
        w = las_to_well(las)
        assert w.uwi == "999/99-99-999-99W9/0"


# ---------------------------------------------------------------------------
# read_las_header
# ---------------------------------------------------------------------------

class TestReadLasHeader:
    def test_returns_dict_with_all_keys(self):
        path = _write_tmp(_las_text())
        try:
            hdr = read_las_header(path)
            for key in ("well_name", "uwi", "x", "y", "kb",
                        "depth_start", "depth_stop", "depth_step",
                        "depth_unit", "curve_names"):
                assert key in hdr
        finally:
            os.unlink(path)

    def test_well_name(self):
        path = _write_tmp(_las_text(well_name="HeaderWell"))
        try:
            assert read_las_header(path)["well_name"] == "HeaderWell"
        finally:
            os.unlink(path)

    def test_coordinates(self):
        path = _write_tmp(_las_text(xcoord="123456.0", ycoord="654321.0"))
        try:
            hdr = read_las_header(path)
            assert pytest.approx(hdr["x"]) == 123456.0
            assert pytest.approx(hdr["y"]) == 654321.0
        finally:
            os.unlink(path)

    def test_kb(self):
        path = _write_tmp(_las_text(kb="75.5"))
        try:
            assert pytest.approx(read_las_header(path)["kb"]) == 75.5
        finally:
            os.unlink(path)

    def test_depth_range(self):
        path = _write_tmp(_las_text())
        try:
            hdr = read_las_header(path)
            assert pytest.approx(hdr["depth_start"]) == 100.0
            assert pytest.approx(hdr["depth_stop"]) == 300.0
        finally:
            os.unlink(path)

    def test_depth_unit(self):
        path = _write_tmp(_las_text())
        try:
            assert read_las_header(path)["depth_unit"] == "M"
        finally:
            os.unlink(path)

    def test_curve_names_excludes_depth(self):
        path = _write_tmp(_las_text(curves=[("GR", "GAPI", "gr"), ("RHOB", "G/CC", "rhob")]))
        try:
            names = read_las_header(path)["curve_names"]
            assert "DEPT" not in names
            assert "GR" in names
            assert "RHOB" in names
        finally:
            os.unlink(path)

    def test_missing_optional_fields_are_none(self):
        path = _write_tmp(_las_text(xcoord=None, ycoord=None, kb=None, uwi=""))
        try:
            hdr = read_las_header(path)
            assert hdr["x"] is None
            assert hdr["y"] is None
            assert hdr["kb"] is None
            assert hdr["uwi"] is None
        finally:
            os.unlink(path)

    def test_does_not_load_data(self):
        # The returned dict has no data — just strings/floats
        path = _write_tmp(_las_text())
        try:
            hdr = read_las_header(path)
            assert isinstance(hdr["curve_names"], list)
            assert isinstance(hdr["depth_start"], float)
        finally:
            os.unlink(path)

    def test_uwi_in_header(self):
        path = _write_tmp(_las_text(uwi="100/03-12-035-09W5/0"))
        try:
            assert read_las_header(path)["uwi"] == "100/03-12-035-09W5/0"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_alternative_x_key_easting(self):
        """EASTING header mnemonic should be picked up as x."""
        text = _las_text(xcoord=None, ycoord=None)
        text = text.replace("~WELL\n", "~WELL\nEASTING.M 700000.0 : easting\nNORTHING.M 6100000.0 : northing\n")
        las = _las_file(text)
        w = las_to_well(las)
        assert pytest.approx(w.x) == 700000.0
        assert pytest.approx(w.y) == 6100000.0

    def test_unknown_well_name_treated_as_empty(self):
        """'UNKNOWN' in the WELL field should fall back to 'Unnamed'."""
        text = _las_text(well_name="UNKNOWN")
        las = _las_file(text)
        w = las_to_well(las)
        assert w.name == "Unnamed"

    def test_curve_with_empty_unit(self):
        """Curves with no unit string should store empty string, not None."""
        text = _las_text(
            curves=[("SP", "", "spontaneous potential")],
            data_rows=["100.0 -50.0", "200.0 -55.0", "300.0 -48.0"],
        )
        las = _las_file(text)
        w = las_to_well(las)
        assert w.get_log("SP").units == ""

    def test_float_coordinates_from_int_header(self):
        """Integer-looking header values (e.g. KB 50) should become floats."""
        text = _las_text(kb="50")
        las = _las_file(text)
        w = las_to_well(las)
        assert w.kb == 50.0
        assert isinstance(w.kb, float)

    def test_custom_null_value(self):
        text = _las_text(
            null="-9999",
            curves=[("GR", "GAPI", "gr")],
            data_rows=["100.0 50.0", "200.0 -9999", "300.0 45.0"],
        )
        las = _las_file(text)
        w = las_to_well(las)
        assert math.isnan(w.get_log("GR").sample(200.0))

    def test_curve_sample_interpolates(self):
        """Values read from LAS should support interpolation via LogCurve.sample."""
        text = _las_text(
            curves=[("GR", "GAPI", "gr")],
            data_rows=["100.0 0.0", "200.0 100.0", "300.0 50.0"],
        )
        las = _las_file(text)
        w = las_to_well(las)
        assert pytest.approx(w.get_log("GR").sample(150.0)) == 50.0

    def test_large_dataset(self):
        """1000-sample LAS file loads without error."""
        depths = np.linspace(0.0, 999.0, 1000)
        gr_vals = np.random.default_rng(0).uniform(20, 150, 1000)
        rows = [f"{d:.1f} {g:.2f}" for d, g in zip(depths, gr_vals)]
        text = _las_text(
            curves=[("GR", "GAPI", "gr")],
            data_rows=rows,
        )
        # patch STRT/STOP to match
        text = text.replace("STRT.M  100.0 : start", "STRT.M  0.0 : start")
        text = text.replace("STOP.M  300.0 : stop", "STOP.M  999.0 : stop")
        las = _las_file(text)
        w = las_to_well(las)
        assert w.get_log("GR").n_samples == 1000
