"""Tests for LAS coordinate parsing, CRS auto-detection, and header extraction."""
from __future__ import annotations

import re
import pytest

from section_tool.io.las import (
    _parse_loc_xy,
    _X_KEYS, _Y_KEYS, _KB_KEYS,
    _LOC_X_RE, _LOC_Y_RE,
    extract_header_full,
)


# ---------------------------------------------------------------------------
# LOC field regex parsing
# ---------------------------------------------------------------------------

class _FakeLASField:
    def __init__(self, value, unit="", descr=""):
        self.value = value
        self.unit = unit
        self.descr = descr


class _FakeLAS:
    """Minimal stand-in for lasio.LASFile for testing _parse_loc_xy."""

    def __init__(self, well_dict: dict):
        self._well = {k: _FakeLASField(v) for k, v in well_dict.items()}

    @property
    def well(self):
        return self._well


def test_loc_regex_standard_format():
    x_match = _LOC_X_RE.search("X = 606554.0000 Y = 6080126.0000")
    y_match = _LOC_Y_RE.search("X = 606554.0000 Y = 6080126.0000")
    assert x_match is not None
    assert y_match is not None
    assert float(x_match.group(1)) == pytest.approx(606554.0)
    assert float(y_match.group(1)) == pytest.approx(6080126.0)


def test_loc_regex_colon_separator():
    loc = "X: 123456.5 Y: 7654321.1"
    assert _LOC_X_RE.search(loc) is not None
    assert _LOC_Y_RE.search(loc) is not None
    assert float(_LOC_X_RE.search(loc).group(1)) == pytest.approx(123456.5)
    assert float(_LOC_Y_RE.search(loc).group(1)) == pytest.approx(7654321.1)


def test_loc_regex_no_spaces():
    loc = "X=500000 Y=5000000"
    assert _LOC_X_RE.search(loc) is not None
    assert float(_LOC_X_RE.search(loc).group(1)) == pytest.approx(500000.0)


def test_loc_regex_negative():
    loc = "X = -73.9857 Y = 40.7484"
    assert _LOC_X_RE.search(loc) is not None
    assert float(_LOC_X_RE.search(loc).group(1)) == pytest.approx(-73.9857)


def test_parse_loc_xy_returns_both():
    las = _FakeLAS({"LOC": "X = 606554.0000 Y = 6080126.0000"})
    x, y = _parse_loc_xy(las)
    assert x == pytest.approx(606554.0)
    assert y == pytest.approx(6080126.0)


def test_parse_loc_xy_empty_field():
    las = _FakeLAS({"LOC": ""})
    x, y = _parse_loc_xy(las)
    assert x is None
    assert y is None


def test_parse_loc_xy_missing_field():
    las = _FakeLAS({})
    x, y = _parse_loc_xy(las)
    assert x is None
    assert y is None


def test_parse_loc_xy_loca_fallback():
    las = _FakeLAS({"LOCA": "X = 1234.0 Y = 5678.0"})
    x, y = _parse_loc_xy(las)
    assert x == pytest.approx(1234.0)
    assert y == pytest.approx(5678.0)


# ---------------------------------------------------------------------------
# _X_KEYS / _Y_KEYS / _KB_KEYS coverage
# ---------------------------------------------------------------------------

def test_x_keys_include_required():
    for key in ("XCOORD", "X", "EASTING", "LONG"):
        assert key in _X_KEYS, f"Missing key: {key}"


def test_y_keys_include_required():
    for key in ("YCOORD", "Y", "NORTHING", "LAT"):
        assert key in _Y_KEYS, f"Missing key: {key}"


def test_kb_keys_include_ekb():
    assert "EKB" in _KB_KEYS
    assert "EKBR" in _KB_KEYS


# ---------------------------------------------------------------------------
# CRS auto-suggest (pure logic — no Qt)
# ---------------------------------------------------------------------------

from section_tool.views.las_import_dialog import _suggest_crs


def test_suggest_crs_geographic():
    suggestion = _suggest_crs(-73.98, 40.74)
    assert "geographic" in suggestion.lower() or "lon" in suggestion.lower()


def test_suggest_crs_utm():
    suggestion = _suggest_crs(606554.0, 6080126.0)
    assert "utm" in suggestion.lower() or "projected" in suggestion.lower()


def test_suggest_crs_no_suggestion():
    suggestion = _suggest_crs(0.0, 0.0)
    assert suggestion == ""


# ---------------------------------------------------------------------------
# extract_header_full — requires lasio
# ---------------------------------------------------------------------------

def test_extract_header_full_f02():
    """Integration test against the real F02-01 LAS file (skipped if absent)."""
    pytest.importorskip("lasio")
    import os
    path = r"J:\data\F3_Demo_2023\Rawdata\Well_data\F02-01_logs.las"
    if not os.path.exists(path):
        pytest.skip("F02-01 LAS file not available")

    import lasio
    las = lasio.read(path)
    h = extract_header_full(las)

    assert h["well_name"] == "F02-01"
    assert h["x"] == pytest.approx(606554.0, abs=1.0)
    assert h["y"] == pytest.approx(6080126.0, abs=1.0)
    assert h["x_source"] == "parsed from LOC field"
    assert h["y_source"] == "parsed from LOC field"
    assert h["kb"] == pytest.approx(30.0, abs=0.1)
    assert h["kb_source"] is not None and "EKB" in h["kb_source"]
    assert h["depth_start"] == pytest.approx(30.0)
    assert h["depth_stop"] == pytest.approx(3150.0)
