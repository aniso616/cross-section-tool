"""Tops import from the F3 markers shape (index, MD, name-with-spaces)."""
from __future__ import annotations

import os

import pytest

from section_tool.core.wells import Well
from section_tool.io.tops_io import parse_markers, load_markers_into_well

_MARKERS = """\
30\tSeasurface
553.6\tMFS11
1025.42\tTruncation 1
1134.73\tFS 3
1285.09\tNMRF (Mid_Mio_Unc)
3150\tSLCL
99
"""


@pytest.fixture
def markers_file(tmp_path):
    p = tmp_path / "F02-01_markers.txt"
    p.write_text(_MARKERS, encoding="utf-8")
    return str(p)


def test_parse_count_and_order(markers_file):
    pairs = parse_markers(markers_file)
    assert len(pairs) == 6   # trailing nameless row "99" skipped
    assert pairs[0] == ("Seasurface", 30.0)
    assert pairs[-1] == ("SLCL", 3150.0)


def test_names_with_spaces_preserved(markers_file):
    names = dict(parse_markers(markers_file))
    assert "FS 3" in names
    assert "Truncation 1" in names
    assert "NMRF (Mid_Mio_Unc)" in names


def test_md_spot_checks(markers_file):
    names = dict(parse_markers(markers_file))
    assert names["MFS11"] == pytest.approx(553.6)
    assert names["NMRF (Mid_Mio_Unc)"] == pytest.approx(1285.09)


def test_load_into_well(markers_file):
    w = Well("F02-01", 0.0, 0.0, kb=30.0)
    n = load_markers_into_well(markers_file, w)
    assert n == 6
    assert len(w.formation_tops) == 6
    assert w.formation_tops["SLCL"] == pytest.approx(3150.0)


def test_reimport_idempotent(markers_file):
    w = Well("F02-01", 0.0, 0.0)
    load_markers_into_well(markers_file, w)
    load_markers_into_well(markers_file, w)
    assert len(w.formation_tops) == 6   # overwrite, not duplicate


_REAL_MARKERS = r"J:\data\F3_Demo_2023\Rawdata\Well_data\F02-01_markers.txt"


@pytest.mark.skipif(not os.path.exists(_REAL_MARKERS), reason="F3 data drive not present")
def test_real_markers_file():
    pairs = parse_markers(_REAL_MARKERS)
    assert len(pairs) == 25   # Seasurface … SLCL
    names = dict(pairs)
    assert names["Seasurface"] == pytest.approx(30.0)
    assert names["SLCL"] == pytest.approx(3150.0)
    assert "NMRF (Mid_Mio_Unc)" in names
