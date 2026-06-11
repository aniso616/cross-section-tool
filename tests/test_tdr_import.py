"""Checkshot + sonic-TDR import: parsing, ms/s detection, and the datum decision."""
from __future__ import annotations

import os

import numpy as np
import pytest

from section_tool.core.wells import Well
from section_tool.core.zdomain import ZDomain
from section_tool.io.tdr_io import (
    load_checkshot, load_sonic_tdr, detect_twt_domain, read_numeric_columns)

# A vertical F02-01-like well: KB 30 m, so TVDSS = MD − 30.
def _well():
    return Well("F02-01", 606554.0, 6080126.0, kb=30.0, td=3200.0)


# Trimmed copy of the real F02-01_TD.txt shape: depth-MD(m), TWT(s). No index col.
_TD_TXT = """\
30\t0
553.6\t0.544
1695\t1.67
3150\t3.234
"""

# Trimmed copy of F02-01_DT_TVDSS.txt: TVDSS(m), TWT(ms).
_DT_TVDSS_TXT = """\
0.0000          1.5693
5.0000          6.7446
1695.0000    1810.0000
3120.0000    3259.3618
"""


@pytest.fixture
def td_file(tmp_path):
    p = tmp_path / "F02-01_TD.txt"
    p.write_text(_TD_TXT, encoding="utf-8")
    return str(p)


@pytest.fixture
def dt_file(tmp_path):
    p = tmp_path / "F02-01_DT_TVDSS.txt"
    p.write_text(_DT_TVDSS_TXT, encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# Checkshot — including the datum decision (MD → TVDSS)
# ---------------------------------------------------------------------------

class TestCheckshot:
    def test_kind_and_reference(self, td_file):
        tdr = load_checkshot(td_file, _well())
        assert tdr.kind == "checkshot"
        assert tdr.depth_reference == "TVDSS"
        assert tdr.source == "F02-01_TD.txt"
        assert tdr.well_uuid == _well().uuid or tdr.well_uuid  # stamped

    def test_datum_resolution_first_last_pairs(self, td_file):
        """The pinned datum decision: MD-from-KB → TVDSS = MD − 30, TWT seconds.
        First (MD 30, 0 s) → (TVDSS 0.0, 0.0 s); last (MD 3150) → (TVDSS 3120)."""
        tdr = load_checkshot(td_file, _well())
        d = tdr.depth_m
        t = tdr.twt_s
        assert d[0] == pytest.approx(0.0)      # MD 30 − KB 30
        assert t[0] == pytest.approx(0.0)
        assert d[-1] == pytest.approx(3120.0)  # MD 3150 − KB 30
        assert t[-1] == pytest.approx(3.234)

    def test_twt_seconds_autodetected(self, td_file):
        # Values ≤ 30 → seconds; no spurious ms division.
        tdr = load_checkshot(td_file, _well())
        assert tdr.twt_at_depth(0.0) == pytest.approx(0.0)
        # MD 553.6 → TVDSS 523.6 → 0.544 s
        assert float(tdr.twt_at_depth(523.6)) == pytest.approx(0.544)

    def test_invertible(self, td_file):
        tdr = load_checkshot(td_file, _well())
        z = float(tdr.depth_at_twt(1.67))
        assert z == pytest.approx(1665.0)  # MD 1695 − 30


# ---------------------------------------------------------------------------
# Sonic TDR — TVDSS native, ms → s
# ---------------------------------------------------------------------------

class TestSonicTdr:
    def test_kind_reference_and_ms_conversion(self, dt_file):
        tdr = load_sonic_tdr(dt_file, _well())
        assert tdr.kind == "sonic_integrated"
        assert tdr.depth_reference == "TVDSS"
        d, t = tdr.depth_m, tdr.twt_s
        assert d[0] == pytest.approx(0.0)
        assert t[0] == pytest.approx(0.0015693)     # 1.5693 ms → s
        assert d[-1] == pytest.approx(3120.0)       # already TVDSS, no datum shift
        assert t[-1] == pytest.approx(3.2593618)    # 3259.3618 ms → s

    def test_no_datum_conversion_applied(self, dt_file):
        # TVDSS 1695 stays 1695 (unlike the checkshot's MD which shifts by KB).
        tdr = load_sonic_tdr(dt_file, _well())
        assert 1695.0 in tdr.depth_m


# ---------------------------------------------------------------------------
# ms/s detection + reader robustness
# ---------------------------------------------------------------------------

class TestDetection:
    def test_detect_seconds(self):
        assert detect_twt_domain([0.0, 0.544, 3.234]) == ZDomain.TWT_S

    def test_detect_milliseconds(self):
        assert detect_twt_domain([1.57, 1810.0, 3259.0]) == ZDomain.TWT_MS

    def test_reader_skips_header_and_blank(self, tmp_path):
        p = tmp_path / "x.txt"
        p.write_text("# comment\nDEPTH TWT\n10 1.0\n\n20 2.0\n", encoding="utf-8")
        arr = read_numeric_columns(str(p))
        assert arr.shape == (2, 2)
        assert np.allclose(arr[:, 0], [10.0, 20.0])


# ---------------------------------------------------------------------------
# Cross-check against the real F3 files when the data drive is present.
# ---------------------------------------------------------------------------

_REAL_TD = r"J:\data\F3_Demo_2023\Rawdata\Well_data\F02-01_TD.txt"
_REAL_DT = r"J:\data\F3_Demo_2023\Rawdata\Well_data\F02-01_DT_TVDSS.txt"


@pytest.mark.skipif(not os.path.exists(_REAL_TD), reason="F3 data drive not present")
def test_real_checkshot_matches_sonic_extent():
    tdr = load_checkshot(_REAL_TD, _well())
    assert tdr.depth_reference == "TVDSS"
    assert tdr.depth_range()[0] == pytest.approx(0.0)
    assert tdr.depth_range()[1] == pytest.approx(3120.0)   # MD 3150 − KB 30
    assert tdr.twt_range()[1] == pytest.approx(3.234, abs=1e-3)


@pytest.mark.skipif(not os.path.exists(_REAL_DT), reason="F3 data drive not present")
def test_real_sonic_tdr_agrees_with_checkshot():
    sonic = load_sonic_tdr(_REAL_DT, _well())
    cs = load_checkshot(_REAL_TD, _well()) if os.path.exists(_REAL_TD) else None
    assert sonic.depth_range()[1] == pytest.approx(3120.0, abs=1.0)
    if cs is not None:
        # Two independent T-D sources should agree to a few ms shallow.
        assert float(sonic.twt_at_depth(523.6)) == pytest.approx(
            float(cs.twt_at_depth(523.6)), abs=0.01)
