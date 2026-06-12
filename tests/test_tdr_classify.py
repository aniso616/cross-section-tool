"""Evidence-based TDR classification, the single-door loader, the integrity
shape-check, and the drift-target self-calibration guard (Test-session fix 01,
Issue 1)."""
from __future__ import annotations

import numpy as np
import pytest

from section_tool.core.wells import Well, LogCurve
from section_tool.core.tdr import TimeDepthRelation
from section_tool.core.velocity_model import VelocityModel
from section_tool.core.grounded_velocity import _US_FT_TO_S_PER_M
from section_tool.io.tdr_io import (
    classify_tdr_table, classify_tdr_file, load_tdr_as, tdr_shape_is_sonic,
)
from section_tool.core.zdomain import ZDomain


def _well():
    return Well("F02-01", 606554.0, 6080126.0, kb=30.0, td=3200.0)


# Real F3 shapes, trimmed/synthesised.
_CHECKSHOT_TXT = """\
30\t0
553.6\t0.544
1695\t1.67
3150\t3.234
"""  # 4 irregular pairs, depth-MD, TWT-s — checkshot shape


def _dense_grid_txt() -> str:
    # 625 points on a regular 5 m grid, TVDSS / TWT-ms — sonic TDR shape.
    z = np.arange(0.0, 3125.0, 5.0)
    t = z / 2000.0 * 2.0 * 1000.0  # v=2000 → twt(ms), strictly increasing
    return "\n".join(f"{zi:.4f}\t{ti:.4f}" for zi, ti in zip(z, t)) + "\n"


# ---------------------------------------------------------------------------
# Classifier heuristic — the two real shapes, plus an ambiguous middle.
# ---------------------------------------------------------------------------

class TestClassifier:
    def test_sparse_irregular_is_checkshot(self, tmp_path):
        p = tmp_path / "F02-01_TD.txt"
        p.write_text(_CHECKSHOT_TXT, encoding="utf-8")
        cls = classify_tdr_file(str(p))
        assert cls.suggested_kind == "checkshot"
        assert cls.suggested_depth_reference == "MD"
        assert cls.n_points == 4
        assert not cls.spacing_regular
        assert cls.twt_domain == ZDomain.TWT_S

    def test_dense_regular_is_sonic_integrated(self, tmp_path):
        p = tmp_path / "F02-01_DT_TVDSS.txt"
        p.write_text(_dense_grid_txt(), encoding="utf-8")
        cls = classify_tdr_file(str(p))
        assert cls.suggested_kind == "sonic_integrated"
        assert cls.suggested_depth_reference == "TVDSS"
        assert cls.n_points == 625
        assert cls.spacing_regular
        assert cls.median_spacing == pytest.approx(5.0)
        assert cls.twt_domain == ZDomain.TWT_MS
        # The evidence sentence must name the contradiction, not just the kind.
        assert "not a checkshot" in cls.evidence

    def test_dense_but_irregular_is_imported(self):
        # 200 points, but log-spaced (not a regular grid) and not <=100 → imported.
        z = np.geomspace(10.0, 3000.0, 200)
        t = z / 2000.0 * 2.0
        table = np.column_stack([z, t])
        cls = classify_tdr_table(table)
        assert cls.suggested_kind == "imported"
        assert not cls.spacing_regular


# ---------------------------------------------------------------------------
# Single-door loader — kind + reference faithfully recorded (provenance).
# ---------------------------------------------------------------------------

class TestLoadTdrAs:
    def test_records_source_and_kind_faithfully(self, tmp_path):
        p = tmp_path / "mystery.txt"
        p.write_text(_CHECKSHOT_TXT, encoding="utf-8")
        tdr = load_tdr_as(str(p), _well(), kind="checkshot", depth_reference="MD")
        assert tdr.kind == "checkshot"
        assert tdr.source == "mystery.txt"
        assert tdr.construction["params"]["imported_as"] == "checkshot"
        # MD 30 → TVDSS 0 (KB resolved).
        assert tdr.depth_m[0] == pytest.approx(0.0)

    def test_override_kind_is_honoured(self, tmp_path):
        # The same file imported as sonic_integrated wears that grade, with no
        # MD→TVDSS shift (depth taken at face value).
        p = tmp_path / "mystery.txt"
        p.write_text(_CHECKSHOT_TXT, encoding="utf-8")
        tdr = load_tdr_as(str(p), _well(), kind="sonic_integrated",
                          depth_reference="TVDSS")
        assert tdr.kind == "sonic_integrated"
        assert tdr.depth_m[0] == pytest.approx(30.0)  # face value, no KB shift


# ---------------------------------------------------------------------------
# Integrity shape-check — used by the panel chip to flag a mislabelled TDR.
# ---------------------------------------------------------------------------

class TestShapeCheck:
    def test_sonic_grid_flagged(self):
        z = np.arange(0.0, 3125.0, 5.0)
        assert tdr_shape_is_sonic(z)

    def test_sparse_checkshot_not_flagged(self):
        assert not tdr_shape_is_sonic([0.0, 523.6, 1665.0, 3120.0])

    def test_the_real_bug_shape(self):
        """The exact seismic_stretch_test situation: a 625-pt regular grid stored
        as kind='checkshot' must be flagged for re-verification."""
        z = np.arange(-30.0, 3095.0, 5.0)  # KB-shifted like the buggy import
        tdr = TimeDepthRelation(z, np.linspace(0.001, 3.2, len(z)),
                                kind="checkshot", depth_reference="TVDSS")
        assert tdr.kind == "checkshot"
        assert tdr_shape_is_sonic(tdr.depth_m)


# ---------------------------------------------------------------------------
# Drift-target guard — a sonic model cannot self-calibrate to a sonic TDR.
# ---------------------------------------------------------------------------

_SLOW_2000_US_FT = (1.0 / 2000.0) / _US_FT_TO_S_PER_M


def _sonic_well():
    w = Well("W", 0.0, 0.0, kb=0.0, td=3000.0)
    md = np.arange(0.0, 3001.0, 5.0)
    w.add_log(LogCurve("DT", "us/ft", md, np.full_like(md, _SLOW_2000_US_FT)))
    return w


def _sonic_integrated_tdr():
    z = np.arange(0.0, 3001.0, 5.0)
    t = z / 1000.0  # v=2000 → twt = z/1000 (s), strictly increasing
    t[0] = 1e-4
    return TimeDepthRelation(z, t, kind="sonic_integrated", depth_reference="TVDSS")


def _checkshot_tdr():
    z = [0.0, 500.0, 1500.0, 2500.0]
    t = [1e-4, 0.5, 1.5, 2.5]
    return TimeDepthRelation(z, t, kind="checkshot", depth_reference="TVDSS")


class TestDriftGuard:
    def test_sonic_integrated_drift_target_refused(self):
        with pytest.raises(ValueError, match="self-calibration"):
            VelocityModel.from_sonic(
                _sonic_well(), drift_target="checkshot",
                checkshot=_sonic_integrated_tdr())

    def test_genuine_checkshot_drift_target_accepted(self):
        m = VelocityModel.from_sonic(
            _sonic_well(), drift_target="checkshot", checkshot=_checkshot_tdr())
        assert m is not None
        assert m.provenance == "well_calibrated"
