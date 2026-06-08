"""Tests for seismic domain handling: TWT detection, display domain, velocity stubs."""
from __future__ import annotations

import numpy as np
import pytest

from section_tool.io.segy import SeismicDataset, detect_domain
from section_tool.core.section import Section
from section_tool.io.database import ProjectDatabase


# ---------------------------------------------------------------------------
# 1. Domain detection from sample interval
# ---------------------------------------------------------------------------

class TestDetectDomain:
    def test_2ms_is_twt(self):
        domain, units = detect_domain(2_000)  # 2 000 µs = 2 ms
        assert domain == "twt"
        assert units == "ms"

    def test_4ms_is_twt(self):
        domain, units = detect_domain(4_000)
        assert domain == "twt"
        assert units == "ms"

    def test_1ms_boundary_is_twt(self):
        domain, units = detect_domain(1_000)
        assert domain == "twt"
        assert units == "ms"

    def test_8ms_boundary_is_twt(self):
        domain, units = detect_domain(8_000)
        assert domain == "twt"
        assert units == "ms"

    def test_500us_is_depth(self):
        """Very short interval → depth-migrated data."""
        domain, units = detect_domain(500)
        assert domain == "depth"
        assert units == "m"

    def test_10ms_is_depth(self):
        """Above 8 ms → unusual, treated as depth."""
        domain, units = detect_domain(10_000)
        assert domain == "depth"
        assert units == "m"

    def test_zero_interval_is_depth(self):
        domain, units = detect_domain(0)
        assert domain == "depth"
        assert units == "m"

    def test_f3_typical_4ms_interval(self):
        """F3 Demo uses 4 ms sample interval."""
        domain, units = detect_domain(4_000)
        assert domain == "twt"
        assert units == "ms"


# ---------------------------------------------------------------------------
# 2. SeismicDataset: time_range and sample_interval_ms
# ---------------------------------------------------------------------------

def _make_dataset(domain="twt", dt_ms=2.0, n_samples=500) -> SeismicDataset:
    samples = np.arange(n_samples, dtype=float) * dt_ms
    return SeismicDataset(
        name="test",
        data=np.zeros((10, n_samples), dtype=np.float32),
        trace_x=np.zeros(10),
        trace_y=np.zeros(10),
        samples=samples,
        sample_interval=dt_ms,
        domain=domain,
        depth_units="ms",
        crs_epsg=32631,
        sample_interval_ms=dt_ms,
    )


class TestSeismicDataset:
    def test_time_range_twt(self):
        ds = _make_dataset(domain="twt", dt_ms=4.0, n_samples=462)
        t_min, t_max = ds.time_range
        assert t_min == pytest.approx(0.0)
        assert t_max == pytest.approx(461 * 4.0)

    def test_time_range_empty(self):
        ds = SeismicDataset(
            name="empty", data=np.empty((0, 0), dtype=np.float32),
            trace_x=np.array([]), trace_y=np.array([]),
            samples=np.array([]), sample_interval=2.0,
            domain="twt", depth_units="ms", crs_epsg=32631,
        )
        assert ds.time_range == (0.0, 0.0)

    def test_sample_interval_ms_stored(self):
        ds = _make_dataset(dt_ms=2.0)
        assert ds.sample_interval_ms == pytest.approx(2.0)

    def test_domain_field(self):
        ds = _make_dataset(domain="twt")
        assert ds.domain == "twt"

    def test_depth_domain_field(self):
        ds = _make_dataset(domain="depth")
        assert ds.domain == "depth"


# ---------------------------------------------------------------------------
# 3. Section: display_domain, y_label, y_range
# ---------------------------------------------------------------------------

def _make_section(**kwargs) -> Section:
    return Section([(0, 0), (10000, 0)], name="S1", **kwargs)


class TestSectionDisplayDomain:
    """Depth-canonical: the section is ALWAYS depth.  The legacy display_domain
    toggle was retired (M3); display_domain is read-only and always 'depth',
    independent of depth_domain, and y_label/y_range are always depth."""

    def test_display_domain_always_depth(self):
        assert _make_section(depth_domain="depth").display_domain == "depth"
        assert _make_section(depth_domain="twt").display_domain == "depth"

    def test_display_domain_is_read_only(self):
        s = _make_section()
        with pytest.raises(AttributeError):
            s.display_domain = "twt"

    def test_y_label_depth_m(self):
        s = _make_section(depth_domain="depth", depth_units="m")
        assert s.y_label == "Depth (m)"

    def test_y_label_depth_ft(self):
        s = _make_section(depth_domain="depth", depth_units="ft")
        assert s.y_label == "Depth (ft)"

    def test_y_label_twt_section_still_depth(self):
        # Even a legacy depth_domain='twt' section labels its axis in depth.
        s = _make_section(depth_domain="twt", depth_units="m")
        assert s.y_label == "Depth (m)"

    def test_y_range_always_depth(self):
        for dd in ("depth", "twt"):
            top, bot = _make_section(depth_domain=dd).y_range
            assert top == pytest.approx(0.0)
            assert bot > 0


# ---------------------------------------------------------------------------
# 4. Velocity conversion stubs (round-trip)
# ---------------------------------------------------------------------------

class TestVelocityStubs:
    _V0 = 2_000.0  # m/s constant assumed by the stubs

    def test_depth_to_twt_zero(self):
        assert Section.depth_to_twt(0.0) == pytest.approx(0.0)

    def test_depth_to_twt_1000m(self):
        # TWT = 2 * 1000 / 2000 * 1000 ms = 1000 ms
        assert Section.depth_to_twt(1000.0) == pytest.approx(1000.0)

    def test_depth_to_twt_2500m(self):
        assert Section.depth_to_twt(2500.0) == pytest.approx(2500.0)

    def test_twt_to_depth_zero(self):
        assert Section.twt_to_depth(0.0) == pytest.approx(0.0)

    def test_twt_to_depth_1000ms(self):
        # depth = 1000 / 1000 * 2000 / 2 = 1000 m
        assert Section.twt_to_depth(1000.0) == pytest.approx(1000.0)

    def test_twt_to_depth_2500ms(self):
        assert Section.twt_to_depth(2500.0) == pytest.approx(2500.0)

    @pytest.mark.parametrize("depth_m", [0.0, 100.0, 500.0, 1000.0, 3150.0, 5000.0])
    def test_round_trip_depth_twt_depth(self, depth_m):
        twt = Section.depth_to_twt(depth_m)
        recovered = Section.twt_to_depth(twt)
        assert recovered == pytest.approx(depth_m, abs=1e-9)

    @pytest.mark.parametrize("twt_ms", [0.0, 500.0, 1000.0, 2000.0, 3000.0])
    def test_round_trip_twt_depth_twt(self, twt_ms):
        depth = Section.twt_to_depth(twt_ms)
        recovered = Section.depth_to_twt(depth)
        assert recovered == pytest.approx(twt_ms, abs=1e-9)

    def test_stubs_accept_xy_args(self):
        """Stub must accept x, y positional args without error."""
        t = Section.depth_to_twt(1000.0, 606554.0, 6080126.0)
        assert t > 0
        d = Section.twt_to_depth(1000.0, 606554.0, 6080126.0)
        assert d > 0

    def test_linearity(self):
        """Conversion should be linear for a constant-velocity model."""
        d1 = Section.depth_to_twt(1000.0)
        d2 = Section.depth_to_twt(2000.0)
        assert d2 == pytest.approx(d1 * 2.0)


# ---------------------------------------------------------------------------
# 5. Database: display_domain stored and retrieved
# ---------------------------------------------------------------------------

class TestDatabaseDisplayDomain:
    def test_display_domain_column_exists(self, tmp_path):
        db = ProjectDatabase(str(tmp_path / "test.sqlite"))
        cols = {r[1] for r in db.conn.execute(
            "PRAGMA table_info(sections)"
        ).fetchall()}
        assert "display_domain" in cols
        db.close()

    def test_display_domain_default_depth(self, tmp_path):
        db = ProjectDatabase(str(tmp_path / "test.sqlite"))
        sec = _make_section(depth_domain="depth")
        db.upsert_section(sec)
        row = db.get_all_sections()[0]
        assert row["display_domain"] == "depth"
        db.close()

    def test_display_domain_always_depth_stored(self, tmp_path):
        # Depth-canonical: a section always persists display_domain == 'depth',
        # even when its source depth_domain is 'twt' (the toggle was retired).
        db = ProjectDatabase(str(tmp_path / "test.sqlite"))
        sec = _make_section(depth_domain="twt")
        db.upsert_section(sec)
        row = db.get_all_sections()[0]
        assert row["display_domain"] == "depth"
        db.close()

    def test_round_trip_display_domain(self, tmp_path):
        path = str(tmp_path / "rt.sqlite")
        db1 = ProjectDatabase(path)
        sec = _make_section(depth_domain="depth")
        db1.upsert_section(sec)
        db1.close()

        db2 = ProjectDatabase(path)
        row = db2.get_all_sections()[0]
        assert row["display_domain"] == "depth"
        db2.close()
