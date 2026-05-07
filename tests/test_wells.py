"""Tests for cross_section_tool.core.wells — LogCurve, DeviationSurvey, Well."""

import math

import numpy as np
import pytest

from cross_section_tool.core.section import Section
from cross_section_tool.core.wells import DeviationSurvey, LogCurve, Well


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def linear_curve(name="GR", lo=0.0, hi=1000.0, n=101) -> LogCurve:
    depths = np.linspace(lo, hi, n)
    values = depths * 0.1  # GR = 0.1 * depth
    return LogCurve(name, "GAPI", depths, values)


def east_section(length: float = 1000.0) -> Section:
    return Section([(0.0, 0.0), (length, 0.0)])


def dogleg_section() -> Section:
    return Section([(0.0, 0.0), (1000.0, 0.0), (1000.0, 1000.0)])


# ---------------------------------------------------------------------------
# LogCurve — construction
# ---------------------------------------------------------------------------

class TestLogCurveConstruction:
    def test_basic(self):
        lc = LogCurve("GR", "GAPI", [0.0, 100.0, 200.0], [10.0, 20.0, 30.0])
        assert lc.n_samples == 3
        assert lc.name == "GR"
        assert lc.units == "GAPI"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            LogCurve("GR", "GAPI", [], [])

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            LogCurve("GR", "GAPI", [0.0, 100.0], [10.0])

    def test_auto_sort(self):
        lc = LogCurve("GR", "GAPI", [200.0, 0.0, 100.0], [30.0, 10.0, 20.0])
        assert list(lc.depths) == [0.0, 100.0, 200.0]
        assert list(lc.values) == [10.0, 20.0, 30.0]

    def test_depths_are_copies(self):
        d = np.array([0.0, 100.0])
        lc = LogCurve("X", "m", d, [1.0, 2.0])
        d[0] = 999.0
        assert lc.depths[0] == 0.0

    def test_repr(self):
        lc = linear_curve()
        assert "GR" in repr(lc)

    def test_depth_range(self):
        lc = LogCurve("X", "m", [50.0, 150.0, 250.0], [1.0, 2.0, 3.0])
        lo, hi = lc.depth_range()
        assert lo == 50.0
        assert hi == 250.0


# ---------------------------------------------------------------------------
# LogCurve — sampling
# ---------------------------------------------------------------------------

class TestLogCurveSample:
    def test_sample_at_first(self):
        lc = linear_curve()
        assert pytest.approx(lc.sample(0.0)) == 0.0

    def test_sample_at_last(self):
        lc = linear_curve()
        assert pytest.approx(lc.sample(1000.0)) == 100.0

    def test_sample_interpolated(self):
        lc = LogCurve("GR", "GAPI", [0.0, 1000.0], [0.0, 100.0])
        assert pytest.approx(lc.sample(500.0)) == 50.0

    def test_sample_before_range_nan(self):
        lc = LogCurve("GR", "GAPI", [100.0, 200.0], [10.0, 20.0])
        assert math.isnan(lc.sample(0.0))

    def test_sample_after_range_nan(self):
        lc = LogCurve("GR", "GAPI", [0.0, 200.0], [0.0, 20.0])
        assert math.isnan(lc.sample(300.0))

    def test_sample_many(self):
        lc = LogCurve("GR", "GAPI", [0.0, 1000.0], [0.0, 1000.0])
        zs = lc.sample_many([0.0, 250.0, 500.0, 750.0, 1000.0])
        np.testing.assert_allclose(zs, [0.0, 250.0, 500.0, 750.0, 1000.0])

    def test_sample_many_nan_outside(self):
        lc = LogCurve("GR", "GAPI", [100.0, 900.0], [10.0, 90.0])
        zs = lc.sample_many([0.0, 500.0, 1000.0])
        assert math.isnan(zs[0])
        assert not math.isnan(zs[1])
        assert math.isnan(zs[2])

    def test_sample_many_empty(self):
        lc = linear_curve()
        assert len(lc.sample_many([])) == 0


# ---------------------------------------------------------------------------
# DeviationSurvey — construction
# ---------------------------------------------------------------------------

class TestDeviationSurveyConstruction:
    def test_minimum_two_stations(self):
        with pytest.raises(ValueError):
            DeviationSurvey([0.0], [0.0], [0.0])

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            DeviationSurvey([0.0, 100.0], [0.0], [0.0, 0.0])

    def test_decreasing_md_raises(self):
        with pytest.raises(ValueError):
            DeviationSurvey([100.0, 0.0], [0.0, 0.0], [0.0, 0.0])

    def test_vertical_classmethod(self):
        dev = DeviationSurvey.vertical(surface_x=100.0, surface_y=200.0, td=3000.0)
        assert dev.max_md == 3000.0
        assert dev.surface_x == 100.0
        assert dev.surface_y == 200.0

    def test_repr(self):
        dev = DeviationSurvey.vertical()
        assert "DeviationSurvey" in repr(dev)

    def test_properties_are_copies(self):
        dev = DeviationSurvey.vertical(0.0, 0.0, td=100.0)
        t = dev.tvd_track
        t[0] = 999.0
        assert dev.tvd_track[0] == 0.0


# ---------------------------------------------------------------------------
# DeviationSurvey — vertical well geometry
# ---------------------------------------------------------------------------

class TestDeviationSurveyVertical:
    def test_tvd_equals_md(self):
        dev = DeviationSurvey.vertical(0.0, 0.0, td=1000.0)
        assert pytest.approx(dev.max_tvd) == 1000.0

    def test_x_constant_at_surface(self):
        dev = DeviationSurvey.vertical(300.0, 400.0, td=1000.0)
        np.testing.assert_allclose(dev.x_track, 300.0)
        np.testing.assert_allclose(dev.y_track, 400.0)

    def test_tvd_at_md(self):
        dev = DeviationSurvey.vertical(0.0, 0.0, td=1000.0)
        assert pytest.approx(dev.tvd_at_md(500.0)) == 500.0
        assert pytest.approx(dev.tvd_at_md(0.0)) == 0.0
        assert pytest.approx(dev.tvd_at_md(1000.0)) == 1000.0

    def test_md_to_tvd_array(self):
        dev = DeviationSurvey.vertical(0.0, 0.0, td=1000.0)
        tvds = dev.md_to_tvd([0.0, 250.0, 500.0, 750.0, 1000.0])
        np.testing.assert_allclose(tvds, [0.0, 250.0, 500.0, 750.0, 1000.0])

    def test_xyz_at_md_vertical(self):
        dev = DeviationSurvey.vertical(50.0, 75.0, td=500.0)
        x, y, tvd = dev.xyz_at_md(250.0)
        assert pytest.approx(x) == 50.0
        assert pytest.approx(y) == 75.0
        assert pytest.approx(tvd) == 250.0

    def test_xyz_at_md_clamps_beyond_td(self):
        dev = DeviationSurvey.vertical(0.0, 0.0, td=1000.0)
        # np.interp clamps to endpoint values beyond range
        x, y, tvd = dev.xyz_at_md(2000.0)
        assert pytest.approx(tvd) == 1000.0

    def test_xyz_at_surface(self):
        dev = DeviationSurvey.vertical(10.0, 20.0, td=100.0)
        x, y, tvd = dev.xyz_at_md(0.0)
        assert pytest.approx(x) == 10.0
        assert pytest.approx(y) == 20.0
        assert pytest.approx(tvd) == 0.0


# ---------------------------------------------------------------------------
# DeviationSurvey — minimum curvature
# ---------------------------------------------------------------------------

class TestDeviationSurveyMinCurvature:
    def test_constant_45_east(self):
        """Constant 45° inclination due east: TVD and easting both = MD * cos(45°)."""
        root2_over2 = math.sqrt(2) / 2
        dev = DeviationSurvey(
            [0.0, 100.0], [45.0, 45.0], [90.0, 90.0], surface_x=0.0, surface_y=0.0
        )
        # Dog-leg = 0, RF = 1 for constant direction
        expected_x = 100.0 * root2_over2
        expected_tvd = 100.0 * root2_over2
        assert pytest.approx(dev._x[-1], rel=1e-6) == expected_x
        assert pytest.approx(dev._y[-1], abs=1e-9) == 0.0
        assert pytest.approx(dev._tvd[-1], rel=1e-6) == expected_tvd

    def test_constant_horizontal_north(self):
        """Constant 90° inc, 0° azi (horizontal north): TVD stays zero, y increases."""
        dev = DeviationSurvey(
            [0.0, 100.0], [90.0, 90.0], [0.0, 0.0], surface_x=0.0, surface_y=0.0
        )
        assert pytest.approx(dev._tvd[-1], abs=1e-9) == 0.0
        assert pytest.approx(dev._x[-1], abs=1e-9) == 0.0
        assert pytest.approx(dev._y[-1], rel=1e-6) == 100.0

    def test_constant_horizontal_east(self):
        """Constant 90° inc, 90° azi (horizontal east): TVD stays zero, x increases."""
        dev = DeviationSurvey(
            [0.0, 100.0], [90.0, 90.0], [90.0, 90.0], surface_x=0.0, surface_y=0.0
        )
        assert pytest.approx(dev._tvd[-1], abs=1e-9) == 0.0
        assert pytest.approx(dev._x[-1], rel=1e-6) == 100.0
        assert pytest.approx(dev._y[-1], abs=1e-9) == 0.0

    def test_build_from_vertical_to_horizontal(self):
        """90° build over 100 m MD: TVD ≈ easting ≈ 200/π (minimum curvature)."""
        dev = DeviationSurvey(
            [0.0, 100.0], [0.0, 90.0], [90.0, 90.0], surface_x=0.0, surface_y=0.0
        )
        # RF = (2/(π/2))*tan(π/4) = 4/π
        expected = 200.0 / math.pi
        assert pytest.approx(dev._x[-1], rel=1e-6) == expected
        assert pytest.approx(dev._tvd[-1], rel=1e-6) == expected

    def test_multi_station_cumulative(self):
        """Three stations: vertical then 45° east. Offsets are cumulative."""
        root2_over2 = math.sqrt(2) / 2
        dev = DeviationSurvey(
            [0.0, 100.0, 200.0],
            [0.0, 0.0, 45.0],   # vertical to station 1, then kick east
            [0.0, 90.0, 90.0],
            surface_x=0.0, surface_y=0.0,
        )
        # Stations 0→1: vertical, no displacement
        assert pytest.approx(dev._x[1], abs=1e-9) == 0.0
        assert pytest.approx(dev._tvd[1]) == 100.0
        # Stations 1→2: transition from 0° to 45° at azi=90° over 100m
        # Expect x[2] > 0, tvd[2] > 100
        assert dev._x[2] > 0.0
        assert dev._tvd[2] > 100.0

    def test_surface_position_offset(self):
        """Surface position is added to all absolute coordinates."""
        dev = DeviationSurvey(
            [0.0, 100.0], [90.0, 90.0], [90.0, 90.0],
            surface_x=500.0, surface_y=200.0,
        )
        assert pytest.approx(dev._x[0]) == 500.0
        assert pytest.approx(dev._y[0]) == 200.0
        assert pytest.approx(dev._x[-1], rel=1e-6) == 600.0  # 500 + 100
        assert pytest.approx(dev._y[-1]) == 200.0

    def test_xyz_at_md_interpolates_between_stations(self):
        """Mid-station MD interpolates linearly between station positions."""
        root2_over2 = math.sqrt(2) / 2
        dev = DeviationSurvey(
            [0.0, 100.0], [45.0, 45.0], [90.0, 90.0], surface_x=0.0, surface_y=0.0
        )
        x, y, tvd = dev.xyz_at_md(50.0)
        expected = 50.0 * root2_over2
        assert pytest.approx(x, rel=1e-6) == expected
        assert pytest.approx(tvd, rel=1e-6) == expected


# ---------------------------------------------------------------------------
# Well — construction and metadata
# ---------------------------------------------------------------------------

class TestWellConstruction:
    def test_basic(self):
        w = Well("W1", 500.0, 200.0)
        assert w.name == "W1"
        assert w.x == 500.0
        assert w.y == 200.0

    def test_default_deviation_is_vertical(self):
        w = Well("W1", 100.0, 200.0)
        # Auto-created vertical survey starts at the well location
        assert pytest.approx(w.deviation.surface_x) == 100.0
        assert pytest.approx(w.deviation.surface_y) == 200.0

    def test_custom_deviation(self):
        dev = DeviationSurvey([0.0, 500.0], [0.0, 0.0], [0.0, 0.0], 10.0, 20.0)
        w = Well("W1", 10.0, 20.0, deviation=dev)
        assert w.deviation is dev

    def test_kb_stored(self):
        w = Well("W1", 0.0, 0.0, kb=55.0)
        assert w.kb == 55.0

    def test_uwi_stored(self):
        w = Well("W1", 0.0, 0.0, uwi="100/03-12-035-09W5/0")
        assert w.uwi == "100/03-12-035-09W5/0"

    def test_repr(self):
        w = Well("Testwell", 0.0, 0.0)
        assert "Testwell" in repr(w)


# ---------------------------------------------------------------------------
# Well — log management
# ---------------------------------------------------------------------------

class TestWellLogs:
    def test_add_and_get(self):
        w = Well("W1", 0.0, 0.0)
        lc = linear_curve("GR")
        w.add_log(lc)
        assert w.get_log("GR") is lc

    def test_log_names(self):
        w = Well("W1", 0.0, 0.0)
        w.add_log(linear_curve("GR"))
        w.add_log(linear_curve("RHOB"))
        assert set(w.log_names) == {"GR", "RHOB"}

    def test_get_missing_raises(self):
        w = Well("W1", 0.0, 0.0)
        with pytest.raises(KeyError):
            w.get_log("GR")

    def test_remove_log(self):
        w = Well("W1", 0.0, 0.0)
        w.add_log(linear_curve("GR"))
        w.remove_log("GR")
        assert "GR" not in w.log_names

    def test_remove_missing_raises(self):
        w = Well("W1", 0.0, 0.0)
        with pytest.raises(KeyError):
            w.remove_log("GR")

    def test_add_replaces_existing(self):
        w = Well("W1", 0.0, 0.0)
        lc1 = linear_curve("GR")
        lc2 = LogCurve("GR", "GAPI", [0.0, 100.0], [50.0, 50.0])
        w.add_log(lc1)
        w.add_log(lc2)
        assert w.get_log("GR") is lc2

    def test_no_logs_initially(self):
        w = Well("W1", 0.0, 0.0)
        assert w.log_names == []


# ---------------------------------------------------------------------------
# Well — formation tops
# ---------------------------------------------------------------------------

class TestWellFormationTops:
    def test_add_and_read(self):
        w = Well("W1", 0.0, 0.0)
        w.add_formation_top("Top Cretaceous", 1500.0)
        assert w.formation_tops["Top Cretaceous"] == 1500.0

    def test_multiple_tops(self):
        w = Well("W1", 0.0, 0.0)
        w.add_formation_top("Top A", 1000.0)
        w.add_formation_top("Top B", 2000.0)
        assert len(w.formation_tops) == 2

    def test_formation_tops_is_copy(self):
        w = Well("W1", 0.0, 0.0)
        w.add_formation_top("Top A", 1000.0)
        tops = w.formation_tops
        tops["Top A"] = 9999.0
        assert w.formation_tops["Top A"] == 1000.0

    def test_remove_top(self):
        w = Well("W1", 0.0, 0.0)
        w.add_formation_top("Top A", 1000.0)
        w.remove_formation_top("Top A")
        assert "Top A" not in w.formation_tops

    def test_remove_missing_raises(self):
        w = Well("W1", 0.0, 0.0)
        with pytest.raises(KeyError):
            w.remove_formation_top("Top A")

    def test_overwrite_top(self):
        w = Well("W1", 0.0, 0.0)
        w.add_formation_top("Top A", 1000.0)
        w.add_formation_top("Top A", 1200.0)
        assert w.formation_tops["Top A"] == 1200.0


# ---------------------------------------------------------------------------
# Well — section projection
# ---------------------------------------------------------------------------

class TestWellSectionProjection:
    def test_project_vertical_well_on_east_section(self):
        sec = east_section(1000.0)
        w = Well("W1", 500.0, 0.0)
        dist, perp = w.project_to_section(sec)
        assert pytest.approx(dist) == 500.0
        assert pytest.approx(perp, abs=1e-9) == 0.0

    def test_project_off_section(self):
        sec = east_section(1000.0)
        w = Well("W1", 500.0, 100.0)  # 100 m north of section
        dist, perp = w.project_to_section(sec)
        assert pytest.approx(dist) == 500.0
        assert pytest.approx(perp) == 100.0

    def test_section_track_vertical_constant_distance(self):
        sec = east_section(1000.0)
        dev = DeviationSurvey.vertical(500.0, 0.0, td=1000.0)
        w = Well("W1", 500.0, 0.0, deviation=dev)
        distances, tvds = w.section_track(sec)
        # Vertical well: all distances should be ~500 m
        np.testing.assert_allclose(distances, 500.0, atol=1e-9)
        assert pytest.approx(tvds[0]) == 0.0
        assert pytest.approx(tvds[-1]) == 1000.0

    def test_section_track_vertical_tvd_span(self):
        sec = east_section(1000.0)
        dev = DeviationSurvey.vertical(200.0, 0.0, td=3000.0)
        w = Well("W1", 200.0, 0.0, deviation=dev)
        distances, tvds = w.section_track(sec)
        assert len(distances) == len(tvds)
        assert pytest.approx(tvds[0]) == 0.0
        assert pytest.approx(tvds[-1]) == 3000.0

    def test_section_track_deviated_east(self):
        """Well kicking 45° east: distance along east section increases with TVD."""
        sec = east_section(1000.0)
        dev = DeviationSurvey(
            [0.0, 200.0], [45.0, 45.0], [90.0, 90.0],
            surface_x=0.0, surface_y=0.0,
        )
        w = Well("W1", 0.0, 0.0, deviation=dev)
        distances, tvds = w.section_track(sec)
        # Deeper station should have larger distance (well kicks east = into section)
        assert distances[-1] > distances[0]
        expected_x = 200.0 * math.sqrt(2) / 2
        assert pytest.approx(distances[-1], rel=1e-5) == expected_x

    def test_section_track_on_dogleg_section(self):
        sec = dogleg_section()
        dev = DeviationSurvey.vertical(0.0, 0.0, td=500.0)
        w = Well("W1", 0.0, 0.0, deviation=dev)
        distances, tvds = w.section_track(sec)
        # Vertical well at section start → all distances ≈ 0
        np.testing.assert_allclose(distances, 0.0, atol=1e-9)

    def test_formation_top_in_section_vertical(self):
        sec = east_section(1000.0)
        dev = DeviationSurvey.vertical(600.0, 0.0, td=2000.0)
        w = Well("W1", 600.0, 0.0, deviation=dev)
        w.add_formation_top("Top Sand", 800.0)
        dist, tvd = w.formation_top_in_section("Top Sand", sec)
        assert pytest.approx(dist) == 600.0
        assert pytest.approx(tvd) == 800.0

    def test_formation_top_in_section_deviated(self):
        """Deviated well: top position shifts with the wellbore trajectory."""
        sec = east_section(1000.0)
        dev = DeviationSurvey(
            [0.0, 100.0], [45.0, 45.0], [90.0, 90.0],
            surface_x=0.0, surface_y=0.0,
        )
        w = Well("W1", 0.0, 0.0, deviation=dev)
        w.add_formation_top("Top A", 100.0)
        dist, tvd = w.formation_top_in_section("Top A", sec)
        expected = 100.0 * math.sqrt(2) / 2
        assert pytest.approx(dist, rel=1e-5) == expected
        assert pytest.approx(tvd, rel=1e-5) == expected

    def test_formation_top_missing_raises(self):
        sec = east_section()
        w = Well("W1", 0.0, 0.0)
        with pytest.raises(KeyError):
            w.formation_top_in_section("Top A", sec)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_log_curve_single_sample(self):
        lc = LogCurve("X", "m", [500.0], [42.0])
        assert lc.sample(500.0) == 42.0
        assert math.isnan(lc.sample(501.0))

    def test_deviation_single_horizontal_segment(self):
        """Purely horizontal well: max_tvd stays near zero."""
        dev = DeviationSurvey(
            [0.0, 1000.0], [90.0, 90.0], [0.0, 0.0],
            surface_x=0.0, surface_y=0.0,
        )
        assert pytest.approx(dev.max_tvd, abs=1e-9) == 0.0
        assert pytest.approx(dev.max_md) == 1000.0
        assert pytest.approx(dev._y[-1]) == 1000.0

    def test_deviation_many_stations(self):
        """10-station survey doesn't crash and has consistent monotonic TVD."""
        mds = np.linspace(0, 1000, 10)
        incs = np.zeros(10)
        azis = np.zeros(10)
        dev = DeviationSurvey(mds, incs, azis, 0.0, 0.0)
        assert all(np.diff(dev.tvd_track) >= 0.0)

    def test_well_project_before_section_start(self):
        """Well behind the start of the section clamps to distance 0."""
        sec = east_section(1000.0)
        w = Well("W1", -200.0, 0.0)
        dist, _ = w.project_to_section(sec)
        assert pytest.approx(dist) == 0.0

    def test_well_project_after_section_end(self):
        """Well beyond the end of the section clamps to total length."""
        sec = east_section(1000.0)
        w = Well("W1", 1500.0, 0.0)
        dist, _ = w.project_to_section(sec)
        assert pytest.approx(dist) == 1000.0

    def test_deviation_zero_md_interval(self):
        """Duplicate MD stations (dMD=0) are handled without division by zero."""
        dev = DeviationSurvey(
            [0.0, 0.0, 100.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
            surface_x=0.0, surface_y=0.0,
        )
        assert pytest.approx(dev._tvd[-1]) == 100.0

    def test_md_to_tvd_array_empty(self):
        dev = DeviationSurvey.vertical(0.0, 0.0, td=1000.0)
        result = dev.md_to_tvd([])
        assert len(result) == 0
