"""Tests for section_tool.core.crs."""

import numpy as np
import pytest

from section_tool.core.crs import (
    CRSInfo,
    get_crs_info,
    is_projected,
    linear_units,
    transform_points,
    transform_section,
    units_are_feet,
    units_are_metres,
    validate_projected_crs,
)
from section_tool.core.section import Section

# ---------------------------------------------------------------------------
# Well-known EPSG codes used in tests
# ---------------------------------------------------------------------------
UTM_32N = 32632       # WGS 84 / UTM zone 32N  (metres, projected)
UTM_33N = 32633       # WGS 84 / UTM zone 33N  (metres, projected)
WGS84   = 4326        # WGS 84                 (degrees, geographic)
BNG     = 27700       # British National Grid  (metres, projected)
NY_SP   = 2263        # NAD83 / NY Long Island (US survey feet, projected)
WEBMERC = 3857        # Web Mercator           (metres, projected)


# ---------------------------------------------------------------------------
# CRSInfo dataclass
# ---------------------------------------------------------------------------

class TestCRSInfo:
    def test_frozen(self):
        info = CRSInfo(epsg=32632, name="test", is_projected=True, linear_units="metre")
        with pytest.raises((AttributeError, TypeError)):
            info.epsg = 99  # type: ignore[misc]

    def test_equality(self):
        a = CRSInfo(epsg=32632, name="test", is_projected=True, linear_units="metre")
        b = CRSInfo(epsg=32632, name="test", is_projected=True, linear_units="metre")
        assert a == b

    def test_inequality(self):
        a = CRSInfo(epsg=32632, name="test", is_projected=True, linear_units="metre")
        b = CRSInfo(epsg=32633, name="test", is_projected=True, linear_units="metre")
        assert a != b


# ---------------------------------------------------------------------------
# get_crs_info
# ---------------------------------------------------------------------------

class TestGetCRSInfo:
    def test_utm_projected(self):
        info = get_crs_info(UTM_32N)
        assert info.epsg == UTM_32N
        assert info.is_projected is True
        assert "UTM" in info.name or "utm" in info.name.lower()

    def test_utm_linear_units(self):
        info = get_crs_info(UTM_32N)
        assert info.linear_units == "metre"

    def test_wgs84_not_projected(self):
        info = get_crs_info(WGS84)
        assert info.is_projected is False

    def test_wgs84_no_linear_units(self):
        info = get_crs_info(WGS84)
        assert info.linear_units == ""

    def test_bng_projected_metres(self):
        info = get_crs_info(BNG)
        assert info.is_projected is True
        assert info.linear_units == "metre"

    def test_ny_state_plane_feet(self):
        info = get_crs_info(NY_SP)
        assert info.is_projected is True
        assert "foot" in info.linear_units.lower()

    def test_invalid_epsg_raises(self):
        with pytest.raises(ValueError, match="Invalid EPSG"):
            get_crs_info(99999)

    def test_name_is_nonempty(self):
        assert len(get_crs_info(UTM_32N).name) > 0

    def test_caching_returns_same_object(self):
        # lru_cache should return the same object instance on repeated calls
        a = get_crs_info(UTM_32N)
        b = get_crs_info(UTM_32N)
        assert a is b


# ---------------------------------------------------------------------------
# is_projected
# ---------------------------------------------------------------------------

class TestIsProjected:
    def test_utm_is_projected(self):
        assert is_projected(UTM_32N) is True

    def test_wgs84_not_projected(self):
        assert is_projected(WGS84) is False

    def test_bng_is_projected(self):
        assert is_projected(BNG) is True

    def test_web_mercator_is_projected(self):
        assert is_projected(WEBMERC) is True


# ---------------------------------------------------------------------------
# linear_units
# ---------------------------------------------------------------------------

class TestLinearUnits:
    def test_utm_is_metre(self):
        assert linear_units(UTM_32N) == "metre"

    def test_bng_is_metre(self):
        assert linear_units(BNG) == "metre"

    def test_ny_sp_is_foot(self):
        assert "foot" in linear_units(NY_SP).lower()

    def test_geographic_is_empty(self):
        assert linear_units(WGS84) == ""


# ---------------------------------------------------------------------------
# units_are_metres / units_are_feet
# ---------------------------------------------------------------------------

class TestUnitHelpers:
    def test_utm_is_metres(self):
        assert units_are_metres(UTM_32N) is True
        assert units_are_feet(UTM_32N) is False

    def test_bng_is_metres(self):
        assert units_are_metres(BNG) is True

    def test_ny_sp_is_feet(self):
        assert units_are_feet(NY_SP) is True
        assert units_are_metres(NY_SP) is False

    def test_web_mercator_is_metres(self):
        assert units_are_metres(WEBMERC) is True

    def test_geographic_neither(self):
        assert units_are_metres(WGS84) is False
        assert units_are_feet(WGS84) is False


# ---------------------------------------------------------------------------
# validate_projected_crs
# ---------------------------------------------------------------------------

class TestValidateProjectedCRS:
    def test_utm_passes(self):
        validate_projected_crs(UTM_32N)  # should not raise

    def test_bng_passes(self):
        validate_projected_crs(BNG)

    def test_web_mercator_passes(self):
        validate_projected_crs(WEBMERC)

    def test_wgs84_raises(self):
        with pytest.raises(ValueError, match="not a projected CRS"):
            validate_projected_crs(WGS84)

    def test_invalid_epsg_raises(self):
        with pytest.raises(ValueError):
            validate_projected_crs(99999)

    def test_error_message_contains_epsg(self):
        with pytest.raises(ValueError, match="4326"):
            validate_projected_crs(WGS84)


# ---------------------------------------------------------------------------
# transform_points
# ---------------------------------------------------------------------------

class TestTransformPoints:
    def test_identity_transform(self):
        xs = np.array([500000.0, 600000.0])
        ys = np.array([5000000.0, 5100000.0])
        tx, ty = transform_points(xs, ys, UTM_32N, UTM_32N)
        np.testing.assert_allclose(tx, xs, rtol=1e-10)
        np.testing.assert_allclose(ty, ys, rtol=1e-10)

    def test_geographic_to_utm_roundtrip(self):
        # Known point: 15°E, 51°N → UTM 32N → back
        lons = np.array([15.0])
        lats = np.array([51.0])
        xs, ys = transform_points(lons, lats, WGS84, UTM_32N)
        lon_back, lat_back = transform_points(xs, ys, UTM_32N, WGS84)
        np.testing.assert_allclose(lon_back, lons, atol=1e-8)
        np.testing.assert_allclose(lat_back, lats, atol=1e-8)

    def test_geographic_to_utm_values(self):
        # 15°E, 51°N should land at ~920857 E, ~5666978 N in UTM 32N
        xs, ys = transform_points([15.0], [51.0], WGS84, UTM_32N)
        assert pytest.approx(xs[0], rel=1e-4) == 920857.1
        assert pytest.approx(ys[0], rel=1e-4) == 5666978.3

    def test_utm_to_utm_different_zones(self):
        # Transform a point in UTM 32N to UTM 33N and back
        xs_orig = np.array([700000.0])
        ys_orig = np.array([5500000.0])
        xs_33, ys_33 = transform_points(xs_orig, ys_orig, UTM_32N, UTM_33N)
        xs_back, ys_back = transform_points(xs_33, ys_33, UTM_33N, UTM_32N)
        np.testing.assert_allclose(xs_back, xs_orig, atol=1e-3)
        np.testing.assert_allclose(ys_back, ys_orig, atol=1e-3)

    def test_multi_point_array(self):
        lons = np.array([9.0, 12.0, 15.0])
        lats = np.array([48.0, 49.0, 51.0])
        xs, ys = transform_points(lons, lats, WGS84, UTM_32N)
        assert xs.shape == (3,)
        assert ys.shape == (3,)
        # All easting values should be positive and southing > 0
        assert np.all(xs > 0)
        assert np.all(ys > 0)

    def test_empty_arrays(self):
        xs, ys = transform_points([], [], WGS84, UTM_32N)
        assert len(xs) == 0
        assert len(ys) == 0

    def test_returns_numpy_arrays(self):
        xs, ys = transform_points([15.0], [51.0], WGS84, UTM_32N)
        assert isinstance(xs, np.ndarray)
        assert isinstance(ys, np.ndarray)

    def test_list_input_accepted(self):
        xs, ys = transform_points([15.0, 16.0], [51.0, 52.0], WGS84, UTM_32N)
        assert xs.shape == (2,)


# ---------------------------------------------------------------------------
# transform_section
# ---------------------------------------------------------------------------

class TestTransformSection:
    def test_new_section_has_target_epsg(self):
        sec = Section([(500000.0, 5500000.0), (600000.0, 5500000.0)], crs_epsg=UTM_32N)
        new_sec = transform_section(sec, UTM_33N)
        assert new_sec.crs_epsg == UTM_33N

    def test_node_count_preserved(self):
        sec = Section(
            [(500000.0, 5500000.0), (550000.0, 5550000.0), (600000.0, 5500000.0)],
            crs_epsg=UTM_32N,
        )
        new_sec = transform_section(sec, UTM_33N)
        assert new_sec.n_nodes == 3

    def test_metadata_preserved(self):
        sec = Section(
            [(500000.0, 5500000.0), (600000.0, 5500000.0)],
            name="TestLine",
            depth_domain="twt",
            depth_units="ft",
            vertical_exaggeration=2.0,
            crs_epsg=UTM_32N,
        )
        new_sec = transform_section(sec, UTM_33N)
        assert new_sec.name == "TestLine"
        assert new_sec.depth_domain == "twt"
        assert new_sec.depth_units == "ft"
        assert new_sec.vertical_exaggeration == 2.0

    def test_roundtrip_preserves_nodes(self):
        nodes = np.array([[500000.0, 5500000.0], [600000.0, 5600000.0]])
        sec = Section(nodes, crs_epsg=UTM_32N)
        via_33 = transform_section(sec, UTM_33N)
        back = transform_section(via_33, UTM_32N)
        np.testing.assert_allclose(back.nodes, nodes, atol=1e-3)

    def test_same_crs_identity(self):
        nodes = np.array([[500000.0, 5500000.0], [600000.0, 5500000.0]])
        sec = Section(nodes, crs_epsg=UTM_32N)
        new_sec = transform_section(sec, UTM_32N)
        np.testing.assert_allclose(new_sec.nodes, nodes, rtol=1e-10)

    def test_does_not_mutate_original(self):
        nodes = np.array([[500000.0, 5500000.0], [600000.0, 5500000.0]])
        sec = Section(nodes.copy(), crs_epsg=UTM_32N)
        _ = transform_section(sec, UTM_33N)
        np.testing.assert_allclose(sec.nodes, nodes)

    def test_dogleg_section_roundtrip(self):
        nodes = np.array([
            [500000.0, 5500000.0],
            [550000.0, 5500000.0],
            [550000.0, 5550000.0],
        ])
        sec = Section(nodes, crs_epsg=UTM_32N)
        via_33 = transform_section(sec, UTM_33N)
        back = transform_section(via_33, UTM_32N)
        np.testing.assert_allclose(back.nodes, nodes, atol=1e-3)
