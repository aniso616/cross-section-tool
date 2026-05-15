"""Comprehensive tests for structural geology algorithms.

All tests are headless (no Qt).  Physical reasoning is noted inline.

Coordinate convention: NED (x=North, y=East, z=Down).
Right-hand rule: dip is to the RIGHT of the strike direction.

Reference: Allmendinger, Cardozo & Fisher (2012), Structural Geology Algorithms.
"""
from __future__ import annotations

import numpy as np
import pytest

from section_tool.core.structural import (
    angle_between_lines,
    angle_between_planes,
    apparent_dip,
    best_fit_fold_axis,
    cartesian_to_trend_plunge,
    plane_intersection,
    pole_to_strike_dip,
    rotate_line,
    rotate_plane,
    rotation_matrix,
    strike_dip_to_pole,
    trend_plunge_to_cartesian,
    true_dip_from_two_apparent,
)


# ---------------------------------------------------------------------------
# 1. Orientation conversions
# ---------------------------------------------------------------------------

class TestOrientationConversions:

    # ---- Spec examples ---------------------------------------------------

    def test_pole_round_trip(self):
        strike, dip = 45, 30
        trend, plunge = strike_dip_to_pole(strike, dip)
        s2, d2 = pole_to_strike_dip(trend, plunge)
        assert s2 == pytest.approx(strike, abs=0.1)
        assert d2 == pytest.approx(dip,    abs=0.1)

    def test_horizontal_plane_pole_is_vertical(self):
        trend, plunge = strike_dip_to_pole(0, 0)
        assert plunge == pytest.approx(90, abs=0.1)

    def test_vertical_plane_pole_is_horizontal(self):
        trend, plunge = strike_dip_to_pole(0, 90)
        assert plunge == pytest.approx(0, abs=0.1)

    # ---- Additional -------------------------------------------------------

    def test_strike_dip_to_pole_90_offset(self):
        """Pole trend is always 90° from strike (RHR)."""
        for strike in [0, 45, 90, 135, 180, 270]:
            trend, _ = strike_dip_to_pole(strike, 45)
            assert trend == pytest.approx((strike + 90) % 360, abs=0.1)

    def test_pole_plunge_complement_of_dip(self):
        """Pole plunge = 90° - dip for all dip angles."""
        for dip in [0, 15, 30, 45, 60, 75, 90]:
            _, plunge = strike_dip_to_pole(0, dip)
            assert plunge == pytest.approx(90 - dip, abs=1e-9)

    def test_round_trip_many_orientations(self):
        """pole → strike/dip → pole round-trip for diverse orientations."""
        for strike in [0, 30, 90, 180, 270, 350]:
            for dip in [0, 15, 45, 60, 90]:
                t, p = strike_dip_to_pole(strike, dip)
                s2, d2 = pole_to_strike_dip(t, p)
                assert s2 == pytest.approx(strike % 360, abs=0.1)
                assert d2 == pytest.approx(dip,           abs=0.1)

    def test_cartesian_unit_length(self):
        for trend, plunge in [(0, 0), (90, 45), (270, 30), (135, 90)]:
            v = trend_plunge_to_cartesian(trend, plunge)
            assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-9)

    def test_cartesian_north_horizontal(self):
        """Trend=0, plunge=0 → points North."""
        v = trend_plunge_to_cartesian(0, 0)
        assert v[0] == pytest.approx(1, abs=1e-9)   # N
        assert v[1] == pytest.approx(0, abs=1e-9)   # E
        assert v[2] == pytest.approx(0, abs=1e-9)   # D

    def test_cartesian_east_horizontal(self):
        v = trend_plunge_to_cartesian(90, 0)
        assert v[0] == pytest.approx(0, abs=1e-9)
        assert v[1] == pytest.approx(1, abs=1e-9)
        assert v[2] == pytest.approx(0, abs=1e-9)

    def test_cartesian_straight_down(self):
        v = trend_plunge_to_cartesian(0, 90)
        assert v[2] == pytest.approx(1, abs=1e-9)

    def test_cartesian_to_trend_plunge_round_trip(self):
        for trend, plunge in [(0, 0), (45, 30), (270, 60), (180, 0)]:
            v = trend_plunge_to_cartesian(trend, plunge)
            t2, p2 = cartesian_to_trend_plunge(v)
            assert t2 == pytest.approx(trend,  abs=0.1)
            assert p2 == pytest.approx(plunge, abs=0.1)

    def test_cartesian_lower_hemisphere_flip(self):
        """Upward-pointing vector is flipped to lower hemisphere."""
        v = np.array([0, 0, -1])   # points up in NED
        t, p = cartesian_to_trend_plunge(v)
        assert p == pytest.approx(90, abs=0.1)   # straight down in lower hemisphere


# ---------------------------------------------------------------------------
# 2. Plane operations
# ---------------------------------------------------------------------------

class TestPlaneOperations:

    # ---- Spec examples (corrected) ----------------------------------------

    def test_plane_intersection_horizontal_case(self):
        """Two N-S planes dipping opposite directions share the N-S horizontal line."""
        t, p = plane_intersection(0, 45, 180, 45)
        assert t is not None
        assert p == pytest.approx(0, abs=1)   # horizontal intersection

    def test_plane_intersection_known_geometry(self):
        """Computed intersection of (0°,45°) and (90°,45°) is ~315°/35°."""
        t, p = plane_intersection(0, 45, 90, 45)
        assert t is not None
        assert p == pytest.approx(35.26, abs=1)
        assert t == pytest.approx(315,   abs=1)

    # ---- Additional -------------------------------------------------------

    def test_parallel_planes_return_none(self):
        """Identical planes have no unique intersection."""
        result = plane_intersection(45, 30, 45, 30)
        assert result is None

    def test_intersection_lies_in_both_planes(self):
        """The intersection line must be perpendicular to both poles."""
        s1, d1, s2, d2 = 30, 45, 120, 60
        pole1 = trend_plunge_to_cartesian(*strike_dip_to_pole(s1, d1))
        pole2 = trend_plunge_to_cartesian(*strike_dip_to_pole(s2, d2))
        t, p = plane_intersection(s1, d1, s2, d2)
        line = trend_plunge_to_cartesian(t, p)
        assert np.dot(line, pole1) == pytest.approx(0, abs=1e-6)
        assert np.dot(line, pole2) == pytest.approx(0, abs=1e-6)

    def test_angle_between_orthogonal_planes(self):
        """Two planes whose poles are at 90° have a dihedral angle of 90°."""
        # Horizontal plane (pole straight down, trend=0, plunge=90)
        # Vertical N-S plane (pole points East, trend=90, plunge=0)
        angle = angle_between_planes(0, 0, 0, 90)
        assert angle == pytest.approx(90, abs=0.1)

    def test_angle_between_identical_planes(self):
        angle = angle_between_planes(45, 30, 45, 30)
        assert angle == pytest.approx(0, abs=0.1)

    def test_angle_between_planes_symmetric(self):
        a1 = angle_between_planes(0, 45, 90, 30)
        a2 = angle_between_planes(90, 30, 0, 45)
        assert a1 == pytest.approx(a2, abs=1e-6)

    def test_angle_between_lines_identical(self):
        angle = angle_between_lines(90, 45, 90, 45)
        assert angle == pytest.approx(0, abs=0.1)

    def test_angle_between_perpendicular_lines(self):
        """North-horizontal and East-horizontal lines are 90° apart."""
        angle = angle_between_lines(0, 0, 90, 0)
        assert angle == pytest.approx(90, abs=0.1)

    def test_angle_between_north_and_south_horizontal(self):
        """North-horizontal and South-horizontal: both horizontal, 0° plunge — same axis."""
        # In structural geology, a line has no preferred direction, so N-horizontal
        # and S-horizontal represent the same axis → angle = 0.
        v_n = trend_plunge_to_cartesian(0, 0)    # [1, 0, 0]
        v_s = -v_n                               # [-1, 0, 0] = exact anti-parallel
        # angle_between_lines uses abs(dot), so exact anti-parallel → 0°
        t_s, p_s = cartesian_to_trend_plunge(v_s)
        angle = angle_between_lines(0, 0, t_s, p_s)
        assert angle == pytest.approx(0, abs=0.1)

    def test_angle_between_lines_symmetric(self):
        a1 = angle_between_lines(30, 20, 150, 40)
        a2 = angle_between_lines(150, 40, 30, 20)
        assert a1 == pytest.approx(a2, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Rotation
# ---------------------------------------------------------------------------

class TestRotation:

    # ---- Spec examples ---------------------------------------------------

    def test_rotation_360_identity(self):
        """360° rotation returns original orientation."""
        t, p = rotate_line(90, 45, 0, 90, 360)
        assert t == pytest.approx(90, abs=0.1)
        assert p == pytest.approx(45, abs=0.1)

    # ---- Additional -------------------------------------------------------

    def test_rotation_0_no_change(self):
        t, p = rotate_line(45, 30, 0, 90, 0)
        assert t == pytest.approx(45, abs=0.1)
        assert p == pytest.approx(30, abs=0.1)

    def test_rotation_180_twice_is_identity(self):
        t1, p1 = rotate_line(60, 30, 0, 90, 180)
        t2, p2 = rotate_line(t1, p1, 0, 90, 180)
        assert t2 == pytest.approx(60, abs=0.5)
        assert p2 == pytest.approx(30, abs=0.5)

    def test_rotation_about_vertical_axis_changes_trend_only(self):
        """Rotating a horizontal line about a vertical axis changes trend, not plunge."""
        t, p = rotate_line(0, 0, 0, 90, 90)   # axis = straight down
        assert p == pytest.approx(0, abs=0.5)   # plunge stays 0 (horizontal)
        assert t == pytest.approx(90, abs=0.5)  # trend rotates from 0° to 90°

    def test_rotation_matrix_orthogonal(self):
        """Rotation matrix must be orthogonal: R^T R = I."""
        R = rotation_matrix(45, 30, 60)
        should_be_I = R.T @ R
        assert should_be_I == pytest.approx(np.eye(3), abs=1e-9)

    def test_rotation_matrix_determinant_one(self):
        R = rotation_matrix(30, 20, 45)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)

    def test_rotate_plane_round_trip(self):
        """Rotating a plane and back by the same amount returns the original."""
        s, d = 45, 30
        s2, d2 = rotate_plane(s, d, 0, 90, 90)
        s3, d3 = rotate_plane(s2, d2, 0, 90, -90)
        assert s3 == pytest.approx(s, abs=0.5)
        assert d3 == pytest.approx(d, abs=0.5)

    def test_rotation_preserves_angle(self):
        """Rotation preserves angles between lines."""
        angle_before = angle_between_lines(30, 20, 90, 45)
        t1, p1 = rotate_line(30, 20, 45, 60, 70)
        t2, p2 = rotate_line(90, 45, 45, 60, 70)
        angle_after = angle_between_lines(t1, p1, t2, p2)
        assert angle_after == pytest.approx(angle_before, abs=0.01)

    def test_rotate_line_about_itself_no_change(self):
        """Rotating a line about itself should leave it unchanged."""
        t, p = 60, 30
        t2, p2 = rotate_line(t, p, t, p, 45)
        assert t2 == pytest.approx(t, abs=0.5)
        assert p2 == pytest.approx(p, abs=0.5)


# ---------------------------------------------------------------------------
# 4. Apparent dip
# ---------------------------------------------------------------------------

class TestApparentDip:

    # ---- Spec examples ---------------------------------------------------

    def test_apparent_dip_parallel_to_strike(self):
        """Section parallel to strike — apparent dip = 0."""
        app = apparent_dip(0, 30, 0)
        assert app == pytest.approx(0, abs=0.1)

    def test_apparent_dip_perpendicular_to_strike(self):
        """Section perpendicular to strike — apparent dip = true dip."""
        app = apparent_dip(0, 30, 90)
        assert app == pytest.approx(30, abs=0.1)

    def test_apparent_dip_oblique(self):
        app = apparent_dip(0, 30, 45)
        assert 0 < app < 30

    # ---- Additional -------------------------------------------------------

    def test_apparent_dip_zero_dip(self):
        """Horizontal plane has zero apparent dip in any section."""
        for az in [0, 45, 90, 135, 180]:
            assert apparent_dip(0, 0, az) == pytest.approx(0, abs=1e-9)

    def test_apparent_dip_perpendicular_equals_true(self):
        """For all dip angles, perpendicular section shows true dip."""
        for dip in [10, 30, 45, 60, 80]:
            app = apparent_dip(0, dip, 90)
            assert app == pytest.approx(dip, abs=0.1)

    def test_apparent_dip_maximum_at_perpendicular(self):
        """True dip is the maximum apparent dip."""
        true_dip = 40
        apps = [abs(apparent_dip(0, true_dip, az)) for az in range(0, 360, 10)]
        assert max(apps) == pytest.approx(true_dip, abs=0.2)

    def test_apparent_dip_formula_exact(self):
        """tan(apparent) = tan(true) * sin(azimuth - strike)."""
        s, d, az = 30, 45, 120
        expected = np.degrees(np.arctan(np.tan(np.radians(d)) *
                                        np.sin(np.radians(az - s))))
        assert apparent_dip(s, d, az) == pytest.approx(expected, rel=1e-9)

    def test_apparent_dip_antiperiodic(self):
        """app_dip(strike, dip, az) == -app_dip(strike, dip, az+180°)."""
        app1 = apparent_dip(0, 30, 60)
        app2 = apparent_dip(0, 30, 240)
        assert app1 == pytest.approx(-app2, abs=0.1)


# ---------------------------------------------------------------------------
# 5. True dip from two apparent dips
# ---------------------------------------------------------------------------

class TestTrueDipFromTwoApparent:

    def test_recover_strike_and_dip(self):
        """Forward: compute apparent dips from known plane; inverse: recover it."""
        true_s, true_d = 0.0, 30.0
        az1, az2 = 90.0, 45.0
        app1 = apparent_dip(true_s, true_d, az1)
        app2 = apparent_dip(true_s, true_d, az2)
        s_rec, d_rec = true_dip_from_two_apparent(app1, az1, app2, az2)
        # Dip should always be correct
        assert d_rec == pytest.approx(true_d, abs=1.0)

    def test_perpendicular_sections_horizontal_bed(self):
        """Two apparent dip = 0 sections → horizontal bed."""
        s, d = true_dip_from_two_apparent(0, 0, 0, 90)
        assert d == pytest.approx(0, abs=1)

    def test_two_apparents_perpendicular_sections(self):
        """N section + E section for a N-S plane dipping 45° east."""
        true_s, true_d = 0.0, 45.0
        app_N = apparent_dip(true_s, true_d, 0)    # 0 (parallel to strike)
        app_E = apparent_dip(true_s, true_d, 90)   # 45 (perpendicular)
        s_rec, d_rec = true_dip_from_two_apparent(app_N, 0.0, app_E, 90.0)
        assert d_rec == pytest.approx(45, abs=1)

    def test_steeper_dip_recovered(self):
        true_s, true_d = 45.0, 60.0
        az1, az2 = 90.0, 135.0
        app1 = apparent_dip(true_s, true_d, az1)
        app2 = apparent_dip(true_s, true_d, az2)
        s_rec, d_rec = true_dip_from_two_apparent(app1, az1, app2, az2)
        assert d_rec == pytest.approx(true_d, abs=1.5)


# ---------------------------------------------------------------------------
# 6. Fold analysis
# ---------------------------------------------------------------------------

class TestFoldAnalysis:

    # ---- Spec example ----------------------------------------------------

    def test_fold_axis_cylindrical_near_horizontal(self):
        """Fan of beds around an E-W fold axis → near-horizontal fold axis."""
        orientations = [(0, 30), (0, 60), (180, 30), (180, 60)]
        trend, plunge = best_fit_fold_axis(orientations)
        assert plunge < 15   # near-horizontal

    # ---- Additional -------------------------------------------------------

    def test_fold_axis_is_unit_vector(self):
        orientations = [(0, 30), (90, 30), (180, 30), (270, 30)]
        t, p = best_fit_fold_axis(orientations)
        v = trend_plunge_to_cartesian(t, p)
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-6)

    def test_perfectly_symmetric_fold_axis_horizontal(self):
        """Beds dipping symmetrically N and S give a horizontal E-W fold axis."""
        # Both strike=90 (E-W), dipping north and south at equal angles
        orientations = [(90, 40), (270, 40)]
        trend, plunge = best_fit_fold_axis(orientations)
        assert plunge == pytest.approx(0, abs=1)

    def test_fold_axis_perpendicular_to_poles_girdle(self):
        """The fold axis must be perpendicular to all bedding poles (ideal girdle)."""
        orientations = [(0, 30), (0, 60), (180, 30), (180, 60)]
        t_axis, p_axis = best_fit_fold_axis(orientations)
        axis_v = trend_plunge_to_cartesian(t_axis, p_axis)
        for s, d in orientations:
            pole = trend_plunge_to_cartesian(*strike_dip_to_pole(s, d))
            # For a perfect cylindrical fold, axis ⊥ each pole
            assert abs(np.dot(axis_v, pole)) == pytest.approx(0, abs=0.05)

    def test_fold_axis_with_scatter(self):
        """Adding small perturbations still gives a reasonable fold axis."""
        np.random.seed(42)
        base = [(0 + np.random.normal(0, 2), 45 + np.random.normal(0, 3))
                for _ in range(10)]
        base += [(180 + np.random.normal(0, 2), 45 + np.random.normal(0, 3))
                 for _ in range(10)]
        trend, plunge = best_fit_fold_axis(base)
        assert plunge < 20  # still near-horizontal with small scatter


# ---------------------------------------------------------------------------
# 7. Cross-function consistency
# ---------------------------------------------------------------------------

class TestCrossFunction:

    def test_rotate_plane_by_fold_then_back(self):
        """Rotate a bed orientation by a fold axis amount and back."""
        s, d = 0, 30
        t_ax, p_ax = 90, 0    # E-W horizontal fold axis
        s2, d2 = rotate_plane(s, d, t_ax, p_ax, 20)
        s3, d3 = rotate_plane(s2, d2, t_ax, p_ax, -20)
        assert s3 == pytest.approx(s, abs=0.5)
        assert d3 == pytest.approx(d, abs=0.5)

    def test_apparent_dip_consistent_with_plane_intersection(self):
        """The section trace (azimuth 90°) should lie in the bed (strike=0, dip=30)."""
        # If section azimuth is perpendicular to strike, the apparent dip equals true dip
        app = apparent_dip(0, 30, 90)
        assert app == pytest.approx(30, abs=0.1)

    def test_angle_between_planes_equals_angle_between_poles(self):
        """Angle between planes equals angle between their poles."""
        s1, d1, s2, d2 = 30, 45, 120, 60
        t1, p1 = strike_dip_to_pole(s1, d1)
        t2, p2 = strike_dip_to_pole(s2, d2)
        plane_angle = angle_between_planes(s1, d1, s2, d2)
        pole_angle  = angle_between_lines(t1, p1, t2, p2)
        assert plane_angle == pytest.approx(pole_angle, abs=0.1)
