"""Structural geology algorithms — Allmendinger, Cardozo & Fisher (2012).

Coordinate convention throughout: NED (x=North, y=East, z=Down).
Angles follow the right-hand rule (RHR): dip is to the RIGHT of the strike
direction when viewed from above.

No GUI or Qt dependencies.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Orientation conversions
# ---------------------------------------------------------------------------

def strike_dip_to_pole(strike_deg: float, dip_deg: float) -> tuple[float, float]:
    """Convert strike/dip to the trend/plunge of the pole to the plane.

    Right-hand rule: dip is to the right of the strike direction.

    Parameters
    ----------
    strike_deg : float
        Strike azimuth (degrees, 0–360 clockwise from North).
    dip_deg : float
        Dip angle (degrees, 0–90).

    Returns
    -------
    (trend_deg, plunge_deg) of the pole to the plane.
    """
    trend  = (strike_deg + 90) % 360   # pole trend is 90° from strike
    plunge = 90.0 - dip_deg
    return trend, plunge


def pole_to_strike_dip(trend_deg: float, plunge_deg: float) -> tuple[float, float]:
    """Convert a pole (trend/plunge) back to strike/dip."""
    strike = (trend_deg - 90) % 360
    dip    = 90.0 - plunge_deg
    return strike, dip


def trend_plunge_to_cartesian(trend_deg: float, plunge_deg: float) -> np.ndarray:
    """Convert trend/plunge to a unit vector in NED (North, East, Down).

    Parameters
    ----------
    trend_deg  : azimuth of the line (degrees, clockwise from North).
    plunge_deg : plunge below horizontal (degrees, 0–90).

    Returns
    -------
    np.ndarray, shape (3,)
        Unit vector ``[N, E, D]``.
    """
    trend  = np.radians(trend_deg)
    plunge = np.radians(plunge_deg)
    x = np.cos(plunge) * np.cos(trend)   # North
    y = np.cos(plunge) * np.sin(trend)   # East
    z = np.sin(plunge)                    # Down (positive)
    return np.array([x, y, z])


def cartesian_to_trend_plunge(v: np.ndarray) -> tuple[float, float]:
    """Convert a unit vector (NED) to trend/plunge.

    Always returns plunge >= 0 (lower hemisphere convention).
    """
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v)
    plunge = float(np.degrees(np.arcsin(v[2])))
    trend  = float(np.degrees(np.arctan2(v[1], v[0])) % 360)
    if plunge < 0:              # flip to lower hemisphere
        plunge = -plunge
        trend  = (trend + 180) % 360
    return trend, plunge


# ---------------------------------------------------------------------------
# Plane operations
# ---------------------------------------------------------------------------

def plane_intersection(
    strike1: float, dip1: float,
    strike2: float, dip2: float,
) -> tuple[float, float] | None:
    """Find the line of intersection of two planes.

    Returns
    -------
    (trend_deg, plunge_deg) of the intersection line, or None for parallel planes.
    """
    pole1 = trend_plunge_to_cartesian(*strike_dip_to_pole(strike1, dip1))
    pole2 = trend_plunge_to_cartesian(*strike_dip_to_pole(strike2, dip2))
    line  = np.cross(pole1, pole2)
    if np.linalg.norm(line) < 1e-10:
        return None   # parallel planes
    return cartesian_to_trend_plunge(line)


def angle_between_planes(
    strike1: float, dip1: float,
    strike2: float, dip2: float,
) -> float:
    """Dihedral angle between two planes (degrees, 0–90)."""
    pole1 = trend_plunge_to_cartesian(*strike_dip_to_pole(strike1, dip1))
    pole2 = trend_plunge_to_cartesian(*strike_dip_to_pole(strike2, dip2))
    cos_a = np.clip(np.dot(pole1, pole2), -1, 1)
    return float(np.degrees(np.arccos(abs(cos_a))))


def angle_between_lines(
    trend1: float, plunge1: float,
    trend2: float, plunge2: float,
) -> float:
    """Angle between two lines (degrees, 0–90)."""
    v1 = trend_plunge_to_cartesian(trend1, plunge1)
    v2 = trend_plunge_to_cartesian(trend2, plunge2)
    cos_a = np.clip(np.dot(v1, v2), -1, 1)
    return float(np.degrees(np.arccos(abs(cos_a))))


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

def rotation_matrix(
    axis_trend: float,
    axis_plunge: float,
    angle_deg: float,
) -> np.ndarray:
    """3×3 rotation matrix about an arbitrary axis using Rodrigues' formula."""
    k = trend_plunge_to_cartesian(axis_trend, axis_plunge)
    theta = np.radians(angle_deg)
    K = np.array([
        [0,    -k[2],  k[1]],
        [k[2],  0,    -k[0]],
        [-k[1], k[0],  0   ],
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def rotate_line(
    trend: float, plunge: float,
    axis_trend: float, axis_plunge: float,
    angle_deg: float,
) -> tuple[float, float]:
    """Rotate a line (trend/plunge) about an axis."""
    v     = trend_plunge_to_cartesian(trend, plunge)
    R     = rotation_matrix(axis_trend, axis_plunge, angle_deg)
    v_rot = R @ v
    return cartesian_to_trend_plunge(v_rot)


def rotate_plane(
    strike: float, dip: float,
    axis_trend: float, axis_plunge: float,
    angle_deg: float,
) -> tuple[float, float]:
    """Rotate a plane (strike/dip) about an axis.

    Returns (strike, dip) of the rotated plane.
    """
    pole     = trend_plunge_to_cartesian(*strike_dip_to_pole(strike, dip))
    R        = rotation_matrix(axis_trend, axis_plunge, angle_deg)
    pole_rot = R @ pole
    t_rot, p_rot = cartesian_to_trend_plunge(pole_rot)
    return pole_to_strike_dip(t_rot, p_rot)


# ---------------------------------------------------------------------------
# Fold analysis
# ---------------------------------------------------------------------------

def best_fit_fold_axis(orientations: list[tuple[float, float]]) -> tuple[float, float]:
    """Find the best-fit fold axis from a set of bedding orientations.

    Uses eigenvalue analysis of the orientation tensor (Allmendinger ch. 12).
    The eigenvector corresponding to the *smallest* eigenvalue is the fold axis
    (the pole to the best-fit girdle of bedding poles for a cylindrical fold).

    Parameters
    ----------
    orientations : list of (strike, dip) tuples.

    Returns
    -------
    (trend_deg, plunge_deg) of the fold axis.
    """
    poles = np.array([
        trend_plunge_to_cartesian(*strike_dip_to_pole(s, d))
        for s, d in orientations
    ])
    T = poles.T @ poles / len(poles)
    eigenvalues, eigenvectors = np.linalg.eigh(T)
    # Smallest eigenvalue → fold axis (perpendicular to the girdle plane)
    fold_axis = eigenvectors[:, 0]
    return cartesian_to_trend_plunge(fold_axis)


# ---------------------------------------------------------------------------
# Apparent dip
# ---------------------------------------------------------------------------

def apparent_dip(
    true_strike: float,
    true_dip: float,
    section_azimuth: float,
) -> float:
    """Compute the apparent dip on a cross-section.

    Parameters
    ----------
    true_strike      : strike of the bed (degrees).
    true_dip         : true dip of the bed (degrees).
    section_azimuth  : azimuth of the section line (degrees from North).

    Returns
    -------
    float
        Apparent dip angle (degrees).  Positive when dipping in the section
        azimuth direction.
    """
    delta    = np.radians(section_azimuth - true_strike)
    dip_rad  = np.radians(true_dip)
    apparent = np.degrees(np.arctan(np.tan(dip_rad) * np.sin(delta)))
    return float(apparent)


def true_dip_from_two_apparent(
    app_dip1: float, azimuth1: float,
    app_dip2: float, azimuth2: float,
) -> tuple[float, float]:
    """Compute true strike and dip from two apparent dips on different sections.

    Each apparent dip + section azimuth defines a line lying in the bedding
    plane.  The cross product of these two lines gives the bedding normal.

    Parameters
    ----------
    app_dip1, azimuth1 : apparent dip (degrees) and section azimuth (degrees)
                         for the first section.
    app_dip2, azimuth2 : same for the second section.

    Returns
    -------
    (true_strike_deg, true_dip_deg)
    """
    a1 = np.radians(azimuth1)
    a2 = np.radians(azimuth2)
    d1 = np.radians(app_dip1)
    d2 = np.radians(app_dip2)

    # Direction vectors of the two apparent-dip lines (lying in the bedding plane)
    v1 = np.array([np.cos(a1), np.sin(a1), np.tan(d1)])
    v2 = np.array([np.cos(a2), np.sin(a2), np.tan(d2)])

    # Normal to the bedding plane
    n = np.cross(v1, v2)
    n = n / np.linalg.norm(n)

    # Ensure the normal points downward in NED (positive z = down)
    if n[2] < 0:
        n = -n

    # The horizontal component of the downward normal points *uphill*;
    # the dip direction is opposite (downhill).
    # dip_azimuth = arctan2(-n[1], -n[0])
    dip_azimuth_deg = float(np.degrees(np.arctan2(-n[1], -n[0])) % 360)
    strike = (dip_azimuth_deg - 90) % 360
    dip    = 90.0 - float(np.degrees(np.arcsin(n[2])))

    return strike, dip
