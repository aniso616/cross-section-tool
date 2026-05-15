from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError


@dataclass(frozen=True)
class CRSInfo:
    """Lightweight metadata snapshot for a CRS."""
    epsg: int
    name: str
    is_projected: bool
    linear_units: str   # raw pyproj unit name, e.g. 'metre', 'US survey foot'; '' for geographic


# ---------------------------------------------------------------------------
# CRS look-up
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def get_crs_info(epsg: int) -> CRSInfo:
    """Return metadata for *epsg*.  Raises ValueError for invalid codes."""
    try:
        crs = CRS.from_epsg(epsg)
    except CRSError as exc:
        raise ValueError(f"Invalid EPSG code {epsg}: {exc}") from exc
    projected = bool(crs.is_projected)
    units = crs.axis_info[0].unit_name if projected else ""
    return CRSInfo(epsg=epsg, name=crs.name, is_projected=projected, linear_units=units)


def is_projected(epsg: int) -> bool:
    """Return True if *epsg* identifies a projected (Cartesian) CRS."""
    return get_crs_info(epsg).is_projected


def linear_units(epsg: int) -> str:
    """Return the raw linear unit name for a projected CRS (e.g. 'metre', 'US survey foot').

    Returns an empty string for geographic CRS.
    """
    return get_crs_info(epsg).linear_units


# ---------------------------------------------------------------------------
# Normalised unit helpers
# ---------------------------------------------------------------------------

_METRE_KEYWORDS = {"metr", "meter"}
_FOOT_KEYWORDS = {"foot", "feet"}


def units_are_metres(epsg: int) -> bool:
    """True if the projected CRS uses metres (or metric-equivalent)."""
    u = linear_units(epsg).lower()
    return any(kw in u for kw in _METRE_KEYWORDS)


def units_are_feet(epsg: int) -> bool:
    """True if the projected CRS uses feet (any foot variant)."""
    u = linear_units(epsg).lower()
    return any(kw in u for kw in _FOOT_KEYWORDS)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_projected_crs(epsg: int) -> None:
    """Raise ValueError if *epsg* is not a valid projected CRS.

    Call this when accepting user-supplied CRS codes for section nodes,
    surface grids, or well locations that require linear (metre/foot) units.
    """
    info = get_crs_info(epsg)   # raises ValueError for invalid EPSG
    if not info.is_projected:
        raise ValueError(
            f"EPSG:{epsg} ({info.name!r}) is not a projected CRS. "
            "Section coordinates require a projected CRS with linear units."
        )


# ---------------------------------------------------------------------------
# Coordinate transformation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _get_transformer(from_epsg: int, to_epsg: int) -> Transformer:
    """Cached Transformer between two EPSG codes (always_xy=True)."""
    return Transformer.from_crs(
        CRS.from_epsg(from_epsg),
        CRS.from_epsg(to_epsg),
        always_xy=True,
    )


def transform_points(
    xs: list | np.ndarray,
    ys: list | np.ndarray,
    from_epsg: int,
    to_epsg: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject coordinate arrays from *from_epsg* to *to_epsg*.

    Uses (easting/longitude, northing/latitude) order (always_xy=True).
    Empty arrays are returned unchanged.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size == 0:
        return xs.copy(), ys.copy()
    t = _get_transformer(from_epsg, to_epsg)
    tx, ty = t.transform(xs, ys)
    return np.asarray(tx, dtype=float), np.asarray(ty, dtype=float)


def transform_section(section, to_epsg: int):
    """Return a new Section with nodes reprojected from section.crs_epsg to *to_epsg*.

    All other Section attributes are preserved.
    """
    # Local import avoids module-level circular dependency if section ever imports crs
    from section_tool.core.section import Section  # noqa: PLC0415

    xs = section._nodes[:, 0]
    ys = section._nodes[:, 1]
    new_xs, new_ys = transform_points(xs, ys, section.crs_epsg, to_epsg)
    return Section(
        np.column_stack([new_xs, new_ys]),
        name=section.name,
        depth_domain=section.depth_domain,
        depth_units=section.depth_units,
        vertical_exaggeration=section.vertical_exaggeration,
        crs_epsg=to_epsg,
    )
