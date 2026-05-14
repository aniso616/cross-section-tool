from __future__ import annotations

import io
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import lasio
from lasio import LASFile

from cross_section_tool.core.wells import LogCurve, Well


# Mnemonics to try (in order) when resolving optional header fields
_X_KEYS = ("XCOORD", "X", "XWELL", "LONG", "LON", "EASTING", "SURF_X", "EAST")
_Y_KEYS = ("YCOORD", "Y", "YWELL", "LAT", "NORTHING", "SURF_Y", "NORTH")
_KB_KEYS = ("KB", "EKBR", "EKB", "KELLY", "DF", "ELEV")

# Sentinel string values that indicate a header field is effectively absent
_EMPTY_STRINGS = frozenset(("", "unknown", "Unknown", "UNKNOWN", "--", "none", "None", "NONE"))

# Patterns to extract X/Y from a free-text LOC field like "X = 606554.0 Y = 6080126.0"
_LOC_X_RE = re.compile(r'(?:^|[\s,;])[Xx]\s*[=:]\s*([-+]?\d+(?:\.\d+)?)', re.IGNORECASE)
_LOC_Y_RE = re.compile(r'(?:^|[\s,;])[Yy]\s*[=:]\s*([-+]?\d+(?:\.\d+)?)', re.IGNORECASE)


def _parse_loc_xy(las: LASFile) -> tuple[float | None, float | None]:
    """Try to extract X/Y from the LOC (well location) free-text header field."""
    loc_val = None
    for key in ("LOC", "LOCA", "LOCATION"):
        if key in las.well:
            v = las.well[key].value
            if v and str(v).strip() not in _EMPTY_STRINGS:
                loc_val = str(v)
                break
    if not loc_val:
        return None, None
    mx = _LOC_X_RE.search(loc_val)
    my = _LOC_Y_RE.search(loc_val)
    x = float(mx.group(1)) if mx else None
    y = float(my.group(1)) if my else None
    return x, y


def read_las(
    path: str | os.PathLike,
    *,
    x: float | None = None,
    y: float | None = None,
    kb: float | None = None,
    name: str | None = None,
    crs_epsg: int = 32632,
) -> Well:
    """Read a LAS 2.0 file from *path* and return a populated :class:`Well`.

    Parameters
    ----------
    path:
        Path to the .las / .LAS file.
    x, y:
        Override surface easting / northing in CRS units.  When omitted the
        reader searches common header mnemonics (XCOORD, X, EASTING …) and
        falls back to 0.0 when nothing is found.
    kb:
        Override Kelly Bushing elevation (metres).  Falls back to the KB
        header, then 0.0.
    name:
        Override the well name.  Falls back to the WELL header, then to the
        file stem, then to "Unnamed".
    crs_epsg:
        EPSG code for the coordinate reference system of *x* / *y*.  LAS
        files carry no CRS information, so the caller must supply this.
    """
    las = lasio.read(str(path))
    return _las_to_well(las, x=x, y=y, kb=kb, name=name, crs_epsg=crs_epsg, path=path)


def las_to_well(
    las: LASFile,
    *,
    x: float | None = None,
    y: float | None = None,
    kb: float | None = None,
    name: str | None = None,
    crs_epsg: int = 32632,
) -> Well:
    """Convert an already-loaded :class:`lasio.LASFile` to a :class:`Well`.

    Useful when the caller has already called :func:`lasio.read` or needs
    to pass custom lasio read options.
    """
    return _las_to_well(las, x=x, y=y, kb=kb, name=name, crs_epsg=crs_epsg, path=None)


def read_las_header(path: str | os.PathLike) -> dict[str, Any]:
    """Read only the header sections of a LAS file (no data arrays loaded).

    Returns a plain dict:

    .. code-block:: python

        {
            "well_name":   str | None,
            "uwi":         str | None,
            "x":           float | None,
            "y":           float | None,
            "kb":          float | None,
            "depth_start": float | None,
            "depth_stop":  float | None,
            "depth_step":  float | None,
            "depth_unit":  str | None,
            "curve_names": list[str],   # all curves except the depth index
        }
    """
    las = lasio.read(str(path), ignore_data=True)
    return _extract_header(las)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _well_value(las: LASFile, *keys: str, default: Any = None) -> Any:
    """Return the first matching well-header value, or *default*."""
    for key in keys:
        if key in las.well:
            val = las.well[key].value
            if val is None:
                continue
            if str(val).strip() in _EMPTY_STRINGS:
                continue
            return val
    return default


def _well_value_with_source(
    las: LASFile, *keys: str, default: Any = None
) -> tuple[Any, str | None]:
    """Return (value, source_key) for the first matching header field."""
    for key in keys:
        if key in las.well:
            val = las.well[key].value
            if val is None:
                continue
            if str(val).strip() in _EMPTY_STRINGS:
                continue
            return val, f"from {key}"
    return default, None


def extract_header_full(las: LASFile) -> dict[str, Any]:
    """Like :func:`_extract_header` but also returns provenance strings.

    Extra keys compared to the basic version:
    ``x_source``, ``y_source``, ``kb_source``, ``gl``.
    """
    well_name = _well_value(las, "WELL", default=None)
    uwi = _well_value(las, "UWI", "API", default=None)

    x, x_src = _well_value_with_source(las, *_X_KEYS)
    y, y_src = _well_value_with_source(las, *_Y_KEYS)
    if x is None or y is None:
        loc_x, loc_y = _parse_loc_xy(las)
        if x is None and loc_x is not None:
            x, x_src = loc_x, "parsed from LOC field"
        if y is None and loc_y is not None:
            y, y_src = loc_y, "parsed from LOC field"
    if x_src is None:
        x_src = "not found — enter manually"
    if y_src is None:
        y_src = "not found — enter manually"

    kb, kb_src = _well_value_with_source(las, *_KB_KEYS)
    gl, _      = _well_value_with_source(las, "EGL", "GL", "GLE", "GROUND")

    def _opt_float(v: Any) -> float | None:
        try:
            return float(v) if v is not None else None
        except (ValueError, TypeError):
            return None

    strt = _well_value(las, "STRT", default=None)
    stop = _well_value(las, "STOP", default=None)
    step = _well_value(las, "STEP", default=None)
    depth_unit = las.index_unit or None
    curve_names = [c.mnemonic for c in las.curves[1:]]

    return {
        "well_name":   str(well_name).strip() if well_name is not None else None,
        "uwi":         str(uwi).strip() if uwi is not None else None,
        "x":           _opt_float(x),
        "y":           _opt_float(y),
        "x_source":    x_src,
        "y_source":    y_src,
        "kb":          _opt_float(kb),
        "kb_source":   kb_src,
        "gl":          _opt_float(gl),
        "depth_start": _opt_float(strt),
        "depth_stop":  _opt_float(stop),
        "depth_step":  _opt_float(step),
        "depth_unit":  str(depth_unit).strip() if depth_unit else None,
        "curve_names": curve_names,
    }


def _extract_header(las: LASFile) -> dict[str, Any]:
    well_name = _well_value(las, "WELL", default=None)
    uwi = _well_value(las, "UWI", "API", default=None)
    x = _well_value(las, *_X_KEYS, default=None)
    y = _well_value(las, *_Y_KEYS, default=None)
    if x is None or y is None:
        loc_x, loc_y = _parse_loc_xy(las)
        if x is None:
            x = loc_x
        if y is None:
            y = loc_y
    kb = _well_value(las, *_KB_KEYS, default=None)
    strt = _well_value(las, "STRT", default=None)
    stop = _well_value(las, "STOP", default=None)
    step = _well_value(las, "STEP", default=None)
    depth_unit = las.index_unit or None
    curve_names = [c.mnemonic for c in las.curves[1:]]  # skip depth index

    def _opt_float(v: Any) -> float | None:
        try:
            return float(v) if v is not None else None
        except (ValueError, TypeError):
            return None

    return {
        "well_name": str(well_name).strip() if well_name is not None else None,
        "uwi": str(uwi).strip() if uwi is not None else None,
        "x": _opt_float(x),
        "y": _opt_float(y),
        "kb": _opt_float(kb),
        "depth_start": _opt_float(strt),
        "depth_stop": _opt_float(stop),
        "depth_step": _opt_float(step),
        "depth_unit": str(depth_unit).strip() if depth_unit else None,
        "curve_names": curve_names,
    }


def _las_to_well(
    las: LASFile,
    *,
    x: float | None,
    y: float | None,
    kb: float | None,
    name: str | None,
    crs_epsg: int,
    path: str | os.PathLike | None,
) -> Well:
    # --- well name ---
    if name:
        well_name = str(name)
    else:
        header_name = _well_value(las, "WELL", default=None)
        if header_name:
            well_name = str(header_name).strip()
        elif path:
            well_name = Path(str(path)).stem
        else:
            well_name = "Unnamed"

    # --- UWI ---
    uwi = _well_value(las, "UWI", "API", default="")
    uwi = str(uwi).strip() if uwi else ""

    # --- location ---
    _x_raw = x if x is not None else _well_value(las, *_X_KEYS, default=None)
    _y_raw = y if y is not None else _well_value(las, *_Y_KEYS, default=None)
    if _x_raw is None or _y_raw is None:
        loc_x, loc_y = _parse_loc_xy(las)
        if _x_raw is None:
            _x_raw = loc_x
        if _y_raw is None:
            _y_raw = loc_y
    x = float(_x_raw) if _x_raw is not None else 0.0
    y = float(_y_raw) if _y_raw is not None else 0.0

    if kb is None:
        kb = float(_well_value(las, *_KB_KEYS, default=0.0))
    else:
        kb = float(kb)

    well = Well(name=well_name, x=x, y=y, kb=kb, uwi=uwi)

    # --- depth index (first curve) ---
    if not las.curves:
        return well

    depth_mnemonic = las.curves[0].mnemonic
    depths = np.asarray(las[depth_mnemonic], dtype=float)

    if len(depths) == 0:
        return well  # no data rows — return well without logs

    # --- log curves ---
    for curve in las.curves[1:]:
        values = np.asarray(las[curve.mnemonic], dtype=float)
        # lasio has already substituted the declared null value with NaN
        lc = LogCurve(
            name=curve.mnemonic,
            units=curve.unit or "",
            depths=depths,
            values=values,
        )
        well.add_log(lc)

    return well
