from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import numpy as np
import lasio
from lasio import LASFile

from cross_section_tool.core.wells import LogCurve, Well


# Mnemonics to try (in order) when resolving optional header fields
_X_KEYS = ("XCOORD", "X", "XWELL", "EASTING", "EAST")
_Y_KEYS = ("YCOORD", "Y", "YWELL", "NORTHING", "NORTH")
_KB_KEYS = ("KB", "KELLY", "DF", "ELEV")

# Sentinel string values that indicate a header field is effectively absent
_EMPTY_STRINGS = frozenset(("", "unknown", "Unknown", "UNKNOWN", "--", "none", "None", "NONE"))


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


def _extract_header(las: LASFile) -> dict[str, Any]:
    well_name = _well_value(las, "WELL", default=None)
    uwi = _well_value(las, "UWI", "API", default=None)
    x = _well_value(las, *_X_KEYS, default=None)
    y = _well_value(las, *_Y_KEYS, default=None)
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
    if x is None:
        x = float(_well_value(las, *_X_KEYS, default=0.0))
    else:
        x = float(x)

    if y is None:
        y = float(_well_value(las, *_Y_KEYS, default=0.0))
    else:
        y = float(y)

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
