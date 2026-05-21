"""XYZ ASCII surface reader.

Reads whitespace / comma / tab-delimited files with at least three numeric
columns interpreted as X, Y, Z.  Comment lines (#), blank lines, and header
rows (any line that cannot be parsed as all-numeric) are silently skipped.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from section_tool.core.surfaces import Surface


def read_xyz(
    path: str | os.PathLike,
    *,
    name: str | None = None,
    x_col: int = 0,
    y_col: int = 1,
    z_col: int = 2,
    delimiter: str | None = None,   # None → any whitespace / auto-detect
    skip_header: int = 0,
    z_units: str = "m",
    crs_epsg: int = 32632,
    display_color: str = "#E87722",
) -> Surface:
    """Read an XYZ ASCII file and return a :class:`Surface`.

    Parameters
    ----------
    path:
        Path to the file.
    name:
        Surface name. Defaults to the file stem.
    x_col, y_col, z_col:
        Zero-based column indices for X, Y, Z. Default: 0, 1, 2.
    delimiter:
        Column separator. ``None`` → try comma, then whitespace.
    skip_header:
        Number of lines to skip unconditionally at the start.
    z_units:
        Units of the Z column (``'m'``, ``'ft'``, ``'km'``, ``'ms'``).
    crs_epsg:
        EPSG code of the horizontal CRS.
    """
    path = Path(path)
    if name is None:
        name = path.stem

    xs, ys, zs = [], [], []

    with open(path, encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh):
            if lineno < skip_header:
                continue
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Auto-detect delimiter on first data line
            if delimiter is None:
                if "," in line:
                    sep = ","
                else:
                    sep = None        # any whitespace
            else:
                sep = delimiter
            try:
                parts = line.split(sep) if sep else line.split()
                x = float(parts[x_col])
                y = float(parts[y_col])
                z = float(parts[z_col])
                if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                    continue
                xs.append(x); ys.append(y); zs.append(z)
            except (ValueError, IndexError):
                continue   # skip header rows / malformed lines

    if len(xs) < 3:
        raise ValueError(
            f"XYZ file contains only {len(xs)} valid data rows (need ≥ 3): {path}"
        )

    return Surface(
        np.array(xs), np.array(ys), np.array(zs),
        name=name,
        z_units=z_units,
        crs_epsg=crs_epsg,
        display_color=display_color,
        source_file=str(path),
        source_format="xyz",
    )
