"""Surface reader dispatch — auto-detect format from file extension / content."""
from __future__ import annotations

import os
from pathlib import Path

from section_tool.core.surfaces import Surface


_XYZ_EXTENSIONS = {".xyz", ".txt", ".csv", ".dat", ".asc"}


def load_surface(
    path: str | os.PathLike,
    *,
    name: str | None = None,
    z_units: str = "m",
    crs_epsg: int = 32632,
    display_color: str = "#E87722",
    **reader_kwargs,
) -> Surface:
    """Load a surface from *path*, auto-detecting the format.

    Supported formats (priority order):
    1. XYZ ASCII  — .xyz, .txt, .csv, .dat, .asc
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in _XYZ_EXTENSIONS:
        return _try_xyz(path, name=name, z_units=z_units,
                        crs_epsg=crs_epsg, display_color=display_color,
                        **reader_kwargs)

    # Unknown extension — try XYZ as a fallback
    return _try_xyz(path, name=name, z_units=z_units,
                    crs_epsg=crs_epsg, display_color=display_color,
                    **reader_kwargs)


def _try_xyz(path, **kw) -> Surface:
    from section_tool.io.xyz import read_xyz
    return read_xyz(path, **kw)


def supported_extensions() -> list[str]:
    """Return all file extensions that load_surface() can handle."""
    return sorted(_XYZ_EXTENSIONS)
