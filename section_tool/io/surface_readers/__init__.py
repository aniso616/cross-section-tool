"""Surface reader registry and auto-dispatch."""
from __future__ import annotations

from .base import SurfaceReader
from .xyz_reader import XYZReader
from .opendtect_reader import OpendTectReader

SURFACE_READERS: list[SurfaceReader] = [
    OpendTectReader(),   # check .hor first (more specific)
    XYZReader(),         # permissive text fallback
]


def read_surface(filepath: str, **options):
    """Auto-detect format and read a surface file."""
    for reader in SURFACE_READERS:
        if reader.can_read(filepath):
            return reader.read(filepath, **options)
    raise ValueError(
        f"No surface reader available for: {filepath}\n"
        f"Supported extensions: {supported_extensions()}"
    )


def supported_extensions() -> list[str]:
    exts = []
    for r in SURFACE_READERS:
        exts.extend(r.extensions)
    return sorted(set(exts))
