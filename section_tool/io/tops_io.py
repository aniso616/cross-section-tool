"""Import formation tops (markers) from an F3 ``*_markers.txt`` table.

Shape: whitespace/tab columns ``MD(m)  name``, no header.  Names may contain
spaces (``FS 3``, ``Lower Low Sonic``, ``NMRF (Mid_Mio_Unc)``), so the name is
the remainder of the line after the MD — not a single token.

Tops are added to an existing well as ``{name: MD}`` (MD in metres, the well's
native top representation).  No TWT is stored: a marker's two-way time is derived
at use time from the well's checkshot/TDR (see ``core/tdr.py``), per the
canonical/derived pattern — the ``well_tops`` schema deliberately has no TWT.
"""
from __future__ import annotations

import os
import re

# MD (float)  name (rest of line)
_ROW_RE = re.compile(r"^\s*([-+]?\d*\.?\d+)\s+(.+?)\s*$")


def parse_markers(path: str | os.PathLike) -> list[tuple[str, float]]:
    """Parse an F3 markers file → list of ``(name, md_m)`` in file order.

    Skips blank lines, comments (``#`` / ``!``) and rows that don't match the
    ``index MD name`` shape (e.g. a trailing index-only line).
    """
    out: list[tuple[str, float]] = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            s = line.rstrip("\n")
            if not s.strip() or s.lstrip()[0] in "#!":
                continue
            m = _ROW_RE.match(s)
            if not m:
                continue
            md = float(m.group(1))
            name = m.group(2).strip()
            if name:
                out.append((name, md))
    return out


def load_markers_into_well(path: str | os.PathLike, well) -> int:
    """Import markers from *path* into *well* as formation tops. Returns the count.

    Existing tops with the same name are overwritten (``add_formation_top`` is a
    dict assignment), so re-importing is idempotent.
    """
    pairs = parse_markers(path)
    for name, md in pairs:
        well.add_formation_top(name, md)
    return len(pairs)
