"""Import time-depth relations from ASCII tables (checkshot / sonic TDR).

Two F3 shapes are supported, but the machinery is generic (column indices +
unit/datum are parameters):

``*_TD.txt`` — checkshot
    Whitespace columns ``depth(m)  TWT(s)``.  Depth is **MD from KB**:
    its values equal the well's markers/welltrack MD column, and the first row
    (30 m → 0 s) is the KB / seismic datum.  We resolve MD → **TVDSS** via the
    well's deviation and KB elevation (``TVDSS = tvd_at_md(MD) − kb``) so the
    stored relation is sea-level referenced — matching the sonic TDR and the
    section's seismic-datum frame.  TWT is in **seconds** in this file.

``*_DT_TVDSS.txt`` — sonic-integrated TDR
    Whitespace columns ``TVDSS(m)  TWT(ms)``.  Depth is already **TVDSS**
    (no datum conversion); TWT is in **milliseconds** (converted at the boundary).

Datum decision (pinned by ``tests/test_tdr_import.py``):
    checkshot MD 30 → 0 s   ⇒  TVDSS 0.0 m → 0.0 s
    checkshot MD 3150 → 3.234 s ⇒ TVDSS 3120.0 m → 3.234 s   (= sonic TDR extent)
If a well's datum is genuinely ambiguous the caller can override the
depth-reference / unit defaults (the import dialog exposes them).
"""
from __future__ import annotations

import os

import numpy as np

from section_tool.core.tdr import (
    TimeDepthRelation, seconds_from, metres_from)
from section_tool.core.zdomain import ZDomain

# Above this magnitude a TWT column is milliseconds, not seconds: a two-way time
# of 30 s would be ~45 km of section — far beyond any well. Safe ms/s splitter.
_TWT_S_CEILING = 30.0


def detect_twt_domain(twt_values) -> ZDomain:
    """Guess whether a TWT column is seconds or milliseconds by magnitude.

    A defensive default only — the import dialog's unit field is authoritative.
    """
    arr = np.asarray(twt_values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size and float(np.nanmax(np.abs(finite))) > _TWT_S_CEILING:
        return ZDomain.TWT_MS
    return ZDomain.TWT_S


def read_numeric_columns(path: str | os.PathLike, *, min_cols: int = 2) -> np.ndarray:
    """Read an ASCII table of whitespace-separated numbers into an (n, ncols) array.

    Skips blank lines, comment lines (``#`` / ``!``), header lines (any row whose
    relevant fields aren't all numeric), and short rows (< *min_cols* numeric
    fields). Ragged rows are truncated to the first *min_cols* — enough columns
    are kept for the configured depth/twt indices.
    """
    rows: list[list[float]] = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s[0] in "#!":
                continue
            parts = s.split()
            if len(parts) < min_cols:
                continue
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue  # header / non-numeric row
            rows.append(vals)
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    width = min(len(r) for r in rows)
    return np.array([r[:width] for r in rows], dtype=float)


def _md_to_tvdss(md, well) -> np.ndarray:
    """MD (m) → TVDSS (m, sea-level referenced) via the well's deviation + KB.

    For a vertical well this is ``MD − kb``; the deviation survey generalises it
    to deviated wells. ``kb`` is the KB elevation above mean sea level.
    """
    md = np.asarray(md, dtype=float)
    tvd_from_kb = np.array([well.deviation.tvd_at_md(float(m)) for m in md])
    return tvd_from_kb - float(well.kb)


def load_tdr(
    path: str | os.PathLike,
    well,
    *,
    kind: str,
    depth_col: int,
    twt_col: int,
    twt_domain: ZDomain | None,
    depth_domain: ZDomain = ZDomain.DEPTH_M,
    depth_input_reference: str = "TVDSS",
    resolve_md_to_tvdss: bool = False,
) -> TimeDepthRelation:
    """Generic TDR loader. Returns a TVDSS-referenced :class:`TimeDepthRelation`.

    *twt_domain* ``None`` auto-detects ms vs s by magnitude.  When
    *resolve_md_to_tvdss* is set the depth column is treated as MD and converted
    to TVDSS via *well*; otherwise it is taken at face value in *depth_domain*.
    """
    table = read_numeric_columns(path, min_cols=max(depth_col, twt_col) + 1)
    depth_raw = table[:, depth_col]
    twt_raw = table[:, twt_col]

    if twt_domain is None:
        twt_domain = detect_twt_domain(twt_raw)
    twt_s = seconds_from(twt_raw, twt_domain)

    if resolve_md_to_tvdss:
        depth_m = _md_to_tvdss(metres_from(depth_raw, depth_domain), well)
        depth_reference = "TVDSS"
    else:
        depth_m = metres_from(depth_raw, depth_domain)
        depth_reference = depth_input_reference

    tdr = TimeDepthRelation.from_pairs(
        depth_m, twt_s,
        kind=kind, depth_reference=depth_reference,
        source=os.path.basename(str(path)), well_uuid=well.uuid,
        construction={"kind": "time_depth_relation", "parents": [well.uuid],
                      "params": {"source": os.path.basename(str(path)),
                                 "imported_as": kind}},
    )
    return tdr


def load_checkshot(path, well, *, twt_domain: ZDomain | None = None) -> TimeDepthRelation:
    """Load an F3 ``*_TD.txt`` checkshot (cols: depth-MD m, TWT s)."""
    return load_tdr(
        path, well, kind="checkshot",
        depth_col=0, twt_col=1, twt_domain=twt_domain,
        depth_domain=ZDomain.DEPTH_M, resolve_md_to_tvdss=True)


def load_sonic_tdr(path, well, *, twt_domain: ZDomain | None = ZDomain.TWT_MS
                   ) -> TimeDepthRelation:
    """Load an F3 ``*_DT_TVDSS.txt`` sonic TDR (cols: TVDSS m, TWT ms)."""
    return load_tdr(
        path, well, kind="sonic_integrated",
        depth_col=0, twt_col=1, twt_domain=twt_domain,
        depth_domain=ZDomain.DEPTH_M, depth_input_reference="TVDSS",
        resolve_md_to_tvdss=False)
