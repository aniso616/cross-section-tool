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
from dataclasses import dataclass

import numpy as np

from section_tool.core.tdr import (
    TimeDepthRelation, seconds_from, metres_from)
from section_tool.core.zdomain import ZDomain

# Above this magnitude a TWT column is milliseconds, not seconds: a two-way time
# of 30 s would be ~45 km of section — far beyond any well. Safe ms/s splitter.
_TWT_S_CEILING = 30.0

# Shape heuristic thresholds. A checkshot is a sparse, irregularly-spaced set of
# measured pairs (F3's is 25 pairs); a sonic-integrated TDR is a dense, regular
# grid (F3's is 625 points on a 5 m grid). The split is on point count + spacing
# regularity, not on the menu item the user happened to click.
_DENSE_POINTS = 100        # ≥ this many points → "dense"
_REGULAR_CV = 0.05         # spacing coefficient-of-variation below this → "regular grid"


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


# ---------------------------------------------------------------------------
# Evidence-based classification — what does this file's *shape* look like?
# One import door classifies; it never lets data wear a grade the evidence
# contradicts without the user overriding it explicitly.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TdrClassification:
    """What the numbers in a time-depth file look like, and the kind they imply.

    ``suggested_kind`` is a heuristic from point count + spacing regularity;
    the import dialog pre-selects it but the user can override.  ``evidence``
    is the one-line human justification shown in the dialog.
    """
    n_points: int
    depth_min: float
    depth_max: float
    median_spacing: float
    spacing_regular: bool
    twt_domain: ZDomain          # detected ms vs s (display units)
    twt_min: float               # in the detected units
    twt_max: float
    suggested_kind: str          # checkshot | sonic_integrated | imported
    suggested_depth_reference: str
    evidence: str


def _spacing_stats(depth: np.ndarray) -> tuple[float, bool]:
    """(median |Δdepth|, is the spacing a near-constant regular grid?)."""
    d = np.sort(np.asarray(depth, dtype=float))
    diffs = np.diff(d)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.0, False
    med = float(np.median(diffs))
    mean = float(np.mean(diffs))
    cv = float(np.std(diffs) / mean) if mean > 0 else float("inf")
    return med, bool(cv < _REGULAR_CV)


def classify_tdr_table(table: np.ndarray, *, depth_col: int = 0,
                       twt_col: int = 1) -> TdrClassification:
    """Classify a parsed (n, ncols) numeric table by its depth/TWT shape."""
    depth = np.asarray(table[:, depth_col], dtype=float)
    twt = np.asarray(table[:, twt_col], dtype=float)
    n = int(len(depth))
    median_spacing, regular = _spacing_stats(depth)
    twt_domain = detect_twt_domain(twt)
    dense = n >= _DENSE_POINTS

    if dense and regular:
        kind, ref = "sonic_integrated", "TVDSS"
        evidence = (f"{n} points on a regular ~{median_spacing:.0f} m grid — this "
                    f"looks like an integrated/sonic TDR, not a checkshot.")
    elif n <= _DENSE_POINTS:
        kind, ref = "checkshot", "MD"
        evidence = (f"{n} irregularly-spaced pairs — this looks like a measured "
                    f"checkshot (depth↔TWT tie).")
    else:
        kind, ref = "imported", "TVDSS"
        evidence = (f"{n} points, {'regular' if regular else 'irregular'} spacing — "
                    f"kind is unclear; defaulting to a generic imported TDR.")

    return TdrClassification(
        n_points=n,
        depth_min=float(np.min(depth)), depth_max=float(np.max(depth)),
        median_spacing=median_spacing, spacing_regular=regular,
        twt_domain=twt_domain,
        twt_min=float(np.min(twt)), twt_max=float(np.max(twt)),
        suggested_kind=kind, suggested_depth_reference=ref, evidence=evidence)


def classify_tdr_file(path: str | os.PathLike, *, depth_col: int = 0,
                      twt_col: int = 1) -> TdrClassification:
    """Read *path* and classify it. Raises if no numeric rows are found."""
    table = read_numeric_columns(path, min_cols=max(depth_col, twt_col) + 1)
    return classify_tdr_table(table, depth_col=depth_col, twt_col=twt_col)


def load_tdr_as(
    path: str | os.PathLike,
    well,
    *,
    kind: str,
    depth_reference: str = "TVDSS",
    twt_domain: ZDomain | None = None,
    depth_col: int = 0,
    twt_col: int = 1,
) -> TimeDepthRelation:
    """Load with a user/heuristic-chosen *kind* and *depth_reference*.

    The single import door's loader.  ``depth_reference == "MD"`` means the depth
    column is measured-depth-from-KB and is resolved to TVDSS via the well
    (matching :func:`load_checkshot`); any other reference is taken at face value
    (matching :func:`load_sonic_tdr`).
    """
    return load_tdr(
        path, well, kind=kind, depth_col=depth_col, twt_col=twt_col,
        twt_domain=twt_domain, depth_domain=ZDomain.DEPTH_M,
        depth_input_reference=depth_reference,
        resolve_md_to_tvdss=(depth_reference == "MD"))


def tdr_shape_is_sonic(depth_m) -> bool:
    """True if a TDR's depth column has the dense, regular shape of a sonic TDR.

    Integrity check for already-loaded relations: a relation stamped
    ``kind="checkshot"`` whose depths look like this was almost certainly a
    sonic/integrated TDR imported through the wrong door (see the panel chip,
    which flags it for re-verification rather than trusting the grade silently).
    """
    depth = np.asarray(depth_m, dtype=float)
    if depth.size < _DENSE_POINTS:
        return False
    _median, regular = _spacing_stats(depth)
    return regular
