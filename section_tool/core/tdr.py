"""TimeDepthRelation (TDR) — a first-class per-well time↔depth control entity.

A TDR is an ordered set of (depth, twt) pairs tying depth to two-way time for one
well: a *checkshot* (sparse, measured), a *sonic_integrated* relationship (dense,
derived from the sonic log), or a generic *imported* table.

SI internal truth: depth in **metres**, two-way time in **SECONDS**.  Unit and
datum conversion happen at the IO boundary (see ``io/tdr_io.py``) via the
``ZDomain`` thin adapter below; the entity itself is always (m, s).  ``ms`` only
appears at import/display.

Monotonic by construction — depth strictly increases down the well and twt
increases with it (velocity > 0), so the relationship is invertible.  Input that
is not monotonic is rejected on load (see :meth:`TimeDepthRelation.from_pairs`).

First-class entity, mirroring ``VelocityModel``: stable ``uuid`` + construction
metadata ``{kind, parents, params}`` + schema-versioned ``to_dict``/``from_dict``.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from section_tool.core.surfaces import new_entity_uuid
from section_tool.core.zdomain import ZDomain

SCHEMA_VERSION = 1

TDRKind = Literal["checkshot", "sonic_integrated", "imported"]
DepthReference = Literal["MD", "TVDSS", "TVD_KB"]

# Display labels for the depth reference — what the depth column is measured from.
DEPTH_REFERENCE_LABEL: dict[str, str] = {
    "MD":      "MD (measured depth from KB)",
    "TVDSS":   "TVDSS (true vertical depth subsea)",
    "TVD_KB":  "TVD (true vertical depth from KB)",
}

KIND_LABEL: dict[str, str] = {
    "checkshot":        "checkshot",
    "sonic_integrated": "sonic-integrated",
    "imported":         "imported",
}


# ---------------------------------------------------------------------------
# ZDomain thin adapter — convert an IO column to SI at the boundary.
# Reuses the existing ZDomain enum; does NOT add a new domain string.
# ---------------------------------------------------------------------------

def seconds_from(values, z: ZDomain) -> np.ndarray:
    """Convert a TWT column expressed in *z* (TWT_S or TWT_MS) to SI seconds."""
    arr = np.asarray(values, dtype=float)
    if z == ZDomain.TWT_S:
        return arr
    if z == ZDomain.TWT_MS:
        return arr / 1000.0
    raise ValueError(f"seconds_from expects a TWT domain, got {z!r}")


def metres_from(values, z: ZDomain) -> np.ndarray:
    """Convert a depth column expressed in *z* (a depth domain) to SI metres."""
    if not z.is_depth():
        raise ValueError(f"metres_from expects a depth domain, got {z!r}")
    return np.asarray(z.to_metres(np.asarray(values, dtype=float)), dtype=float)


class TimeDepthRelation:
    """Ordered (depth_m, twt_s) control for one well.

    Parameters
    ----------
    depth_m, twt_s:
        Equal-length SI arrays (metres, seconds), already datum-resolved.
    kind:
        ``checkshot`` | ``sonic_integrated`` | ``imported``.
    depth_reference:
        What the depth column is measured from — ``MD`` | ``TVDSS`` | ``TVD_KB``.
        Explicit so downstream conversions never guess the datum.
    source:
        Originating filename (provenance).
    well_uuid:
        Stable identity of the owning :class:`~section_tool.core.wells.Well`.
    """

    def __init__(
        self,
        depth_m,
        twt_s,
        *,
        kind: TDRKind = "imported",
        depth_reference: DepthReference = "MD",
        source: str = "",
        well_uuid: str = "",
        uuid: str | None = None,
        construction: dict | None = None,
    ) -> None:
        depth = np.asarray(depth_m, dtype=float)
        twt = np.asarray(twt_s, dtype=float)
        if depth.ndim != 1 or twt.ndim != 1:
            raise ValueError("depth_m and twt_s must be 1D arrays")
        if len(depth) != len(twt):
            raise ValueError("depth_m and twt_s must have the same length")
        if len(depth) < 2:
            raise ValueError("TimeDepthRelation requires at least 2 points")
        self._validate_monotonic(depth, twt)
        self._depth = depth.copy()
        self._twt = twt.copy()
        self.kind: TDRKind = kind
        self.depth_reference: DepthReference = depth_reference
        self.source = str(source)
        self.well_uuid = str(well_uuid)
        self.uuid = uuid or new_entity_uuid()
        self.construction = construction or {
            "kind": "time_depth_relation", "parents": [], "params": {}}
        self.schema_version = SCHEMA_VERSION

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_monotonic(depth: np.ndarray, twt: np.ndarray) -> None:
        """Reject non-monotonic input. Both depth and twt must strictly increase
        together so the relationship is single-valued and invertible."""
        if np.any(np.isnan(depth)) or np.any(np.isnan(twt)):
            raise ValueError("TimeDepthRelation rejects NaN in depth or twt")
        if np.any(np.diff(depth) <= 0.0):
            raise ValueError(
                "TimeDepthRelation depth column must be strictly increasing "
                "(sorted, no duplicates)")
        if np.any(np.diff(twt) <= 0.0):
            raise ValueError(
                "TimeDepthRelation twt column must be strictly increasing with "
                "depth (implies velocity > 0); non-monotonic input rejected")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def depth_m(self) -> np.ndarray:
        return self._depth.copy()

    @property
    def twt_s(self) -> np.ndarray:
        return self._twt.copy()

    @property
    def n_points(self) -> int:
        return len(self._depth)

    def depth_range(self) -> tuple[float, float]:
        return float(self._depth[0]), float(self._depth[-1])

    def twt_range(self) -> tuple[float, float]:
        return float(self._twt[0]), float(self._twt[-1])

    # ------------------------------------------------------------------
    # Interpolation (monotone piecewise-linear, clamped at the ends)
    # ------------------------------------------------------------------

    def twt_at_depth(self, z):
        """TWT (s) at depth *z* (m). Clamps to the end values outside the range."""
        return np.interp(np.asarray(z, dtype=float), self._depth, self._twt)

    def depth_at_twt(self, t):
        """Depth (m) at TWT *t* (s). Clamps to the end values outside the range."""
        return np.interp(np.asarray(t, dtype=float), self._twt, self._depth)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "uuid": self.uuid,
            "well_uuid": self.well_uuid,
            "kind": self.kind,
            "depth_reference": self.depth_reference,
            "source": self.source,
            "construction": self.construction,
            "depth_m": self._depth.tolist(),
            "twt_s": self._twt.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TimeDepthRelation":
        return cls(
            d["depth_m"], d["twt_s"],
            kind=d.get("kind", "imported"),
            depth_reference=d.get("depth_reference", "MD"),
            source=d.get("source", ""),
            well_uuid=d.get("well_uuid", ""),
            uuid=d.get("uuid"),
            construction=d.get("construction"),
        )

    @classmethod
    def from_pairs(cls, depth_m, twt_s, **kw) -> "TimeDepthRelation":
        """Build from raw (already-SI) pairs, sorting by depth first.

        Convenience for importers: sorts on depth so caller need not pre-order,
        then the constructor validates strict monotonicity.
        """
        depth = np.asarray(depth_m, dtype=float)
        twt = np.asarray(twt_s, dtype=float)
        order = np.argsort(depth, kind="stable")
        return cls(depth[order], twt[order], **kw)

    def __repr__(self) -> str:
        lo, hi = self.depth_range()
        return (f"TimeDepthRelation(kind={self.kind!r}, ref={self.depth_reference!r}, "
                f"n={self.n_points}, depth=[{lo:.1f}, {hi:.1f}]m)")
