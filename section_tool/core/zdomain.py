"""ZDomain — unified depth/time domain + units enum for surfaces and sections."""
from __future__ import annotations

from enum import Enum

import numpy as np


class ZDomain(str, Enum):
    """Carries domain and unit together; knows how to convert to metres."""

    DEPTH_M  = "depth_m"
    DEPTH_FT = "depth_ft"
    DEPTH_KM = "depth_km"
    TWT_MS   = "twt_ms"
    TWT_S    = "twt_s"
    ELEV_M   = "elev_m"   # elevation, positive up

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    def is_twt(self) -> bool:
        return self in (ZDomain.TWT_MS, ZDomain.TWT_S)

    def is_depth(self) -> bool:
        return self in (ZDomain.DEPTH_M, ZDomain.DEPTH_FT,
                        ZDomain.DEPTH_KM, ZDomain.ELEV_M)

    def units_label(self) -> str:
        _labels = {
            ZDomain.DEPTH_M:  "m",
            ZDomain.DEPTH_FT: "ft",
            ZDomain.DEPTH_KM: "km",
            ZDomain.TWT_MS:   "ms",
            ZDomain.TWT_S:    "s",
            ZDomain.ELEV_M:   "m",
        }
        return _labels[self]

    def display_label(self) -> str:
        _labels = {
            ZDomain.DEPTH_M:  "Depth (m)",
            ZDomain.DEPTH_FT: "Depth (ft)",
            ZDomain.DEPTH_KM: "Depth (km)",
            ZDomain.TWT_MS:   "TWT (ms)",
            ZDomain.TWT_S:    "TWT (s)",
            ZDomain.ELEV_M:   "Elevation (m)",
        }
        return _labels[self]

    # ------------------------------------------------------------------
    # Unit conversion
    # ------------------------------------------------------------------

    def to_metres(self, z: np.ndarray | float) -> np.ndarray | float:
        """Convert *z* to metres (or metres-equivalent for TWT)."""
        z = np.asarray(z, dtype=float) if not isinstance(z, float) else z
        if self == ZDomain.DEPTH_FT:
            return z * 0.3048
        if self == ZDomain.DEPTH_KM:
            return z * 1000.0
        if self == ZDomain.TWT_S:
            return z * 1000.0   # to ms — caller must do TWT→depth separately
        return z                # already m, or ms stays as ms

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def from_strings(cls, domain: str, units: str) -> "ZDomain":
        """Construct from legacy separate domain/units strings."""
        d = domain.lower().strip()
        u = units.lower().strip()
        if d in ("twt", "time"):
            return cls.TWT_MS if u in ("ms", "milliseconds") else cls.TWT_S
        if u == "ft":
            return cls.DEPTH_FT
        if u == "km":
            return cls.DEPTH_KM
        if u == "m" and d in ("elevation", "elev"):
            return cls.ELEV_M
        return cls.DEPTH_M
