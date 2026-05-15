"""Phase 4 — Velocity model for depth ↔ TWT conversion."""
from __future__ import annotations

import math
from typing import Literal


class VelocityModel:
    """Simple velocity model supporting constant and linear-gradient (V0+Kz) types.

    Linear gradient (v0k):  v(z) = V0 + K*z
        depth_to_twt(d) = (1/K) * ln(1 + K*d/V0)
        twt_to_depth(t) = (V0/K) * (exp(K*t) - 1)

    Constant:  v = V0
        depth_to_twt(d) = 2 * d / V0
        twt_to_depth(t) = t * V0 / 2
    """

    def __init__(
        self,
        model_type: Literal["constant", "linear_v0k"] = "constant",
        v0: float = 1500.0,
        k:  float = 0.5,
    ) -> None:
        self.model_type: Literal["constant", "linear_v0k"] = model_type
        self.v0:  float = float(v0)
        self.k:   float = float(k)

    # ------------------------------------------------------------------

    def depth_to_twt(self, depth_m: float) -> float:
        """Return two-way travel time (seconds) for *depth_m* (metres)."""
        if self.model_type == "constant":
            return 2.0 * depth_m / self.v0
        # linear_v0k
        if self.k == 0.0:
            return 2.0 * depth_m / self.v0
        return (2.0 / self.k) * math.log(1.0 + self.k * depth_m / self.v0)

    def twt_to_depth(self, twt_s: float) -> float:
        """Return depth (metres) for *twt_s* (seconds)."""
        if self.model_type == "constant":
            return twt_s * self.v0 / 2.0
        if self.k == 0.0:
            return twt_s * self.v0 / 2.0
        return (self.v0 / self.k) * (math.exp(self.k * twt_s / 2.0) - 1.0)

    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"model_type": self.model_type, "v0": self.v0, "k": self.k}

    @classmethod
    def from_dict(cls, d: dict) -> "VelocityModel":
        return cls(
            model_type=d.get("model_type", "constant"),
            v0=d.get("v0", 1500.0),
            k=d.get("k", 0.5),
        )

    def __repr__(self) -> str:
        return f"VelocityModel(type={self.model_type!r}, V0={self.v0}, K={self.k})"
