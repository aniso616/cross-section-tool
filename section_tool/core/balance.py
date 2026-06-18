"""Balance analysis for cross-section restoration — pure geometry, no UI.

Dahlstrom's principles for a balanced, admissible section:

* **Area balance** — bed area is conserved under plane strain (no material
  leaves the section plane), so a deformed polygon and its restored equivalent
  should have equal area; a large discrepancy flags a non-balanced section.
* **Line-length balance** — a bed's length is conserved (flexural slip / no
  layer-parallel strain), so deformed and restored bed lengths should match.
* **Depth to detachment** — the excess area above a regional datum divided by the
  shortening estimates the detachment depth (``d = excess_area / shortening``).

Geometry is ``(x, y)`` in section metres: x = along-section distance, y = depth
(positive DOWN). SI internal. Everything returns a typed result object so a
caller can display intermediate values and show its work, not just a number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Default fractional area discrepancy above which a section is flagged unbalanced.
DEFAULT_AREA_TOLERANCE = 0.05      # 5%


# ---------------------------------------------------------------------------
# Primitive measurements
# ---------------------------------------------------------------------------

def horizon_line_length(points) -> float:
    """Cumulative arc length of an ``(N, 2)`` polyline of ``(x, y)`` metres.

    Domain-agnostic: deformed and restored copies are both in section metres, so
    the same Euclidean arc length applies to each.
    """
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2:
        return 0.0
    return float(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1])).sum())


def polygon_area(points) -> float:
    """SIGNED area of a closed polygon via the shoelace formula (section metres²).

    Sign convention — with x = along-section (rightward +) and y = depth (DOWN +,
    i.e. the on-screen frame), a vertex order that traces the polygon CLOCKWISE
    ON SCREEN yields a POSITIVE area. Concretely
    ``[(0,0),(W,0),(W,H),(0,H)]`` (right along the top, down the right side, left
    along the bottom) returns ``+W·H``. The polygon is implicitly closed (the
    last vertex connects back to the first). Callers needing only magnitude take
    ``abs()``; the sign is exposed so a winding/order bug surfaces instead of
    being silently absorbed.
    """
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


# ---------------------------------------------------------------------------
# Balance comparisons (deformed vs restored)
# ---------------------------------------------------------------------------

@dataclass
class AreaBalance:
    """Dahlstrom area balance for one polygon: areas conserved under plane strain."""
    name: str
    deformed_area: float
    restored_area: float

    @property
    def discrepancy(self) -> float:
        """Fractional area discrepancy ``|deformed − restored| / |restored|``.

        0 = perfectly balanced; ``inf`` if restored is ~0 but deformed is not.
        """
        denom = abs(self.restored_area)
        if denom < 1e-12:
            return 0.0 if abs(self.deformed_area) < 1e-12 else float("inf")
        return abs(self.deformed_area - self.restored_area) / denom

    def is_balanced(self, tol: float = DEFAULT_AREA_TOLERANCE) -> bool:
        return self.discrepancy <= tol


def area_balance(deformed_polygon, restored_polygon, *, name: str = "") -> AreaBalance:
    """Area balance between a deformed polygon and its restored equivalent.

    Both arguments are ``(N, 2)`` vertex arrays in section metres. Magnitudes are
    compared (winding-independent); :func:`polygon_area`'s sign is only for
    diagnosing order bugs.
    """
    return AreaBalance(name=name,
                       deformed_area=abs(polygon_area(deformed_polygon)),
                       restored_area=abs(polygon_area(restored_polygon)))


@dataclass
class LineLengthBalance:
    """Dahlstrom line-length balance for one bed: bed length conserved."""
    name: str
    deformed_length: float
    restored_length: float

    @property
    def shortening(self) -> float:
        """``restored − deformed`` (m): positive = the bed is shorter when deformed."""
        return self.restored_length - self.deformed_length

    @property
    def discrepancy(self) -> float:
        denom = abs(self.restored_length)
        if denom < 1e-12:
            return 0.0 if abs(self.deformed_length) < 1e-12 else float("inf")
        return abs(self.deformed_length - self.restored_length) / denom

    def is_balanced(self, tol: float = DEFAULT_AREA_TOLERANCE) -> bool:
        return self.discrepancy <= tol


def line_length_balance(deformed: dict, restored: dict,
                        names: dict | None = None) -> "list[LineLengthBalance]":
    """Per-bed line-length comparison (Dahlstrom line-length conservation).

    *deformed* / *restored* map a bed key → its ``(N, 2)`` ``(x, y)`` polyline.
    The key is the bed's UUID — Step-3's preserved-UUID snapshot policy lets a
    restored bed pair to its deformed self by equality. *names* optionally maps a
    key → display name. Returns one :class:`LineLengthBalance` per bed present in
    BOTH, in *deformed* iteration order.
    """
    names = names or {}
    out: list[LineLengthBalance] = []
    for key, dpts in deformed.items():
        if key in restored:
            out.append(LineLengthBalance(
                name=names.get(key, str(key)),
                deformed_length=horizon_line_length(dpts),
                restored_length=horizon_line_length(restored[key])))
    return out


# ---------------------------------------------------------------------------
# Depth to detachment (excess-area method)
# ---------------------------------------------------------------------------

@dataclass
class DetachmentDepth:
    """Excess-area detachment-depth estimate ``d = excess_area / shortening``.

    Records its inputs and formula so a display can show its work — a geologist
    should be able to audit the number, not just read it.
    """
    excess_area: float          # m² between the deformed bed(s) and the datum
    shortening: float           # m of horizontal shortening

    @property
    def depth(self) -> float:
        if abs(self.shortening) < 1e-12:
            return float("nan")
        return self.excess_area / self.shortening

    @property
    def formula(self) -> str:
        return "d = excess_area / shortening"

    def explain(self) -> str:
        return (f"d = {self.excess_area:,.0f} m² / {self.shortening:,.1f} m "
                f"= {self.depth:,.0f} m")


def depth_to_detachment(excess_area: float, shortening: float) -> DetachmentDepth:
    """Classic Dahlstrom excess-area estimate of detachment depth.

    *excess_area* (m²) is the area between the deformed bed and the regional
    datum; *shortening* (m) the horizontal shortening. Returns a
    :class:`DetachmentDepth` carrying the inputs + formula.
    """
    return DetachmentDepth(float(excess_area), float(shortening))
