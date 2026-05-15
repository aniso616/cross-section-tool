"""Live topological intersection graph for section-space interpretation lines.

All lines on a section (horizons, faults, reference lines, boundaries) are
maintained as a planar graph.  When any line changes, all intersections
involving that line are recomputed automatically.

Usage
-----
    topo = SectionTopology("S1", section_length=10_000, max_depth=5_000)
    topo.update_line("horizon_0", "horizon", [(500, 1000), (9500, 1200)])
    topo.update_line("fault_0",   "fault",   [(4000, 0),  (5000, 3000)])
    pts = topo.intersections          # list[IntersectionPoint]
    polys = topo.get_all_faces()      # list[shapely.Polygon]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from shapely.geometry import LineString
from shapely.ops import polygonize, unary_union


@dataclass
class IntersectionPoint:
    """A computed intersection between two named section lines."""
    x: float        # distance along section
    y: float        # depth
    line_a: str     # name of first line
    line_b: str     # name of second line
    type: str = "unknown"


class SectionTopology:
    """Live planar graph of all lines on a section.

    Parameters
    ----------
    section_name:
        Name of the section this topology belongs to.
    section_length:
        Full horizontal extent (distance axis) in data units.
    max_depth:
        Lower depth bound for the boundary rectangle.
    """

    def __init__(self,
                 section_name: str,
                 section_length: float = 10_000.0,
                 max_depth: float = 5_000.0) -> None:
        self.section_name = section_name
        self._section_length = float(section_length)
        self._max_depth = float(max_depth)
        # name → (line_type, LineString)
        self._lines: Dict[str, tuple[str, LineString]] = {}
        self._intersections: List[IntersectionPoint] = []
        self._dirty = True
        self._update_boundaries()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def update_bounds(self, section_length: float, max_depth: float) -> None:
        """Update the section extent, rebuild boundaries, and re-extend user lines."""
        self._section_length = float(section_length)
        self._max_depth = float(max_depth)
        self._update_boundaries()
        # Re-extend all user lines so they still reach the new edges
        for name, (ltype, ls) in list(self._lines.items()):
            if not name.startswith("__"):
                extended = self._extend_to_edges(list(ls.coords))
                self._lines[name] = (ltype, LineString(extended))
        self._dirty = True

    # ------------------------------------------------------------------
    # Line management
    # ------------------------------------------------------------------

    def update_line(self, name: str, line_type: str,
                    coords: list[tuple[float, float]]) -> None:
        """Add or replace a named line.  Extends the line to section boundaries."""
        if len(coords) < 2:
            self.remove_line(name)
            return
        extended = self._extend_to_edges(coords)
        self._lines[name] = (line_type, LineString(extended))
        self._dirty = True

    def remove_line(self, name: str) -> None:
        """Remove a user line (boundary lines are not removable)."""
        if name in self._lines and not name.startswith("__"):
            del self._lines[name]
            self._dirty = True

    def clear_user_lines(self) -> None:
        """Remove all non-boundary lines."""
        keys = [k for k in self._lines if not k.startswith("__")]
        for k in keys:
            del self._lines[k]
        self._dirty = True

    # ------------------------------------------------------------------
    # Intersection queries
    # ------------------------------------------------------------------

    @property
    def intersections(self) -> List[IntersectionPoint]:
        if self._dirty:
            self.recompute_all()
        return list(self._intersections)

    def get_intersections_for(self, name: str) -> List[IntersectionPoint]:
        return [p for p in self.intersections
                if p.line_a == name or p.line_b == name]

    def get_snap_targets(self) -> List[tuple[float, float]]:
        """Interior (non-boundary) intersection coordinates — snap targets during editing."""
        return [(p.x, p.y) for p in self.intersections
                if "boundary" not in p.type]

    # ------------------------------------------------------------------
    # Face detection
    # ------------------------------------------------------------------

    def get_all_faces(self):
        """Return closed Shapely Polygons representing all bounded faces.

        Always includes the section boundary rectangle, even when no
        interpretation lines exist (returns a single polygon covering the
        whole section).  Slivers smaller than 0.1 % of the section area
        are filtered out.
        """
        if self._dirty:
            self.recompute_all()

        all_ls = [ls for _, ls in self._lines.values()]
        if not all_ls:
            return []

        # Ensure the boundary ring is present even if it was somehow cleared
        xl, xr = 0.0, self._section_length
        yt, yb = 0.0, self._max_depth
        boundary_ring = LineString(
            [(xl, yt), (xr, yt), (xr, yb), (xl, yb), (xl, yt)]
        )
        # Only add if our stored boundary is the 4-segment variant; the single
        # ring replaces them to give polygonize clean shared nodes.
        user_ls = [ls for n, (_, ls) in self._lines.items()
                   if not n.startswith("__")]
        lines_to_poly = [boundary_ring] + user_ls

        try:
            merged = unary_union(lines_to_poly)
            polys = list(polygonize(merged))
        except Exception:
            return []

        # Filter slivers (< 0.1 % of section bounding box)
        bbox_area = self._section_length * self._max_depth
        min_area = bbox_area * 0.001 if bbox_area > 0 else 1.0
        return [p for p in polys if p.area >= min_area]

    # ------------------------------------------------------------------
    # Full recompute
    # ------------------------------------------------------------------

    def recompute_all(self) -> None:
        """Recompute all pairwise intersections from scratch."""
        self._intersections = []
        names = list(self._lines.keys())
        non_bnd = [n for n in names if not n.startswith("__")]
        bnd = [n for n in names if n.startswith("__")]

        # Non-boundary pairs
        for i, na in enumerate(non_bnd):
            ta, la = self._lines[na]
            for nb in non_bnd[i + 1:]:
                tb, lb = self._lines[nb]
                self._intersect_pair(na, ta, la, nb, tb, lb)
            # Non-boundary vs boundary
            for nb in bnd:
                tb, lb = self._lines[nb]
                self._intersect_pair(na, ta, la, nb, tb, lb)

        self._dirty = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_boundaries(self) -> None:
        xl, xr = 0.0, self._section_length
        yt, yb = 0.0, self._max_depth
        self._lines["__left__"]   = ("boundary", LineString([(xl, yt), (xl, yb)]))
        self._lines["__right__"]  = ("boundary", LineString([(xr, yt), (xr, yb)]))
        self._lines["__top__"]    = ("boundary", LineString([(xl, yt), (xr, yt)]))
        self._lines["__bottom__"] = ("boundary", LineString([(xl, yb), (xr, yb)]))

    def _extend_to_edges(self,
                         coords: list[tuple[float, float]],
                         ) -> list[tuple[float, float]]:
        """Extrapolate the line to reach x=0 and x=section_length.

        Uses linear extrapolation from the nearest two picks.  When only a
        single pick exists the line is held constant at that pick's depth.
        The resulting z-values are clamped to [0, max_depth].
        """
        coords = sorted(coords, key=lambda p: p[0])
        xl, xr = 0.0, self._section_length
        yt, yb = 0.0, self._max_depth
        result = list(coords)

        def _clamp(z: float) -> float:
            return max(yt, min(yb, z))

        def _extrapolate(d0, z0, d1, z1, x_target):
            dd = d1 - d0
            if abs(dd) < 1e-9:
                return z0
            return z0 + (z1 - z0) * (x_target - d0) / dd

        # Extend left — skip for vertical/near-vertical lines (dx ≈ 0)
        if coords[0][0] > xl:
            dx_left = coords[1][0] - coords[0][0] if len(coords) >= 2 else 0.0
            if abs(dx_left) > 1e-9:
                z_xl = _extrapolate(coords[0][0], coords[0][1],
                                     coords[1][0], coords[1][1], xl)
                result = [(xl, _clamp(z_xl))] + result

        # Extend right — skip for vertical/near-vertical lines (dx ≈ 0)
        if coords[-1][0] < xr:
            dx_right = coords[-1][0] - coords[-2][0] if len(coords) >= 2 else 0.0
            if abs(dx_right) > 1e-9:
                z_xr = _extrapolate(coords[-2][0], coords[-2][1],
                                     coords[-1][0], coords[-1][1], xr)
                result = result + [(xr, _clamp(z_xr))]

        return result

    def _intersect_pair(self, na: str, ta: str, la: LineString,
                        nb: str, tb: str, lb: LineString) -> None:
        try:
            inter = la.intersection(lb)
        except Exception:
            return
        if inter.is_empty:
            return

        # Classify
        if ta == "boundary":
            itype = f"{tb}_boundary"
        elif tb == "boundary":
            itype = f"{ta}_boundary"
        else:
            itype = f"{ta}_{tb}"

        if inter.geom_type == "Point":
            pts: list[tuple[float, float]] = [(inter.x, inter.y)]
        elif inter.geom_type == "MultiPoint":
            pts = [(p.x, p.y) for p in inter.geoms]
        else:
            return  # collinear overlap — skip

        for x, y in pts:
            self._intersections.append(
                IntersectionPoint(float(x), float(y), na, nb, itype)
            )

    def __repr__(self) -> str:
        n = len([k for k in self._lines if not k.startswith("__")])
        return (f"SectionTopology(section={self.section_name!r}, "
                f"lines={n}, intersections={len(self._intersections)})")
