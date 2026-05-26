from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np


@dataclasses.dataclass
class PolygonBoundary:
    """Reference to a horizon or fault pick that bounds a SectionPolygon.

    When the referenced entity changes, AppState re-runs
    ``compute_polygon_points`` and cascades ``polygon_modified`` to all views.

    Attributes
    ----------
    category:
        ``"Horizons"`` or ``"Faults"`` — which pick list to look up.
    index:
        Index into ``project.horizon_picks`` or ``project.fault_picks``.
    reversed:
        If ``True``, traverse the picks on this boundary in reverse order.
        Useful for forming closed loops (top boundary left→right, bottom
        boundary right→left).
    """

    category: str
    index: int
    reversed: bool = False


class SectionPolygon:
    """A closed polygon in section-space: (distance_along, depth) vertices.

    Vertices are stored in the section coordinate frame.  Convert to map
    coordinates via :meth:`to_map_coords` using the associated
    :class:`~section_tool.core.section.Section`.

    Polygons can be defined in two ways:

    * **Free-form** (legacy / user-drawn): explicit vertex array passed as
      ``vertices``.  Stored in :attr:`free_points`; :attr:`bounds` is empty.
    * **Reference-based**: :attr:`bounds` lists the entity picks that define
      the perimeter.  :attr:`compute_polygon_points` resolves these to
      vertices on demand; :attr:`free_points` is ``None``.

    Parameters
    ----------
    vertices:
        (N, 2) array of ``(distance, depth)`` pairs.  The polygon is
        implicitly closed — the first vertex is NOT repeated at the end.
    bounds:
        Optional list of :class:`PolygonBoundary` references.  When non-empty
        and *project* + *section_name* are supplied, :meth:`compute_polygon_points`
        assembles vertices from the referenced entity picks.
    """

    def __init__(
        self,
        vertices: list | np.ndarray,
        name: str = "",
        fill_color: str = "#9467bd",
        fill_alpha: float = 0.6,
        edge_color: str = "#555555",
        edge_width: float = 1.0,
        formation: str = "",
        section_name: str = "",
        bounds: list[PolygonBoundary] | None = None,
    ) -> None:
        verts = np.asarray(vertices, dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 2:
            raise ValueError("vertices must be an (N, 2) array of (distance, depth)")
        if len(verts) < 3:
            raise ValueError("SectionPolygon requires at least 3 vertices")
        self._vertices = verts.copy()
        # free_points: explicit vertex array — source of truth for free-form polygons.
        # For all polygons constructed via the vertices argument (including those
        # migrated from the database), free_points == _vertices.
        self.free_points: np.ndarray = self._vertices
        # bounds: entity references that define the polygon perimeter.
        # Empty for free-form polygons; non-empty for reference-based polygons.
        self.bounds: list[PolygonBoundary] = list(bounds) if bounds else []
        self.name = name
        self.fill_color = fill_color
        self.fill_alpha = float(np.clip(fill_alpha, 0.0, 1.0))
        self.edge_color = edge_color
        self.edge_width = float(edge_width)
        self.formation: str = formation
        self.section_name: str = section_name
        # Kinematic restoration: how this polygon was constructed
        self.construction_rule = None  # ConstructionRule | None
        self.visible: bool = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices.copy()

    @property
    def n_vertices(self) -> int:
        return len(self._vertices)

    def is_bound(self) -> bool:
        """True if this polygon is defined by entity boundary references."""
        return len(self.bounds) > 0

    def is_free(self) -> bool:
        """True if this polygon is defined by an explicit vertex array."""
        return self.free_points is not None and not self.is_bound()

    @property
    def area(self) -> float:
        """Signed area in section-space units² (Shoelace formula).

        Always returns the absolute value (positive).
        """
        v = self._vertices
        n = len(v)
        a = 0.0
        for i in range(n):
            j = (i + 1) % n
            a += v[i, 0] * v[j, 1] - v[j, 0] * v[i, 1]
        return abs(a) * 0.5

    # ------------------------------------------------------------------
    # Reference-based geometry
    # ------------------------------------------------------------------

    def compute_polygon_points(
        self, project=None, section_name: str = ""
    ) -> np.ndarray:
        """Return the effective ``(N, 2)`` vertex array for this polygon.

        If :attr:`bounds` is non-empty and *project* / *section_name* are
        supplied, assembles vertices from the referenced entity picks.
        Falls back to :attr:`free_points` on any failure or when bounds are
        empty.
        """
        if self.bounds and project is not None and section_name:
            try:
                return self._resolve_bounds(project, section_name)
            except Exception:
                pass
        return self.free_points

    def _resolve_bounds(self, project, section_name: str) -> np.ndarray:
        """Concatenate picks from each boundary reference into a polygon."""
        segments: list[np.ndarray] = []
        for bound in self.bounds:
            picks = (
                project.horizon_picks
                if bound.category == "Horizons"
                else project.fault_picks
            )
            if bound.index >= len(picks):
                raise ValueError(
                    f"PolygonBoundary index {bound.index} out of range "
                    f"for {bound.category}"
                )
            hp = picks[bound.index]
            idxs = hp.section_indices(section_name)
            if len(idxs) < 1:
                raise ValueError(
                    f"No picks on section '{section_name}' for {bound.category}[{bound.index}]"
                )
            seg = np.column_stack([hp._distances[idxs], hp._depths[idxs]])
            if bound.reversed:
                seg = seg[::-1]
            segments.append(seg)
        if not segments:
            raise ValueError("No boundary segments")
        combined = np.concatenate(segments, axis=0)
        if len(combined) < 3:
            raise ValueError("Too few points to form a polygon")
        return combined

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def to_map_coords(self, section) -> list[tuple[float, float, float]]:
        """Convert vertices to ``(x, y, z)`` map coordinates via *section*.

        Returns a list of ``(easting, northing, -depth)`` tuples (elevation
        convention: depth negated to give elevation).
        """
        result = []
        for dist, depth in self._vertices:
            x, y = section.section_to_map(float(dist))
            result.append((x, y, -float(depth)))
        return result

    # ------------------------------------------------------------------
    # Closed-polygon helpers
    # ------------------------------------------------------------------

    def closed_distances(self) -> np.ndarray:
        """Distance axis with the first vertex appended to close the polygon."""
        return np.append(self._vertices[:, 0], self._vertices[0, 0])

    def closed_depths(self) -> np.ndarray:
        """Depth axis with the first vertex appended to close the polygon."""
        return np.append(self._vertices[:, 1], self._vertices[0, 1])

    def __repr__(self) -> str:
        bound_info = f", {len(self.bounds)} bounds" if self.bounds else ""
        return (
            f"SectionPolygon(name={self.name!r}, n_vertices={self.n_vertices}"
            f"{bound_info}, fill={self.fill_color!r})"
        )
