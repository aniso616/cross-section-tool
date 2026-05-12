from __future__ import annotations

from typing import Literal

import numpy as np


class SectionPolygon:
    """A closed polygon in section-space: (distance_along, depth) vertices.

    Vertices are stored in the section coordinate frame.  Convert to map
    coordinates via :meth:`to_map_coords` using the associated
    :class:`~cross_section_tool.core.section.Section`.

    Parameters
    ----------
    vertices:
        (N, 2) array of ``(distance, depth)`` pairs.  The polygon is
        implicitly closed — the first vertex is NOT repeated at the end.
    name:
        Human-readable label.
    fill_color:
        Fill colour as a CSS hex string (e.g. ``"#9467bd"``).
    fill_alpha:
        Fill opacity in ``[0, 1]``.
    edge_color:
        Outline colour as a CSS hex string.
    edge_width:
        Outline width in points.
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
    ) -> None:
        verts = np.asarray(vertices, dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 2:
            raise ValueError("vertices must be an (N, 2) array of (distance, depth)")
        if len(verts) < 3:
            raise ValueError("SectionPolygon requires at least 3 vertices")
        self._vertices = verts.copy()
        self.name = name
        self.fill_color = fill_color
        self.fill_alpha = float(np.clip(fill_alpha, 0.0, 1.0))
        self.edge_color = edge_color
        self.edge_width = float(edge_width)
        self.formation: str = formation   # Phase 5: reference to Formation.name

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices.copy()

    @property
    def n_vertices(self) -> int:
        return len(self._vertices)

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
        return (
            f"SectionPolygon(name={self.name!r}, n_vertices={self.n_vertices}, "
            f"fill={self.fill_color!r})"
        )
