from __future__ import annotations

import math
from typing import Literal

import numpy as np


class Section:
    """Geometric primitive representing a cross-section polyline in a projected CRS."""

    def __init__(
        self,
        nodes: list[tuple[float, float]] | np.ndarray,
        name: str = "",
        depth_domain: Literal["depth", "twt"] = "depth",
        depth_units: str = "m",
        vertical_exaggeration: float = 1.0,
        crs_epsg: int = 32632,
    ) -> None:
        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError("nodes must be an (N, 2) array of (x, y) pairs")
        if len(nodes) < 2:
            raise ValueError("A section requires at least 2 nodes")
        self._nodes: np.ndarray = nodes.copy()
        self.name = name
        self.depth_domain: Literal["depth", "twt"] = depth_domain
        self.depth_units: str = str(depth_units)
        self.vertical_exaggeration: float = float(vertical_exaggeration)
        self.crs_epsg: int = int(crs_epsg)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> np.ndarray:
        return self._nodes.copy()

    @property
    def n_nodes(self) -> int:
        return len(self._nodes)

    @property
    def n_segments(self) -> int:
        return len(self._nodes) - 1

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def segment_lengths(self) -> np.ndarray:
        """Length of each segment in CRS units (shape: n_segments)."""
        deltas = np.diff(self._nodes, axis=0)
        return np.hypot(deltas[:, 0], deltas[:, 1])

    def cumulative_distances(self) -> np.ndarray:
        """Cumulative distance from node 0 to each node (shape: n_nodes)."""
        lengths = self.segment_lengths()
        return np.concatenate([[0.0], np.cumsum(lengths)])

    def total_length(self) -> float:
        return float(self.segment_lengths().sum())

    def segment_azimuths(self) -> np.ndarray:
        """Azimuth in degrees from north (clockwise) for each segment (shape: n_segments)."""
        deltas = np.diff(self._nodes, axis=0)
        # atan2 measured from east; convert to bearing from north
        azimuths = np.degrees(np.arctan2(deltas[:, 0], deltas[:, 1]))
        return azimuths % 360.0

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def _project_point_onto_segment(
        self,
        px: float,
        py: float,
        ax: float,
        ay: float,
        bx: float,
        by: float,
    ) -> tuple[float, float, float]:
        """Return (t, dist_along, perp_signed) for point P onto segment A-B.

        t         – parametric position on segment [0, 1]
        dist_along – distance from A to projection along the segment
        perp_signed – perpendicular distance, positive on the left of A→B
        """
        dx, dy = bx - ax, by - ay
        seg_len = math.hypot(dx, dy)
        if seg_len == 0.0:
            return 0.0, 0.0, math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / (seg_len * seg_len)
        t_clamped = max(0.0, min(1.0, t))
        proj_x = ax + t_clamped * dx
        proj_y = ay + t_clamped * dy
        # Signed perpendicular: positive to the left of the direction of travel
        # D × (P-A) = dx*(py-ay) - dy*(px-ax)
        perp = (dx * (py - ay) - dy * (px - ax)) / seg_len
        dist_along = t_clamped * seg_len
        return t_clamped, dist_along, perp

    def map_to_section(self, x: float, y: float) -> tuple[float, float]:
        """Convert map (x, y) → (distance_along_section, y_offset).

        distance_along_section is measured along the polyline from node 0.
        y_offset is the signed perpendicular distance from the nearest point
        on the polyline (positive = left of the direction of travel).
        """
        cum = self.cumulative_distances()
        best_dist = math.inf
        best_s = 0.0
        best_perp = 0.0

        for i in range(self.n_segments):
            ax, ay = self._nodes[i]
            bx, by = self._nodes[i + 1]
            t, d_along, perp = self._project_point_onto_segment(x, y, ax, ay, bx, by)
            px = ax + t * (bx - ax)
            py = ay + t * (by - ay)
            dist_to_line = math.hypot(x - px, y - py)
            if dist_to_line < best_dist:
                best_dist = dist_to_line
                best_s = cum[i] + d_along
                best_perp = perp

        return best_s, best_perp

    def section_to_map(self, distance_along: float) -> tuple[float, float]:
        """Convert distance_along_section → (x, y) map coordinates.

        Clamps to [0, total_length] if out of range.
        """
        cum = self.cumulative_distances()
        distance_along = float(np.clip(distance_along, 0.0, cum[-1]))

        for i in range(self.n_segments):
            if cum[i + 1] >= distance_along - 1e-10:
                seg_d = distance_along - cum[i]
                ax, ay = self._nodes[i]
                bx, by = self._nodes[i + 1]
                seg_len = cum[i + 1] - cum[i]
                if seg_len == 0.0:
                    return float(ax), float(ay)
                t = seg_d / seg_len
                return float(ax + t * (bx - ax)), float(ay + t * (by - ay))

        # Exactly at the last node
        return float(self._nodes[-1, 0]), float(self._nodes[-1, 1])

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, x: float, y: float) -> None:
        """Append a node at the end of the polyline."""
        new = np.array([[x, y]], dtype=float)
        self._nodes = np.vstack([self._nodes, new])

    def insert_node(self, index: int, x: float, y: float) -> None:
        """Insert a node before *index* (0 = before first node)."""
        if not (0 <= index <= len(self._nodes)):
            raise IndexError(f"index {index} out of range for {len(self._nodes)} nodes")
        self._nodes = np.insert(self._nodes, index, [x, y], axis=0)

    def insert_node_on_segment(self, segment_index: int, x: float, y: float) -> None:
        """Split segment *segment_index* by inserting (x, y) into the node list."""
        if not (0 <= segment_index < self.n_segments):
            raise IndexError(f"segment_index {segment_index} out of range")
        self.insert_node(segment_index + 1, x, y)

    def delete_node(self, index: int) -> None:
        """Delete node at *index*. Raises if fewer than 3 nodes (would leave < 2)."""
        if self.n_nodes <= 2:
            raise ValueError("Cannot delete a node: section must have at least 2 nodes")
        if not (0 <= index < self.n_nodes):
            raise IndexError(f"index {index} out of range")
        self._nodes = np.delete(self._nodes, index, axis=0)

    def move_node(self, index: int, x: float, y: float) -> None:
        """Move node at *index* to (x, y)."""
        if not (0 <= index < self.n_nodes):
            raise IndexError(f"index {index} out of range")
        self._nodes[index] = [x, y]

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_azimuth_length(
        cls,
        start_x: float,
        start_y: float,
        segments: list[tuple[float, float]],
        **kwargs,
    ) -> "Section":
        """Build a Section from a start point and (azimuth_degrees, length) tuples.

        Azimuth is measured clockwise from north (geographic convention).
        """
        nodes = [(start_x, start_y)]
        x, y = start_x, start_y
        for azimuth_deg, length in segments:
            az_rad = math.radians(azimuth_deg)
            dx = length * math.sin(az_rad)
            dy = length * math.cos(az_rad)
            x += dx
            y += dy
            nodes.append((x, y))
        return cls(nodes, **kwargs)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Phase 5: snapshot / restore (foundation for restoration workflows)
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Serialise the section's geometry and metadata to a plain dict.

        Can be deep-copied, modified, and reloaded with :meth:`load_snapshot`
        without touching the original section.

        Returns a dictionary with keys: nodes, name, depth_domain, depth_units,
        vertical_exaggeration, crs_epsg.
        """
        return {
            "nodes":                  self._nodes.copy(),
            "name":                   self.name,
            "depth_domain":           self.depth_domain,
            "depth_units":            self.depth_units,
            "vertical_exaggeration":  self.vertical_exaggeration,
            "crs_epsg":               self.crs_epsg,
        }

    def load_snapshot(self, snap: dict) -> None:
        """Restore geometry and metadata from a snapshot dict in-place."""
        import numpy as _np
        nodes = _np.asarray(snap["nodes"], dtype=float)
        if nodes.ndim != 2 or nodes.shape[1] != 2 or len(nodes) < 2:
            raise ValueError("snapshot 'nodes' must be an (N≥2, 2) array")
        self._nodes               = nodes.copy()
        self.name                 = snap.get("name", self.name)
        self.depth_domain         = snap.get("depth_domain", self.depth_domain)
        self.depth_units          = snap.get("depth_units", self.depth_units)
        self.vertical_exaggeration = float(snap.get("vertical_exaggeration",
                                                     self.vertical_exaggeration))
        self.crs_epsg             = int(snap.get("crs_epsg", self.crs_epsg))

    def __repr__(self) -> str:
        return (
            f"Section(name={self.name!r}, n_nodes={self.n_nodes}, "
            f"length={self.total_length():.1f} {self.depth_units}, "
            f"crs_epsg={self.crs_epsg})"
        )
