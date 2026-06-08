from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Elevation / depth helpers (module-level)
# ---------------------------------------------------------------------------

def depth_to_elevation(depth: float) -> float:
    """Convert depth-positive-down to elevation-positive-up.

    depth_to_elevation(1000) → -1000  (1000 m below surface)
    depth_to_elevation(0)    →  0     (at sea level)
    """
    return -depth


def elevation_to_depth(elevation: float) -> float:
    """Convert elevation-positive-up to depth-positive-down.

    elevation_to_depth(-1000) → 1000
    elevation_to_depth(50)    → -50  (50 m above datum)
    """
    return -elevation


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
        # Phase 5: per-section seismic display settings
        from section_tool.core.seismic_settings import SeismicDisplaySettings
        self.seismic_display: SeismicDisplaySettings = SeismicDisplaySettings()
        # User-overridable display domain (None → follows depth_domain)
        self._display_domain: Literal["depth", "twt"] | None = None

    # ------------------------------------------------------------------
    # Display domain and axis properties
    # ------------------------------------------------------------------

    @property
    def display_domain(self) -> Literal["depth", "twt"]:
        """The domain shown on the Y axis: 'twt' or 'depth'.

        If the user has not overridden this, it follows :attr:`depth_domain`.
        """
        return self._display_domain if self._display_domain is not None else self.depth_domain

    @display_domain.setter
    def display_domain(self, value: Literal["depth", "twt"]) -> None:
        if value not in ("depth", "twt"):
            raise ValueError(f"display_domain must be 'depth' or 'twt', got {value!r}")
        self._display_domain = value

    @property
    def y_label(self) -> str:
        """Axis label string appropriate for :attr:`display_domain`."""
        if self.display_domain == "twt":
            return "TWT (ms)"
        units = self.depth_units or "m"
        return f"Depth ({units})"

    @property
    def y_range(self) -> tuple[float, float]:
        """Default Y-axis range (top, bottom) appropriate for the display domain.

        Returns (0, 3000) ms for TWT and (0, 5000) m for depth by default.
        These are intentionally conservative starting values; the section view
        overrides them based on loaded data.
        """
        if self.display_domain == "twt":
            return (0.0, 3000.0)   # ms
        return (0.0, 5000.0)       # m (or ft)

    # ------------------------------------------------------------------
    # Velocity conversion stubs
    # ------------------------------------------------------------------

    @staticmethod
    def depth_to_twt(depth_m: float, x: float = 0.0, y: float = 0.0) -> float:
        """Convert depth (m) to two-way travel time (ms).

        **Stub implementation** — uses a constant velocity of 2 000 m/s
        (v_interval = 2 000 m/s → TWT = 2 * depth / v_interval * 1 000 ms/s).
        The full implementation will call the project VelocityModel.
        """
        _V0 = 2_000.0   # m/s — placeholder
        return depth_m * 2.0 / _V0 * 1_000.0

    @staticmethod
    def twt_to_depth(twt_ms: float, x: float = 0.0, y: float = 0.0) -> float:
        """Convert two-way travel time (ms) to depth (m).

        **Stub implementation** — uses a constant velocity of 2 000 m/s.
        The full implementation will call the project VelocityModel.
        """
        _V0 = 2_000.0   # m/s — placeholder
        return twt_ms / 1_000.0 * _V0 / 2.0

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

    def project_point(self, x: float, y: float) -> tuple[float, float]:
        """Project map point (x, y) onto the section, returning an *unclamped* result.

        Unlike :meth:`map_to_section`, ``distance_along`` is not clamped to
        ``[0, total_length]``:

        * A point before the section start yields ``distance_along < 0``.
        * A point past the section end yields ``distance_along > total_length``.
        * Interior points are projected onto the nearest segment (clamped per
          segment for segment selection, then unclamped for the chosen segment's
          first and last positions).

        Parameters
        ----------
        x, y : float
            Map-space coordinates (same CRS as the section nodes).  Works
            correctly with large projected coordinates (e.g. UTM 6 000 000+).

        Returns
        -------
        (distance_along, perpendicular_offset)
            *distance_along* — signed distance from node 0 along the polyline.
            *perpendicular_offset* — signed perpendicular offset; positive is
            to the left of the direction of travel (A → B convention).
        """
        cum = self.cumulative_distances()
        n = self.n_segments

        best_euclidean = math.inf
        best_s = 0.0
        best_perp = 0.0

        for i in range(n):
            ax = float(self._nodes[i, 0])
            ay = float(self._nodes[i, 1])
            bx = float(self._nodes[i + 1, 0])
            by = float(self._nodes[i + 1, 1])
            dx, dy = bx - ax, by - ay
            seg_len = math.hypot(dx, dy)
            if seg_len < 1e-12:
                continue

            # Work with offsets from A to avoid precision loss with large coords
            ox, oy = x - ax, y - ay

            # Unclamped parametric position along this segment
            t = (ox * dx + oy * dy) / (seg_len * seg_len)

            # Signed perpendicular distance (positive = left of A→B)
            perp = (dx * oy - dy * ox) / seg_len

            # Euclidean distance to nearest point on the FINITE segment
            t_clamped = max(0.0, min(1.0, t))
            dist = math.hypot(ox - t_clamped * dx, oy - t_clamped * dy)

            if dist < best_euclidean:
                best_euclidean = dist
                best_perp = perp

                if i == 0 and t < 0.0:
                    # Before section start: extend first segment backwards
                    best_s = t * seg_len          # negative
                elif i == n - 1 and t > 1.0:
                    # Past section end: extend last segment forwards
                    best_s = cum[i] + t * seg_len  # > total_length
                else:
                    best_s = cum[i] + t_clamped * seg_len

        return best_s, best_perp

    def project_points(self, xs, ys) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized :meth:`project_point` over arrays of points.

        Returns ``(distance_along, perp_offset)`` arrays with identical semantics
        to the scalar version — including the unclamped first/last-segment
        extension.  Used by seismic extraction so projecting a multi-million-trace
        survey onto the section does not pay one Python call (and a fresh
        ``cumulative_distances``) per trace, which froze the UI.
        """
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        p = xs.shape[0]
        n = self.n_segments
        if n == 0 or p == 0:
            return np.zeros(p), np.zeros(p)

        nodes = self._nodes
        cum = self.cumulative_distances()
        t_all    = np.zeros((n, p))
        perp_all = np.zeros((n, p))
        dist_all = np.full((n, p), np.inf)
        seg_len  = np.zeros(n)
        for i in range(n):                     # loops over segments (few), not points
            ax, ay = float(nodes[i, 0]), float(nodes[i, 1])
            bx, by = float(nodes[i + 1, 0]), float(nodes[i + 1, 1])
            dx, dy = bx - ax, by - ay
            sl = math.hypot(dx, dy)
            seg_len[i] = sl
            if sl < 1e-12:
                continue
            ox, oy = xs - ax, ys - ay
            t = (ox * dx + oy * dy) / (sl * sl)
            perp = (dx * oy - dy * ox) / sl
            tc = np.clip(t, 0.0, 1.0)
            t_all[i]    = t
            perp_all[i] = perp
            dist_all[i] = np.hypot(ox - tc * dx, oy - tc * dy)

        best = np.argmin(dist_all, axis=0)
        cols = np.arange(p)
        t_b    = t_all[best, cols]
        perp_b = perp_all[best, cols]
        sl_b   = seg_len[best]
        cum_b  = cum[best]
        s = cum_b + np.clip(t_b, 0.0, 1.0) * sl_b
        s = np.where((best == 0) & (t_b < 0.0), t_b * sl_b, s)              # before start
        s = np.where((best == n - 1) & (t_b > 1.0), cum_b + t_b * sl_b, s)  # past end
        return s, perp_b

    def section_to_map(self, distance_along: float,
                       extrapolate: bool = False) -> tuple[float, float]:
        """Convert distance_along_section → (x, y) map coordinates.

        By default clamps to ``[0, total_length]`` (the long-standing contract).
        With *extrapolate=True* this is the exact inverse of :meth:`project_point`
        for out-of-range distances: a distance < 0 extends the FIRST segment
        backwards and a distance > total_length extends the LAST segment forwards
        (for a bent section, along that segment's bearing). This is what
        beyond-section picks need so their real-world XY follows the azimuth.
        """
        cum = self.cumulative_distances()

        if extrapolate and distance_along < 0.0:
            ax, ay = self._nodes[0]
            bx, by = self._nodes[1]
            seg_len = float(cum[1])              # length of the first segment
            if seg_len == 0.0:
                return float(ax), float(ay)
            t = distance_along / seg_len         # negative
            return float(ax + t * (bx - ax)), float(ay + t * (by - ay))

        if extrapolate and distance_along > cum[-1]:
            ax, ay = self._nodes[-2]
            bx, by = self._nodes[-1]
            seg_len = float(cum[-1] - cum[-2])   # length of the last segment
            if seg_len == 0.0:
                return float(bx), float(by)
            t = (distance_along - cum[-2]) / seg_len   # > 1
            return float(ax + t * (bx - ax)), float(ay + t * (by - ay))

        if not extrapolate:
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

    def pick_to_world(self, distance_along: float) -> tuple[float, float]:
        """World XY for a PICK at *distance_along* — always extrapolates past the
        section ends (``section_to_map(d, extrapolate=True)``). The single entry
        point every model-writing pick path uses, so no call site can forget the
        flag and clamp a beyond-section pick back to the endpoint. Non-pick
        callers that genuinely want clamping keep calling section_to_map directly.
        """
        return self.section_to_map(distance_along, extrapolate=True)

    # ------------------------------------------------------------------
    # Slice protocol (a Section is a *vertical* slice)
    # ------------------------------------------------------------------

    @property
    def kind(self) -> str:
        return "section"

    def to_world(self, distance_along: float, depth: float) -> tuple[float, float, float]:
        """Slice coords ``(distance_along, depth)`` → world ``(x, y, z)``.

        x, y come from the trace at *distance_along*; z is positive-up
        (``z = −depth``).
        """
        x, y = self.section_to_map(distance_along)
        return float(x), float(y), float(-depth)

    def from_world(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """World ``(x, y, z)`` → ``(distance_along, depth, residual)``.

        *residual* is the absolute perpendicular offset from the trace.
        """
        dist, perp = self.project_point(x, y)
        return float(dist), float(-z), abs(float(perp))

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
    # Phase 4: domain conversion
    # ------------------------------------------------------------------

    def convert_domain(self, target_domain: str, velocity_model) -> "Section":
        """Return a copy of this section in *target_domain* ("depth" or "twt").

        *velocity_model* must be a :class:`~core.velocity_model.VelocityModel`.
        The section geometry (map nodes) is unchanged; only ``depth_domain`` and
        ``depth_units`` are updated.  Pick conversion is handled by callers.
        """
        copy_ = copy.deepcopy(self)
        if target_domain == self.depth_domain:
            return copy_
        copy_.depth_domain = target_domain
        if target_domain == "twt":
            copy_.depth_units = "ms"
        else:
            copy_.depth_units = "m"
        return copy_

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


# ---------------------------------------------------------------------------
# Well-section projection
# ---------------------------------------------------------------------------

@dataclass
class WellSectionProjection:
    """Result of projecting a well collar onto a section line.

    Attributes
    ----------
    well_name : str
    section_name : str
    distance_along : float
        Signed distance from section node 0 (m).  Negative if the well projects
        before the section start; greater than the section total length if past
        the end.
    perpendicular_offset : float
        Signed perpendicular distance from the section plane (m).  Positive =
        left of the direction of travel (A → B convention), i.e. roughly north
        for an east-ward section.
    display_tier : str
        One of ``"on_plane"``, ``"near"``, ``"far"``, ``"hidden"`` — based on
        the absolute perpendicular offset.

    Tier thresholds (with default *tolerance* = 2 000 m):
        on_plane  |offset| < 100 m
        near      |offset| < tolerance (2 000 m)
        far       |offset| < tolerance × 2.5 (5 000 m)
        hidden    otherwise
    """

    well_name: str
    section_name: str
    distance_along: float
    perpendicular_offset: float
    display_tier: str

    @staticmethod
    def compute(well, section: Section, tolerance: float = 2_000.0) -> "WellSectionProjection":
        """Compute the projection of *well* onto *section*.

        Parameters
        ----------
        well :
            Any object with ``.name``, ``.x``, ``.y`` attributes.
        section : Section
            The section to project onto.
        tolerance : float
            Near-plane threshold in the same units as the section CRS (default
            2 000 m).  ``far`` extends to ``tolerance × 2.5``.
        """
        dist, offset = section.project_point(well.x, well.y)
        abs_off = abs(offset)
        if abs_off < 100.0:
            tier = "on_plane"
        elif abs_off < tolerance:
            tier = "near"
        elif abs_off < tolerance * 2.5:
            tier = "far"
        else:
            tier = "hidden"
        return WellSectionProjection(
            well_name=well.name,
            section_name=section.name,
            distance_along=dist,
            perpendicular_offset=offset,
            display_tier=tier,
        )
