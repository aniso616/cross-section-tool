"""Slice abstraction — the surface on which an observation (fault/horizon trace)
is drawn.

A *slice* is any 2D frame embedded in 3D world space that observations live on.
Two implementers today:

* :class:`~section_tool.core.section.Section` — a **vertical** slice: a plan
  polyline (trace) with a depth axis. Slice coords are ``(distance_along, depth)``.
* :class:`HorizontalSlice` — a **horizontal** plan slice at a fixed elevation.
  Slice coords are world ``(easting, northing)``; the transform is trivial.

The protocol names two transforms that already implicitly existed for sections,
so one fault/horizon entity can carry observations on either kind of slice
through a single, uniform interface:

    to_world(u, v)      -> (x, y, z)
    from_world(x, y, z) -> (u, v, residual)     # residual = off-plane distance

This module is deliberately tiny: it is the backbone for slice-agnostic
observations, not a geometry engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class Slice(Protocol):
    """A 2D observation frame embedded in world space.

    Implementers expose ``kind`` ('section' | 'horizontal') and ``name``, plus
    the two coordinate transforms below. ``(u, v)`` are the slice's native 2D
    coordinates: ``(distance_along, depth)`` for a section, ``(easting,
    northing)`` for a horizontal slice.
    """

    kind: str
    name: str

    def to_world(self, u: float, v: float) -> tuple[float, float, float]:
        """Map slice coords ``(u, v)`` to world ``(x, y, z)`` (z positive up)."""
        ...

    def from_world(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """Map world ``(x, y, z)`` to ``(u, v, residual)``.

        ``residual`` is the off-plane distance (perpendicular for a section,
        ``|z - elevation|`` for a horizontal slice) — for projection/tolerance.
        """
        ...


@dataclass
class HorizontalSlice:
    """A horizontal plan slice at a fixed elevation (z positive up).

    Slice coords are world ``(easting, northing)``; the transform to/from world
    is the identity in x/y with a constant z. No trace, no projection.

    Parameters
    ----------
    name:       Stable identifier/label for the slice (used as the observation
                slice-ref, like a Section's name).
    elevation:  z₀ in metres, positive up (e.g. -1500 for 1500 m below datum).
    crs_epsg:   Projected CRS of the easting/northing coords.
    """

    name: str
    elevation: float
    crs_epsg: int = 32632
    kind: str = field(default="horizontal", init=False)

    def to_world(self, easting: float, northing: float) -> tuple[float, float, float]:
        return float(easting), float(northing), float(self.elevation)

    def from_world(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        # Slice coords ARE world (x, y); residual is vertical distance off-plane.
        return float(x), float(y), abs(float(z) - float(self.elevation))
