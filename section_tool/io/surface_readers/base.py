"""Abstract base class for surface readers."""
from __future__ import annotations

from abc import ABC, abstractmethod


class SurfaceReader(ABC):
    name: str = "Unknown"
    extensions: list[str] = []
    description: str = ""

    @abstractmethod
    def can_read(self, filepath: str) -> bool:
        """Fast check: extension + optional magic-byte / first-line sniff."""
        ...

    @abstractmethod
    def read(self, filepath: str, **options):
        """Parse the file and return a :class:`~section_tool.core.surfaces.Surface`."""
        ...

    def validate(self, surface) -> list[str]:
        warnings = []
        if surface.n_points == 0:
            warnings.append("Surface contains no points")
        elif surface.n_points < 3:
            warnings.append(f"Only {surface.n_points} points — interpolation may fail")
        return warnings
