"""Read XYZ ASCII surface files.

Accepts space / tab / comma delimited files with ≥3 numeric columns (X Y Z).
Comment lines starting with ``#`` or ``//`` and non-numeric header rows are
silently skipped.  Null values (< -1e6, > 1e6, -999, -9999) are removed.
"""
from __future__ import annotations

import os

import numpy as np

from .base import SurfaceReader
from section_tool.core.surfaces import Surface, detect_grid


class XYZReader(SurfaceReader):
    name = "XYZ ASCII"
    extensions = ["xyz", "txt", "dat", "csv", "asc"]
    description = "ASCII text file with X Y Z columns"

    def can_read(self, filepath: str) -> bool:
        if not os.path.isfile(filepath):
            return False
        ext = os.path.splitext(filepath)[1].lower().lstrip(".")
        if ext not in self.extensions:
            return False
        # Sniff first few parseable lines
        try:
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                checked = 0
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#") or s.startswith("//"):
                        continue
                    parts = s.replace(",", " ").split()
                    try:
                        [float(p) for p in parts[:3]]
                        return True
                    except ValueError:
                        checked += 1
                        if checked > 5:
                            return False
        except Exception:
            pass
        return False

    def read(
        self,
        filepath: str,
        *,
        crs_epsg: int = 0,
        z_domain: str = "depth_m",
        x_col: int = 0,
        y_col: int = 1,
        z_col: int = 2,
        **options,
    ) -> Surface:
        rows = []
        delimiter = None

        with open(filepath, encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or s.startswith("//"):
                    continue
                # Auto-detect delimiter from first data line
                if delimiter is None:
                    delimiter = "\t" if "\t" in s else ("," if "," in s else " ")
                parts = s.split(delimiter) if delimiter != " " else s.split()
                parts = [p.strip() for p in parts]
                try:
                    nums = [float(parts[c]) for c in (x_col, y_col, z_col)]
                    rows.append(nums)
                except (ValueError, IndexError):
                    continue   # skip header / malformed lines

        if not rows:
            raise ValueError(f"No valid XYZ data in {filepath}")

        points = np.array(rows, dtype=np.float64)

        # Strip null / out-of-range Z values
        z = points[:, 2]
        bad = (z < -1e6) | (z > 1e6) | np.isin(z, [-999.0, -9999.0])
        if bad.any():
            points = points[~bad]

        if len(points) < 3:
            raise ValueError(f"Fewer than 3 valid points in {filepath}")

        name = os.path.splitext(os.path.basename(filepath))[0]
        z_units = "ms" if "twt" in z_domain else "m"

        surf = Surface(
            name=name,
            points=points,
            crs_epsg=crs_epsg,
            z_domain=z_domain,
            z_units=z_units,
            source_file=filepath,
            source_format="XYZ ASCII",
        )
        surf.grid_info = detect_grid(points)
        return surf
