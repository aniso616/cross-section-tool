"""Read OpendTect horizon files (.hor).

The primary export format for OpendTect V7 is a text file with
Inline / Crossline / Z columns.  A separate ``.survey`` (or ``.par``)
file may provide the affine transform from survey bin coordinates to
geographic CRS.

Survey affine transform
-----------------------
OpendTect stores the transform as:

    Coord-X-BinID: X0  dX/dIL  dX/dXL
    Coord-Y-BinID: Y0  dY/dIL  dY/dXL

where (IL, XL) are bin numbers and (X, Y) are CRS coordinates.
"""
from __future__ import annotations

import os
import re

import numpy as np

from .base import SurfaceReader
from section_tool.core.surfaces import Surface, GridInfo


class OpendTectReader(SurfaceReader):
    name = "OpendTect Horizon"
    extensions = ["hor"]
    description = "OpendTect horizon export (IL XL Z)"

    def can_read(self, filepath: str) -> bool:
        if not os.path.isfile(filepath):
            return False
        if os.path.splitext(filepath)[1].lower() != ".hor":
            return False
        try:
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                head = f.read(4096)
            return "Horizon" in head or "OpendTect" in head or self._looks_ilxl(head)
        except Exception:
            return False

    @staticmethod
    def _looks_ilxl(text: str) -> bool:
        for line in text.split("\n")[:30]:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    il, xl = int(parts[0]), int(parts[1])
                    float(parts[2])
                    if 0 < il < 100_000 and 0 < xl < 100_000:
                        return True
                except ValueError:
                    continue
        return False

    # ------------------------------------------------------------------

    def read(
        self,
        filepath: str,
        *,
        crs_epsg: int = 0,
        survey_transform: dict | None = None,
        **options,
    ) -> Surface:
        """Read an OpendTect .hor file.

        Parameters
        ----------
        survey_transform:
            If provided, convert IL/XL to geographic XY.  May be:
            - ``{'matrix': 2×3 ndarray, 'origin': (il0, xl0)}``
            - ``{'Coord-X-BinID': [X0, dX/dIL, dX/dXL],
                  'Coord-Y-BinID': [Y0, dY/dIL, dY/dXL]}``
            If None, IL/XL are used as X/Y directly (useful when no
            survey geometry is available).
        """
        # Try to auto-load survey transform from a sidecar file
        if survey_transform is None:
            survey_transform = self._find_survey_transform(filepath)

        il_list, xl_list, z_list = [], [], []
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or s.startswith('"'):
                    continue
                parts = s.split()
                if len(parts) < 3:
                    continue
                try:
                    il = float(parts[0])
                    xl = float(parts[1])
                    z  = float(parts[2])
                    if z > 1e6 or z < -1e6:
                        continue
                    il_list.append(il); xl_list.append(xl); z_list.append(z)
                except ValueError:
                    continue

        if not il_list:
            raise ValueError(f"No valid horizon data in {filepath}")

        il = np.array(il_list)
        xl = np.array(xl_list)
        z  = np.array(z_list)

        # Apply survey transform
        if survey_transform:
            x, y = self._apply_transform(il, xl, survey_transform)
        else:
            x, y = il, xl

        # Infer Z domain from magnitude
        z_max = float(z.max())
        if z_max < 15.0:          # seconds (TWT)
            z *= 1000.0
            z_domain, z_units = "twt_ms", "ms"
        elif z_max < 20_000.0:    # ms (TWT) or shallow depth
            # Heuristic: OpendTect typically exports TWT in ms
            z_domain, z_units = "twt_ms", "ms"
        else:
            z_domain, z_units = "depth_m", "m"

        points = np.column_stack([x, y, z])
        name = os.path.splitext(os.path.basename(filepath))[0]

        il_arr_i = il.astype(int)
        xl_arr_i = xl.astype(int)

        surf = Surface(
            name=name,
            points=points,
            crs_epsg=crs_epsg,
            z_domain=z_domain,
            z_units=z_units,
            source_file=filepath,
            source_format="OpendTect Horizon",
            grid_info=GridInfo(
                origin=(float(x.min()), float(y.min())),
                step_x=(1.0, 0.0),
                step_y=(0.0, 1.0),
                nx=int(il_arr_i.max() - il_arr_i.min() + 1),
                ny=int(xl_arr_i.max() - xl_arr_i.min() + 1),
                inline_range=(int(il_arr_i.min()), int(il_arr_i.max())),
                xline_range=(int(xl_arr_i.min()), int(xl_arr_i.max())),
            ),
        )
        return surf

    # ------------------------------------------------------------------

    @staticmethod
    def _apply_transform(il, xl, t: dict):
        """Convert IL/XL to geographic X/Y."""
        if "matrix" in t:
            m = np.asarray(t["matrix"])
            il0, xl0 = t.get("origin", (0, 0))
            x = m[0, 0] * (il - il0) + m[0, 1] * (xl - xl0) + m[0, 2]
            y = m[1, 0] * (il - il0) + m[1, 1] * (xl - xl0) + m[1, 2]
            return x, y
        # OpendTect Coord-X/Y-BinID format
        if "Coord-X-BinID" in t:
            cx = t["Coord-X-BinID"]   # [X0, dX/dIL, dX/dXL]
            cy = t["Coord-Y-BinID"]   # [Y0, dY/dIL, dY/dXL]
            x = cx[0] + cx[1] * il + cx[2] * xl
            y = cy[0] + cy[1] * il + cy[2] * xl
            return x, y
        return il, xl

    @staticmethod
    def _find_survey_transform(hor_path: str) -> dict | None:
        """Look for a .survey sidecar file and parse the coordinate transform."""
        # Try survey file in parent directories (OpendTect project structure)
        base = os.path.dirname(hor_path)
        for candidate in [
            os.path.join(base, "survey"),
            os.path.join(base, "..", "survey"),
            os.path.join(base, "..", "..", "survey"),
        ]:
            if os.path.isfile(candidate):
                return OpendTectReader._parse_survey_file(candidate)
        return None

    @staticmethod
    def _parse_survey_file(path: str) -> dict | None:
        """Parse an OpendTect survey file for coordinate transform parameters."""
        result = {}
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    for key in ("Coord-X-BinID", "Coord-Y-BinID"):
                        if line.startswith(key):
                            nums = [float(v) for v in line.split(":")[1].split()]
                            result[key] = nums
        except Exception:
            return None
        return result if len(result) == 2 else None
