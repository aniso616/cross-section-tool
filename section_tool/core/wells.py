from __future__ import annotations

import numpy as np


class LogCurve:
    """A single log curve: values sampled as a function of measured depth (MD)."""

    def __init__(
        self,
        name: str,
        units: str,
        depths: list | np.ndarray,
        values: list | np.ndarray,
    ) -> None:
        depths = np.asarray(depths, dtype=float)
        values = np.asarray(values, dtype=float)
        if depths.ndim != 1 or values.ndim != 1:
            raise ValueError("depths and values must be 1D arrays")
        if len(depths) != len(values):
            raise ValueError("depths and values must have the same length")
        if len(depths) == 0:
            raise ValueError("LogCurve requires at least one sample")
        order = np.argsort(depths, kind="stable")
        self._depths = depths[order].copy()
        self._values = values[order].copy()
        self.name = name
        self.units = units

    @property
    def depths(self) -> np.ndarray:
        return self._depths.copy()

    @property
    def values(self) -> np.ndarray:
        return self._values.copy()

    @property
    def n_samples(self) -> int:
        return len(self._depths)

    def depth_range(self) -> tuple[float, float]:
        """Return (min_md, max_md)."""
        return float(self._depths[0]), float(self._depths[-1])

    def sample(self, depth: float) -> float:
        """Linearly interpolate value at *depth* (MD). Returns nan outside range."""
        return float(
            np.interp(depth, self._depths, self._values, left=np.nan, right=np.nan)
        )

    def sample_many(self, depths: list | np.ndarray) -> np.ndarray:
        """Linearly interpolate values at an array of MD values."""
        depths = np.asarray(depths, dtype=float)
        return np.interp(
            depths, self._depths, self._values, left=np.nan, right=np.nan
        ).astype(float)

    def __repr__(self) -> str:
        lo, hi = self.depth_range()
        return (
            f"LogCurve(name={self.name!r}, units={self.units!r}, "
            f"n={self.n_samples}, depth=[{lo:.1f}, {hi:.1f}])"
        )


class DeviationSurvey:
    """Wellbore trajectory computed from (MD, inclination, azimuth) stations
    using the minimum curvature method.

    Inclination is measured from vertical (0° = vertical, 90° = horizontal).
    Azimuth is measured clockwise from north (0° = north, 90° = east).
    x = easting, y = northing.
    TVD is measured downward from surface (KB).
    """

    def __init__(
        self,
        md: list | np.ndarray,
        inc_deg: list | np.ndarray,
        azi_deg: list | np.ndarray,
        surface_x: float = 0.0,
        surface_y: float = 0.0,
    ) -> None:
        md = np.asarray(md, dtype=float)
        inc_rad = np.radians(np.asarray(inc_deg, dtype=float))
        azi_rad = np.radians(np.asarray(azi_deg, dtype=float))
        if not (len(md) == len(inc_rad) == len(azi_rad)):
            raise ValueError("md, inc_deg, and azi_deg must have the same length")
        if len(md) < 2:
            raise ValueError("DeviationSurvey requires at least 2 stations")
        if np.any(np.diff(md) < 0.0):
            raise ValueError("md must be non-decreasing")

        n = len(md)
        x = np.empty(n)
        y = np.empty(n)
        tvd = np.empty(n)
        x[0] = float(surface_x)
        y[0] = float(surface_y)
        tvd[0] = 0.0

        for i in range(n - 1):
            dMD = md[i + 1] - md[i]
            I1, I2 = inc_rad[i], inc_rad[i + 1]
            A1, A2 = azi_rad[i], azi_rad[i + 1]
            # Minimum curvature: dog-leg angle between successive tangent vectors
            cos_dl = np.cos(I2 - I1) - np.sin(I1) * np.sin(I2) * (1.0 - np.cos(A2 - A1))
            DL = np.arccos(np.clip(cos_dl, -1.0, 1.0))
            RF = (2.0 / DL) * np.tan(DL / 2.0) if DL > 1e-10 else 1.0
            dE = dMD / 2.0 * (np.sin(I1) * np.sin(A1) + np.sin(I2) * np.sin(A2)) * RF
            dN = dMD / 2.0 * (np.sin(I1) * np.cos(A1) + np.sin(I2) * np.cos(A2)) * RF
            dTVD = dMD / 2.0 * (np.cos(I1) + np.cos(I2)) * RF
            x[i + 1] = x[i] + dE
            y[i + 1] = y[i] + dN
            tvd[i + 1] = tvd[i] + dTVD

        self._md = md
        self._x = x
        self._y = y
        self._tvd = tvd
        self._inc_deg = np.degrees(inc_rad)
        self._azi_deg = np.degrees(azi_rad)
        self.surface_x = float(surface_x)
        self.surface_y = float(surface_y)

    @classmethod
    def vertical(
        cls,
        surface_x: float = 0.0,
        surface_y: float = 0.0,
        td: float = 5000.0,
    ) -> "DeviationSurvey":
        """Create a vertical survey (inc=0, azi=0) down to total depth *td*."""
        return cls([0.0, td], [0.0, 0.0], [0.0, 0.0], surface_x, surface_y)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def md(self) -> np.ndarray:
        return self._md.copy()

    @property
    def x_track(self) -> np.ndarray:
        """Easting at each deviation station."""
        return self._x.copy()

    @property
    def y_track(self) -> np.ndarray:
        """Northing at each deviation station."""
        return self._y.copy()

    @property
    def tvd_track(self) -> np.ndarray:
        """TVD (depth below surface/KB) at each deviation station."""
        return self._tvd.copy()

    @property
    def inc_deg(self) -> np.ndarray:
        """Inclination from vertical (degrees) at each station."""
        return self._inc_deg.copy()

    @property
    def azi_deg(self) -> np.ndarray:
        """Azimuth clockwise from north (degrees) at each station."""
        return self._azi_deg.copy()

    @property
    def max_md(self) -> float:
        return float(self._md[-1])

    @property
    def max_tvd(self) -> float:
        return float(self._tvd[-1])

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def xyz_at_md(self, md_val: float) -> tuple[float, float, float]:
        """Return (x, y, tvd) at *md_val* by linear interpolation between stations."""
        x = float(np.interp(md_val, self._md, self._x))
        y = float(np.interp(md_val, self._md, self._y))
        tvd = float(np.interp(md_val, self._md, self._tvd))
        return x, y, tvd

    def tvd_at_md(self, md_val: float) -> float:
        return float(np.interp(md_val, self._md, self._tvd))

    def md_to_tvd(self, mds: list | np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(mds, dtype=float), self._md, self._tvd).astype(float)

    def __repr__(self) -> str:
        return (
            f"DeviationSurvey(n_stations={len(self._md)}, "
            f"max_md={self.max_md:.1f}, max_tvd={self.max_tvd:.1f})"
        )


class Well:
    """A borehole: surface location, deviation survey, log curves, and formation tops."""

    def __init__(
        self,
        name: str,
        x: float,
        y: float,
        kb: float = 0.0,
        deviation: DeviationSurvey | None = None,
        uwi: str = "",
        td: float | None = None,
    ) -> None:
        self.name = name
        self.x = float(x)
        self.y = float(y)
        self.kb = float(kb)
        self.uwi = uwi
        self.color: str = "#E8C46A"   # default warm-gold well colour
        # Original (pre-transformation) coordinates — set by importer when CRS differs
        self.original_x: float | None = None
        self.original_y: float | None = None
        self.original_crs_epsg: int | None = None
        if deviation is not None:
            self.deviation = deviation
        elif td is not None:
            self.deviation = DeviationSurvey.vertical(x, y, td=float(td))
        else:
            self.deviation = DeviationSurvey.vertical(x, y)
        self._logs: dict[str, LogCurve] = {}
        self._formation_tops: dict[str, float] = {}  # name → MD

    # ------------------------------------------------------------------
    # Log management
    # ------------------------------------------------------------------

    def add_log(self, curve: LogCurve) -> None:
        """Register *curve* keyed by its name, replacing any existing curve with that name."""
        self._logs[curve.name] = curve

    def get_log(self, name: str) -> LogCurve:
        if name not in self._logs:
            raise KeyError(f"No log curve named {name!r}")
        return self._logs[name]

    def remove_log(self, name: str) -> None:
        if name not in self._logs:
            raise KeyError(f"No log curve named {name!r}")
        del self._logs[name]

    @property
    def log_names(self) -> list[str]:
        return list(self._logs.keys())

    # ------------------------------------------------------------------
    # Formation tops
    # ------------------------------------------------------------------

    def add_formation_top(self, name: str, md: float) -> None:
        self._formation_tops[name] = float(md)

    def remove_formation_top(self, name: str) -> None:
        if name not in self._formation_tops:
            raise KeyError(f"No formation top named {name!r}")
        del self._formation_tops[name]

    @property
    def formation_tops(self) -> dict[str, float]:
        """Copy of the {name: MD} formation top dictionary."""
        return dict(self._formation_tops)

    # ------------------------------------------------------------------
    # Section projection
    # ------------------------------------------------------------------

    def project_to_section(self, section) -> tuple[float, float]:
        """Return (distance_along_section, perp_offset) for the well collar."""
        return section.map_to_section(self.x, self.y)

    def section_track(self, section) -> tuple[np.ndarray, np.ndarray]:
        """Return (distances, tvds) projecting the wellbore path onto *section*.

        For a vertical well, distances are constant at the collar projection.
        For a deviated well, distances vary as the trajectory departs from vertical.
        """
        distances = np.array([
            section.map_to_section(float(xi), float(yi))[0]
            for xi, yi in zip(self.deviation._x, self.deviation._y)
        ])
        return distances, self.deviation._tvd.copy()

    def formation_top_in_section(
        self, top_name: str, section
    ) -> tuple[float, float]:
        """Return (distance_along_section, tvd) for a named formation top.

        The wellbore position at the top's MD is projected onto the section.
        """
        if top_name not in self._formation_tops:
            raise KeyError(f"No formation top named {top_name!r}")
        md = self._formation_tops[top_name]
        x, y, tvd = self.deviation.xyz_at_md(md)
        dist, _ = section.map_to_section(x, y)
        return dist, tvd

    def __repr__(self) -> str:
        return (
            f"Well(name={self.name!r}, uwi={self.uwi!r}, "
            f"n_logs={len(self._logs)}, n_tops={len(self._formation_tops)})"
        )
