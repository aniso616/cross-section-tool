from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import segyio
from segyio import TraceField


def detect_domain(sample_interval_us: float) -> tuple[Literal["twt", "depth"], Literal["ms", "m", "ft"]]:
    """Infer seismic domain from the binary-header sample interval.

    SEG-Y stores the sample interval in *microseconds*.

    * 1 000 – 8 000 µs (1 – 8 ms): typical TWT recording, returns ``("twt", "ms")``.
    * Values outside that range (or 0): assumed depth-migrated, returns
      ``("depth", "m")``.

    Parameters
    ----------
    sample_interval_us:
        Binary-header sample interval in **microseconds**.

    Returns
    -------
    (domain, depth_units)
    """
    if 1_000 <= sample_interval_us <= 8_000:
        return "twt", "ms"
    return "depth", "m"


@dataclass
class SeismicDataset:
    """In-memory 2D seismic dataset read from a SEG-Y file.

    data:               float32, shape (n_traces, n_samples)
    trace_x / y:        float64, CRS easting/northing for each trace
    samples:            float64, sample-axis positions (ms for TWT, m for depth)
    sample_interval:    float, dt in the same units as *samples*
    sample_interval_ms: float, dt **always** in milliseconds regardless of domain
    """

    name: str
    data: np.ndarray
    trace_x: np.ndarray
    trace_y: np.ndarray
    samples: np.ndarray
    sample_interval: float
    domain: Literal["twt", "depth"]
    depth_units: Literal["ms", "m", "ft"]
    crs_epsg: int
    # Sample interval stored in ms for consistent comparison across domains
    sample_interval_ms: float = 0.0

    @property
    def time_range(self) -> tuple[float, float]:
        """(t_min, t_max) of the sample axis in its native units (ms for TWT)."""
        if len(self.samples) == 0:
            return (0.0, 0.0)
        return (float(self.samples[0]), float(self.samples[-1]))

    @property
    def n_traces(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    def project_onto_section(self, section) -> tuple[np.ndarray, np.ndarray]:
        """Return *(distances, perp_offsets)* for each trace projected onto *section*.

        Both arrays have length :attr:`n_traces`.  Uses
        ``Section.map_to_section`` for each trace coordinate.
        """
        if self.n_traces == 0:
            return np.array([]), np.array([])
        pairs = np.array([
            section.map_to_section(float(x), float(y))
            for x, y in zip(self.trace_x, self.trace_y)
        ])
        return pairs[:, 0], pairs[:, 1]

    def traces_sorted_by_section(
        self, section
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return *(distances, data, perp_offsets)* sorted by distance along *section*.

        Convenient for building a display-ready section image where the
        horizontal axis is distance along the interpretation line.
        """
        distances, perps = self.project_onto_section(section)
        order = np.argsort(distances, kind="stable")
        return distances[order], self.data[order], perps[order]

    def __repr__(self) -> str:
        return (
            f"SeismicDataset(name={self.name!r}, n_traces={self.n_traces}, "
            f"n_samples={self.n_samples}, domain={self.domain!r}, "
            f"dt={self.sample_interval} {self.depth_units})"
        )


# ---------------------------------------------------------------------------
# Public readers
# ---------------------------------------------------------------------------

def read_segy(
    path: str | os.PathLike,
    *,
    x_field: int = TraceField.CDP_X,
    y_field: int = TraceField.CDP_Y,
    scalar_field: int = TraceField.SourceGroupScalar,
    apply_scalar: bool = True,
    domain: Literal["twt", "depth"] | None = None,
    depth_units: Literal["ms", "m", "ft"] | None = None,
    crs_epsg: int = 32632,
    progress_callback=None,
) -> SeismicDataset:
    """Read a SEG-Y file and return a :class:`SeismicDataset`.

    Parameters
    ----------
    path:
        Path to the SEG-Y / SGY file.
    x_field, y_field:
        Trace-header fields for horizontal coordinates.  Defaults to CDP_X
        (bytes 181-184) and CDP_Y (bytes 185-188).
    scalar_field:
        Trace-header field for the coordinate scalar (bytes 71-72).
    apply_scalar:
        Apply the SEG-Y coordinate scalar to *x* and *y* (default True).
        Pass ``False`` to use raw integer header values.
    domain:
        ``'twt'`` for two-way time data, ``'depth'`` for depth-migrated data.
    depth_units:
        Unit of the sample axis: ``'ms'`` (TWT milliseconds), ``'m'``, or
        ``'ft'``.  The reader does not convert units; this is stored as metadata.
    crs_epsg:
        EPSG code of the coordinate reference system for *trace_x / trace_y*.
        SEG-Y carries no CRS; the caller must supply it.
    """
    path = str(path)
    with segyio.open(path, ignore_geometry=True) as f:
        n_traces = f.tracecount
        samples = np.asarray(f.samples, dtype=float)
        dt_us = float(segyio.tools.dt(f))            # microseconds
        dt_ms = dt_us / 1000.0                       # milliseconds

        # Auto-detect domain from sample interval if caller did not override
        detected_domain, detected_units = detect_domain(dt_us)
        if domain is None:
            domain = detected_domain
        if depth_units is None:
            depth_units = detected_units

        x_raw = f.attributes(x_field)[:].astype(float)
        y_raw = f.attributes(y_field)[:].astype(float)

        if apply_scalar and n_traces > 0:
            scalars = f.attributes(scalar_field)[:].astype(float)
            trace_x = _apply_scalar(x_raw, scalars)
            trace_y = _apply_scalar(y_raw, scalars)
        else:
            trace_x = x_raw.copy()
            trace_y = y_raw.copy()

        if n_traces == 0:
            data = np.empty((0, len(samples)), dtype=np.float32)
        elif progress_callback is not None:
            # Chunked read so the caller can update a progress bar
            data = np.empty((n_traces, len(samples)), dtype=np.float32)
            chunk = max(1, n_traces // 100)
            for start in range(0, n_traces, chunk):
                end = min(start + chunk, n_traces)
                data[start:end] = f.trace.raw[start:end]
                progress_callback(int(end * 100 / n_traces))
        else:
            data = f.trace.raw[:].astype(np.float32)

    return SeismicDataset(
        name=Path(path).stem,
        data=data,
        trace_x=trace_x,
        trace_y=trace_y,
        samples=samples,
        sample_interval=dt_ms,
        domain=domain,
        depth_units=depth_units,
        crs_epsg=crs_epsg,
        sample_interval_ms=dt_ms,
    )


def read_segy_header(
    path: str | os.PathLike,
    *,
    x_field: int = TraceField.CDP_X,
    y_field: int = TraceField.CDP_Y,
    scalar_field: int = TraceField.SourceGroupScalar,
    apply_scalar: bool = True,
) -> dict[str, Any]:
    """Read SEG-Y header metadata without loading trace amplitudes.

    Returns a plain dict with keys:
    ``n_traces``, ``n_samples``, ``sample_interval_us``,
    ``sample_interval_ms``, ``samples_start``,
    ``x_range``, ``y_range``, ``text_header``.
    """
    path = str(path)
    with segyio.open(path, ignore_geometry=True) as f:
        n_traces = f.tracecount
        n_samples = len(f.samples)
        dt_us = float(segyio.tools.dt(f))
        samples_start = float(f.samples[0]) if n_samples > 0 else 0.0

        x_raw = f.attributes(x_field)[:].astype(float)
        y_raw = f.attributes(y_field)[:].astype(float)

        if apply_scalar and n_traces > 0:
            scalars = f.attributes(scalar_field)[:].astype(float)
            x = _apply_scalar(x_raw, scalars)
            y = _apply_scalar(y_raw, scalars)
        else:
            x, y = x_raw.copy(), y_raw.copy()

        try:
            text_header = segyio.tools.wrap(f.text[0])
        except Exception:
            text_header = ""

    x_range = (float(x.min()), float(x.max())) if len(x) > 0 else (0.0, 0.0)
    y_range = (float(y.min()), float(y.max())) if len(y) > 0 else (0.0, 0.0)

    return {
        "n_traces": n_traces,
        "n_samples": n_samples,
        "sample_interval_us": dt_us,
        "sample_interval_ms": dt_us / 1000.0,
        "samples_start": samples_start,
        "x_range": x_range,
        "y_range": y_range,
        "text_header": text_header,
    }


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _apply_scalar(raw: np.ndarray, scalars: np.ndarray) -> np.ndarray:
    """Apply the SEG-Y coordinate scalar to raw integer header values.

    * scalar < 0  ->  coordinate = raw / abs(scalar)
    * scalar > 0  ->  coordinate = raw * scalar
    * scalar == 0 ->  coordinate = raw  (no scaling)
    """
    # Replace zeros with 1 so that -1/scalar is safe to evaluate everywhere;
    # the np.where selector still picks 1.0 for the zero case.
    safe = np.where(scalars == 0, 1.0, scalars.astype(float))
    scale = np.where(safe < 0, -1.0 / safe, safe)
    return raw.astype(float) * scale
