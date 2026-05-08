"""Tests for cross_section_tool.io.segy."""

import math

import numpy as np
import pytest
import segyio
from segyio import TraceField

from cross_section_tool.core.section import Section
from cross_section_tool.io.segy import SeismicDataset, _apply_scalar, read_segy, read_segy_header

# ---------------------------------------------------------------------------
# SEG-Y file factory
# ---------------------------------------------------------------------------

N_TRACES = 5
N_SAMPLES = 10
DT_US = 4000  # 4 ms sample interval


def _write_segy(
    path: str,
    *,
    cdp_x: list[int] | None = None,
    cdp_y: list[int] | None = None,
    scalar: int = -100,
    n_traces: int = N_TRACES,
    n_samples: int = N_SAMPLES,
    dt_us: int = DT_US,
    amplitude_fn=None,
    x_field: int = TraceField.CDP_X,
    y_field: int = TraceField.CDP_Y,
) -> None:
    """Write a minimal synthetic SEG-Y file at *path*."""
    if cdp_x is None:
        cdp_x = [(i + 1) * 10000 for i in range(n_traces)]
    if cdp_y is None:
        cdp_y = [0] * n_traces
    if amplitude_fn is None:
        amplitude_fn = lambda i: np.ones(n_samples, dtype=np.float32) * float(i + 1)

    spec = segyio.spec()
    spec.sorting = None
    spec.format = 1  # IBM float
    spec.samples = np.arange(n_samples, dtype=np.float32) * (dt_us / 1000.0)
    spec.tracecount = n_traces

    with segyio.create(path, spec) as f:
        f.bin.update(hdt=dt_us, dto=dt_us)
        for i in range(n_traces):
            f.header[i].update({
                x_field: cdp_x[i],
                y_field: cdp_y[i],
                TraceField.SourceGroupScalar: scalar,
                TraceField.TRACE_SEQUENCE_FILE: i + 1,
                TraceField.CDP: i + 1,
            })
            f.trace[i] = amplitude_fn(i)


@pytest.fixture
def segy_file(tmp_path):
    """Standard 5-trace, 10-sample SEG-Y with scalar=-100 and CDP_X in 100m steps."""
    path = str(tmp_path / "test.segy")
    _write_segy(path)
    return path


@pytest.fixture
def segy_ds(segy_file):
    return read_segy(segy_file)


# ---------------------------------------------------------------------------
# _apply_scalar unit tests
# ---------------------------------------------------------------------------

class TestApplyScalar:
    def test_negative_scalar_divides(self):
        raw = np.array([10000.0])
        sc = np.array([-100.0])
        assert pytest.approx(_apply_scalar(raw, sc)[0]) == 100.0

    def test_positive_scalar_multiplies(self):
        raw = np.array([100.0])
        sc = np.array([10.0])
        assert pytest.approx(_apply_scalar(raw, sc)[0]) == 1000.0

    def test_zero_scalar_identity(self):
        raw = np.array([12345.0])
        sc = np.array([0.0])
        assert pytest.approx(_apply_scalar(raw, sc)[0]) == 12345.0

    def test_mixed_scalars(self):
        raw = np.array([10000.0, 5000.0, 999.0])
        sc = np.array([-100.0, 10.0, 0.0])
        result = _apply_scalar(raw, sc)
        assert pytest.approx(result[0]) == 100.0
        assert pytest.approx(result[1]) == 50000.0
        assert pytest.approx(result[2]) == 999.0

    def test_array_output_shape(self):
        raw = np.array([1.0, 2.0, 3.0])
        sc = np.array([-10.0, -10.0, -10.0])
        assert _apply_scalar(raw, sc).shape == (3,)


# ---------------------------------------------------------------------------
# SeismicDataset properties
# ---------------------------------------------------------------------------

class TestSeismicDataset:
    def test_n_traces(self, segy_ds):
        assert segy_ds.n_traces == N_TRACES

    def test_n_samples(self, segy_ds):
        assert segy_ds.n_samples == N_SAMPLES

    def test_data_shape(self, segy_ds):
        assert segy_ds.data.shape == (N_TRACES, N_SAMPLES)

    def test_data_dtype(self, segy_ds):
        assert segy_ds.data.dtype == np.float32

    def test_repr_contains_name(self, segy_ds):
        assert "test" in repr(segy_ds)

    def test_repr_contains_domain(self, segy_ds):
        assert "twt" in repr(segy_ds)


# ---------------------------------------------------------------------------
# read_segy — coordinates
# ---------------------------------------------------------------------------

class TestReadSegyCoordinates:
    def test_cdp_x_scaled(self, segy_ds):
        # Raw CDP_X = [10000, 20000, ..., 50000], scalar=-100 -> [100, 200, ..., 500]
        expected = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        np.testing.assert_allclose(segy_ds.trace_x, expected)

    def test_cdp_y_zero(self, segy_ds):
        np.testing.assert_allclose(segy_ds.trace_y, 0.0)

    def test_positive_scalar(self, tmp_path):
        path = str(tmp_path / "pos_scalar.segy")
        _write_segy(path, cdp_x=[10, 20, 30, 40, 50], scalar=10)
        ds = read_segy(path)
        expected = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        np.testing.assert_allclose(ds.trace_x, expected)

    def test_zero_scalar_no_scaling(self, tmp_path):
        path = str(tmp_path / "zero_scalar.segy")
        _write_segy(path, cdp_x=[100, 200, 300, 400, 500], scalar=0)
        ds = read_segy(path)
        expected = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        np.testing.assert_allclose(ds.trace_x, expected)

    def test_apply_scalar_false(self, tmp_path):
        path = str(tmp_path / "raw.segy")
        _write_segy(path, cdp_x=[10000, 20000, 30000, 40000, 50000], scalar=-100)
        ds = read_segy(path, apply_scalar=False)
        expected = np.array([10000.0, 20000.0, 30000.0, 40000.0, 50000.0])
        np.testing.assert_allclose(ds.trace_x, expected)

    def test_custom_x_field(self, tmp_path):
        path = str(tmp_path / "custom_x.segy")
        # Store x in FieldRecord (bytes 9-12) instead of CDP_X
        _write_segy(
            path,
            cdp_x=[5000, 6000, 7000, 8000, 9000],
            scalar=-10,
            x_field=TraceField.FieldRecord,
        )
        ds = read_segy(path, x_field=TraceField.FieldRecord, scalar_field=TraceField.SourceGroupScalar)
        expected = np.array([500.0, 600.0, 700.0, 800.0, 900.0])
        np.testing.assert_allclose(ds.trace_x, expected)

    def test_trace_x_dtype_float64(self, segy_ds):
        assert segy_ds.trace_x.dtype == np.float64

    def test_trace_y_dtype_float64(self, segy_ds):
        assert segy_ds.trace_y.dtype == np.float64


# ---------------------------------------------------------------------------
# read_segy — amplitudes
# ---------------------------------------------------------------------------

class TestReadSegyAmplitudes:
    def test_trace_values(self, tmp_path):
        path = str(tmp_path / "amp.segy")
        # Each trace i has value (i+1) for all samples
        _write_segy(path)
        ds = read_segy(path)
        for i in range(N_TRACES):
            np.testing.assert_allclose(ds.data[i], float(i + 1), rtol=1e-5)

    def test_custom_amplitudes(self, tmp_path):
        path = str(tmp_path / "custom.segy")
        rng = np.random.default_rng(42)
        known = rng.standard_normal((N_TRACES, N_SAMPLES)).astype(np.float32)
        _write_segy(path, amplitude_fn=lambda i: known[i])
        ds = read_segy(path)
        np.testing.assert_allclose(ds.data, known, rtol=1e-5)


# ---------------------------------------------------------------------------
# read_segy — samples and metadata
# ---------------------------------------------------------------------------

class TestReadSegyMetadata:
    def test_sample_interval_ms(self, segy_ds):
        assert pytest.approx(segy_ds.sample_interval) == 4.0  # 4000 us = 4 ms

    def test_samples_array(self, segy_ds):
        expected = np.arange(N_SAMPLES, dtype=float) * 4.0  # 0, 4, 8, ... ms
        np.testing.assert_allclose(segy_ds.samples, expected)

    def test_samples_start_zero(self, segy_ds):
        assert segy_ds.samples[0] == 0.0

    def test_name_from_path_stem(self, segy_file, segy_ds):
        assert segy_ds.name == "test"

    def test_domain_default_twt(self, segy_ds):
        assert segy_ds.domain == "twt"

    def test_domain_depth(self, tmp_path):
        path = str(tmp_path / "depth.segy")
        _write_segy(path)
        ds = read_segy(path, domain="depth", depth_units="m")
        assert ds.domain == "depth"
        assert ds.depth_units == "m"

    def test_depth_units_default_ms(self, segy_ds):
        assert segy_ds.depth_units == "ms"

    def test_crs_epsg_stored(self, tmp_path):
        path = str(tmp_path / "crs.segy")
        _write_segy(path)
        ds = read_segy(path, crs_epsg=27700)
        assert ds.crs_epsg == 27700

    def test_crs_epsg_default(self, segy_ds):
        assert segy_ds.crs_epsg == 32632


# ---------------------------------------------------------------------------
# SeismicDataset — section projection
# ---------------------------------------------------------------------------

class TestProjectOntSection:
    def test_traces_on_east_section(self, segy_ds):
        # Traces at x=[100,200,300,400,500], y=0 on a section going east
        sec = Section([(0.0, 0.0), (600.0, 0.0)])
        distances, perps = segy_ds.project_onto_section(sec)
        expected = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        np.testing.assert_allclose(distances, expected, atol=1e-9)
        np.testing.assert_allclose(perps, 0.0, atol=1e-9)

    def test_traces_offset_from_section(self, tmp_path):
        # Traces 50 m north of an east section -> all perps = 50
        path = str(tmp_path / "offset.segy")
        _write_segy(path, cdp_x=[10000, 20000, 30000, 40000, 50000],
                    cdp_y=[5000, 5000, 5000, 5000, 5000], scalar=-100)
        ds = read_segy(path)
        sec = Section([(0.0, 0.0), (600.0, 0.0)])
        _, perps = ds.project_onto_section(sec)
        np.testing.assert_allclose(perps, 50.0, atol=1e-9)

    def test_project_empty_dataset(self, tmp_path):
        path = str(tmp_path / "empty.segy")
        _write_segy(path, n_traces=1)  # minimum 1 trace, then use slice
        ds = read_segy(path)
        # Manually make empty
        empty_ds = SeismicDataset(
            name="empty", data=np.empty((0, 10), dtype=np.float32),
            trace_x=np.array([]), trace_y=np.array([]),
            samples=np.arange(10.0), sample_interval=4.0,
            domain="twt", depth_units="ms", crs_epsg=32632,
        )
        sec = Section([(0.0, 0.0), (600.0, 0.0)])
        distances, perps = empty_ds.project_onto_section(sec)
        assert len(distances) == 0
        assert len(perps) == 0

    def test_traces_sorted_by_section_order(self, tmp_path):
        # Write traces in reverse distance order
        path = str(tmp_path / "rev.segy")
        _write_segy(path,
                    cdp_x=[50000, 40000, 30000, 20000, 10000],
                    cdp_y=[0, 0, 0, 0, 0], scalar=-100)
        ds = read_segy(path)
        sec = Section([(0.0, 0.0), (600.0, 0.0)])
        distances, data, perps = ds.traces_sorted_by_section(sec)
        # Distances should be ascending
        assert np.all(np.diff(distances) >= 0)
        expected = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        np.testing.assert_allclose(distances, expected, atol=1e-9)

    def test_traces_sorted_preserves_data_correspondence(self, tmp_path):
        # Trace at x=100 has amplitude 5.0; trace at x=500 has amplitude 1.0 (reversed)
        path = str(tmp_path / "assoc.segy")
        _write_segy(
            path,
            cdp_x=[50000, 40000, 30000, 20000, 10000],
            cdp_y=[0, 0, 0, 0, 0],
            scalar=-100,
            amplitude_fn=lambda i: np.ones(N_SAMPLES, dtype=np.float32) * float(i + 1),
        )
        ds = read_segy(path)
        sec = Section([(0.0, 0.0), (600.0, 0.0)])
        distances, data, _ = ds.traces_sorted_by_section(sec)
        # After sorting, closest trace (x=100) was originally last (i=4), amp=5
        assert pytest.approx(float(data[0, 0])) == 5.0
        # Farthest trace (x=500) was originally first (i=0), amp=1
        assert pytest.approx(float(data[-1, 0])) == 1.0

    def test_project_onto_dogleg_section(self, tmp_path):
        # Traces along a dogleg section — all on-line, perp should be ~0
        path = str(tmp_path / "dogleg.segy")
        # Traces at (100,0),(200,0),(300,0) on first leg and (300,100),(300,200) on second
        _write_segy(
            path,
            cdp_x=[10000, 20000, 30000, 30000, 30000],
            cdp_y=[0, 0, 0, 10000, 20000],
            scalar=-100,
        )
        ds = read_segy(path)
        sec = Section([(0.0, 0.0), (300.0, 0.0), (300.0, 300.0)])
        distances, perps = ds.project_onto_section(sec)
        np.testing.assert_allclose(perps, 0.0, atol=1e-9)
        expected = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        np.testing.assert_allclose(distances, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# read_segy_header
# ---------------------------------------------------------------------------

class TestReadSegyHeader:
    def test_required_keys(self, segy_file):
        hdr = read_segy_header(segy_file)
        for key in ("n_traces", "n_samples", "sample_interval_us",
                    "sample_interval_ms", "samples_start",
                    "x_range", "y_range", "text_header"):
            assert key in hdr

    def test_n_traces(self, segy_file):
        assert read_segy_header(segy_file)["n_traces"] == N_TRACES

    def test_n_samples(self, segy_file):
        assert read_segy_header(segy_file)["n_samples"] == N_SAMPLES

    def test_sample_interval_us(self, segy_file):
        assert pytest.approx(read_segy_header(segy_file)["sample_interval_us"]) == float(DT_US)

    def test_sample_interval_ms(self, segy_file):
        assert pytest.approx(read_segy_header(segy_file)["sample_interval_ms"]) == 4.0

    def test_samples_start_zero(self, segy_file):
        assert read_segy_header(segy_file)["samples_start"] == 0.0

    def test_x_range(self, segy_file):
        xmin, xmax = read_segy_header(segy_file)["x_range"]
        assert pytest.approx(xmin) == 100.0
        assert pytest.approx(xmax) == 500.0

    def test_y_range_all_zero(self, segy_file):
        ymin, ymax = read_segy_header(segy_file)["y_range"]
        assert ymin == 0.0
        assert ymax == 0.0

    def test_text_header_is_string(self, segy_file):
        assert isinstance(read_segy_header(segy_file)["text_header"], str)

    def test_apply_scalar_false_in_header(self, tmp_path):
        path = str(tmp_path / "hdr.segy")
        _write_segy(path, cdp_x=[10000, 20000, 30000, 40000, 50000], scalar=-100)
        hdr = read_segy_header(path, apply_scalar=False)
        xmin, xmax = hdr["x_range"]
        assert pytest.approx(xmin) == 10000.0
        assert pytest.approx(xmax) == 50000.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_trace(self, tmp_path):
        path = str(tmp_path / "single.segy")
        _write_segy(path, n_traces=1, cdp_x=[10000], cdp_y=[0])
        ds = read_segy(path)
        assert ds.n_traces == 1
        assert ds.data.shape == (1, N_SAMPLES)

    def test_single_sample(self, tmp_path):
        path = str(tmp_path / "onesample.segy")
        _write_segy(path, n_samples=1)
        ds = read_segy(path)
        assert ds.n_samples == 1
        assert ds.data.shape == (N_TRACES, 1)

    def test_large_negative_scalar(self, tmp_path):
        path = str(tmp_path / "large_sc.segy")
        _write_segy(path, cdp_x=[1000000, 2000000, 3000000, 4000000, 5000000], scalar=-1000)
        ds = read_segy(path)
        np.testing.assert_allclose(ds.trace_x, [1000.0, 2000.0, 3000.0, 4000.0, 5000.0])

    def test_path_as_pathlib(self, tmp_path):
        import pathlib
        path = tmp_path / "pl.segy"
        _write_segy(str(path))
        ds = read_segy(path)
        assert ds.n_traces == N_TRACES

    def test_non_uniform_scalars(self, tmp_path):
        """Each trace can carry a different scalar."""
        path = str(tmp_path / "mixed_sc.segy")
        spec = segyio.spec()
        spec.sorting = None
        spec.format = 1
        spec.samples = np.arange(N_SAMPLES, dtype=np.float32) * 4.0
        spec.tracecount = 2
        with segyio.create(str(path), spec) as f:
            f.bin.update(hdt=DT_US, dto=DT_US)
            f.header[0].update({
                TraceField.CDP_X: 10000,
                TraceField.CDP_Y: 0,
                TraceField.SourceGroupScalar: -100,
                TraceField.TRACE_SEQUENCE_FILE: 1,
            })
            f.header[1].update({
                TraceField.CDP_X: 5000,
                TraceField.CDP_Y: 0,
                TraceField.SourceGroupScalar: -10,
                TraceField.TRACE_SEQUENCE_FILE: 2,
            })
            f.trace[0] = np.ones(N_SAMPLES, dtype=np.float32)
            f.trace[1] = np.ones(N_SAMPLES, dtype=np.float32) * 2.0
        ds = read_segy(path)
        assert pytest.approx(ds.trace_x[0]) == 100.0   # 10000 / 100
        assert pytest.approx(ds.trace_x[1]) == 500.0   # 5000  / 10
