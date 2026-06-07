"""M1 — layered velocity model: math, schema back-compat, identity, persistence."""
from __future__ import annotations

import math

import pytest

from section_tool.core.velocity_model import (
    VelocityModel, VelocityLayer, VelocityFunction, SCHEMA_VERSION)
from section_tool.core.formation import LITHOLOGY_DEFAULTS, Formation


# ---------------------------------------------------------------------------
# Units (SI internal truth) + matrix_velocity sanity (values read as m/s)
# ---------------------------------------------------------------------------

def test_function_is_si_internal():
    assert VelocityFunction().units == "m/s"
    with pytest.raises(ValueError):
        VelocityFunction(units="ft/s")

def test_linear_vt_reserved_not_implemented():
    with pytest.raises(NotImplementedError):
        VelocityFunction("linear_vt", v0=2000.0)

def test_matrix_velocity_defaults_read_as_ms():
    # The existing matrix_velocity defaults must sit in a sane m/s range — they
    # are interval velocities, not ft/s or km/s. No values change.
    for lith, props in LITHOLOGY_DEFAULTS.items():
        v = props["matrix_velocity"]
        assert 1000.0 <= v <= 7000.0, f"{lith}: {v} not a sane m/s velocity"
    assert 1000.0 <= Formation("X").matrix_velocity <= 7000.0


# ---------------------------------------------------------------------------
# Single-layer analytic cases (hand-computed)
# ---------------------------------------------------------------------------

def test_constant_analytic():
    m = VelocityModel.bulk(2000.0)
    assert m.twt_to_depth(1.0) == pytest.approx(1000.0)   # z = V·t/2
    assert m.depth_to_twt(1000.0) == pytest.approx(1.0)   # t = 2z/V

def test_linear_v0k_analytic():
    m = VelocityModel.average_vz(v0=2000.0, k=0.5)
    # T = (2/k)·ln(1 + k·z/v0) = 4·ln(1.25)
    assert m.depth_to_twt(1000.0) == pytest.approx(4.0 * math.log(1.25))
    # z = (v0/k)·(exp(k·T/2) − 1)
    assert m.twt_to_depth(4.0 * math.log(1.25)) == pytest.approx(1000.0)

def test_k_to_zero_limit_equals_constant():
    lin0 = VelocityModel.average_vz(v0=2500.0, k=0.0)
    const = VelocityModel.bulk(2500.0)
    for t in (0.5, 1.0, 2.0):
        assert lin0.twt_to_depth(t) == pytest.approx(const.twt_to_depth(t))


# ---------------------------------------------------------------------------
# Round-trip + monotonicity
# ---------------------------------------------------------------------------

def test_single_layer_roundtrip():
    m = VelocityModel.average_vz(v0=1800.0, k=0.6)
    for t in (0.2, 0.8, 1.5, 3.0):
        assert m.depth_to_twt(m.twt_to_depth(t)) == pytest.approx(t, abs=1e-9)

def test_monotonic_depth_increases_with_twt():
    m = VelocityModel.average_vz(v0=1600.0, k=0.4)
    zs = [m.twt_to_depth(t) for t in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)]
    assert all(b > a for a, b in zip(zs, zs[1:]))


# ---------------------------------------------------------------------------
# Layer-cake: continuity at boundaries + multi-layer round-trip
# ---------------------------------------------------------------------------

def _two_layer():
    return VelocityModel(layers=[
        VelocityLayer(VelocityFunction("constant", v0=2000.0), top_twt_s=0.0,
                      name="L0"),
        VelocityLayer(VelocityFunction("constant", v0=3000.0), top_twt_s=1.0,
                      name="L1"),
    ])

def test_layer_boundary_continuity():
    m = _two_layer()
    # depth at the boundary (t=1.0 s) is single-valued from above and below
    z_boundary = m.twt_to_depth(1.0)
    assert z_boundary == pytest.approx(1000.0)            # 1.0·2000/2
    assert m.twt_to_depth(0.999999) == pytest.approx(z_boundary, abs=1e-2)
    assert m.depth_to_twt(z_boundary) == pytest.approx(1.0, abs=1e-9)

def test_multilayer_values_and_roundtrip():
    m = _two_layer()
    assert m.twt_to_depth(2.0) == pytest.approx(2500.0)   # 1000 + 1.0·3000/2
    assert m.depth_to_twt(2500.0) == pytest.approx(2.0)
    for t in (0.3, 1.0, 1.7, 2.5):
        assert m.depth_to_twt(m.twt_to_depth(t)) == pytest.approx(t, abs=1e-9)


# ---------------------------------------------------------------------------
# Identity, provenance, method label
# ---------------------------------------------------------------------------

def test_uuid_and_construction_metadata():
    m = VelocityModel.bulk(2400.0)
    assert isinstance(m.uuid, str) and len(m.uuid) > 0
    assert set(m.construction) >= {"kind", "parents", "params"}
    assert m.construction["kind"] == "velocity_model"

def test_provenance_default_assumed_and_weakest_dominates():
    m = VelocityModel(layers=[
        VelocityLayer(VelocityFunction("constant", v0=2000.0), provenance="well_calibrated"),
        VelocityLayer(VelocityFunction("constant", v0=3000.0), top_twt_s=1.0),  # assumed
    ])
    assert m.layers[1].provenance == "assumed"             # default
    assert m.provenance == "assumed"                       # weakest dominates

def test_method_label_identifiable():
    assert "bulk" in VelocityModel.bulk(2400.0).method_label
    assert "V(z)" in VelocityModel.average_vz(1800.0, 0.6).method_label
    assert "layered" in _two_layer().method_label


# ---------------------------------------------------------------------------
# Schema: v1 back-compat + v2 round-trip
# ---------------------------------------------------------------------------

def test_v1_single_function_migrates_to_one_assumed_layer():
    old = {"model_type": "linear_v0k", "v0": 1800.0, "k": 0.6}
    m = VelocityModel.from_dict(old)
    assert len(m.layers) == 1
    assert m.layers[0].provenance == "assumed"
    assert m.layers[0].function.method == "linear_v0k"
    assert m.layers[0].function.v0 == pytest.approx(1800.0)
    # converts identically to the original single function (v0=1800, k=0.6)
    assert m.depth_to_twt(1000.0) == pytest.approx(
        VelocityModel.average_vz(1800.0, 0.6).depth_to_twt(1000.0))

def test_v2_roundtrip_preserves_layers_uuid_construction():
    m = _two_layer()
    m.construction = {"kind": "velocity_model", "parents": ["abc"], "params": {"setting": "marine"}}
    d = m.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    m2 = VelocityModel.from_dict(d)
    assert m2.uuid == m.uuid
    assert m2.construction == m.construction
    assert len(m2.layers) == 2
    assert m2.twt_to_depth(2.0) == pytest.approx(m.twt_to_depth(2.0))

def test_empty_default_roundtrips():
    m = VelocityModel()
    assert m.is_empty
    assert VelocityModel.from_dict(m.to_dict()).is_empty
    with pytest.raises(ValueError):
        m.twt_to_depth(1.0)        # empty/unconverted model can't convert


# ---------------------------------------------------------------------------
# SQLite persistence round-trip (KV blob in the velocity_model table)
# ---------------------------------------------------------------------------

def test_sqlite_velocity_model_roundtrip(tmp_path):
    from section_tool.io.database import ProjectDatabase
    db = ProjectDatabase(str(tmp_path / "p.db"))
    m = _two_layer()
    db.save_velocity_model(m)
    loaded = VelocityModel.from_dict(db.load_velocity_model())
    assert len(loaded.layers) == 2
    assert loaded.uuid == m.uuid
    assert loaded.twt_to_depth(2.0) == pytest.approx(2500.0)
    db.close()
