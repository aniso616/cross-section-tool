"""M3 — stretch-setup controller (the depth-stretch tool's headless logic)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from section_tool.core.stretch_setup import StretchSetup, WATER_VELOCITY_MS
from section_tool.core.formation import StratigraphicColumn, Formation
from section_tool.core.surfaces import HorizonPick
from section_tool.core.conversion import set_anchors, build_bulk


def test_onshore_bulk_single_layer():
    m = StretchSetup(setting="onshore", method="bulk", bulk_v=2400.0).build_model()
    assert len(m.layers) == 1
    assert m.layers[0].function.v0 == pytest.approx(2400.0)
    assert "bulk" in m.method_label

def test_marine_adds_water_top_layer():
    s = StretchSetup(setting="marine", method="bulk", bulk_v=2600.0,
                     seafloor_twt_s=0.4)
    m = s.build_model()
    assert len(m.layers) == 2
    assert m.layers[0].name == "Water"
    assert m.layers[0].function.v0 == pytest.approx(WATER_VELOCITY_MS)
    assert m.layers[0].top_twt_s == pytest.approx(0.0)
    assert m.layers[1].top_twt_s == pytest.approx(0.4)   # bulk starts at seafloor
    assert m.layers[1].function.v0 == pytest.approx(2600.0)

def test_average_vz_method():
    m = StretchSetup(method="average_vz", v0=1700.0, k=0.5).build_model()
    assert m.layers[0].function.method == "linear_v0k"
    assert m.layers[0].function.k == pytest.approx(0.5)

def test_layered_gated_then_builds():
    s = StretchSetup(method="layered_from_formations")
    assert not s.method_available(has_zone_tops=False)
    assert s.method_available(has_zone_tops=True)
    with pytest.raises(ValueError):
        s.build_model(zone_tops=None)
    col = StratigraphicColumn()
    a = Formation("A"); a.matrix_velocity = 2500.0
    b = Formation("B"); b.matrix_velocity = 4000.0
    col.add_formation(a); col.add_formation(b)
    m = s.build_model(zone_tops=[(0.0, "A"), (0.5, "B")], strat_column=col)
    assert len(m.layers) == 2
    assert m.layers[1].function.v0 == pytest.approx(4000.0)

def test_construction_records_method_and_setting():
    m = StretchSetup(setting="marine", method="bulk", seafloor_twt_s=0.3).build_model()
    p = m.construction["params"]
    assert p["setting"] == "marine"
    assert p["method"] == "bulk"
    assert p["seafloor_twt_s"] == pytest.approx(0.3)

def test_apply_installs_model_and_restretches_tied():
    # a tied horizon follows the applied model; a depth-native one does not
    tied = HorizonPick(np.array([0.0, 500.0]), np.array([1000.0, 1200.0]),
                       name="T", section_names=np.array(["L1", "L1"], dtype=object))
    set_anchors(tied, build_bulk(2000.0))
    native = HorizonPick(np.array([0.0, 500.0]), np.array([800.0, 900.0]),
                         name="N", section_names=np.array(["L1", "L1"], dtype=object))
    native_before = native._depths.copy()
    proj = SimpleNamespace(horizon_picks=[tied, native], fault_picks=[],
                           strat_column=None, velocity_model=None)

    model = StretchSetup(method="average_vz", v0=1800.0, k=0.5).apply(proj)
    assert proj.velocity_model is model
    assert np.allclose(native._depths, native_before)             # native fixed
    assert np.allclose(tied._depths,
                       [model.twt_to_depth(a) for a in tied._twt_anchor])  # tied follows
