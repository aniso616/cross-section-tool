"""M2 — conversion engine + the TWT-anchor contract (well-free, headless)."""
from __future__ import annotations

import numpy as np
import pytest

from section_tool.core.conversion import (
    build_bulk, build_average_vz, build_layered_from_formations,
    stretch_trace_to_depth, stretch_image_to_depth,
    recover_anchors, set_anchors, derive_depths, apply_depths_from_anchors)
from section_tool.core.surfaces import HorizonPick
from section_tool.core.formation import StratigraphicColumn, Formation


# ---------------------------------------------------------------------------
# The ladder: bulk / average need no picks; layered is interpretation-gated
# ---------------------------------------------------------------------------

def test_bulk_and_average_need_no_picks():
    assert build_bulk(2400.0).twt_to_depth(1.0) == pytest.approx(1200.0)
    assert "bulk" in build_bulk(2400.0).method_label
    m = build_average_vz(1800.0, 0.6)
    assert m.twt_to_depth(1.0) > 0 and "V(z)" in m.method_label

def test_layered_unavailable_without_picks():
    with pytest.raises(ValueError):
        build_layered_from_formations([], strat_column=None)

def test_layered_from_formations_seeds_matrix_velocity():
    col = StratigraphicColumn()
    a = Formation("A"); a.matrix_velocity = 2500.0
    b = Formation("B"); b.matrix_velocity = 4000.0
    col.add_formation(a); col.add_formation(b)
    m = build_layered_from_formations([(0.0, "A"), (0.6, "B")], col)
    assert len(m.layers) == 2
    assert m.layers[0].function.v0 == pytest.approx(2500.0)
    assert m.layers[1].function.v0 == pytest.approx(4000.0)
    assert m.layers[0].provenance == "assumed"
    assert "layered" in m.method_label


# ---------------------------------------------------------------------------
# Seismic stretch (twt → depth): sample-count / energy sanity
# ---------------------------------------------------------------------------

def test_trace_stretch_sanity():
    amp = np.array([0, 1, 0, -1, 0, 1, 0, -1, 0, 1], dtype=float)
    m = build_bulk(2000.0)
    z_axis, d = stretch_trace_to_depth(amp, dt_s=0.004, model=m, z_max=20.0, dz=1.0)
    assert len(d) == len(z_axis)
    assert np.all(np.isfinite(d))
    assert np.max(np.abs(d)) <= np.max(np.abs(amp)) + 1e-9   # interp can't amplify

def test_image_stretch_shape():
    img = np.random.RandomState(0).randn(8, 50)
    m = build_average_vz(1800.0, 0.5)
    z_axis, dimg = stretch_image_to_depth(img, dt_s=0.004, model=m, z_max=200.0, dz=5.0)
    assert dimg.shape == (8, len(z_axis))
    assert np.all(np.isfinite(dimg))


# ---------------------------------------------------------------------------
# The TWT-anchor contract — the heart of pick-once-refine-forever
# ---------------------------------------------------------------------------

def _depth_horizon():
    # depth-native to start; set_anchors ties it through the active model
    return HorizonPick(np.array([0.0, 500.0]), np.array([1000.0, 1200.0]),
                       name="H", section_names=np.array(["L1", "L1"], dtype=object))

def test_anchor_recovery_exact_under_same_model():
    hp = _depth_horizon()
    m = build_bulk(2000.0)
    set_anchors(hp, m)
    assert hp.seismic_tied
    # anchor = depth_to_twt(z); re-deriving depth under the SAME model is exact
    assert np.allclose(derive_depths(hp, m), [1000.0, 1200.0], atol=1e-9)
    assert np.allclose(hp._twt_anchor,
                       [m.depth_to_twt(1000.0), m.depth_to_twt(1200.0)])

def test_anchor_invariant_across_model_swap():
    # Pick under bulk, switch to a layered model: depth RE-DERIVES, anchor does NOT.
    hp = _depth_horizon()
    m_bulk = build_bulk(2000.0)
    set_anchors(hp, m_bulk)
    saved = hp._twt_anchor.copy()

    m_new = build_average_vz(1800.0, 0.5)
    apply_depths_from_anchors(hp, m_new)
    assert np.allclose(hp._twt_anchor, saved)                 # anchor invariant
    assert np.allclose(hp._depths, [m_new.twt_to_depth(a) for a in saved])
    assert not np.allclose(hp._depths, [1000.0, 1200.0])      # depth actually moved

def test_depth_native_geometry_untouched_by_iteration():
    hp = _depth_horizon()                 # never tied
    before = hp._depths.copy()
    apply_depths_from_anchors(hp, build_bulk(5000.0))   # no-op: not seismic_tied
    assert np.allclose(hp._depths, before)


# ---------------------------------------------------------------------------
# Anchor persists across save/load (SQLite KV — the primary store)
# ---------------------------------------------------------------------------

def test_anchor_persists_via_db(tmp_path):
    from section_tool.io.database import ProjectDatabase
    db = ProjectDatabase(str(tmp_path / "p.db"))
    hp = _depth_horizon()
    set_anchors(hp, build_bulk(2000.0))
    db.upsert_horizon(hp)
    h = db.get_all_horizons()[0]
    assert h["seismic_tied"] == 1
    anchors = sorted(p["twt_anchor"] for p in h["picks"])
    assert anchors == pytest.approx(sorted(hp._twt_anchor.tolist()))
    db.close()
