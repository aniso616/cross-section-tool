"""M2 — conversion engine + the TWT-anchor contract (well-free, headless)."""
from __future__ import annotations

import numpy as np
import pytest

from section_tool.core.conversion import (
    build_bulk, build_average_vz, build_layered_from_formations,
    zone_tops_from_picks, restretch_project,
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

def test_restretch_moves_tied_leaves_native():
    """The re-stretch contract: applying a new model re-derives seismic-tied
    geometry from anchors (it follows the backdrop) and leaves depth-native
    geometry fixed."""
    from types import SimpleNamespace
    from section_tool.core.conversion import restretch_project

    tied = _depth_horizon()
    set_anchors(tied, build_bulk(2000.0))      # anchors at t = z/1000
    native = _depth_horizon()                  # never tied
    native_before = native._depths.copy()

    proj = SimpleNamespace(horizon_picks=[tied, native], fault_picks=[])
    n = restretch_project(proj, build_average_vz(1800.0, 0.5))
    assert n == 1                              # only the tied one moved
    assert np.allclose(native._depths, native_before)        # native fixed
    assert not np.allclose(tied._depths, [1000.0, 1200.0])   # tied re-derived

def test_live_iteration_keeps_horizon_glued():
    """Tuning velocity twice: depth tracks the model each time, anchor invariant."""
    from types import SimpleNamespace
    from section_tool.core.conversion import restretch_project

    hp = _depth_horizon()
    set_anchors(hp, build_bulk(2000.0))
    anchors = hp._twt_anchor.copy()
    proj = SimpleNamespace(horizon_picks=[hp], fault_picks=[])
    for v in (2200.0, 2600.0, 3000.0):
        restretch_project(proj, build_bulk(v))
        assert np.allclose(hp._twt_anchor, anchors)                      # invariant
        assert np.allclose(hp._depths, [build_bulk(v).twt_to_depth(a) for a in anchors])


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


# ---------------------------------------------------------------------------
# Layered-from-formations (Prompt 2): zone tops from anchors + formation seeding
# ---------------------------------------------------------------------------

def _tied_zone_horizon(depth_m, fa, fb, name):
    """A seismic-tied horizon whose anchor (under bulk 2000) is depth_m/1000 s."""
    hp = HorizonPick(np.array([0.0, 500.0]), np.array([depth_m, depth_m]),
                     name=name, formation_above=fa, formation_below=fb)
    set_anchors(hp, build_bulk(2000.0))     # anchor = depth_to_twt(z) = z/1000
    return hp


def _strat():
    col = StratigraphicColumn()
    for nm, v in [("Overburden", 2500.0), ("Reservoir", 4000.0), ("Basement", 5500.0)]:
        f = Formation(nm); f.matrix_velocity = v; col.add_formation(f)
    return col


def test_zone_tops_from_picks_caps_with_formation_above():
    picks = [_tied_zone_horizon(800.0, "Overburden", "Reservoir", "TopRes"),
             _tied_zone_horizon(1200.0, "Reservoir", "Basement", "BaseRes")]
    tops = zone_tops_from_picks(picks, base_twt_s=0.0)
    # cap at datum (formation_above of shallowest), then each anchor / formation_below
    # anchor = depth_to_twt(z) = 2z/2000 = z/1000  → 800 m = 0.8 s, 1200 m = 1.2 s
    assert tops[0][0] == pytest.approx(0.0) and tops[0][1] == "Overburden"
    assert tops[1][0] == pytest.approx(0.8) and tops[1][1] == "Reservoir"
    assert tops[2][0] == pytest.approx(1.2) and tops[2][1] == "Basement"


def test_zone_tops_from_picks_marine_base_seafloor():
    picks = [_tied_zone_horizon(1000.0, "Overburden", "Reservoir", "H1")]
    tops = zone_tops_from_picks(picks, base_twt_s=0.3)   # seafloor at 0.3 s
    assert tops[0] == (pytest.approx(0.3), "Overburden")  # cap starts at seafloor
    assert tops[1][0] == pytest.approx(1.0)               # 1000 m → 1.0 s


def test_zone_tops_from_picks_ignores_depth_native_and_empty():
    native = HorizonPick(np.array([0.0, 500.0]), np.array([900.0, 900.0]), name="N")
    assert zone_tops_from_picks([native], 0.0) == []
    assert zone_tops_from_picks([], 0.0) == []


def test_layered_build_from_picks_seeds_formation_velocities():
    picks = [_tied_zone_horizon(800.0, "Overburden", "Reservoir", "TopRes"),
             _tied_zone_horizon(1200.0, "Reservoir", "Basement", "BaseRes")]
    tops = zone_tops_from_picks(picks, base_twt_s=0.0)
    m = build_layered_from_formations(tops, _strat())
    assert [round(l.top_twt_s, 6) for l in m.layers] == [0.0, 0.8, 1.2]
    assert [l.function.v0 for l in m.layers] == [2500.0, 4000.0, 5500.0]
    assert m.method_label == "layered — formation matrix velocities"
    assert m.provenance == "assumed"


def test_layered_tune_keeps_horizon_glued():
    """Changing a zone's interval velocity re-stretches; the seismic-tied horizon
    follows through its invariant anchor."""
    picks = [_tied_zone_horizon(800.0, "Overburden", "Reservoir", "TopRes"),
             _tied_zone_horizon(1200.0, "Reservoir", "Basement", "BaseRes")]
    tops = zone_tops_from_picks(picks, 0.0)
    strat = _strat()
    m_a = build_layered_from_formations(tops, strat)
    from types import SimpleNamespace
    proj = SimpleNamespace(horizon_picks=picks, fault_picks=[])
    restretch_project(proj, m_a)
    anchors = [hp._twt_anchor.copy() for hp in picks]
    # "Tune" the Reservoir interval velocity up and re-stretch.
    strat.get_formation("Reservoir").matrix_velocity = 4600.0
    m_b = build_layered_from_formations(tops, strat)
    restretch_project(proj, m_b)
    for hp, a in zip(picks, anchors):
        assert np.allclose(hp._twt_anchor, a)                                  # invariant
        assert np.allclose(hp._depths, [m_b.twt_to_depth(t) for t in hp._twt_anchor])


# ---------------------------------------------------------------------------
# Glue regression extended to the grounded rungs (Prompt 06): a seismic-tied
# horizon stays glued (anchor invariant; depth follows the model) when the model
# is swapped to a from_tdr / from_sonic model — same contract, new methods.
# ---------------------------------------------------------------------------

def _grounded_well():
    from section_tool.core.wells import Well, LogCurve
    from section_tool.core.grounded_velocity import _US_FT_TO_S_PER_M
    w = Well("W", 0.0, 0.0, kb=0.0, td=3000.0)
    md = np.arange(0.0, 3001.0, 5.0)
    slow = (1.0 / 2000.0) / _US_FT_TO_S_PER_M          # 152 µs/ft → v=2000
    w.add_log(LogCurve("DT", "us/ft", md, np.full_like(md, slow)))
    return w


def _checkshot_tdr():
    from section_tool.core.tdr import TimeDepthRelation
    return TimeDepthRelation([0.0, 1000.0, 2000.0, 3000.0],
                             [0.0, 0.95, 1.90, 2.85],
                             kind="checkshot", depth_reference="TVDSS")


def test_from_tdr_model_keeps_horizon_glued():
    from section_tool.core.velocity_model import VelocityModel
    from types import SimpleNamespace
    hp = _depth_horizon()
    set_anchors(hp, build_bulk(2000.0))
    anchors = hp._twt_anchor.copy()

    m = VelocityModel.from_tdr(_grounded_well(), _checkshot_tdr(), setting="onshore")
    proj = SimpleNamespace(horizon_picks=[hp], fault_picks=[])
    restretch_project(proj, m)
    assert np.allclose(hp._twt_anchor, anchors)                           # invariant
    assert np.allclose(hp._depths, [m.twt_to_depth(t) for t in anchors])  # follows model


def test_from_sonic_model_keeps_horizon_glued():
    from section_tool.core.velocity_model import VelocityModel
    from types import SimpleNamespace
    w = _grounded_well()
    w.add_tdr(_checkshot_tdr())
    hp = _depth_horizon()
    set_anchors(hp, build_bulk(2000.0))
    anchors = hp._twt_anchor.copy()

    for target in ("none", "checkshot"):
        m = VelocityModel.from_sonic(w, drift_target=target, setting="onshore")
        proj = SimpleNamespace(horizon_picks=[hp], fault_picks=[])
        restretch_project(proj, m)
        assert np.allclose(hp._twt_anchor, anchors)                       # invariant
        assert np.allclose(hp._depths, [m.twt_to_depth(t) for t in anchors])
