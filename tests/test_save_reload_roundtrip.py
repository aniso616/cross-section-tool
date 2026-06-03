"""Phase 2 scenario 10 — save / reload roundtrip (SQLite project).

This is the data-loss guard: what a user interprets must survive a save+reopen.
The harness builds a DB-backed project, populates it with sections, horizon &
fault picks (carrying construction rules), and polygons, then reopens it in a
fresh AppState and asserts everything came back — with the construction
metadata checked *deeply* (rule type AND parameters), since silently dropping a
DipConstrainedRule's angle or a ParallelToBedRule's offset would lose work
without any error.

Two known limitations are encoded as xfail(strict=True): they wrap the
*desired* behaviour (the thing IS restored) which currently fails, so the day
either is fixed the strict-xfail flips to a failure that says "delete this
marker / the gap is closed."
"""
from __future__ import annotations

import numpy as np
import pytest

from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick
from section_tool.core.polygons import PolygonBoundary, SectionPolygon
from section_tool.core.construction import (
    DipConstrainedRule, KinkBandRule, ParallelToBedRule,
)


def _hp(name, d0, d1, z0, z1, rule=None, color="#2ca02c"):
    hp = HorizonPick(np.array([d0, d1], dtype=float), np.array([z0, z1], dtype=float),
                     name=name, color=color, section_names=["L1", "L1"])
    if rule is not None:
        hp.construction_rule = rule
    return hp


def _populate(state):
    """Build a fully-featured project in *state* (which already has an open DB)."""
    sec = Section([(0.0, 0.0), (2000.0, 0.0)], name="L1",
                  depth_domain="depth", depth_units="m",
                  vertical_exaggeration=2.0, crs_epsg=32631)
    state.add_section(sec)
    state.set_active_section(sec)

    state.add_horizon_pick(_hp("DipH", 0, 2000, 100, 200,
                               rule=DipConstrainedRule(dip_deg=23.5)))        # idx 0
    state.add_horizon_pick(_hp("KinkH", 0, 2000, 300, 400,
                               rule=KinkBandRule(axial_surface_dip_deg=45.0,
                                                 fore_dip_deg=30.0, back_dip_deg=5.0)))  # idx 1
    state.add_horizon_pick(_hp("ParaH", 0, 2000, 500, 600,
                               rule=ParallelToBedRule(reference_name="DipH",
                                                      offset_m=120.0)))       # idx 2
    state.add_fault_pick(_hp("F1", 0, 2000, 700, 800,
                             rule=DipConstrainedRule(dip_deg=60.0), color="#d62728"))

    # bound polygon (Top=DipH fwd, Bot=ParaH reversed)
    bound_poly = SectionPolygon(
        [[0, 100], [2000, 200], [2000, 600], [0, 500]],
        name="Layer", section_name="L1",
        bounds=[PolygonBoundary("Horizons", 0, reversed=False),
                PolygonBoundary("Horizons", 2, reversed=True)])
    state.add_polygon(bound_poly)

    # free polygon carrying its own construction rule
    ruled_poly = SectionPolygon([[0, 100], [100, 100], [100, 200]],
                                name="RuleP", section_name="L1")
    ruled_poly.construction_rule = ParallelToBedRule(reference_name="DipH", offset_m=80.0)
    state.add_polygon(ruled_poly)
    return sec


def _roundtrip(tmp_path):
    """Create+save a project, return a fresh AppState that has reopened it."""
    folder = str(tmp_path / "proj")
    src = AppState()
    src.new_project(name="RT", crs_epsg=32631, folder_path=folder)
    _populate(src)
    src.save_project()

    dst = AppState()
    dst.open_project(folder)
    return dst


def _by_name(picks, name):
    return next(p for p in picks if p.name == name)


# ---------------------------------------------------------------------------
# Core geometry survives
# ---------------------------------------------------------------------------

class TestGeometryRoundtrip:

    def test_section_survives(self, tmp_path):
        proj = _roundtrip(tmp_path).project
        assert [s.name for s in proj.sections] == ["L1"]
        sec = proj.sections[0]
        assert sec.nodes == pytest.approx(np.array([[0, 0], [2000, 0]]))
        assert sec.depth_units == "m"
        assert sec.crs_epsg == 32631
        assert sec.vertical_exaggeration == pytest.approx(2.0)

    def test_horizons_survive(self, tmp_path):
        proj = _roundtrip(tmp_path).project
        assert {h.name for h in proj.horizon_picks} == {"DipH", "KinkH", "ParaH"}
        dip = _by_name(proj.horizon_picks, "DipH")
        assert dip._distances == pytest.approx([0.0, 2000.0])
        assert dip._depths == pytest.approx([100.0, 200.0])

    def test_faults_survive(self, tmp_path):
        proj = _roundtrip(tmp_path).project
        f = _by_name(proj.fault_picks, "F1")
        assert f._depths == pytest.approx([700.0, 800.0])

    def test_polygons_survive(self, tmp_path):
        proj = _roundtrip(tmp_path).project
        names = {p.name for p in proj.polygons}
        assert {"Layer", "RuleP"} <= names
        layer = _by_name(proj.polygons, "Layer")
        assert layer.vertices == pytest.approx(
            np.array([[0, 100], [2000, 200], [2000, 600], [0, 500]]))


# ---------------------------------------------------------------------------
# Construction metadata survives — DEEP (type + parameters)
# ---------------------------------------------------------------------------

class TestConstructionMetadataRoundtrip:

    def test_dip_constrained_rule_params(self, tmp_path):
        dip = _by_name(_roundtrip(tmp_path).project.horizon_picks, "DipH")
        assert isinstance(dip.construction_rule, DipConstrainedRule)
        assert dip.construction_rule.dip_deg == pytest.approx(23.5)

    def test_kink_band_rule_params(self, tmp_path):
        kink = _by_name(_roundtrip(tmp_path).project.horizon_picks, "KinkH")
        r = kink.construction_rule
        assert isinstance(r, KinkBandRule)
        assert r.axial_surface_dip_deg == pytest.approx(45.0)
        assert r.fore_dip_deg == pytest.approx(30.0)
        assert r.back_dip_deg == pytest.approx(5.0)

    def test_parallel_rule_params(self, tmp_path):
        para = _by_name(_roundtrip(tmp_path).project.horizon_picks, "ParaH")
        r = para.construction_rule
        assert isinstance(r, ParallelToBedRule)
        assert r.reference_name == "DipH"
        assert r.offset_m == pytest.approx(120.0)

    def test_fault_rule_params(self, tmp_path):
        f = _by_name(_roundtrip(tmp_path).project.fault_picks, "F1")
        assert isinstance(f.construction_rule, DipConstrainedRule)
        assert f.construction_rule.dip_deg == pytest.approx(60.0)

    def test_polygon_construction_rule_params(self, tmp_path):
        rp = _by_name(_roundtrip(tmp_path).project.polygons, "RuleP")
        assert isinstance(rp.construction_rule, ParallelToBedRule)
        assert rp.construction_rule.reference_name == "DipH"
        assert rp.construction_rule.offset_m == pytest.approx(80.0)

    def test_polygon_bounds_survive_roundtrip(self, tmp_path):
        """Bound polygons reload as bound (bounds_json persisted) so scenario-7
        auto-update keeps working after save/reload."""
        layer = _by_name(_roundtrip(tmp_path).project.polygons, "Layer")
        assert len(layer.bounds) == 2
        assert layer.bounds[0].category == "Horizons" and layer.bounds[0].index == 0
        assert layer.bounds[1].reversed is True

    def test_auto_update_still_works_after_reload(self, tmp_path):
        """The whole point of persisting bounds: editing a bounding horizon
        after a reload still reshapes the polygon (scenario-7 survives)."""
        dst = _roundtrip(tmp_path)
        dst.set_active_section(dst.project.sections[0])
        layer = _by_name(dst.project.polygons, "Layer")
        assert layer.vertices[:, 1].min() == pytest.approx(100.0)   # DipH top edge

        dst.update_horizon_pick(0, _hp("DipH", 0, 2000, 40, 40))    # raise DipH
        assert layer.vertices[:, 1].min() == pytest.approx(40.0)     # polygon followed


# ---------------------------------------------------------------------------
# Stable entity identity (Step 1): UUIDs survive + are stable across reload
# ---------------------------------------------------------------------------

class TestEntityIdentity:

    def test_every_entity_has_unique_uuid_after_reload(self, tmp_path):
        proj = _roundtrip(tmp_path).project
        entities = proj.horizon_picks + proj.fault_picks
        assert all(getattr(e, "uuid", None) for e in entities)   # non-empty
        uuids = [e.uuid for e in entities]
        assert len(uuids) == len(set(uuids))                     # unique

    def test_uuid_is_stable_across_save_reload(self, tmp_path):
        """Identity must be preserved, not regenerated — the whole point."""
        folder = str(tmp_path / "proj")
        src = AppState()
        src.new_project(name="RT", crs_epsg=32631, folder_path=folder)
        _populate(src)
        before = {hp.name: hp.uuid for hp in src.project.horizon_picks}
        src.save_project()

        dst = AppState()
        dst.open_project(folder)
        after = {hp.name: hp.uuid for hp in dst.project.horizon_picks}
        assert after == before

    def test_pickless_entity_survives_reload(self, tmp_path):
        """A horizon created but not yet drawn must reload (regression: an empty
        HorizonPick used to crash reconstruct with 'requires at least one point').
        """
        folder = str(tmp_path / "proj")
        src = AppState()
        src.new_project(name="RT", crs_epsg=32631, folder_path=folder)
        sec = Section([(0.0, 0.0), (2000.0, 0.0)], name="L1")
        src.add_section(sec)
        src.set_active_section(sec)
        undrawn = HorizonPick.empty(name="Undrawn")
        saved_uuid = undrawn.uuid
        src.add_horizon_pick(undrawn)
        src.save_project()

        dst = AppState()
        dst.open_project(folder)
        reloaded = _by_name(dst.project.horizon_picks, "Undrawn")
        assert reloaded.n_picks == 0
        assert reloaded.uuid == saved_uuid


# ---------------------------------------------------------------------------
# Construction parent refs are id-based and rename-safe (Step 2)
# ---------------------------------------------------------------------------

class TestConstructionParentRefs:

    def test_parallel_reference_uuid_backfilled_on_reload(self, tmp_path):
        """A rule stored with only reference_name resolves to the bed's UUID on
        open (migration backfill), while keeping the name as a display label."""
        dst = _roundtrip(tmp_path)
        dip  = _by_name(dst.project.horizon_picks, "DipH")
        para = _by_name(dst.project.horizon_picks, "ParaH")
        assert isinstance(para.construction_rule, ParallelToBedRule)
        assert para.construction_rule.reference_name == "DipH"     # label kept
        assert para.construction_rule.reference_uuid == dip.uuid    # id link resolved
        assert dip.uuid                                             # sanity

    def test_polygon_parallel_reference_uuid_backfilled(self, tmp_path):
        dst = _roundtrip(tmp_path)
        dip = _by_name(dst.project.horizon_picks, "DipH")
        rp  = _by_name(dst.project.polygons, "RuleP")
        assert rp.construction_rule.reference_uuid == dip.uuid

    def test_new_parallel_tool_records_reference_uuid(self):
        """New constructions store the parent's UUID at creation (not just name)."""
        from section_tool.tools.construction_tools import ParallelOffsetTool
        ref = HorizonPick(np.array([0.0, 1000.0]), np.array([100.0, 100.0]),
                          name="Bed", section_names=["L1", "L1"])
        tool = ParallelOffsetTool()
        tool.set_reference(ref, "L1")
        new = tool.handle_placement(500.0, 250.0, "L1")
        assert new.construction_rule.reference_uuid == ref.uuid
        assert new.construction_rule.reference_name == "Bed"

    def test_link_survives_rename(self):
        """The id link is immutable across a rename; only the name label goes
        stale. This is the rename-safety the id-based ref buys us."""
        bed = HorizonPick(np.array([0.0, 1.0]), np.array([0.0, 1.0]), name="X")
        para = HorizonPick(np.array([0.0, 1.0]), np.array([2.0, 3.0]), name="P")
        para.construction_rule = ParallelToBedRule(
            reference_name=bed.name, reference_uuid=bed.uuid, offset_m=10.0)
        bed.name = "Y"                                   # rename the bed
        assert para.construction_rule.reference_uuid == bed.uuid   # link intact
        assert para.construction_rule.reference_name == "X"        # label now stale


# ---------------------------------------------------------------------------
# Known limitations — aspirational tests, xfail(strict) so a fix flips them red
# ---------------------------------------------------------------------------

class TestKnownLimitations:

    @pytest.mark.xfail(
        strict=True,
        reason="Extracted seismic is not restored on project open: it lives only "
               "in the in-memory _section_seismic dict and is never persisted/"
               "reloaded (no disk re-read in _load_from_sqlite). Known limitation "
               "from the seismic cache work — user must re-extract after reopen.")
    def test_extracted_seismic_survives_roundtrip(self, tmp_path):
        folder = str(tmp_path / "proj_seis")
        src = AppState()
        src.new_project(name="S", crs_epsg=32631, folder_path=folder)
        sec = Section([(0.0, 0.0), (2000.0, 0.0)], name="L1")
        src.add_section(sec)
        src.set_active_section(sec)
        data = np.zeros((10, 20), dtype=np.float32)
        meta = {"dist_min": 0.0, "dist_max": 2000.0,
                "samples": list(range(10)), "domain": "twt"}
        src.set_seismic_for_section("L1", data, meta)
        src.save_project()

        dst = AppState()
        dst.open_project(folder)
        ex_data, ex_meta = dst.get_seismic_for_section("L1")
        assert ex_data is not None   # currently None — extracted seismic not restored
