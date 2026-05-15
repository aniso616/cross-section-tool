"""Tests for automatic polygon detection (Phase 4)."""
import numpy as np
import pytest

from section_tool.core.polygon_detection import detect_polygons
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick


def _sec() -> Section:
    return Section([(0.0, 0.0), (1000.0, 0.0)], name="S1")


def _hpick(distances, depths, sec_name="S1") -> HorizonPick:
    hp = HorizonPick(
        distances, depths,
        section_names=[sec_name] * len(distances),
    )
    return hp


class TestDetectPolygons:
    """Two horizontal lines + section boundary → three rectangular regions."""

    def test_two_horizons_make_three_polygons(self):
        sec = _sec()
        # Horizon at depth 500 and 1500
        h1 = _hpick([0.0, 1000.0], [500.0, 500.0])
        h2 = _hpick([0.0, 1000.0], [1500.0, 1500.0])
        polys = detect_polygons([h1, h2], [], [], sec, "S1",
                                min_area=1.0)
        # Three rectangular regions: above h1, between h1 and h2, below h2
        assert len(polys) == 3

    def test_no_picks_no_polygons(self):
        sec = _sec()
        polys = detect_polygons([], [], [], sec, "S1")
        assert polys == []

    def test_single_horizon_makes_two_polygons(self):
        sec = _sec()
        h = _hpick([0.0, 1000.0], [1000.0, 1000.0])
        polys = detect_polygons([h], [], [], sec, "S1", min_area=1.0)
        assert len(polys) == 2

    def test_polygon_areas_are_positive(self):
        sec = _sec()
        h = _hpick([0.0, 1000.0], [1000.0, 1000.0])
        polys = detect_polygons([h], [], [], sec, "S1", min_area=1.0)
        for p in polys:
            assert p.area > 0

    def test_min_area_filter(self):
        sec = _sec()
        h = _hpick([0.0, 1000.0], [1000.0, 1000.0])
        polys_all    = detect_polygons([h], [], [], sec, "S1", min_area=0.0)
        polys_large  = detect_polygons([h], [], [], sec, "S1", min_area=1e9)
        assert len(polys_large) < len(polys_all)

    def test_section_picks_filter(self):
        """Picks from a different section should not affect detection."""
        sec = _sec()
        h_other = _hpick([0.0, 1000.0], [500.0, 500.0], sec_name="Other")
        polys = detect_polygons([h_other], [], [], sec, "S1", min_area=1.0)
        # No picks for S1 → no closed regions (only boundary)
        assert polys == []


class TestFormation:
    def test_formation_fields(self):
        from section_tool.core.formation import Formation
        f = Formation(name="TopSand", primary_lithology="sandstone", age_top_ma=65.0)
        assert f.name == "TopSand"
        assert f.primary_lithology == "sandstone"
        assert f.age_top_ma == 65.0

    def test_formation_roundtrip(self):
        from section_tool.core.formation import Formation
        f = Formation(name="Shale", porosity_surface=0.4)
        assert Formation.from_dict(f.to_dict()).porosity_surface == pytest.approx(0.4)


class TestStratigraphicColumn:
    def test_add_and_order(self):
        from section_tool.core.formation import Formation, StratigraphicColumn
        col = StratigraphicColumn()
        col.add_formation(Formation("A"))
        col.add_formation(Formation("B"))
        assert col.formations[0].name == "A"
        assert col.formations[1].name == "B"

    def test_is_above(self):
        from section_tool.core.formation import Formation, StratigraphicColumn
        col = StratigraphicColumn()
        col.add_formation(Formation("A"))
        col.add_formation(Formation("B"))
        assert col.is_above("A", "B")
        assert not col.is_above("B", "A")

    def test_reorder(self):
        from section_tool.core.formation import Formation, StratigraphicColumn
        col = StratigraphicColumn()
        for n in ("A", "B", "C"):
            col.add_formation(Formation(n))
        col.reorder("C", 0)
        assert col.formations[0].name == "C"

    def test_roundtrip(self):
        from section_tool.core.formation import Formation, StratigraphicColumn
        col = StratigraphicColumn()
        col.add_formation(Formation("X", age_top_ma=100.0))
        col2 = StratigraphicColumn.from_list(col.to_list())
        assert col2.formations[0].name == "X"
        assert col2.formations[0].age_top_ma == 100.0


class TestSectionSnapshot:
    def test_snapshot_and_restore(self):
        import copy
        sec = _sec()
        snap = sec.snapshot()
        assert "nodes" in snap
        # Modify, then restore
        sec.name = "Modified"
        sec.load_snapshot(snap)
        assert sec.name == "S1"

    def test_snapshot_preserves_nodes(self):
        import copy
        sec = _sec()
        snap = copy.deepcopy(sec.snapshot())
        sec.name = "Changed"
        sec.load_snapshot(snap)
        np.testing.assert_array_equal(sec.nodes, _sec().nodes)


class TestFormationPhysics:
    def test_athy_law_surface(self):
        from section_tool.core.formation import Formation
        f = Formation("Shale", porosity_surface=0.63, compaction_coeff=0.00051)
        assert pytest.approx(f.porosity_at_depth(0)) == 0.63

    def test_athy_law_depth_decreases(self):
        from section_tool.core.formation import Formation
        f = Formation("Shale", porosity_surface=0.63, compaction_coeff=0.00051)
        assert f.porosity_at_depth(1000) < f.porosity_at_depth(0)

    def test_athy_law_incompressible(self):
        from section_tool.core.formation import Formation
        f = Formation("Salt", porosity_surface=0.01, compaction_coeff=0.0)
        assert pytest.approx(f.porosity_at_depth(5000)) == 0.01

    def test_bulk_density_at_surface(self):
        from section_tool.core.formation import Formation
        f = Formation("Sand", porosity_surface=0.4, compaction_coeff=0.0,
                      grain_density=2650.0)
        rho = f.bulk_density_at_depth(0.0, fluid_density=1000.0)
        assert pytest.approx(rho) == 2650.0 * 0.6 + 1000.0 * 0.4

    def test_decompaction_greater_than_current(self):
        """Decompacted thickness must be ≥ current (porosity increases at shallower depth)."""
        from section_tool.core.formation import Formation
        f = Formation("Shale", porosity_surface=0.63, compaction_coeff=0.00051)
        restored = f.decompacted_thickness(50.0, 3000.0, target_depth=0.0)
        assert restored > 50.0

    def test_decompaction_positive(self):
        from section_tool.core.formation import Formation
        f = Formation("Shale", porosity_surface=0.63, compaction_coeff=0.00051)
        assert f.decompacted_thickness(100.0, 1000.0) > 0

    def test_decompaction_incompressible_unchanged(self):
        from section_tool.core.formation import Formation
        f = Formation("Salt", porosity_surface=0.01, compaction_coeff=0.0)
        assert pytest.approx(f.decompacted_thickness(200.0, 3000.0)) == 200.0

    def test_lithology_defaults_populate(self):
        from section_tool.core.formation import Formation, LITHOLOGY_DEFAULTS
        f = Formation("SS")
        f.populate_from_lithology("sandstone")
        assert f.porosity_surface == LITHOLOGY_DEFAULTS["sandstone"]["porosity_surface"]
        assert f.grain_density   == LITHOLOGY_DEFAULTS["sandstone"]["grain_density"]

    def test_lithology_defaults_change_primary(self):
        from section_tool.core.formation import Formation
        f = Formation("X", primary_lithology="shale")
        f.populate_from_lithology("limestone")
        assert f.primary_lithology == "limestone"


class TestContactTypeStyle:
    """Verify that the right rendering path is selected per contact_type."""

    def _pick(self, ct: str):
        from section_tool.core.surfaces import HorizonPick
        hp = HorizonPick([0.0, 500.0, 1000.0], [100.0, 120.0, 150.0],
                         name="H", contact_type=ct,
                         section_names=["S"] * 3)
        return hp

    def test_conformable_has_correct_type(self):
        hp = self._pick("conformable")
        assert hp.contact_type == "conformable"

    def test_unconformity_stored(self):
        hp = self._pick("unconformity")
        assert hp.contact_type == "unconformity"

    def test_sequence_boundary_stored(self):
        hp = self._pick("sequence_boundary")
        assert hp.contact_type == "sequence_boundary"

    def test_contact_types_all_valid(self):
        from section_tool.core.surfaces import HorizonPick
        from section_tool.views.horizon_dialog import CONTACT_TYPES
        for ct in CONTACT_TYPES:
            hp = self._pick(ct)
            assert hp.contact_type == ct


class TestFaultAttributes:
    def test_fault_type_default(self):
        from section_tool.core.surfaces import HorizonPick
        hp = HorizonPick([0.0], [100.0])
        assert hp.fault_type == "normal"
        assert hp.dip_direction == "right"

    def test_fault_type_stored(self):
        from section_tool.core.surfaces import HorizonPick
        hp = HorizonPick([0.0], [100.0], fault_type="reverse",
                         dip_direction="left")
        assert hp.fault_type == "reverse"
        assert hp.dip_direction == "left"

    def test_fault_type_in_empty(self):
        from section_tool.core.surfaces import HorizonPick
        hp = HorizonPick.empty(name="F1")
        assert hp.fault_type == "normal"


class TestSectionPolygonArea:
    def test_rectangle_area(self):
        from section_tool.core.polygons import SectionPolygon
        # 100 × 200 rectangle
        poly = SectionPolygon(
            [(0, 0), (100, 0), (100, 200), (0, 200)], name="R"
        )
        assert pytest.approx(poly.area) == 20_000.0

    def test_area_positive(self):
        from section_tool.core.polygons import SectionPolygon
        poly = SectionPolygon(
            [(0, 0), (50, 0), (50, 50), (0, 50)]
        )
        assert poly.area > 0
