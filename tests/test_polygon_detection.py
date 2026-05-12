"""Tests for automatic polygon detection (Phase 4)."""
import numpy as np
import pytest

from cross_section_tool.core.polygon_detection import detect_polygons
from cross_section_tool.core.section import Section
from cross_section_tool.core.surfaces import HorizonPick


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
        from cross_section_tool.core.formation import Formation
        f = Formation(name="TopSand", lithology="sandstone", age_ma=65.0)
        assert f.name == "TopSand"
        assert f.lithology == "sandstone"
        assert f.age_ma == 65.0

    def test_formation_roundtrip(self):
        from cross_section_tool.core.formation import Formation
        f = Formation(name="Shale", porosity_initial=0.4)
        assert Formation.from_dict(f.to_dict()).porosity_initial == 0.4


class TestStratigraphicColumn:
    def test_add_and_order(self):
        from cross_section_tool.core.formation import Formation, StratigraphicColumn
        col = StratigraphicColumn()
        col.add_formation(Formation("A"))
        col.add_formation(Formation("B"))
        assert col.formations[0].name == "A"
        assert col.formations[1].name == "B"

    def test_is_above(self):
        from cross_section_tool.core.formation import Formation, StratigraphicColumn
        col = StratigraphicColumn()
        col.add_formation(Formation("A"))
        col.add_formation(Formation("B"))
        assert col.is_above("A", "B")
        assert not col.is_above("B", "A")

    def test_reorder(self):
        from cross_section_tool.core.formation import Formation, StratigraphicColumn
        col = StratigraphicColumn()
        for n in ("A", "B", "C"):
            col.add_formation(Formation(n))
        col.reorder("C", 0)
        assert col.formations[0].name == "C"

    def test_roundtrip(self):
        from cross_section_tool.core.formation import Formation, StratigraphicColumn
        col = StratigraphicColumn()
        col.add_formation(Formation("X", age_ma=100.0))
        col2 = StratigraphicColumn.from_list(col.to_list())
        assert col2.formations[0].name == "X"
        assert col2.formations[0].age_ma == 100.0


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


class TestSectionPolygonArea:
    def test_rectangle_area(self):
        from cross_section_tool.core.polygons import SectionPolygon
        # 100 × 200 rectangle
        poly = SectionPolygon(
            [(0, 0), (100, 0), (100, 200), (0, 200)], name="R"
        )
        assert pytest.approx(poly.area) == 20_000.0

    def test_area_positive(self):
        from cross_section_tool.core.polygons import SectionPolygon
        poly = SectionPolygon(
            [(0, 0), (50, 0), (50, 50), (0, 50)]
        )
        assert poly.area > 0
