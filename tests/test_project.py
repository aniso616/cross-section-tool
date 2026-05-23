"""Tests for section_tool.io.project — Project save/load round-trips."""

import numpy as np
import pytest

from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick, Surface
from section_tool.core.wells import DeviationSurvey, LogCurve, Well
from section_tool.io.project import FORMAT_VERSION, Project, SeismicRef


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_section(**kw) -> Section:
    return Section(
        [(0.0, 0.0), (1000.0, 0.0)],
        name=kw.get("name", "Line1"),
        depth_domain=kw.get("depth_domain", "twt"),
        depth_units=kw.get("depth_units", "m"),
        vertical_exaggeration=kw.get("ve", 1.0),
        crs_epsg=kw.get("crs_epsg", 32632),
    )


def _make_dogleg_section() -> Section:
    return Section([(0.0, 0.0), (500.0, 0.0), (500.0, 500.0)], name="Dogleg")


def _make_scattered_surface(**kw) -> Surface:
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    z = x + y
    return Surface.from_xyz(x, y, z, name=kw.get("name", "Surf"), kind="horizon", z_units="m")


def _make_grid_surface(**kw) -> Surface:
    xc = np.linspace(0, 100, 5)
    yc = np.linspace(0, 100, 5)
    xx, yy = np.meshgrid(xc, yc)
    return Surface.from_grid(xc, yc, xx + yy, name=kw.get("name", "Grid"), z_units="m")


def _make_horizon_pick(**kw) -> HorizonPick:
    return HorizonPick(
        [0.0, 500.0, 1000.0], [100.0, 200.0, 150.0],
        name=kw.get("name", "TopSand"),
        z_units="m",
        color="#ff0000",
    )


def _make_well(*, deviated: bool = False, with_logs: bool = True,
               with_tops: bool = True, **kw) -> Well:
    name = kw.get("name", "W-1")
    x, y = kw.get("x", 500.0), kw.get("y", 200.0)
    if deviated:
        dev = DeviationSurvey(
            [0.0, 100.0], [45.0, 45.0], [90.0, 90.0], x, y
        )
    else:
        dev = DeviationSurvey.vertical(x, y, td=2000.0)
    well = Well(name=name, x=x, y=y, kb=55.0, uwi="100/01-02-003-04W5/0", deviation=dev)
    if with_logs:
        depths = np.linspace(100.0, 2000.0, 50)
        well.add_log(LogCurve("GR", "GAPI", depths, depths * 0.05))
        well.add_log(LogCurve("RHOB", "G/CC", depths, np.ones(50) * 2.35))
    if with_tops:
        well.add_formation_top("Top Cretaceous", 800.0)
        well.add_formation_top("Top Triassic", 1500.0)
    return well


def _make_seismic_ref(**kw) -> SeismicRef:
    return SeismicRef(
        path=kw.get("path", "/data/line1.segy"),
        name=kw.get("name", "Line1"),
        x_field=181,
        y_field=185,
        scalar_field=71,
        apply_scalar=True,
        domain="twt",
        depth_units="ms",
        crs_epsg=32632,
    )


def _roundtrip(project: Project, tmp_path) -> Project:
    path = tmp_path / "project.h5"
    project.save(path)
    return Project.load(path)


# ---------------------------------------------------------------------------
# Project construction
# ---------------------------------------------------------------------------

class TestProjectConstruction:
    def test_default_name_empty(self):
        assert Project().name == ""

    def test_name_stored(self):
        assert Project(name="TestProj").name == "TestProj"

    def test_crs_epsg_stored(self):
        assert Project(crs_epsg=27700).crs_epsg == 27700

    def test_lists_start_empty(self):
        p = Project()
        assert p.sections == []
        assert p.surfaces == []
        assert p.horizon_picks == []
        assert p.wells == []
        assert p.seismic_refs == []

    def test_repr(self):
        p = Project(name="Demo")
        assert "Demo" in repr(p)


# ---------------------------------------------------------------------------
# Empty project round-trip
# ---------------------------------------------------------------------------

class TestEmptyProjectRoundtrip:
    def test_name_survives(self, tmp_path):
        p = _roundtrip(Project(name="EmptyProj"), tmp_path)
        assert p.name == "EmptyProj"

    def test_crs_epsg_survives(self, tmp_path):
        p = _roundtrip(Project(crs_epsg=27700), tmp_path)
        assert p.crs_epsg == 27700

    def test_all_lists_empty(self, tmp_path):
        p = _roundtrip(Project(), tmp_path)
        assert len(p.sections) == 0
        assert len(p.surfaces) == 0
        assert len(p.horizon_picks) == 0
        assert len(p.wells) == 0
        assert len(p.seismic_refs) == 0

    def test_format_version_in_file(self, tmp_path):
        import h5py
        path = tmp_path / "ver.h5"
        Project().save(path)
        with h5py.File(str(path), "r") as f:
            assert f.attrs["format_version"] == FORMAT_VERSION

    def test_created_at_in_file(self, tmp_path):
        import h5py
        path = tmp_path / "ts.h5"
        Project().save(path)
        with h5py.File(str(path), "r") as f:
            assert "created_at" in f.attrs

    def test_overwrite_existing(self, tmp_path):
        path = tmp_path / "over.h5"
        Project(name="First").save(path)
        Project(name="Second").save(path)
        p = Project.load(path)
        assert p.name == "Second"


# ---------------------------------------------------------------------------
# Section round-trips
# ---------------------------------------------------------------------------

class TestSectionRoundtrip:
    def test_nodes_preserved(self, tmp_path):
        sec = _make_section()
        p = Project()
        p.sections.append(sec)
        loaded = _roundtrip(p, tmp_path)
        np.testing.assert_allclose(loaded.sections[0].nodes, sec.nodes)

    def test_name_preserved(self, tmp_path):
        p = Project()
        p.sections.append(_make_section(name="Seismic Line A"))
        assert _roundtrip(p, tmp_path).sections[0].name == "Seismic Line A"

    def test_depth_domain_preserved(self, tmp_path):
        p = Project()
        p.sections.append(_make_section(depth_domain="twt"))
        assert _roundtrip(p, tmp_path).sections[0].depth_domain == "twt"

    def test_depth_units_preserved(self, tmp_path):
        p = Project()
        p.sections.append(_make_section(depth_units="ft"))
        assert _roundtrip(p, tmp_path).sections[0].depth_units == "ft"

    def test_vertical_exaggeration_preserved(self, tmp_path):
        p = Project()
        p.sections.append(_make_section(ve=2.5))
        assert pytest.approx(_roundtrip(p, tmp_path).sections[0].vertical_exaggeration) == 2.5

    def test_crs_epsg_preserved(self, tmp_path):
        p = Project()
        p.sections.append(_make_section(crs_epsg=27700))
        assert _roundtrip(p, tmp_path).sections[0].crs_epsg == 27700

    def test_dogleg_nodes(self, tmp_path):
        p = Project()
        p.sections.append(_make_dogleg_section())
        loaded = _roundtrip(p, tmp_path)
        assert loaded.sections[0].n_nodes == 3
        np.testing.assert_allclose(loaded.sections[0].nodes[1], [500.0, 0.0])

    def test_multiple_sections_order(self, tmp_path):
        p = Project()
        p.sections.append(_make_section(name="First"))
        p.sections.append(_make_section(name="Second"))
        p.sections.append(_make_section(name="Third"))
        loaded = _roundtrip(p, tmp_path)
        assert len(loaded.sections) == 3
        assert [s.name for s in loaded.sections] == ["First", "Second", "Third"]


# ---------------------------------------------------------------------------
# Surface round-trips
# ---------------------------------------------------------------------------

class TestSurfaceRoundtrip:
    def test_scattered_xyz_preserved(self, tmp_path):
        surf = _make_scattered_surface()
        p = Project()
        p.surfaces.append(surf)
        loaded_surf = _roundtrip(p, tmp_path).surfaces[0]
        np.testing.assert_allclose(loaded_surf._x, surf._x)
        np.testing.assert_allclose(loaded_surf._y, surf._y)
        np.testing.assert_allclose(loaded_surf._z, surf._z)

    def test_scattered_is_not_grid(self, tmp_path):
        p = Project()
        p.surfaces.append(_make_scattered_surface())
        assert not _roundtrip(p, tmp_path).surfaces[0]._is_grid

    def test_grid_flag_preserved(self, tmp_path):
        p = Project()
        p.surfaces.append(_make_grid_surface())
        assert _roundtrip(p, tmp_path).surfaces[0]._is_grid

    def test_grid_data_preserved(self, tmp_path):
        surf = _make_grid_surface()
        p = Project()
        p.surfaces.append(surf)
        loaded = _roundtrip(p, tmp_path).surfaces[0]
        np.testing.assert_allclose(loaded._grid_x, surf._grid_x)
        np.testing.assert_allclose(loaded._grid_y, surf._grid_y)
        np.testing.assert_allclose(loaded._grid_z, surf._grid_z)

    def test_surface_name_preserved(self, tmp_path):
        p = Project()
        p.surfaces.append(_make_scattered_surface(name="Top Jurassic"))
        assert _roundtrip(p, tmp_path).surfaces[0].name == "Top Jurassic"

    def test_surface_kind_preserved(self, tmp_path):
        p = Project()
        surf = Surface.from_xyz([0, 1, 2], [0, 1, 2], [0, 1, 2], kind="fault")
        p.surfaces.append(surf)
        assert _roundtrip(p, tmp_path).surfaces[0].kind == "fault"

    def test_surface_z_units_preserved(self, tmp_path):
        p = Project()
        surf = Surface.from_xyz([0, 1, 2], [0, 1, 2], [0, 1, 2], z_units="ft")
        p.surfaces.append(surf)
        assert _roundtrip(p, tmp_path).surfaces[0].z_units == "ft"

    def test_grid_surface_samples_correctly(self, tmp_path):
        surf = _make_grid_surface()
        p = Project()
        p.surfaces.append(surf)
        loaded = _roundtrip(p, tmp_path).surfaces[0]
        # Tilted plane z=x+y; at (50, 50) should give 100
        assert pytest.approx(loaded.sample(50.0, 50.0), rel=1e-4) == 100.0

    def test_multiple_surfaces_order(self, tmp_path):
        p = Project()
        for name in ["A", "B", "C"]:
            p.surfaces.append(_make_scattered_surface(name=name))
        loaded = _roundtrip(p, tmp_path)
        assert [s.name for s in loaded.surfaces] == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# HorizonPick round-trips
# ---------------------------------------------------------------------------

class TestHorizonPickRoundtrip:
    def test_distances_preserved(self, tmp_path):
        hp = _make_horizon_pick()
        p = Project()
        p.horizon_picks.append(hp)
        loaded = _roundtrip(p, tmp_path).horizon_picks[0]
        np.testing.assert_allclose(loaded.distances, hp.distances)

    def test_depths_preserved(self, tmp_path):
        hp = _make_horizon_pick()
        p = Project()
        p.horizon_picks.append(hp)
        loaded = _roundtrip(p, tmp_path).horizon_picks[0]
        np.testing.assert_allclose(loaded.depths, hp.depths)

    def test_name_preserved(self, tmp_path):
        p = Project()
        p.horizon_picks.append(_make_horizon_pick(name="Base Chalk"))
        assert _roundtrip(p, tmp_path).horizon_picks[0].name == "Base Chalk"

    def test_color_preserved(self, tmp_path):
        hp = HorizonPick([0.0, 1.0], [10.0, 20.0], color="#aabbcc")
        p = Project()
        p.horizon_picks.append(hp)
        assert _roundtrip(p, tmp_path).horizon_picks[0].color == "#aabbcc"

    def test_z_units_preserved(self, tmp_path):
        hp = HorizonPick([0.0, 1.0], [10.0, 20.0], z_units="ft")
        p = Project()
        p.horizon_picks.append(hp)
        assert _roundtrip(p, tmp_path).horizon_picks[0].z_units == "ft"

    def test_sample_still_works(self, tmp_path):
        hp = _make_horizon_pick()
        p = Project()
        p.horizon_picks.append(hp)
        loaded = _roundtrip(p, tmp_path).horizon_picks[0]
        assert pytest.approx(loaded.sample(500.0)) == 200.0

    def test_multiple_picks_order(self, tmp_path):
        p = Project()
        for name in ["X", "Y", "Z"]:
            p.horizon_picks.append(_make_horizon_pick(name=name))
        loaded = _roundtrip(p, tmp_path)
        assert [hp.name for hp in loaded.horizon_picks] == ["X", "Y", "Z"]


# ---------------------------------------------------------------------------
# Well round-trips
# ---------------------------------------------------------------------------

class TestWellRoundtrip:
    def test_name_preserved(self, tmp_path):
        p = Project()
        p.wells.append(_make_well(name="W-99"))
        assert _roundtrip(p, tmp_path).wells[0].name == "W-99"

    def test_xy_preserved(self, tmp_path):
        p = Project()
        p.wells.append(_make_well(x=123456.0, y=654321.0))
        w = _roundtrip(p, tmp_path).wells[0]
        assert pytest.approx(w.x) == 123456.0
        assert pytest.approx(w.y) == 654321.0

    def test_kb_preserved(self, tmp_path):
        p = Project()
        p.wells.append(_make_well())
        assert pytest.approx(_roundtrip(p, tmp_path).wells[0].kb) == 55.0

    def test_uwi_preserved(self, tmp_path):
        p = Project()
        p.wells.append(_make_well())
        assert _roundtrip(p, tmp_path).wells[0].uwi == "100/01-02-003-04W5/0"

    def test_deviation_md_preserved(self, tmp_path):
        p = Project()
        p.wells.append(_make_well())
        loaded = _roundtrip(p, tmp_path).wells[0]
        np.testing.assert_allclose(loaded.deviation._md, [0.0, 2000.0])

    def test_deviation_vertical_geometry(self, tmp_path):
        p = Project()
        p.wells.append(_make_well())
        loaded = _roundtrip(p, tmp_path).wells[0]
        # Vertical well: x constant, tvd == md
        np.testing.assert_allclose(loaded.deviation._x, 500.0)
        assert pytest.approx(loaded.deviation.max_tvd) == 2000.0

    def test_deviation_deviated_inc_azi_preserved(self, tmp_path):
        p = Project()
        p.wells.append(_make_well(deviated=True))
        loaded = _roundtrip(p, tmp_path).wells[0]
        np.testing.assert_allclose(loaded.deviation._inc_deg, [45.0, 45.0])
        np.testing.assert_allclose(loaded.deviation._azi_deg, [90.0, 90.0])

    def test_deviation_deviated_geometry_preserved(self, tmp_path):
        p = Project()
        p.wells.append(_make_well(deviated=True))
        loaded = _roundtrip(p, tmp_path).wells[0]
        import math
        expected = 100.0 * math.sqrt(2) / 2
        assert pytest.approx(loaded.deviation.max_tvd, rel=1e-5) == expected

    def test_log_names_preserved(self, tmp_path):
        p = Project()
        p.wells.append(_make_well())
        loaded = _roundtrip(p, tmp_path).wells[0]
        assert set(loaded.log_names) == {"GR", "RHOB"}

    def test_log_values_preserved(self, tmp_path):
        well = _make_well()
        p = Project()
        p.wells.append(well)
        loaded = _roundtrip(p, tmp_path).wells[0]
        np.testing.assert_allclose(
            loaded.get_log("GR").values, well.get_log("GR").values
        )

    def test_log_units_preserved(self, tmp_path):
        p = Project()
        p.wells.append(_make_well())
        assert _roundtrip(p, tmp_path).wells[0].get_log("GR").units == "GAPI"

    def test_formation_tops_preserved(self, tmp_path):
        p = Project()
        p.wells.append(_make_well())
        tops = _roundtrip(p, tmp_path).wells[0].formation_tops
        assert pytest.approx(tops["Top Cretaceous"]) == 800.0
        assert pytest.approx(tops["Top Triassic"]) == 1500.0

    def test_no_logs_no_tops(self, tmp_path):
        p = Project()
        p.wells.append(_make_well(with_logs=False, with_tops=False))
        loaded = _roundtrip(p, tmp_path).wells[0]
        assert loaded.log_names == []
        assert loaded.formation_tops == {}

    def test_multiple_wells_order(self, tmp_path):
        p = Project()
        for name in ["Alpha", "Beta", "Gamma"]:
            p.wells.append(_make_well(name=name, with_logs=False, with_tops=False))
        loaded = _roundtrip(p, tmp_path)
        assert [w.name for w in loaded.wells] == ["Alpha", "Beta", "Gamma"]

    def test_log_curve_sample_works_after_load(self, tmp_path):
        p = Project()
        p.wells.append(_make_well())
        loaded = _roundtrip(p, tmp_path).wells[0]
        gr = loaded.get_log("GR")
        # GR = depth * 0.05; at depth 100 -> 5.0
        assert pytest.approx(gr.sample(100.0), rel=1e-5) == 5.0


# ---------------------------------------------------------------------------
# SeismicRef round-trips
# ---------------------------------------------------------------------------

class TestSeismicRefRoundtrip:
    def test_path_preserved(self, tmp_path):
        p = Project()
        p.seismic_refs.append(_make_seismic_ref(path="/data/survey/line1.segy"))
        loaded = _roundtrip(p, tmp_path).seismic_refs[0]
        assert loaded.path == "/data/survey/line1.segy"

    def test_name_preserved(self, tmp_path):
        p = Project()
        p.seismic_refs.append(_make_seismic_ref(name="Dip Line"))
        assert _roundtrip(p, tmp_path).seismic_refs[0].name == "Dip Line"

    def test_fields_preserved(self, tmp_path):
        ref = SeismicRef(path="/x.segy", x_field=9, y_field=21, scalar_field=69)
        p = Project()
        p.seismic_refs.append(ref)
        loaded = _roundtrip(p, tmp_path).seismic_refs[0]
        assert loaded.x_field == 9
        assert loaded.y_field == 21
        assert loaded.scalar_field == 69

    def test_apply_scalar_false_preserved(self, tmp_path):
        ref = SeismicRef(path="/x.segy", apply_scalar=False)
        p = Project()
        p.seismic_refs.append(ref)
        assert not _roundtrip(p, tmp_path).seismic_refs[0].apply_scalar

    def test_domain_depth_preserved(self, tmp_path):
        ref = SeismicRef(path="/x.segy", domain="depth", depth_units="m")
        p = Project()
        p.seismic_refs.append(ref)
        loaded = _roundtrip(p, tmp_path).seismic_refs[0]
        assert loaded.domain == "depth"
        assert loaded.depth_units == "m"

    def test_crs_epsg_preserved(self, tmp_path):
        ref = SeismicRef(path="/x.segy", crs_epsg=27700)
        p = Project()
        p.seismic_refs.append(ref)
        assert _roundtrip(p, tmp_path).seismic_refs[0].crs_epsg == 27700

    def test_multiple_refs_order(self, tmp_path):
        p = Project()
        for name in ["L1", "L2", "L3"]:
            p.seismic_refs.append(_make_seismic_ref(name=name))
        loaded = _roundtrip(p, tmp_path)
        assert [r.name for r in loaded.seismic_refs] == ["L1", "L2", "L3"]


# ---------------------------------------------------------------------------
# Full project round-trip
# ---------------------------------------------------------------------------

class TestFullProjectRoundtrip:
    def test_all_types_survive(self, tmp_path):
        p = Project(name="Full Test", crs_epsg=32633)
        p.sections.append(_make_section())
        p.sections.append(_make_dogleg_section())
        p.surfaces.append(_make_scattered_surface())
        p.surfaces.append(_make_grid_surface())
        p.horizon_picks.append(_make_horizon_pick())
        p.wells.append(_make_well())
        p.wells.append(_make_well(name="W-2", deviated=True))
        p.seismic_refs.append(_make_seismic_ref())

        loaded = _roundtrip(p, tmp_path)
        assert loaded.name == "Full Test"
        assert loaded.crs_epsg == 32633
        assert len(loaded.sections) == 2
        assert len(loaded.surfaces) == 2
        assert len(loaded.horizon_picks) == 1
        assert len(loaded.wells) == 2
        assert len(loaded.seismic_refs) == 1

    def test_project_is_independent_after_load(self, tmp_path):
        """Modifying the loaded project must not affect the saved file."""
        p = Project(name="Orig")
        p.sections.append(_make_section(name="Original"))
        path = tmp_path / "proj.h5"
        p.save(path)
        loaded = Project.load(path)
        loaded.name = "Modified"
        loaded.sections[0].name = "Changed"
        # Re-load from same file — original values intact
        reloaded = Project.load(path)
        assert reloaded.name == "Orig"
        assert reloaded.sections[0].name == "Original"

    def test_pathlib_path_accepted(self, tmp_path):
        import pathlib
        path = pathlib.Path(tmp_path) / "proj.h5"
        p = Project(name="Pathlib")
        p.save(path)
        loaded = Project.load(path)
        assert loaded.name == "Pathlib"
