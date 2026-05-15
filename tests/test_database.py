"""Tests for ProjectDatabase CRUD, ProjectManager, and round-trip data integrity."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from section_tool.io.database import ProjectDatabase
from section_tool.io.project_manager import ProjectManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    db = ProjectDatabase(str(tmp_path / "test.sqlite"))
    yield db
    db.close()


@pytest.fixture
def tmp_pm(tmp_path):
    pm = ProjectManager()
    pm.new_project(str(tmp_path / "TestProject"), "TestProject", crs_epsg=32631)
    yield pm
    pm.close()


# ---------------------------------------------------------------------------
# ProjectDatabase — project metadata
# ---------------------------------------------------------------------------

def test_project_meta_set_get(tmp_db):
    tmp_db.set_meta("name", "My Survey")
    assert tmp_db.get_meta("name") == "My Survey"


def test_project_meta_default(tmp_db):
    assert tmp_db.get_meta("nonexistent", "fallback") == "fallback"


def test_project_settings_round_trip(tmp_db):
    tmp_db.set_project_settings(
        name="Test", crs_epsg=32631,
        depth_units="m", depth_domain="md",
        default_depth_min=0.0, default_depth_max=4000.0,
    )
    assert tmp_db.get_meta("name") == "Test"
    assert tmp_db.get_meta("crs_epsg") == "32631"
    assert tmp_db.get_meta("default_depth_max") == "4000.0"


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def test_section_upsert_and_get(tmp_db):
    class FakeSection:
        name = "S1"
        nodes = np.array([[0, 0], [10000, 0]])
        depth_domain = "md"
        depth_units = "m"
        vertical_exaggeration = 1.0
        crs_epsg = 32631

    tmp_db.upsert_section(FakeSection())
    rows = tmp_db.get_all_sections()
    assert len(rows) == 1
    assert rows[0]["name"] == "S1"
    nodes = json.loads(rows[0]["nodes_json"])
    assert nodes == [[0, 0], [10000, 0]]


def test_section_update(tmp_db):
    class FakeSection:
        name = "S1"
        nodes = np.array([[0, 0], [10000, 0]])
        depth_domain = "md"
        depth_units = "m"
        vertical_exaggeration = 1.0
        crs_epsg = 32631

    tmp_db.upsert_section(FakeSection())
    FakeSection.vertical_exaggeration = 2.0
    tmp_db.upsert_section(FakeSection())
    rows = tmp_db.get_all_sections()
    assert len(rows) == 1
    assert float(rows[0]["vertical_exaggeration"]) == 2.0


def test_section_delete(tmp_db):
    class FakeSection:
        name = "S1"
        nodes = np.array([[0, 0], [10000, 0]])
        depth_domain = "md"
        depth_units = "m"
        vertical_exaggeration = 1.0
        crs_epsg = 32631

    tmp_db.upsert_section(FakeSection())
    tmp_db.delete_section("S1")
    assert tmp_db.get_all_sections() == []


# ---------------------------------------------------------------------------
# Horizons + picks
# ---------------------------------------------------------------------------

class _FakePick:
    def __init__(self, name, color, snames, dists, depths):
        self.name = name
        self.color = color
        self._section_names = np.array(snames, dtype=object)
        self._distances = np.array(dists)
        self._depths = np.array(depths)
        self.line_width = 1.5
        self.line_style = "solid"
        self.contact_type = "conformable"
        self.formation_above = ""
        self.formation_below = ""
        self.confidence = 1.0

    def section_names(self):
        return list(set(self._section_names))

    def section_indices(self, sec_name):
        return np.where(self._section_names == sec_name)[0]


def test_horizon_upsert_and_get(tmp_db):
    pick = _FakePick("H1", "#2ca02c",
                     ["S1", "S1", "S1"],
                     [100, 500, 900],
                     [1000, 1100, 1200])
    tmp_db.upsert_horizon(pick)
    rows = tmp_db.get_all_horizons()
    assert len(rows) == 1
    assert rows[0]["name"] == "H1"
    assert len(rows[0]["picks"]) == 3


def test_horizon_picks_preserve_order(tmp_db):
    pick = _FakePick("H1", "#2ca02c",
                     ["S1", "S1", "S1"],
                     [100, 500, 900],
                     [1000, 1100, 1200])
    tmp_db.upsert_horizon(pick)
    rows = tmp_db.get_all_horizons()
    dists = [p["distance_along"] for p in rows[0]["picks"]]
    assert dists == [100.0, 500.0, 900.0]


def test_horizon_delete_cascades_picks(tmp_db):
    pick = _FakePick("H1", "#2ca02c", ["S1"], [100], [1000])
    tmp_db.upsert_horizon(pick)
    tmp_db.delete_horizon("H1")
    # Table should be empty (cascade)
    rows = tmp_db.conn.execute("SELECT * FROM horizon_picks").fetchall()
    assert len(rows) == 0


def test_horizon_update_replaces_picks(tmp_db):
    pick = _FakePick("H1", "#2ca02c",
                     ["S1", "S1"],
                     [100, 500],
                     [1000, 1100])
    tmp_db.upsert_horizon(pick)
    # Update with fewer picks
    pick2 = _FakePick("H1", "#2ca02c", ["S1"], [200], [1050])
    tmp_db.upsert_horizon(pick2)
    rows = tmp_db.get_all_horizons()
    assert len(rows[0]["picks"]) == 1


# ---------------------------------------------------------------------------
# Wells
# ---------------------------------------------------------------------------

class _FakeWell:
    def __init__(self, name, x, y):
        self.name = name
        self.x = float(x)
        self.y = float(y)
        self.kb = 30.0
        self.uwi = "F02-01"
        self.original_x = None
        self.original_y = None
        self.original_crs_epsg = None
        self._formation_tops = {}
        self.log_names = []
        self._deviation_max_tvd = 3150.0

        class _Dev:
            max_tvd = 3150.0
        self.deviation = _Dev()
        self._logs = {}

    @property
    def formation_tops(self):
        return self._formation_tops

    def add_formation_top(self, name, md):
        self._formation_tops[name] = md

    def get_log(self, name):
        return self._logs[name]


def test_well_upsert_and_get(tmp_db):
    well = _FakeWell("F02-01", 606554.0, 6080126.0)
    tmp_db.upsert_well(well)
    rows = tmp_db.get_all_wells()
    assert len(rows) == 1
    assert rows[0]["name"] == "F02-01"
    assert float(rows[0]["x"]) == pytest.approx(606554.0)
    assert float(rows[0]["y"]) == pytest.approx(6080126.0)


def test_well_delete(tmp_db):
    well = _FakeWell("F02-01", 606554.0, 6080126.0)
    tmp_db.upsert_well(well)
    tmp_db.delete_well("F02-01")
    assert tmp_db.get_all_wells() == []


# ---------------------------------------------------------------------------
# ProjectManager — folder creation
# ---------------------------------------------------------------------------

def test_pm_creates_folder_structure(tmp_path):
    pm = ProjectManager()
    folder = str(tmp_path / "MyProject")
    pm.new_project(folder, "MyProject", crs_epsg=32631)
    for sub in ("seismic", "wells", "images", "exports", "cache", "autosave"):
        assert os.path.isdir(os.path.join(folder, sub))
    assert os.path.isfile(os.path.join(folder, "project.sqlite"))
    pm.close()


def test_pm_open_invalid_folder(tmp_path):
    pm = ProjectManager()
    with pytest.raises(FileNotFoundError):
        pm.open_project(str(tmp_path / "NotAProject"))
    pm.close()


def test_pm_settings_persisted(tmp_path):
    folder = str(tmp_path / "P1")
    pm = ProjectManager()
    pm.new_project(folder, "Survey X", crs_epsg=32631,
                   depth_units="m", depth_domain="twt",
                   default_depth_min=0.0, default_depth_max=3000.0)
    pm.close()

    pm2 = ProjectManager()
    pm2.open_project(folder)
    assert pm2.db.get_meta("name") == "Survey X"
    assert pm2.db.get_meta("crs_epsg") == "32631"
    assert pm2.db.get_meta("default_depth_max") == "3000.0"
    pm2.close()


# ---------------------------------------------------------------------------
# ProjectManager — autosave
# ---------------------------------------------------------------------------

def test_pm_autosave_creates_backup(tmp_pm, tmp_path):
    tmp_pm.autosave()
    bak = os.path.join(tmp_pm.project_path, "autosave", "project.sqlite.bak")
    assert os.path.isfile(bak)


def test_pm_autosave_newer_detection(tmp_pm):
    import time
    # No backup yet → not newer
    assert not tmp_pm.autosave_is_newer()
    tmp_pm.autosave()
    # Backup just written — might be same timestamp on fast machines
    # Touch backup to guarantee it's newer
    bak = os.path.join(tmp_pm.project_path, "autosave", "project.sqlite.bak")
    import time; time.sleep(0.01)
    Path(bak).touch()
    assert tmp_pm.autosave_is_newer()


# ---------------------------------------------------------------------------
# File import
# ---------------------------------------------------------------------------

def test_pm_import_file_copy(tmp_pm, tmp_path):
    src = tmp_path / "test.segy"
    src.write_bytes(b"SEGY_HEADER_DATA")
    dest = tmp_pm.import_file(str(src), "segy", copy=True)
    assert os.path.isfile(dest)
    assert os.path.dirname(dest).endswith("seismic")


def test_pm_import_file_reference(tmp_pm, tmp_path):
    src = tmp_path / "test.segy"
    src.write_bytes(b"SEGY_HEADER_DATA")
    dest = tmp_pm.import_file(str(src), "segy", copy=False)
    # Returns the original absolute path unchanged
    assert dest == str(src)


# ---------------------------------------------------------------------------
# Seismic extraction cache
# ---------------------------------------------------------------------------

def test_pm_seismic_cache_round_trip(tmp_pm):
    distances = np.array([0.0, 1000.0, 5000.0])
    data = np.random.rand(3, 100).astype(np.float32)
    samples = np.linspace(0, 2000, 100)

    tmp_pm.save_seismic_extract("S1", "F3_Demo", distances, data, samples)
    cached = tmp_pm.load_seismic_extract("S1", "F3_Demo")
    assert cached is not None
    np.testing.assert_array_almost_equal(cached["distances"], distances)
    np.testing.assert_array_almost_equal(cached["data"], data, decimal=5)
    np.testing.assert_array_almost_equal(cached["samples"], samples)


def test_pm_seismic_cache_miss(tmp_pm):
    assert tmp_pm.load_seismic_extract("NoSection", "NoSeismic") is None


def test_pm_seismic_cache_invalidation(tmp_pm):
    distances = np.array([0.0, 1000.0])
    data = np.random.rand(2, 50).astype(np.float32)
    samples = np.linspace(0, 1000, 50)
    tmp_pm.save_seismic_extract("S1", "F3", distances, data, samples)
    tmp_pm.invalidate_seismic_cache(section_name="S1")
    assert tmp_pm.load_seismic_extract("S1", "F3") is None
