"""Tests for vector_layers table and AppState.add_vector_layer / get_vector_layers.

All tests are headless (no Qt display needed).
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from section_tool.io.database import ProjectDatabase
from section_tool.app_state import AppState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path: Path) -> ProjectDatabase:
    """Create a fresh ProjectDatabase in *tmp_path*."""
    return ProjectDatabase(tmp_path / "test.db")


def _sample_layer(filepath: str = "/fake/path/roads.shp",
                  name: str | None = None) -> dict:
    """Return a dict matching the schema AppState.add_vector_layer would create."""
    import os
    resolved_name = name or os.path.splitext(os.path.basename(filepath))[0]
    return {
        "name":      resolved_name,
        "filepath":  filepath,
        "features":  [],
        "crs":       "EPSG:32632",
        "geom_type": "LineString",
        "color":     "#FFAA00",
        "visible":   True,
    }


# ---------------------------------------------------------------------------
# 1. vector_layers table present in schema
# ---------------------------------------------------------------------------

class TestVectorLayersTableInSchema:

    def test_table_exists_after_init(self, tmp_path):
        db = _make_db(tmp_path)
        # Check via sqlite3 directly (schema introspection)
        con = sqlite3.connect(str(tmp_path / "test.db"))
        tables = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        con.close()
        assert "vector_layers" in tables

    def test_table_has_expected_columns(self, tmp_path):
        db = _make_db(tmp_path)
        con = sqlite3.connect(str(tmp_path / "test.db"))
        cols = {
            row[1]
            for row in con.execute("PRAGMA table_info(vector_layers)").fetchall()
        }
        con.close()
        expected = {"id", "name", "filepath", "crs", "geom_type", "color", "visible"}
        assert expected.issubset(cols)

    def test_table_initially_empty(self, tmp_path):
        db = _make_db(tmp_path)
        rows = db.conn.execute("SELECT COUNT(*) FROM vector_layers").fetchone()[0]
        assert rows == 0


# ---------------------------------------------------------------------------
# 2. upsert and load vector layer via ProjectDatabase
# ---------------------------------------------------------------------------

class TestVectorLayerAddAndRetrieve:

    def test_upsert_and_load(self, tmp_path):
        db = _make_db(tmp_path)
        layer = _sample_layer("/data/fault_traces.gpkg")
        db.upsert_vector_layer(layer)

        rows = db.conn.execute("SELECT * FROM vector_layers").fetchall()
        assert len(rows) == 1
        row = dict(rows[0])
        assert row["name"] == "fault_traces"
        assert row["filepath"] == "/data/fault_traces.gpkg"
        assert row["crs"] == "EPSG:32632"
        assert row["geom_type"] == "LineString"
        assert row["color"] == "#FFAA00"
        assert row["visible"] == 1

    def test_upsert_update_existing(self, tmp_path):
        """Upserting with the same filepath should update, not insert a duplicate."""
        db = _make_db(tmp_path)
        layer = _sample_layer("/data/wells.gpkg")
        db.upsert_vector_layer(layer)

        updated = dict(layer)
        updated["color"] = "#FF0000"
        updated["visible"] = False
        db.upsert_vector_layer(updated)

        rows = db.conn.execute("SELECT * FROM vector_layers").fetchall()
        assert len(rows) == 1  # still one row
        row = dict(rows[0])
        assert row["color"] == "#FF0000"
        assert row["visible"] == 0

    def test_load_vector_layers_returns_list(self, tmp_path):
        db = _make_db(tmp_path)
        layer = _sample_layer("/nonexistent/file.shp")
        db.upsert_vector_layer(layer)

        result = db.load_vector_layers()
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["name"] == "file"

    def test_load_vector_layers_visible_is_bool(self, tmp_path):
        db = _make_db(tmp_path)
        layer = _sample_layer("/data/a.shp")
        db.upsert_vector_layer(layer)

        result = db.load_vector_layers()
        assert isinstance(result[0]["visible"], bool)

    def test_load_vector_layers_features_list_on_missing_file(self, tmp_path):
        """load_vector_layers should not crash when the source file is missing.
        It should set features=[] for unreadable files (fiona unavailable or missing).
        """
        db = _make_db(tmp_path)
        layer = _sample_layer("/this/path/does/not/exist.shp")
        db.upsert_vector_layer(layer)

        result = db.load_vector_layers()
        assert isinstance(result[0].get("features"), list)


# ---------------------------------------------------------------------------
# 3. AppState add_vector_layer and get_vector_layers (in-memory, no project file)
# ---------------------------------------------------------------------------

class TestAppStateVectorLayer:

    def test_add_vector_layer_increases_internal_list(self):
        state = AppState()
        assert len(state._vector_layers) == 0
        state.add_vector_layer("/data/geology.gpkg", [], "EPSG:32632", "Polygon")
        assert len(state._vector_layers) == 1

    def test_add_vector_layer_fields(self):
        state = AppState()
        state.add_vector_layer("/data/geology.gpkg", [], "EPSG:32632", "Polygon")
        lyr = state._vector_layers[0]
        assert lyr["name"] == "geology"
        assert lyr["filepath"] == "/data/geology.gpkg"
        assert lyr["geom_type"] == "Polygon"
        assert lyr["visible"] is True
        assert "color" in lyr

    def test_get_vector_layers_returns_visible_only(self):
        state = AppState()
        state.add_vector_layer("/data/a.shp", [], "EPSG:4326", "Point")
        state.add_vector_layer("/data/b.shp", [], "EPSG:4326", "Point")
        # Hide the second layer manually
        state._vector_layers[1]["visible"] = False

        visible = state.get_vector_layers()
        assert len(visible) == 1
        assert visible[0]["name"] == "a"

    def test_get_vector_layers_all_hidden_returns_empty(self):
        state = AppState()
        state.add_vector_layer("/data/x.shp", [], "EPSG:4326", "Line")
        state._vector_layers[0]["visible"] = False
        assert state.get_vector_layers() == []

    def test_add_vector_layer_emits_project_changed(self):
        state = AppState()
        signals_received = []
        state.project_changed.connect(lambda: signals_received.append(1))
        state.add_vector_layer("/data/test.shp", [], "EPSG:32632", "Line")
        assert len(signals_received) == 1

    def test_add_multiple_vector_layers(self):
        state = AppState()
        for i in range(3):
            state.add_vector_layer(f"/data/layer_{i}.shp", [], "EPSG:4326", "Point")
        assert len(state._vector_layers) == 3


# ---------------------------------------------------------------------------
# 4. Visibility field can be toggled
# ---------------------------------------------------------------------------

class TestVectorLayerVisibilityField:

    def test_visible_defaults_to_true(self):
        state = AppState()
        state.add_vector_layer("/data/roads.shp", [], "EPSG:32632", "Line")
        assert state._vector_layers[0]["visible"] is True

    def test_toggle_visibility_updates_get_result(self):
        state = AppState()
        state.add_vector_layer("/data/roads.shp", [], "EPSG:32632", "Line")
        assert len(state.get_vector_layers()) == 1

        state._vector_layers[0]["visible"] = False
        assert len(state.get_vector_layers()) == 0

        state._vector_layers[0]["visible"] = True
        assert len(state.get_vector_layers()) == 1

    def test_db_upsert_visible_false(self, tmp_path):
        db = _make_db(tmp_path)
        layer = _sample_layer("/data/topo.shp")
        layer["visible"] = False
        db.upsert_vector_layer(layer)
        rows = db.conn.execute("SELECT visible FROM vector_layers").fetchall()
        assert rows[0][0] == 0  # stored as integer 0


# ---------------------------------------------------------------------------
# 5. Multiple vector layers preserved
# ---------------------------------------------------------------------------

class TestMultipleVectorLayersPreserved:

    def test_three_layers_in_db(self, tmp_path):
        db = _make_db(tmp_path)
        for i in range(3):
            db.upsert_vector_layer(_sample_layer(f"/data/layer_{i}.shp"))
        rows = db.conn.execute("SELECT COUNT(*) FROM vector_layers").fetchone()[0]
        assert rows == 3

    def test_three_layers_via_app_state(self):
        state = AppState()
        paths = ["/data/geology.shp", "/data/faults.gpkg", "/data/wells.geojson"]
        for p in paths:
            state.add_vector_layer(p, [], "EPSG:32632", "Mixed")
        assert len(state._vector_layers) == 3
        names = {lyr["name"] for lyr in state._vector_layers}
        assert names == {"geology", "faults", "wells"}

    def test_all_layers_visible_by_default(self):
        state = AppState()
        for i in range(3):
            state.add_vector_layer(f"/data/layer_{i}.shp", [], "EPSG:32632", "Point")
        visible = state.get_vector_layers()
        assert len(visible) == 3

    def test_partial_visibility_filters_correctly(self):
        state = AppState()
        for i in range(3):
            state.add_vector_layer(f"/data/layer_{i}.shp", [], "EPSG:32632", "Point")
        # Hide the middle one
        state._vector_layers[1]["visible"] = False
        visible = state.get_vector_layers()
        assert len(visible) == 2
        visible_names = {lyr["name"] for lyr in visible}
        assert "layer_1" not in visible_names


# ---------------------------------------------------------------------------
# 6. Vector layer serialization — save to DB + reload
# ---------------------------------------------------------------------------

class TestVectorLayerSerialization:

    def test_upsert_and_reload_name_preserved(self, tmp_path):
        db = _make_db(tmp_path)
        layer = _sample_layer("/data/faults.gpkg", name="My Faults")
        layer["name"] = "My Faults"
        db.upsert_vector_layer(layer)

        # Re-open DB and re-check
        db2 = ProjectDatabase(tmp_path / "test.db")
        rows = db2.conn.execute("SELECT name FROM vector_layers").fetchall()
        assert rows[0][0] == "My Faults"

    def test_upsert_multiple_then_reload_count(self, tmp_path):
        db = _make_db(tmp_path)
        for i in range(4):
            db.upsert_vector_layer(_sample_layer(f"/data/lyr_{i}.shp"))

        db2 = ProjectDatabase(tmp_path / "test.db")
        count = db2.conn.execute("SELECT COUNT(*) FROM vector_layers").fetchone()[0]
        assert count == 4
