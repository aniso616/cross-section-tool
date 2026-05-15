"""SQLite project database — schema definition and CRUD access layer.

Every mutation in AppState calls the corresponding method here so the
database is always consistent with in-memory state.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Schema SQL
# ---------------------------------------------------------------------------

_SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS project_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS sections (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    name                  TEXT    UNIQUE NOT NULL,
    nodes_json            TEXT    NOT NULL,
    depth_domain          TEXT    DEFAULT 'md',
    depth_units           TEXT    DEFAULT 'm',
    vertical_exaggeration REAL    DEFAULT 1.0,
    crs_epsg              INTEGER DEFAULT 32632,
    created_date          TEXT,
    modified_date         TEXT
);

CREATE TABLE IF NOT EXISTS horizons (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    name              TEXT    UNIQUE NOT NULL,
    contact_type      TEXT    DEFAULT 'conformable',
    color             TEXT    DEFAULT '#2ca02c',
    line_width        REAL    DEFAULT 1.5,
    line_style        TEXT    DEFAULT 'solid',
    formation_above   TEXT    DEFAULT '',
    formation_below   TEXT    DEFAULT '',
    age_ma            REAL,
    confidence        REAL    DEFAULT 1.0,
    event_id          INTEGER
);

CREATE TABLE IF NOT EXISTS horizon_picks (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    horizon_id     INTEGER NOT NULL REFERENCES horizons(id) ON DELETE CASCADE,
    section_name   TEXT    NOT NULL,
    distance_along REAL    NOT NULL,
    depth          REAL    NOT NULL,
    confidence     REAL    DEFAULT 1.0,
    quality        TEXT    DEFAULT 'picked',
    note           TEXT,
    sort_order     INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS faults (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    name               TEXT    UNIQUE NOT NULL,
    fault_type         TEXT    DEFAULT 'normal',
    dip_direction      TEXT    DEFAULT 'right',
    color              TEXT    DEFAULT '#d62728',
    line_width         REAL    DEFAULT 1.5,
    line_style         TEXT    DEFAULT 'solid',
    displacement       REAL,
    age_activation_ma  REAL,
    age_cessation_ma   REAL,
    confidence         REAL    DEFAULT 1.0,
    event_id           INTEGER
);

CREATE TABLE IF NOT EXISTS fault_picks (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    fault_id       INTEGER NOT NULL REFERENCES faults(id) ON DELETE CASCADE,
    section_name   TEXT    NOT NULL,
    distance_along REAL    NOT NULL,
    depth          REAL    NOT NULL,
    confidence     REAL    DEFAULT 1.0,
    quality        TEXT    DEFAULT 'picked',
    note           TEXT,
    sort_order     INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS wells (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT    UNIQUE NOT NULL,
    uwi                 TEXT    DEFAULT '',
    x                   REAL    DEFAULT 0.0,
    y                   REAL    DEFAULT 0.0,
    kb_elevation        REAL    DEFAULT 0.0,
    td                  REAL    DEFAULT 5000.0,
    original_x          REAL,
    original_y          REAL,
    original_crs_epsg   INTEGER,
    las_file_path       TEXT,
    created_date        TEXT
);

CREATE TABLE IF NOT EXISTS well_tops (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    well_id        INTEGER NOT NULL REFERENCES wells(id) ON DELETE CASCADE,
    formation_name TEXT    NOT NULL,
    md             REAL    NOT NULL,
    tvd            REAL,
    confidence     REAL    DEFAULT 1.0,
    note           TEXT
);

CREATE TABLE IF NOT EXISTS well_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    well_id     INTEGER NOT NULL REFERENCES wells(id) ON DELETE CASCADE,
    mnemonic    TEXT    NOT NULL,
    unit        TEXT    DEFAULT '',
    description TEXT    DEFAULT '',
    depth_min   REAL,
    depth_max   REAL,
    data_min    REAL,
    data_max    REAL,
    data_json   TEXT
);

CREATE TABLE IF NOT EXISTS formations (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    name                 TEXT    UNIQUE NOT NULL,
    strat_order          INTEGER DEFAULT 0,
    age_top_ma           REAL,
    age_base_ma          REAL,
    color                TEXT    DEFAULT '#888888',
    opacity              REAL    DEFAULT 0.6,
    lithology_pattern    TEXT    DEFAULT 'none',
    primary_lithology    TEXT    DEFAULT 'shale',
    sand_fraction        REAL    DEFAULT 0.0,
    shale_fraction       REAL    DEFAULT 1.0,
    carbonate_fraction   REAL    DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS polygons (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    section_name    TEXT    NOT NULL,
    formation_name  TEXT    DEFAULT '',
    vertices_json   TEXT    NOT NULL,
    fill_color      TEXT    DEFAULT '#9467bd',
    fill_opacity    REAL    DEFAULT 0.6,
    outline_width   REAL    DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS reference_lines (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL,
    section_name TEXT,
    line_type    TEXT    NOT NULL DEFAULT 'horizontal',
    value        REAL,
    color        TEXT    DEFAULT '#999999',
    visible      INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS seismic (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    name              TEXT    NOT NULL,
    file_path         TEXT,
    is_external       INTEGER DEFAULT 0,
    domain            TEXT    DEFAULT 'twt',
    depth_units       TEXT    DEFAULT 'ms',
    sample_interval   REAL,
    n_traces          INTEGER DEFAULT 0,
    extent_xmin       REAL    DEFAULT 0.0,
    extent_xmax       REAL    DEFAULT 0.0,
    extent_ymin       REAL    DEFAULT 0.0,
    extent_ymax       REAL    DEFAULT 0.0,
    x_field           INTEGER DEFAULT 181,
    y_field           INTEGER DEFAULT 185,
    scalar_field      INTEGER DEFAULT 71,
    apply_scalar      INTEGER DEFAULT 1,
    crs_epsg          INTEGER DEFAULT 32632,
    colormap          TEXT    DEFAULT 'seismic_red_blue',
    gain              REAL    DEFAULT 1.0,
    clip_percentile   REAL    DEFAULT 99.0,
    opacity           REAL    DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS annotations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    section_name    TEXT,
    text            TEXT    NOT NULL,
    distance        REAL    NOT NULL,
    depth           REAL    NOT NULL,
    font_size       INTEGER DEFAULT 10,
    rotation        REAL    DEFAULT 0.0,
    color           TEXT    DEFAULT '#000000',
    anchor_distance REAL,
    anchor_depth    REAL
);

CREATE TABLE IF NOT EXISTS events (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    name                 TEXT    NOT NULL,
    event_type           TEXT,
    age_ma               REAL,
    related_objects_json TEXT
);

CREATE TABLE IF NOT EXISTS velocity_model (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now().isoformat()


def _dumps(obj: Any) -> str:
    """Serialize numpy-aware JSON."""
    if isinstance(obj, np.ndarray):
        return json.dumps(obj.tolist())
    return json.dumps(obj)


def _loads(s: str | None) -> Any:
    if s is None:
        return None
    return json.loads(s)


# ---------------------------------------------------------------------------
# ProjectDatabase
# ---------------------------------------------------------------------------

class ProjectDatabase:
    """SQLite-backed storage for a single project.

    All mutations are committed immediately (auto-commit pattern).
    Use :meth:`transaction` for batching multiple writes.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = str(db_path)
        self.conn = sqlite3.connect(self._path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Context manager for batched writes
    # ------------------------------------------------------------------

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    # ------------------------------------------------------------------
    # Project metadata
    # ------------------------------------------------------------------

    def set_meta(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO project_meta(key, value) VALUES (?, ?)",
            (key, str(value))
        )
        self.conn.commit()

    def get_meta(self, key: str, default: str = "") -> str:
        row = self.conn.execute(
            "SELECT value FROM project_meta WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def set_project_settings(self, name: str, crs_epsg: int, depth_units: str,
                              depth_domain: str, default_depth_min: float,
                              default_depth_max: float) -> None:
        data = {
            "name": name,
            "crs_epsg": str(crs_epsg),
            "depth_units": depth_units,
            "depth_domain": depth_domain,
            "default_depth_min": str(default_depth_min),
            "default_depth_max": str(default_depth_max),
            "created_date": self.get_meta("created_date") or _now(),
            "modified_date": _now(),
        }
        for k, v in data.items():
            self.conn.execute(
                "INSERT OR REPLACE INTO project_meta(key,value) VALUES(?,?)", (k, v)
            )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def upsert_section(self, section) -> int:
        nodes_json = _dumps(section.nodes.tolist())
        row = self.conn.execute(
            "SELECT id FROM sections WHERE name=?", (section.name,)
        ).fetchone()
        if row:
            self.conn.execute(
                """UPDATE sections SET nodes_json=?, depth_domain=?, depth_units=?,
                   vertical_exaggeration=?, crs_epsg=?, modified_date=?
                   WHERE name=?""",
                (nodes_json, getattr(section, "depth_domain", "md"),
                 getattr(section, "depth_units", "m"),
                 getattr(section, "vertical_exaggeration", 1.0),
                 getattr(section, "crs_epsg", 32632),
                 _now(), section.name)
            )
            sid = row["id"]
        else:
            cur = self.conn.execute(
                """INSERT INTO sections(name, nodes_json, depth_domain, depth_units,
                   vertical_exaggeration, crs_epsg, created_date, modified_date)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (section.name, nodes_json,
                 getattr(section, "depth_domain", "md"),
                 getattr(section, "depth_units", "m"),
                 getattr(section, "vertical_exaggeration", 1.0),
                 getattr(section, "crs_epsg", 32632),
                 _now(), _now())
            )
            sid = cur.lastrowid
        self.conn.commit()
        return sid

    def get_all_sections(self) -> list[dict]:
        return [dict(r) for r in
                self.conn.execute("SELECT * FROM sections ORDER BY id").fetchall()]

    def delete_section(self, name: str) -> None:
        self.conn.execute("DELETE FROM sections WHERE name=?", (name,))
        # Also cascade horizon/fault picks referencing this section
        self.conn.execute("DELETE FROM horizon_picks WHERE section_name=?", (name,))
        self.conn.execute("DELETE FROM fault_picks   WHERE section_name=?", (name,))
        self.conn.execute("DELETE FROM polygons      WHERE section_name=?", (name,))
        self.conn.execute("DELETE FROM reference_lines WHERE section_name=?", (name,))
        self.conn.execute("DELETE FROM annotations   WHERE section_name=?", (name,))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Horizons + picks
    # ------------------------------------------------------------------

    def upsert_horizon(self, pick) -> int:
        row = self.conn.execute(
            "SELECT id FROM horizons WHERE name=?", (pick.name,)
        ).fetchone()
        if row:
            hid = row["id"]
            self.conn.execute(
                """UPDATE horizons SET color=?, line_width=?, line_style=?,
                   contact_type=?, formation_above=?, formation_below=?,
                   confidence=?
                   WHERE id=?""",
                (pick.color,
                 getattr(pick, "line_width", 1.5),
                 getattr(pick, "line_style", "solid"),
                 getattr(pick, "contact_type", "conformable"),
                 getattr(pick, "formation_above", ""),
                 getattr(pick, "formation_below", ""),
                 getattr(pick, "confidence", 1.0),
                 hid)
            )
        else:
            cur = self.conn.execute(
                """INSERT INTO horizons(name, color, line_width, line_style,
                   contact_type, formation_above, formation_below, confidence)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (pick.name, pick.color,
                 getattr(pick, "line_width", 1.5),
                 getattr(pick, "line_style", "solid"),
                 getattr(pick, "contact_type", "conformable"),
                 getattr(pick, "formation_above", ""),
                 getattr(pick, "formation_below", ""),
                 getattr(pick, "confidence", 1.0))
            )
            hid = cur.lastrowid

        # Replace all picks
        self.conn.execute("DELETE FROM horizon_picks WHERE horizon_id=?", (hid,))
        for sec_name in pick.section_names():
            idxs = pick.section_indices(sec_name)
            if len(idxs) == 0:
                continue
            dists = pick._distances[idxs]
            depths = pick._depths[idxs]
            for order, (d, z) in enumerate(zip(dists, depths)):
                self.conn.execute(
                    """INSERT INTO horizon_picks
                       (horizon_id, section_name, distance_along, depth, sort_order)
                       VALUES(?,?,?,?,?)""",
                    (hid, sec_name, float(d), float(z), order)
                )
        self.conn.commit()
        return hid

    def get_all_horizons(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM horizons ORDER BY id"
        ).fetchall()
        result = []
        for h in rows:
            hd = dict(h)
            hd["picks"] = [dict(p) for p in
                           self.conn.execute(
                               """SELECT * FROM horizon_picks
                                  WHERE horizon_id=?
                                  ORDER BY section_name, sort_order""",
                               (h["id"],)).fetchall()]
            result.append(hd)
        return result

    def delete_horizon(self, name: str) -> None:
        self.conn.execute("DELETE FROM horizons WHERE name=?", (name,))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Faults + picks (same pattern as horizons)
    # ------------------------------------------------------------------

    def upsert_fault(self, pick) -> int:
        row = self.conn.execute(
            "SELECT id FROM faults WHERE name=?", (pick.name,)
        ).fetchone()
        if row:
            fid = row["id"]
            self.conn.execute(
                """UPDATE faults SET color=?, line_width=?, line_style=?,
                   fault_type=?, dip_direction=?, confidence=?
                   WHERE id=?""",
                (pick.color,
                 getattr(pick, "line_width", 1.5),
                 getattr(pick, "line_style", "solid"),
                 getattr(pick, "fault_type", "normal"),
                 getattr(pick, "dip_direction", "right"),
                 getattr(pick, "confidence", 1.0),
                 fid)
            )
        else:
            cur = self.conn.execute(
                """INSERT INTO faults(name, color, line_width, line_style,
                   fault_type, dip_direction, confidence)
                   VALUES(?,?,?,?,?,?,?)""",
                (pick.name, pick.color,
                 getattr(pick, "line_width", 1.5),
                 getattr(pick, "line_style", "solid"),
                 getattr(pick, "fault_type", "normal"),
                 getattr(pick, "dip_direction", "right"),
                 getattr(pick, "confidence", 1.0))
            )
            fid = cur.lastrowid

        self.conn.execute("DELETE FROM fault_picks WHERE fault_id=?", (fid,))
        for sec_name in pick.section_names():
            idxs = pick.section_indices(sec_name)
            if len(idxs) == 0:
                continue
            dists  = pick._distances[idxs]
            depths = pick._depths[idxs]
            for order, (d, z) in enumerate(zip(dists, depths)):
                self.conn.execute(
                    """INSERT INTO fault_picks
                       (fault_id, section_name, distance_along, depth, sort_order)
                       VALUES(?,?,?,?,?)""",
                    (fid, sec_name, float(d), float(z), order)
                )
        self.conn.commit()
        return fid

    def get_all_faults(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM faults ORDER BY id").fetchall()
        result = []
        for f in rows:
            fd = dict(f)
            fd["picks"] = [dict(p) for p in
                           self.conn.execute(
                               """SELECT * FROM fault_picks
                                  WHERE fault_id=?
                                  ORDER BY section_name, sort_order""",
                               (f["id"],)).fetchall()]
            result.append(fd)
        return result

    def delete_fault(self, name: str) -> None:
        self.conn.execute("DELETE FROM faults WHERE name=?", (name,))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Wells
    # ------------------------------------------------------------------

    def upsert_well(self, well) -> int:
        row = self.conn.execute(
            "SELECT id FROM wells WHERE name=?", (well.name,)
        ).fetchone()
        if row:
            wid = row["id"]
            self.conn.execute(
                """UPDATE wells SET uwi=?, x=?, y=?, kb_elevation=?, td=?,
                   original_x=?, original_y=?, original_crs_epsg=?
                   WHERE id=?""",
                (well.uwi, well.x, well.y, well.kb,
                 well.deviation.max_tvd,
                 getattr(well, "original_x", None),
                 getattr(well, "original_y", None),
                 getattr(well, "original_crs_epsg", None),
                 wid)
            )
        else:
            cur = self.conn.execute(
                """INSERT INTO wells(name, uwi, x, y, kb_elevation, td,
                   original_x, original_y, original_crs_epsg, created_date)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (well.name, well.uwi, well.x, well.y, well.kb,
                 well.deviation.max_tvd,
                 getattr(well, "original_x", None),
                 getattr(well, "original_y", None),
                 getattr(well, "original_crs_epsg", None),
                 _now())
            )
            wid = cur.lastrowid

        # Formation tops
        self.conn.execute("DELETE FROM well_tops WHERE well_id=?", (wid,))
        for top_name, md in well.formation_tops.items():
            self.conn.execute(
                "INSERT INTO well_tops(well_id, formation_name, md) VALUES(?,?,?)",
                (wid, top_name, float(md))
            )

        # Log metadata + data
        self.conn.execute("DELETE FROM well_logs WHERE well_id=?", (wid,))
        for log_name in well.log_names:
            try:
                log = well.get_log(log_name)
                dmin, dmax = log.depth_range()
                vmin = float(np.nanmin(log.values))
                vmax = float(np.nanmax(log.values))
                depths_json = _dumps(log._depths.tolist())
                values_json = _dumps(log._values.tolist())
                data_json = json.dumps({"depths": depths_json, "values": values_json})
                self.conn.execute(
                    """INSERT INTO well_logs
                       (well_id, mnemonic, unit, depth_min, depth_max,
                        data_min, data_max, data_json)
                       VALUES(?,?,?,?,?,?,?,?)""",
                    (wid, log.name, log.units, dmin, dmax, vmin, vmax, data_json)
                )
            except Exception:
                pass

        self.conn.commit()
        return wid

    def get_all_wells(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM wells ORDER BY id").fetchall()
        result = []
        for w in rows:
            wd = dict(w)
            wd["tops"] = [dict(t) for t in
                          self.conn.execute(
                              "SELECT * FROM well_tops WHERE well_id=?",
                              (w["id"],)).fetchall()]
            wd["logs"] = [dict(lg) for lg in
                          self.conn.execute(
                              "SELECT * FROM well_logs WHERE well_id=?",
                              (w["id"],)).fetchall()]
            result.append(wd)
        return result

    def delete_well(self, name: str) -> None:
        self.conn.execute("DELETE FROM wells WHERE name=?", (name,))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Seismic refs
    # ------------------------------------------------------------------

    def upsert_seismic(self, ref) -> int:
        row = self.conn.execute(
            "SELECT id FROM seismic WHERE name=?", (ref.name,)
        ).fetchone()
        if row:
            sid = row["id"]
            self.conn.execute(
                """UPDATE seismic SET file_path=?, domain=?, depth_units=?,
                   n_traces=?, extent_xmin=?, extent_xmax=?,
                   extent_ymin=?, extent_ymax=?,
                   x_field=?, y_field=?, scalar_field=?, apply_scalar=?, crs_epsg=?
                   WHERE id=?""",
                (ref.path, ref.domain, ref.depth_units,
                 getattr(ref, "n_traces_total", 0),
                 ref.extent_x_min, ref.extent_x_max,
                 ref.extent_y_min, ref.extent_y_max,
                 ref.x_field, ref.y_field, ref.scalar_field,
                 int(ref.apply_scalar), ref.crs_epsg, sid)
            )
        else:
            cur = self.conn.execute(
                """INSERT INTO seismic(name, file_path, domain, depth_units,
                   n_traces, extent_xmin, extent_xmax, extent_ymin, extent_ymax,
                   x_field, y_field, scalar_field, apply_scalar, crs_epsg)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (ref.name, ref.path, ref.domain, ref.depth_units,
                 getattr(ref, "n_traces_total", 0),
                 ref.extent_x_min, ref.extent_x_max,
                 ref.extent_y_min, ref.extent_y_max,
                 ref.x_field, ref.y_field, ref.scalar_field,
                 int(ref.apply_scalar), ref.crs_epsg)
            )
            sid = cur.lastrowid
        self.conn.commit()
        return sid

    def get_all_seismic(self) -> list[dict]:
        return [dict(r) for r in
                self.conn.execute("SELECT * FROM seismic ORDER BY id").fetchall()]

    def delete_seismic(self, name: str) -> None:
        self.conn.execute("DELETE FROM seismic WHERE name=?", (name,))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Polygons
    # ------------------------------------------------------------------

    def upsert_polygon(self, poly, section_name: str, poly_idx: int) -> int:
        name = getattr(poly, "name", f"Polygon {poly_idx}")
        # Use name + section_name as identifier
        row = self.conn.execute(
            "SELECT id FROM polygons WHERE name=? AND section_name=?",
            (name, section_name)
        ).fetchone()
        verts_json = _dumps(poly.vertices.tolist())
        if row:
            pid = row["id"]
            self.conn.execute(
                """UPDATE polygons SET vertices_json=?, fill_color=?,
                   fill_opacity=?, formation_name=? WHERE id=?""",
                (verts_json, getattr(poly, "fill_color", "#9467bd"),
                 getattr(poly, "fill_alpha", 0.6),
                 getattr(poly, "formation", ""),
                 pid)
            )
        else:
            cur = self.conn.execute(
                """INSERT INTO polygons
                   (name, section_name, vertices_json, fill_color, fill_opacity, formation_name)
                   VALUES(?,?,?,?,?,?)""",
                (name, section_name, verts_json,
                 getattr(poly, "fill_color", "#9467bd"),
                 getattr(poly, "fill_alpha", 0.6),
                 getattr(poly, "formation", ""))
            )
            pid = cur.lastrowid
        self.conn.commit()
        return pid

    def replace_all_polygons(self, polygons, section_name: str | None = None) -> None:
        """Replace all polygons (optionally for a specific section)."""
        if section_name:
            self.conn.execute("DELETE FROM polygons WHERE section_name=?", (section_name,))
        else:
            self.conn.execute("DELETE FROM polygons")
        for i, poly in enumerate(polygons):
            sec = getattr(poly, "section_name", section_name or "")
            self.upsert_polygon(poly, sec, i)

    def get_all_polygons(self) -> list[dict]:
        return [dict(r) for r in
                self.conn.execute("SELECT * FROM polygons ORDER BY id").fetchall()]

    def delete_polygon_by_name(self, name: str, section_name: str) -> None:
        self.conn.execute(
            "DELETE FROM polygons WHERE name=? AND section_name=?",
            (name, section_name)
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Reference lines
    # ------------------------------------------------------------------

    def replace_all_reference_lines(self, reference_lines) -> None:
        self.conn.execute("DELETE FROM reference_lines")
        for rl in reference_lines:
            self.conn.execute(
                """INSERT INTO reference_lines(name, line_type, value, color, visible)
                   VALUES(?,?,?,?,?)""",
                (getattr(rl, "name", ""),
                 getattr(rl, "kind", "horizontal"),
                 getattr(rl, "value", 0.0),
                 getattr(rl, "color", "#999999"),
                 int(getattr(rl, "visible", True)))
            )
        self.conn.commit()

    def get_all_reference_lines(self) -> list[dict]:
        return [dict(r) for r in
                self.conn.execute("SELECT * FROM reference_lines ORDER BY id").fetchall()]

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------

    def replace_all_annotations(self, annotations) -> None:
        self.conn.execute("DELETE FROM annotations")
        for ann in annotations:
            self.conn.execute(
                """INSERT INTO annotations
                   (section_name, text, distance, depth, font_size, rotation,
                    color, anchor_distance, anchor_depth)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (getattr(ann, "section_name", ""),
                 getattr(ann, "text", ""),
                 getattr(ann, "distance", 0.0),
                 getattr(ann, "depth", 0.0),
                 getattr(ann, "font_size", 10),
                 getattr(ann, "rotation", 0.0),
                 getattr(ann, "color", "#000000"),
                 getattr(ann, "anchor_distance", None),
                 getattr(ann, "anchor_depth", None))
            )
        self.conn.commit()

    def get_all_annotations(self) -> list[dict]:
        return [dict(r) for r in
                self.conn.execute("SELECT * FROM annotations ORDER BY id").fetchall()]

    # ------------------------------------------------------------------
    # Bulk-load helper (used when opening project)
    # ------------------------------------------------------------------

    def load_all(self) -> dict:
        """Return a dict with all project data, ready for AppState to consume."""
        return {
            "meta": {
                k: v for k, v in
                self.conn.execute("SELECT key, value FROM project_meta").fetchall()
            },
            "sections":        self.get_all_sections(),
            "horizons":        self.get_all_horizons(),
            "faults":          self.get_all_faults(),
            "wells":           self.get_all_wells(),
            "seismic":         self.get_all_seismic(),
            "polygons":        self.get_all_polygons(),
            "reference_lines": self.get_all_reference_lines(),
            "annotations":     self.get_all_annotations(),
        }
