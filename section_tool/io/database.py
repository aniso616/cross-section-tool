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
    depth_domain          TEXT    DEFAULT 'depth',
    display_domain        TEXT    DEFAULT 'depth',  -- user-visible axis domain (twt or depth)
    depth_units           TEXT    DEFAULT 'm',
    vertical_exaggeration REAL    DEFAULT 1.0,
    crs_epsg              INTEGER DEFAULT 32632,
    created_date          TEXT,
    modified_date         TEXT
);

CREATE TABLE IF NOT EXISTS horizons (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    name                   TEXT    UNIQUE NOT NULL,
    contact_type           TEXT    DEFAULT 'conformable',
    color                  TEXT    DEFAULT '#2ca02c',
    line_width             REAL    DEFAULT 1.5,
    line_style             TEXT    DEFAULT 'solid',
    formation_above        TEXT    DEFAULT '',
    formation_below        TEXT    DEFAULT '',
    age_ma                 REAL,
    confidence             REAL    DEFAULT 1.0,
    event_id               INTEGER,
    construction_rule_json TEXT
);

CREATE TABLE IF NOT EXISTS horizon_picks (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    horizon_id     INTEGER NOT NULL REFERENCES horizons(id) ON DELETE CASCADE,
    section_name   TEXT    NOT NULL,
    distance_along REAL    NOT NULL,
    depth          REAL    NOT NULL,
    -- elevation is positive-up (source of truth when populated);
    -- depth = -elevation for display. Populated in a future migration.
    elevation      REAL,
    -- map-space source of truth: reproject onto section when geometry changes
    x              REAL,
    y              REAL,
    confidence     REAL    DEFAULT 1.0,
    quality        TEXT    DEFAULT 'picked',
    note           TEXT,
    sort_order     INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS faults (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    name                   TEXT    UNIQUE NOT NULL,
    fault_type             TEXT    DEFAULT 'normal',
    dip_direction          TEXT    DEFAULT 'right',
    color                  TEXT    DEFAULT '#d62728',
    line_width             REAL    DEFAULT 1.5,
    line_style             TEXT    DEFAULT 'solid',
    displacement           REAL,
    age_activation_ma      REAL,
    age_cessation_ma       REAL,
    confidence             REAL    DEFAULT 1.0,
    event_id               INTEGER,
    construction_rule_json TEXT
);

CREATE TABLE IF NOT EXISTS fault_picks (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    fault_id       INTEGER NOT NULL REFERENCES faults(id) ON DELETE CASCADE,
    section_name   TEXT    NOT NULL,
    distance_along REAL    NOT NULL,
    depth          REAL    NOT NULL,
    elevation      REAL,
    -- map-space source of truth: reproject onto section when geometry changes
    x              REAL,
    y              REAL,
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
    status              TEXT    DEFAULT 'actual',       -- actual, planned, hypothetical
    purpose             TEXT    DEFAULT 'exploration',  -- exploration, production, injection, geothermal, observation, model_only
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

CREATE TABLE IF NOT EXISTS lithologies (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    name                        TEXT    UNIQUE NOT NULL,
    porosity_surface            REAL,   -- phi0, dimensionless (Sclater & Christie 1980)
    compaction_coeff            REAL,   -- c, 1/m
    grain_density               REAL,   -- kg/m³
    matrix_thermal_conductivity REAL,   -- W/(m·K)
    matrix_velocity             REAL,   -- m/s (interval velocity at zero porosity)
    specific_heat_capacity      REAL,   -- J/(kg·K)
    radiogenic_heat_production  REAL    -- µW/m³
);

CREATE TABLE IF NOT EXISTS formations (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    name                  TEXT    UNIQUE NOT NULL,
    uuid                  TEXT    UNIQUE,               -- UUID4 for cross-referencing
    rank                  TEXT    DEFAULT 'formation',  -- supergroup, group, formation, member, bed
    parent_id             INTEGER REFERENCES formations(id),
    primary_lithology_id  INTEGER REFERENCES lithologies(id),
    strat_order           INTEGER DEFAULT 0,
    age_top_ma            REAL,
    age_base_ma           REAL,
    color                 TEXT    DEFAULT '#888888',
    opacity               REAL    DEFAULT 0.6,
    lithology_pattern     TEXT    DEFAULT 'none',
    -- Petrophysical overrides (NULL → inherit from referenced lithology)
    porosity_surface      REAL,
    compaction_coeff      REAL,
    grain_density         REAL,
    matrix_thermal_conductivity REAL,
    sand_fraction         REAL    DEFAULT 0.0,
    shale_fraction        REAL    DEFAULT 1.0,
    carbonate_fraction    REAL    DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS polygons (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    name                   TEXT    NOT NULL,
    section_name           TEXT    NOT NULL,
    formation_name         TEXT    DEFAULT '',
    vertices_json          TEXT    NOT NULL,
    fill_color             TEXT    DEFAULT '#9467bd',
    fill_opacity           REAL    DEFAULT 0.6,
    outline_width          REAL    DEFAULT 1.0,
    construction_rule_json TEXT,
    bounds_json            TEXT
);

CREATE TABLE IF NOT EXISTS reference_lines (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL,
    section_name TEXT,
    line_type    TEXT    NOT NULL DEFAULT 'horizontal',
    value        REAL,
    color        TEXT    DEFAULT '#999999',
    visible      INTEGER DEFAULT 1,
    -- map-space source of truth for vertical lines (recompute value on section change)
    map_x        REAL,
    map_y        REAL
);

CREATE TABLE IF NOT EXISTS aoi (
    id           INTEGER PRIMARY KEY,
    name         TEXT    NOT NULL DEFAULT 'AOI',
    polygon_wkt  TEXT    NOT NULL,
    crs_epsg     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS surfaces (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT    UNIQUE NOT NULL,
    kind           TEXT    DEFAULT 'horizon',
    z_domain       TEXT    DEFAULT 'depth_m',
    z_units        TEXT    DEFAULT 'm',
    crs_epsg       INTEGER NOT NULL DEFAULT 0,
    color_r        INTEGER DEFAULT 255,
    color_g        INTEGER DEFAULT 165,
    color_b        INTEGER DEFAULT 0,
    line_width     REAL    DEFAULT 1.5,
    visible        INTEGER DEFAULT 1,
    interpolation  TEXT    DEFAULT 'linear',
    source_file    TEXT,
    source_format  TEXT,
    point_count    INTEGER DEFAULT 0,
    x_min          REAL,  x_max REAL,
    y_min          REAL,  y_max REAL,
    z_min          REAL,  z_max REAL,
    points_file    TEXT,       -- relative path to .npy file in surfaces/ subdir
    created_date   TEXT
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

CREATE TABLE IF NOT EXISTS measurements (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    well_id     INTEGER REFERENCES wells(id) ON DELETE CASCADE,
    kind        TEXT    NOT NULL,
    -- kind is one of: vitrinite_ro, aft_age, aft_length, ahe_age, zhe_age,
    --                 bht, dst_temp, fluid_inclusion, clumped_isotope, cai
    depth_md    REAL    NOT NULL,
    depth_tvd   REAL,
    value       REAL    NOT NULL,
    uncertainty REAL,
    units       TEXT,
    sample_id   TEXT,
    lab         TEXT,
    method      TEXT,
    note        TEXT
);

CREATE TABLE IF NOT EXISTS well_sections (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    well_id              INTEGER REFERENCES wells(id)    ON DELETE CASCADE,
    section_id           INTEGER REFERENCES sections(id) ON DELETE CASCADE,
    distance_along       REAL,
    perpendicular_offset REAL,
    nearest_segment      INTEGER,
    display_mode         TEXT    DEFAULT 'auto',   -- on_plane, near, far, hidden
    projection_tolerance REAL    DEFAULT 2000
);

CREATE TABLE IF NOT EXISTS velocity_model (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS section_sets (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    name              TEXT    UNIQUE NOT NULL,
    description       TEXT    DEFAULT '',
    sort_order_field  TEXT    DEFAULT 'distance',
    created_date      TEXT
);

CREATE TABLE IF NOT EXISTS section_set_members (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    set_id       INTEGER NOT NULL REFERENCES section_sets(id)  ON DELETE CASCADE,
    section_id   INTEGER NOT NULL REFERENCES sections(id)       ON DELETE CASCADE,
    sort_index   INTEGER NOT NULL,
    UNIQUE(set_id, section_id)
);

CREATE TABLE IF NOT EXISTS vector_layers (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT    NOT NULL,
    filepath  TEXT    NOT NULL,
    crs       TEXT,
    geom_type TEXT,
    color     TEXT    DEFAULT '#FFAA00',
    visible   INTEGER DEFAULT 1
);
"""

# ---------------------------------------------------------------------------
# Lithology defaults — Sclater & Christie 1980 + common additions
# ---------------------------------------------------------------------------

_LITHOLOGY_DEFAULTS: list[dict] = [
    {"name": "Sandstone",      "porosity_surface": 0.49, "compaction_coeff": 0.00027, "grain_density": 2650, "matrix_thermal_conductivity": 3.0},
    {"name": "Shale",          "porosity_surface": 0.63, "compaction_coeff": 0.00051, "grain_density": 2720, "matrix_thermal_conductivity": 1.5},
    {"name": "Limestone",      "porosity_surface": 0.40, "compaction_coeff": 0.00040, "grain_density": 2710, "matrix_thermal_conductivity": 2.5},
    {"name": "Dolomite",       "porosity_surface": 0.35, "compaction_coeff": 0.00038, "grain_density": 2850, "matrix_thermal_conductivity": 3.5},
    {"name": "Chalk",          "porosity_surface": 0.70, "compaction_coeff": 0.00071, "grain_density": 2710, "matrix_thermal_conductivity": 2.0},
    {"name": "Salt/Halite",    "porosity_surface": 0.01, "compaction_coeff": 0.0,     "grain_density": 2170, "matrix_thermal_conductivity": 6.0},
    {"name": "Anhydrite",      "porosity_surface": 0.01, "compaction_coeff": 0.0,     "grain_density": 2960, "matrix_thermal_conductivity": 5.5},
    {"name": "Coal",           "porosity_surface": 0.30, "compaction_coeff": 0.00030, "grain_density": 1500, "matrix_thermal_conductivity": 0.3},
    {"name": "Basement/Granite","porosity_surface": 0.02, "compaction_coeff": 0.0,    "grain_density": 2750, "matrix_thermal_conductivity": 2.8},
    {"name": "Basalt",         "porosity_surface": 0.05, "compaction_coeff": 0.0,     "grain_density": 2950, "matrix_thermal_conductivity": 1.7},
    {"name": "Siltstone",      "porosity_surface": 0.56, "compaction_coeff": 0.00039, "grain_density": 2680, "matrix_thermal_conductivity": 2.0},
    {"name": "Marl",           "porosity_surface": 0.50, "compaction_coeff": 0.00045, "grain_density": 2700, "matrix_thermal_conductivity": 2.0},
    {"name": "Conglomerate",   "porosity_surface": 0.30, "compaction_coeff": 0.00020, "grain_density": 2650, "matrix_thermal_conductivity": 3.0},
    {"name": "Volcanic tuff",  "porosity_surface": 0.40, "compaction_coeff": 0.00035, "grain_density": 2400, "matrix_thermal_conductivity": 1.5},
    {"name": "Gypsum",         "porosity_surface": 0.01, "compaction_coeff": 0.0,     "grain_density": 2350, "matrix_thermal_conductivity": 1.5},
]

# Hardcoded property defaults (level 3 of the inheritance chain)
_PROPERTY_DEFAULTS: dict[str, float] = {
    "porosity_surface":            0.30,
    "compaction_coeff":            0.00027,
    "grain_density":               2650.0,
    "matrix_thermal_conductivity": 2.0,
    "matrix_velocity":             3500.0,
    "specific_heat_capacity":      800.0,
    "radiogenic_heat_production":  1.0,
}

# ---------------------------------------------------------------------------
# Depth / elevation conversion helpers
# ---------------------------------------------------------------------------

def depth_to_elevation(depth: float) -> float:
    """Convert depth-positive-down to elevation-positive-up."""
    return -depth


def elevation_to_depth(elevation: float) -> float:
    """Convert elevation-positive-up to depth-positive-down."""
    return -elevation

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
        self._migrate()
        self._seed_lithologies()
        self.conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after initial schema without breaking existing DBs."""
        col_migrations = [
            ("horizon_picks",  "x",          "REAL"),
            ("horizon_picks",  "y",          "REAL"),
            ("fault_picks",    "x",          "REAL"),
            ("fault_picks",    "y",          "REAL"),
            ("reference_lines","map_x",      "REAL"),
            ("reference_lines","map_y",      "REAL"),
            # surfaces table redesign — add new columns if table already exists
            ("surfaces",       "kind",        "TEXT DEFAULT 'horizon'"),
            ("surfaces",       "z_domain",    "TEXT DEFAULT 'depth_m'"),
            ("surfaces",       "color_r",     "INTEGER DEFAULT 255"),
            ("surfaces",       "color_g",     "INTEGER DEFAULT 165"),
            ("surfaces",       "color_b",     "INTEGER DEFAULT 0"),
            ("surfaces",       "line_width",  "REAL DEFAULT 1.5"),
            ("surfaces",       "visible",     "INTEGER DEFAULT 1"),
            ("surfaces",       "interpolation","TEXT DEFAULT 'linear'"),
            ("surfaces",       "points_file", "TEXT"),
            ("surfaces",       "created_date","TEXT"),
            # Kinematic restoration — construction metadata
            ("horizons",  "construction_rule_json", "TEXT"),
            ("faults",    "construction_rule_json", "TEXT"),
            ("polygons",  "construction_rule_json", "TEXT"),
            # Reference-based polygon perimeter (PolygonBoundary list)
            ("polygons",  "bounds_json", "TEXT"),
        ]
        for table, col, coltype in col_migrations:
            try:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
            except Exception:
                pass  # column already exists
        # Drop obsolete blob column if it exists (ignore if already gone)
        try:
            self.conn.execute("ALTER TABLE surfaces DROP COLUMN points_blob")
        except Exception:
            pass

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
        dd = getattr(section, "depth_domain", "depth")
        disp_dd = getattr(section, "display_domain", dd)
        if row:
            self.conn.execute(
                """UPDATE sections SET nodes_json=?, depth_domain=?, display_domain=?,
                   depth_units=?, vertical_exaggeration=?, crs_epsg=?, modified_date=?
                   WHERE name=?""",
                (nodes_json, dd, disp_dd,
                 getattr(section, "depth_units", "m"),
                 getattr(section, "vertical_exaggeration", 1.0),
                 getattr(section, "crs_epsg", 32632),
                 _now(), section.name)
            )
            sid = row["id"]
        else:
            cur = self.conn.execute(
                """INSERT INTO sections(name, nodes_json, depth_domain, display_domain,
                   depth_units, vertical_exaggeration, crs_epsg, created_date, modified_date)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (section.name, nodes_json, dd, disp_dd,
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
        from section_tool.core.construction import serialize_rule
        rule_json = serialize_rule(getattr(pick, "construction_rule", None))
        row = self.conn.execute(
            "SELECT id FROM horizons WHERE name=?", (pick.name,)
        ).fetchone()
        if row:
            hid = row["id"]
            self.conn.execute(
                """UPDATE horizons SET color=?, line_width=?, line_style=?,
                   contact_type=?, formation_above=?, formation_below=?,
                   confidence=?, construction_rule_json=?
                   WHERE id=?""",
                (pick.color,
                 getattr(pick, "line_width", 1.5),
                 getattr(pick, "line_style", "solid"),
                 getattr(pick, "contact_type", "conformable"),
                 getattr(pick, "formation_above", ""),
                 getattr(pick, "formation_below", ""),
                 getattr(pick, "confidence", 1.0),
                 rule_json,
                 hid)
            )
        else:
            cur = self.conn.execute(
                """INSERT INTO horizons(name, color, line_width, line_style,
                   contact_type, formation_above, formation_below, confidence,
                   construction_rule_json)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (pick.name, pick.color,
                 getattr(pick, "line_width", 1.5),
                 getattr(pick, "line_style", "solid"),
                 getattr(pick, "contact_type", "conformable"),
                 getattr(pick, "formation_above", ""),
                 getattr(pick, "formation_below", ""),
                 getattr(pick, "confidence", 1.0),
                 rule_json)
            )
            hid = cur.lastrowid

        # Replace all picks
        self.conn.execute("DELETE FROM horizon_picks WHERE horizon_id=?", (hid,))
        for sec_name in pick.section_names():
            idxs = pick.section_indices(sec_name)
            if len(idxs) == 0:
                continue
            dists  = pick._distances[idxs]
            depths = pick._depths[idxs]
            map_xs = pick._map_x[idxs]
            map_ys = pick._map_y[idxs]
            for order, (d, z, mx, my) in enumerate(zip(dists, depths, map_xs, map_ys)):
                self.conn.execute(
                    """INSERT INTO horizon_picks
                       (horizon_id, section_name, distance_along, depth, x, y, sort_order)
                       VALUES(?,?,?,?,?,?,?)""",
                    (hid, sec_name, float(d), float(z),
                     None if (mx != mx) else float(mx),
                     None if (my != my) else float(my),
                     order)
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
        from section_tool.core.construction import serialize_rule
        rule_json = serialize_rule(getattr(pick, "construction_rule", None))
        row = self.conn.execute(
            "SELECT id FROM faults WHERE name=?", (pick.name,)
        ).fetchone()
        if row:
            fid = row["id"]
            self.conn.execute(
                """UPDATE faults SET color=?, line_width=?, line_style=?,
                   fault_type=?, dip_direction=?, confidence=?,
                   construction_rule_json=?
                   WHERE id=?""",
                (pick.color,
                 getattr(pick, "line_width", 1.5),
                 getattr(pick, "line_style", "solid"),
                 getattr(pick, "fault_type", "normal"),
                 getattr(pick, "dip_direction", "right"),
                 getattr(pick, "confidence", 1.0),
                 rule_json,
                 fid)
            )
        else:
            cur = self.conn.execute(
                """INSERT INTO faults(name, color, line_width, line_style,
                   fault_type, dip_direction, confidence,
                   construction_rule_json)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (pick.name, pick.color,
                 getattr(pick, "line_width", 1.5),
                 getattr(pick, "line_style", "solid"),
                 getattr(pick, "fault_type", "normal"),
                 getattr(pick, "dip_direction", "right"),
                 getattr(pick, "confidence", 1.0),
                 rule_json)
            )
            fid = cur.lastrowid

        self.conn.execute("DELETE FROM fault_picks WHERE fault_id=?", (fid,))
        for sec_name in pick.section_names():
            idxs = pick.section_indices(sec_name)
            if len(idxs) == 0:
                continue
            dists  = pick._distances[idxs]
            depths = pick._depths[idxs]
            map_xs = pick._map_x[idxs]
            map_ys = pick._map_y[idxs]
            for order, (d, z, mx, my) in enumerate(zip(dists, depths, map_xs, map_ys)):
                self.conn.execute(
                    """INSERT INTO fault_picks
                       (fault_id, section_name, distance_along, depth, x, y, sort_order)
                       VALUES(?,?,?,?,?,?,?)""",
                    (fid, sec_name, float(d), float(z),
                     None if (mx != mx) else float(mx),
                     None if (my != my) else float(my),
                     order)
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
        status  = getattr(well, "status",  "actual")
        purpose = getattr(well, "purpose", "exploration")
        if row:
            wid = row["id"]
            self.conn.execute(
                """UPDATE wells SET uwi=?, x=?, y=?, kb_elevation=?, td=?,
                   original_x=?, original_y=?, original_crs_epsg=?,
                   status=?, purpose=?
                   WHERE id=?""",
                (well.uwi, well.x, well.y, well.kb,
                 well.deviation.max_tvd,
                 getattr(well, "original_x", None),
                 getattr(well, "original_y", None),
                 getattr(well, "original_crs_epsg", None),
                 status, purpose, wid)
            )
        else:
            cur = self.conn.execute(
                """INSERT INTO wells(name, uwi, x, y, kb_elevation, td,
                   original_x, original_y, original_crs_epsg,
                   status, purpose, created_date)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (well.name, well.uwi, well.x, well.y, well.kb,
                 well.deviation.max_tvd,
                 getattr(well, "original_x", None),
                 getattr(well, "original_y", None),
                 getattr(well, "original_crs_epsg", None),
                 status, purpose, _now())
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
        from section_tool.core.construction import serialize_rule
        name = getattr(poly, "name", f"Polygon {poly_idx}")
        rule_json = serialize_rule(getattr(poly, "construction_rule", None))
        # Reference-based perimeter: persist the PolygonBoundary list so bound
        # polygons reload as bound (not free-form) and keep auto-updating when
        # their bounding horizons/faults are edited.
        bounds = getattr(poly, "bounds", None)
        bounds_json = _dumps(
            [{"category": b.category, "index": b.index, "reversed": bool(b.reversed)}
             for b in bounds]
        ) if bounds else None
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
                   fill_opacity=?, formation_name=?,
                   construction_rule_json=?, bounds_json=? WHERE id=?""",
                (verts_json, getattr(poly, "fill_color", "#9467bd"),
                 getattr(poly, "fill_alpha", 0.6),
                 getattr(poly, "formation", ""),
                 rule_json,
                 bounds_json,
                 pid)
            )
        else:
            cur = self.conn.execute(
                """INSERT INTO polygons
                   (name, section_name, vertices_json, fill_color, fill_opacity,
                    formation_name, construction_rule_json, bounds_json)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (name, section_name, verts_json,
                 getattr(poly, "fill_color", "#9467bd"),
                 getattr(poly, "fill_alpha", 0.6),
                 getattr(poly, "formation", ""),
                 rule_json,
                 bounds_json)
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
    # Restoration sequence
    # ------------------------------------------------------------------

    def set_restoration_sequence(self, sequence) -> None:
        """Persist the project restoration sequence as a JSON project_meta entry."""
        self.set_meta("restoration_sequence", sequence.to_json())

    def get_restoration_sequence(self):
        """Load the restoration sequence from project_meta, or return an empty one."""
        from section_tool.core.restoration import RestorationSequence
        raw = self.get_meta("restoration_sequence", "")
        if not raw:
            return RestorationSequence()
        try:
            return RestorationSequence.from_json(raw)
        except Exception:
            return RestorationSequence()

    # ------------------------------------------------------------------
    # Reference lines
    # ------------------------------------------------------------------

    def replace_all_reference_lines(self, reference_lines) -> None:
        self.conn.execute("DELETE FROM reference_lines")
        for rl in reference_lines:
            self.conn.execute(
                """INSERT INTO reference_lines(name, line_type, value, color, visible, map_x, map_y)
                   VALUES(?,?,?,?,?,?,?)""",
                (getattr(rl, "name", ""),
                 getattr(rl, "kind", "horizontal"),
                 getattr(rl, "value", 0.0),
                 getattr(rl, "color", "#999999"),
                 int(getattr(rl, "visible", True)),
                 getattr(rl, "map_x", None),
                 getattr(rl, "map_y", None))
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
    # Lithology library
    # ------------------------------------------------------------------

    def _seed_lithologies(self) -> None:
        """Insert standard lithologies if the table is empty."""
        n = self.conn.execute("SELECT COUNT(*) FROM lithologies").fetchone()[0]
        if n > 0:
            return
        for lith in _LITHOLOGY_DEFAULTS:
            self.conn.execute(
                """INSERT OR IGNORE INTO lithologies
                   (name, porosity_surface, compaction_coeff, grain_density,
                    matrix_thermal_conductivity)
                   VALUES (?, ?, ?, ?, ?)""",
                (lith["name"], lith["porosity_surface"], lith["compaction_coeff"],
                 lith["grain_density"], lith["matrix_thermal_conductivity"])
            )

    def add_lithology(self, name: str, **props) -> int:
        allowed = {"porosity_surface", "compaction_coeff", "grain_density",
                   "matrix_thermal_conductivity", "matrix_velocity",
                   "specific_heat_capacity", "radiogenic_heat_production"}
        cols = ["name"] + [k for k in props if k in allowed]
        vals = [name] + [props[k] for k in cols[1:]]
        placeholders = ", ".join("?" * len(cols))
        col_list = ", ".join(cols)
        cur = self.conn.execute(
            f"INSERT OR REPLACE INTO lithologies({col_list}) VALUES({placeholders})",
            vals
        )
        self.conn.commit()
        return cur.lastrowid

    def get_lithology(self, name: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM lithologies WHERE name=?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def get_lithology_by_id(self, lid: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM lithologies WHERE id=?", (lid,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_lithologies(self) -> list[dict]:
        return [dict(r) for r in
                self.conn.execute("SELECT * FROM lithologies ORDER BY name").fetchall()]

    def delete_lithology(self, name: str) -> None:
        self.conn.execute("DELETE FROM lithologies WHERE name=?", (name,))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Formation property inheritance
    # ------------------------------------------------------------------

    def get_formation_property(self, formation_id: int, prop: str) -> float | None:
        """Walk the three-level inheritance chain for a petrophysical property.

        Level 1: formation record (if the column is not NULL)
        Level 2: referenced lithology (primary_lithology_id)
        Level 3: hardcoded default in _PROPERTY_DEFAULTS
        """
        # Level 1: check formation record
        row = self.conn.execute(
            "SELECT * FROM formations WHERE id=?", (formation_id,)
        ).fetchone()
        if row is None:
            return _PROPERTY_DEFAULTS.get(prop)

        row = dict(row)
        if prop in row and row[prop] is not None:
            return float(row[prop])

        # Level 2: referenced lithology
        lith_id = row.get("primary_lithology_id")
        if lith_id is not None:
            lith = self.get_lithology_by_id(int(lith_id))
            if lith and lith.get(prop) is not None:
                return float(lith[prop])

        # Level 3: hardcoded default
        return _PROPERTY_DEFAULTS.get(prop)

    # ------------------------------------------------------------------
    # Measurements (thermochronology / thermal data)
    # ------------------------------------------------------------------

    def add_measurement(
        self,
        well_id: int,
        kind: str,
        depth_md: float,
        value: float,
        *,
        depth_tvd: float | None = None,
        uncertainty: float | None = None,
        units: str | None = None,
        sample_id: str | None = None,
        lab: str | None = None,
        method: str | None = None,
        note: str | None = None,
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO measurements
               (well_id, kind, depth_md, depth_tvd, value, uncertainty,
                units, sample_id, lab, method, note)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (well_id, kind, depth_md, depth_tvd, value, uncertainty,
             units, sample_id, lab, method, note)
        )
        self.conn.commit()
        return cur.lastrowid

    def get_measurements(self, well_id: int, kind: str | None = None) -> list[dict]:
        if kind:
            rows = self.conn.execute(
                "SELECT * FROM measurements WHERE well_id=? AND kind=? ORDER BY depth_md",
                (well_id, kind)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM measurements WHERE well_id=? ORDER BY depth_md",
                (well_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_measurement(self, measurement_id: int) -> None:
        self.conn.execute("DELETE FROM measurements WHERE id=?", (measurement_id,))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Well-section projection metadata
    # ------------------------------------------------------------------

    def upsert_well_section(
        self,
        well_id: int,
        section_id: int,
        distance_along: float,
        perpendicular_offset: float,
        *,
        nearest_segment: int | None = None,
        display_mode: str = "auto",
        projection_tolerance: float = 2000.0,
    ) -> int:
        row = self.conn.execute(
            "SELECT id FROM well_sections WHERE well_id=? AND section_id=?",
            (well_id, section_id)
        ).fetchone()
        if row:
            self.conn.execute(
                """UPDATE well_sections
                   SET distance_along=?, perpendicular_offset=?,
                       nearest_segment=?, display_mode=?, projection_tolerance=?
                   WHERE well_id=? AND section_id=?""",
                (distance_along, perpendicular_offset, nearest_segment,
                 display_mode, projection_tolerance, well_id, section_id)
            )
            rid = row["id"]
        else:
            cur = self.conn.execute(
                """INSERT INTO well_sections
                   (well_id, section_id, distance_along, perpendicular_offset,
                    nearest_segment, display_mode, projection_tolerance)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (well_id, section_id, distance_along, perpendicular_offset,
                 nearest_segment, display_mode, projection_tolerance)
            )
            rid = cur.lastrowid
        self.conn.commit()
        return rid

    def get_well_sections(self, well_id: int) -> list[dict]:
        return [dict(r) for r in
                self.conn.execute(
                    "SELECT * FROM well_sections WHERE well_id=?", (well_id,)
                ).fetchall()]

    def delete_well_section(self, well_id: int, section_id: int) -> None:
        self.conn.execute(
            "DELETE FROM well_sections WHERE well_id=? AND section_id=?",
            (well_id, section_id)
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Section sets
    # ------------------------------------------------------------------

    def add_section_set(self, name: str, description: str = "",
                        sort_order_field: str = "distance") -> int:
        """Create a new section set and return its ID."""
        cur = self.conn.execute(
            """INSERT INTO section_sets(name, description, sort_order_field, created_date)
               VALUES(?, ?, ?, ?)""",
            (name, description, sort_order_field, _now())
        )
        self.conn.commit()
        return cur.lastrowid

    def delete_section_set(self, set_id: int) -> None:
        """Delete a section set and all its member entries (CASCADE)."""
        self.conn.execute("DELETE FROM section_sets WHERE id=?", (set_id,))
        self.conn.commit()

    def add_section_to_set(self, set_id: int, section_id: int,
                           sort_index: int) -> int:
        """Add a section to a set at *sort_index*. Returns member row ID."""
        cur = self.conn.execute(
            """INSERT INTO section_set_members(set_id, section_id, sort_index)
               VALUES(?, ?, ?)""",
            (set_id, section_id, sort_index)
        )
        self.conn.commit()
        return cur.lastrowid

    def remove_section_from_set(self, set_id: int, section_id: int) -> None:
        """Remove a section from a set (set itself is not deleted)."""
        self.conn.execute(
            "DELETE FROM section_set_members WHERE set_id=? AND section_id=?",
            (set_id, section_id)
        )
        self.conn.commit()

    def get_section_set(self, set_id: int) -> dict | None:
        """Return set metadata + ordered member list, or None if not found."""
        row = self.conn.execute(
            "SELECT * FROM section_sets WHERE id=?", (set_id,)
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        members = self.conn.execute(
            """SELECT ssm.section_id, s.name AS section_name, ssm.sort_index
               FROM section_set_members ssm
               JOIN sections s ON s.id = ssm.section_id
               WHERE ssm.set_id = ?
               ORDER BY ssm.sort_index""",
            (set_id,)
        ).fetchall()
        result["members"] = [dict(m) for m in members]
        return result

    def get_all_section_sets(self) -> list[dict]:
        """Return all section sets with their ordered member lists."""
        rows = self.conn.execute(
            "SELECT * FROM section_sets ORDER BY id"
        ).fetchall()
        return [self.get_section_set(r["id"]) for r in rows]

    def reorder_set_member(self, set_id: int, section_id: int,
                           new_sort_index: int) -> None:
        """Update the sort_index of a member within its set."""
        self.conn.execute(
            """UPDATE section_set_members SET sort_index=?
               WHERE set_id=? AND section_id=?""",
            (new_sort_index, set_id, section_id)
        )
        self.conn.commit()

    def get_adjacent_sections(
        self, set_id: int, section_id: int
    ) -> tuple[dict | None, dict | None]:
        """Return (previous_member, next_member) for *section_id* in the set.

        Members are ordered by sort_index.  Returns ``None`` for the
        prev/next when *section_id* is at the start/end of the set, or when
        the section is not a member of the set.

        Each returned member dict has keys: section_id, section_name, sort_index.
        """
        members = self.conn.execute(
            """SELECT ssm.section_id, s.name AS section_name, ssm.sort_index
               FROM section_set_members ssm
               JOIN sections s ON s.id = ssm.section_id
               WHERE ssm.set_id = ?
               ORDER BY ssm.sort_index""",
            (set_id,)
        ).fetchall()
        members = [dict(m) for m in members]
        idx = next((i for i, m in enumerate(members)
                    if m["section_id"] == section_id), None)
        if idx is None:
            return None, None
        prev_m = members[idx - 1] if idx > 0 else None
        next_m = members[idx + 1] if idx < len(members) - 1 else None
        return prev_m, next_m

    def update_section_set(self, set_id: int, name: str | None = None,
                           description: str | None = None,
                           sort_order_field: str | None = None) -> None:
        """Update editable fields on a section set."""
        updates = []
        vals = []
        if name is not None:
            updates.append("name=?"); vals.append(name)
        if description is not None:
            updates.append("description=?"); vals.append(description)
        if sort_order_field is not None:
            updates.append("sort_order_field=?"); vals.append(sort_order_field)
        if not updates:
            return
        vals.append(set_id)
        self.conn.execute(
            f"UPDATE section_sets SET {', '.join(updates)} WHERE id=?", vals
        )
        self.conn.commit()

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
            "aoi":             self.get_aoi(),
            "surfaces":        self.get_all_surfaces(),
        }

    # ------------------------------------------------------------------
    # AOI
    # ------------------------------------------------------------------

    def upsert_aoi(self, aoi) -> None:
        """Insert or replace the single AOI row (id=1 always)."""
        if aoi is None:
            self.conn.execute("DELETE FROM aoi WHERE id = 1")
        else:
            self.conn.execute(
                """INSERT OR REPLACE INTO aoi(id, name, polygon_wkt, crs_epsg)
                   VALUES(1, ?, ?, ?)""",
                (aoi.name, aoi.polygon_wkt, aoi.crs_epsg),
            )
        self.conn.commit()

    def get_aoi(self) -> dict | None:
        row = self.conn.execute("SELECT * FROM aoi WHERE id = 1").fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Vector layers
    # ------------------------------------------------------------------

    def upsert_vector_layer(self, layer: dict) -> int:
        row = self.conn.execute(
            "SELECT id FROM vector_layers WHERE filepath=?", (layer["filepath"],)
        ).fetchone()
        if row:
            self.conn.execute(
                """UPDATE vector_layers
                   SET name=?, crs=?, geom_type=?, color=?, visible=?
                   WHERE id=?""",
                (layer["name"], layer.get("crs", ""), layer.get("geom_type", ""),
                 layer.get("color", "#FFAA00"), int(layer.get("visible", True)),
                 row["id"]),
            )
            lid = row["id"]
        else:
            cur = self.conn.execute(
                """INSERT INTO vector_layers(name, filepath, crs, geom_type, color, visible)
                   VALUES(?, ?, ?, ?, ?, ?)""",
                (layer["name"], layer["filepath"], layer.get("crs", ""),
                 layer.get("geom_type", ""), layer.get("color", "#FFAA00"),
                 int(layer.get("visible", True))),
            )
            lid = cur.lastrowid
        self.conn.commit()
        return lid

    def load_vector_layers(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT name, filepath, crs, geom_type, color, visible FROM vector_layers"
        ).fetchall()
        result = []
        for r in rows:
            layer = dict(r)
            layer["visible"] = bool(layer["visible"])
            try:
                import fiona
                with fiona.open(layer["filepath"]) as src:
                    layer["features"] = [dict(f) for f in src]
            except Exception:
                layer["features"] = []
            result.append(layer)
        return result

    # ------------------------------------------------------------------
    # Surfaces
    # ------------------------------------------------------------------

    def upsert_surface(self, surface, points_file: str | None = None) -> int:
        """Insert or replace a Surface row (metadata only; point data is in .npy)."""
        from datetime import datetime as _dt
        b = surface.bounds()
        zr = surface.z_range()
        color = getattr(surface, "color", (255, 165, 0))
        row = self.conn.execute(
            "SELECT id FROM surfaces WHERE name=?", (surface.name,)
        ).fetchone()
        params = (
            surface.name,
            getattr(surface, "kind", "horizon"),
            getattr(surface, "z_domain", "depth_m"),
            getattr(surface, "z_units", "m"),
            getattr(surface, "crs_epsg", 0),
            int(color[0]), int(color[1]), int(color[2]),
            float(getattr(surface, "line_width", 1.5)),
            int(getattr(surface, "visible", True)),
            getattr(surface, "interpolation", "linear"),
            getattr(surface, "source_file", None),
            getattr(surface, "source_format", None),
            surface.n_points,
            float(b[0]), float(b[2]), float(b[1]), float(b[3]),
            float(zr[0]), float(zr[1]),
            points_file,
        )
        if row:
            self.conn.execute(
                """UPDATE surfaces SET kind=?,z_domain=?,z_units=?,crs_epsg=?,
                   color_r=?,color_g=?,color_b=?,line_width=?,visible=?,
                   interpolation=?,source_file=?,source_format=?,point_count=?,
                   x_min=?,x_max=?,y_min=?,y_max=?,z_min=?,z_max=?,points_file=?
                   WHERE name=?""",
                params[1:] + (surface.name,),
            )
            self.conn.commit()
            return row["id"]
        cur = self.conn.execute(
            """INSERT INTO surfaces(name,kind,z_domain,z_units,crs_epsg,
               color_r,color_g,color_b,line_width,visible,interpolation,
               source_file,source_format,point_count,
               x_min,x_max,y_min,y_max,z_min,z_max,points_file,created_date)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            params + (_dt.now().isoformat(),),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_all_surfaces(self) -> list[dict]:
        return [dict(r) for r in
                self.conn.execute("SELECT * FROM surfaces ORDER BY id").fetchall()]

    def delete_surface_by_name(self, name: str) -> None:
        self.conn.execute("DELETE FROM surfaces WHERE name=?", (name,))
        self.conn.commit()
