from __future__ import annotations

import os

from PySide6.QtCore import QObject, Signal

from section_tool.core.annotation import Annotation
from section_tool.core.commands import Command, CommandStack
from section_tool.core.intersection import FaultHorizonIntersection
from section_tool.core.polygons import PolygonBoundary, SectionPolygon
from section_tool.core.reference_line import ReferenceLine
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick, Surface
from section_tool.core.topology import SectionTopology
from section_tool.core.wells import LogCurve, Well
from section_tool.io.project import Project, SeismicRef
import numpy as np


_DOMAIN_MIGRATION = {"md": "depth", "twt": "time"}


def _migrate_domain(value: str) -> str:
    """Silently migrate legacy depth-domain strings to current values."""
    return _DOMAIN_MIGRATION.get(value, value)


def _restore_construction_rule(obj, json_str: str | None) -> None:
    """Attach a deserialized ConstructionRule to *obj* (in-place, silent on error)."""
    if not json_str:
        return
    try:
        from section_tool.core.construction import deserialize_rule
        obj.construction_rule = deserialize_rule(json_str)
    except Exception:
        pass


def _surface_from_db_row(row: dict, points: np.ndarray) -> Surface:
    """Reconstruct a Surface from a surfaces table row + pre-loaded points array."""
    color = (
        int(row.get("color_r", 255)),
        int(row.get("color_g", 165)),
        int(row.get("color_b", 0)),
    )
    return Surface(
        name=row.get("name", ""),
        points=points,
        crs_epsg=int(row.get("crs_epsg", 0)),
        z_domain=row.get("z_domain", "depth_m"),
        z_units=row.get("z_units", "m"),
        color=color,
        line_width=float(row.get("line_width", 1.5)),
        visible=bool(row.get("visible", 1)),
        interpolation=row.get("interpolation", "linear"),
        source_file=row.get("source_file"),
        source_format=row.get("source_format"),
        kind=row.get("kind", "horizon"),
    )


class AppState(QObject):
    """Central application state with Qt signals for reactive UI updates.

    Views connect to signals to learn about state changes; they never
    mutate project data directly.  All mutations go through the methods
    on this class, which update the :class:`Project` and emit the
    appropriate signal(s).

    Signals
    -------
    project_changed
        The entire project was replaced (new / open).
    project_path_changed(str)
        The current file path changed.  Emits ``""`` when there is no path.
    project_modified_changed(bool)
        The unsaved-changes flag flipped.

    section_added(object)        / section_removed(object)
    section_modified(int, object)
        Index and new value when a section is replaced in-place.
    active_section_changed(object)
        The actively-viewed section changed (may be ``None``).

    surface_added / surface_removed / surface_modified(int, object)
    horizon_pick_added / horizon_pick_removed / horizon_pick_modified(int, object)

    well_added(object) / well_removed(object)
    well_modified(int, object)
    active_well_changed(object)
        The selected well changed (may be ``None``).

    seismic_ref_added(object) / seismic_ref_removed(object)
    """

    # Project-level
    project_changed = Signal()
    project_settings_changed = Signal()  # emitted only by set_project_properties
    project_path_changed = Signal(str)   # "" means no file
    project_modified_changed = Signal(bool)

    # Sections
    section_added = Signal(object)
    section_removed = Signal(object)
    section_modified = Signal(int, object)
    active_section_changed = Signal(object)
    # The active *workspace slice* — a Section OR a HorizontalSlice (Model A).
    # active_section stays Section-typed and untouched; this is the parallel,
    # additive selection concept that the z-slice workspace routes on.
    active_slice_changed = Signal(object)

    # AOI
    aoi_changed = Signal(object)   # emits AOI or None

    # Surfaces
    surface_added = Signal(object)
    surface_removed = Signal(object)
    surface_modified = Signal(int, object)

    # Horizon picks
    horizon_pick_added = Signal(object)
    horizon_pick_removed = Signal(object)
    horizon_pick_modified = Signal(int, object)

    # Wells
    well_added = Signal(object)
    well_removed = Signal(object)
    well_modified = Signal(int, object)
    active_well_changed = Signal(object)

    # Seismic refs
    seismic_ref_added = Signal(object)
    seismic_ref_removed = Signal(object)

    # Section polygons
    polygon_added = Signal(object)
    polygon_removed = Signal(object)
    polygon_modified = Signal(int, object)

    # Fault picks (separate from horizon picks)
    fault_pick_added    = Signal(object)
    fault_pick_removed  = Signal(object)
    fault_pick_modified = Signal(int, object)

    # Reference lines
    reference_line_added    = Signal(object)
    reference_line_removed  = Signal(object)
    reference_line_modified = Signal(int, object)

    # Phase 2/6 additions
    intersection_added   = Signal(object)
    annotation_added     = Signal(object)
    annotation_removed   = Signal(object)
    annotation_modified  = Signal(int, object)
    # Phase 7: undo/redo status
    undo_performed       = Signal(str)   # description
    redo_performed       = Signal(str)

    # Live topology graph for the active section
    topology_changed = Signal(str)   # section_name whose topology was rebuilt

    # Active pick target: which horizon/fault picks go into
    active_pick_target_changed = Signal(str, int)  # category_name, index

    # Active tool (mirrors ToolPalette; stored here so views can read it)
    tool_changed = Signal(str)

    # Theme switch — emitted after set_theme() is called
    theme_changed = Signal(str)   # new theme name

    # Extracted seismic: emitted when new data is loaded for a section
    seismic_extracted = Signal(str)   # section_name

    # Selection: which entity in the section view is highlighted
    # category is "Horizons" / "Faults" / "Polygons" / "Wells" / "" (none)
    # index is -1 when nothing is selected
    selected_entity_changed = Signal(str, int)

    # ------------------------------------------------------------------

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._project: Project = Project()
        self._project_path: str | None = None
        self._active_section: Section | None = None
        self._active_slice = None   # Section | HorizontalSlice | None (Model A)
        self._active_well: Well | None = None
        self._is_modified: bool = False
        self._active_tool: str = "select"
        # Phase 7: command stack
        self._cmd_stack: CommandStack = CommandStack()
        self._active_pick_category: str | None = None
        self._active_pick_index: int | None = None
        # Live topology (one per active section)
        self._topology: SectionTopology | None = None
        # Extracted seismic per section: {section_name: (data, meta)}
        self._section_seismic: dict[str, tuple[np.ndarray, dict]] = {}
        self._vector_layers: list[dict] = []
        # Selection state
        self._selected_entity_cat: str = ""
        self._selected_entity_idx: int = -1
        # Cross-section ghost markers visibility
        self.show_cross_section_ghosts: bool = True
        # SQLite project manager (None when working with legacy HDF5 projects)
        from section_tool.io.project_manager import ProjectManager
        self._pm: ProjectManager = ProjectManager()

    # ------------------------------------------------------------------
    # DB write-through helper
    # ------------------------------------------------------------------

    @property
    def project_manager(self):
        return self._pm

    def _db_write(self, fn) -> None:
        """Call *fn* only when a ProjectManager database is open."""
        if self._pm.is_open:
            try:
                fn()
            except Exception:
                pass

    def get_meta(self, key: str, default: str = "") -> str:
        """Read a per-project metadata value (project_meta key-value store).

        Returns *default* for in-memory projects with no database open."""
        if self._pm.is_open:
            try:
                return self._pm.db.get_meta(key, default)
            except Exception:
                return default
        return default

    def set_meta(self, key: str, value) -> None:
        """Persist a per-project metadata value (no-op for in-memory projects)."""
        self._db_write(lambda: self._pm.db.set_meta(key, str(value)))

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def project(self) -> Project:
        return self._project

    @property
    def project_path(self) -> str | None:
        return self._project_path

    @property
    def is_modified(self) -> bool:
        return self._is_modified

    @property
    def active_tool(self) -> str:
        return self._active_tool

    @property
    def active_pick_category(self) -> str | None:
        return self._active_pick_category

    @property
    def active_pick_index(self) -> int | None:
        return self._active_pick_index

    @property
    def command_stack(self) -> CommandStack:
        return self._cmd_stack

    @property
    def selected_entity_category(self) -> str:
        return self._selected_entity_cat

    @property
    def selected_entity_index(self) -> int:
        return self._selected_entity_idx

    def set_selected_entity(self, category: str, index: int) -> None:
        """Set the currently-selected entity by category and list index.

        Pass category="" and index=-1 to clear the selection.
        Emits selected_entity_changed only when the value actually changes.
        """
        if category != self._selected_entity_cat or index != self._selected_entity_idx:
            self._selected_entity_cat = category
            self._selected_entity_idx = index
            self.selected_entity_changed.emit(category, index)

    # ------------------------------------------------------------------
    # Extracted seismic
    # ------------------------------------------------------------------

    def get_seismic_for_section(
        self, section_name: str
    ) -> tuple["np.ndarray | None", "dict | None"]:
        """Return (data, meta) for extracted seismic, or (None, None)."""
        pair = self._section_seismic.get(section_name)
        if pair is None:
            return None, None
        return pair

    def set_seismic_for_section(
        self, section_name: str, data: "np.ndarray", meta: dict
    ) -> None:
        """Register extracted seismic for a section and emit seismic_extracted."""
        self._section_seismic[section_name] = (data, meta)
        self.seismic_extracted.emit(section_name)

    # ------------------------------------------------------------------
    # Vector layers (shapefiles, geopackages, GeoJSON)
    # ------------------------------------------------------------------

    def add_vector_layer(self, filepath: str, features: list,
                         crs, geom_type: str) -> None:
        import os
        name = os.path.splitext(os.path.basename(filepath))[0]
        layer = {
            "name":      name,
            "filepath":  filepath,
            "features":  features,
            "crs":       str(crs),
            "geom_type": geom_type,
            "color":     "#FFAA00",
            "visible":   True,
        }
        self._vector_layers.append(layer)
        self._db_write(lambda lyr=layer: self._pm.db.upsert_vector_layer(lyr))
        self.project_changed.emit()

    def get_vector_layers(self) -> list[dict]:
        return [lyr for lyr in self._vector_layers if lyr.get("visible", True)]

    # ------------------------------------------------------------------

    def execute_command(self, command: Command) -> None:
        """Execute *command* via the stack (calls do(), pushes, clears redo)."""
        self._cmd_stack.execute(command)
        if command.description:
            self.undo_performed.emit("")   # signal that stack changed

    def record_command(self, description: str,
                       undo, redo=None) -> None:
        """Record an already-applied operation for undo/redo (backward compat).

        Equivalent to building a Command with ``do=redo`` and calling push().
        The operation is NOT re-executed — it must already have been applied.
        """
        self._cmd_stack.push(Command(description=description,
                                     do=redo or (lambda: None),
                                     undo=undo))

    def undo(self) -> None:
        desc = self._cmd_stack.undo()
        if desc is not None:
            self.undo_performed.emit(desc)

    def redo(self) -> None:
        desc = self._cmd_stack.redo()
        if desc is not None:
            self.redo_performed.emit(desc)

    def set_active_pick_target(self, category: str, index: int) -> None:
        self._active_pick_category = category
        self._active_pick_index    = index
        self.active_pick_target_changed.emit(category, index)

    def set_active_tool(self, tool_id: str) -> None:
        """Set the active tool; emits :attr:`tool_changed` if it changed."""
        if self._active_tool != tool_id:
            self._active_tool = tool_id
            self.tool_changed.emit(tool_id)

    @property
    def active_section(self) -> Section | None:
        return self._active_section

    @property
    def active_well(self) -> Well | None:
        return self._active_well

    @property
    def topology(self) -> SectionTopology | None:
        """Live topology graph for the currently active section (may be None)."""
        return self._topology

    # ------------------------------------------------------------------
    # Project-level operations
    # ------------------------------------------------------------------

    def new_project(
        self,
        name: str = "",
        crs_epsg: int = 32632,
        depth_units: str = "m",
        depth_domain: str = "depth",
        default_depth_min: float = 0.0,
        default_depth_max: float = 5000.0,
        folder_path: str | None = None,
    ) -> None:
        """Replace the current project with a fresh empty one."""
        self._pm.close()
        self._project = Project(
            name=name,
            crs_epsg=crs_epsg,
            depth_units=depth_units,
            depth_domain=depth_domain,
            default_depth_min=default_depth_min,
            default_depth_max=default_depth_max,
        )
        self._project_path = folder_path
        self._active_section = None
        self._active_slice = None
        self._active_well = None
        self._is_modified = False
        self._topology = None
        self._cmd_stack.clear()
        if folder_path:
            self._pm.new_project(
                folder_path, name, crs_epsg,
                depth_units=depth_units,
                depth_domain=depth_domain,
                default_depth_min=default_depth_min,
                default_depth_max=default_depth_max,
            )
        self.project_path_changed.emit(folder_path or "")
        self.project_changed.emit()
        self.project_modified_changed.emit(False)

    def open_project(self, path: str | os.PathLike) -> None:
        """Load a project from *path*.

        Accepts either a project folder (SQLite) or a legacy .h5 file.
        """
        path = str(path)
        self._pm.close()
        self._vector_layers = []   # clear before load so stale layers don't linger
        if os.path.isdir(path):
            # SQLite folder-based project
            self._pm.open_project(path)
            self._project = self._load_from_sqlite(path)
            self._project_path = path
        else:
            # Legacy HDF5 file
            self._project = Project.load(path)
            self._project_path = path
        self._active_section = None
        self._active_slice = None
        self._active_well = None
        self._is_modified = False
        self._topology = None
        self.project_path_changed.emit(self._project_path)
        self.project_changed.emit()
        self.project_modified_changed.emit(False)

    def _load_from_sqlite(self, folder_path: str) -> "Project":
        """Reconstruct in-memory Project from an open SQLite database."""
        import json as _json
        db = self._pm.db
        data = db.load_all()
        meta = data["meta"]

        # (debug prints removed)

        proj = Project(
            name=meta.get("name", ""),
            crs_epsg=int(meta.get("crs_epsg", 32632)),
            depth_units=meta.get("depth_units", "m"),
            depth_domain=_migrate_domain(meta.get("depth_domain", "depth")),
            default_depth_min=float(meta.get("default_depth_min", 0.0)),
            default_depth_max=float(meta.get("default_depth_max", 5000.0)),
        )

        # Sections
        for row in data["sections"]:
            nodes = np.array(_json.loads(row["nodes_json"]))
            sec = Section(
                nodes,
                name=row["name"],
                depth_domain=_migrate_domain(row.get("depth_domain", "depth")),
                depth_units=row.get("depth_units", "m"),
                vertical_exaggeration=float(row.get("vertical_exaggeration", 1.0)),
                crs_epsg=int(row.get("crs_epsg", 32632)),
            )
            proj.sections.append(sec)

        # Horizontal slices (plan-slice registry)
        from section_tool.core.slices import HorizontalSlice
        for hsrow in data.get("horizontal_slices", []):
            proj.horizontal_slices.append(HorizontalSlice(
                name=hsrow["name"],
                elevation=float(hsrow["elevation"]),
                crs_epsg=int(hsrow.get("crs_epsg", 32632)),
            ))

        # Horizons + picks
        for hrow in data["horizons"]:
            picks = hrow.get("picks", [])
            if picks:
                dists   = np.array([p["distance_along"] for p in picks], dtype=float)
                depths  = np.array([p["depth"]          for p in picks], dtype=float)
                # slice_ref is the generalized reference; fall back to the legacy
                # section_name. slice_kind defaults to 'section' for old rows.
                snames  = np.array([p.get("slice_ref") or p["section_name"]
                                    for p in picks], dtype=object)
                skinds  = np.array([p.get("slice_kind") or "section"
                                    for p in picks], dtype=object)
                map_xs  = np.array([p.get("x") if p.get("x") is not None else float("nan")
                                    for p in picks], dtype=float)
                map_ys  = np.array([p.get("y") if p.get("y") is not None else float("nan")
                                    for p in picks], dtype=float)
                anchors = np.array([p.get("twt_anchor") if p.get("twt_anchor") is not None
                                    else float("nan") for p in picks], dtype=float)
                hp = HorizonPick(
                    dists, depths,
                    name=hrow["name"],
                    color=hrow.get("color", "#2ca02c"),
                    line_width=float(hrow.get("line_width", 1.5)),
                    line_style=hrow.get("line_style", "solid"),
                    section_names=snames,
                    map_x=map_xs,
                    map_y=map_ys,
                    slice_kinds=skinds,
                    twt_anchor=anchors,
                    seismic_tied=bool(hrow.get("seismic_tied", 0)),
                    uuid=hrow.get("uuid"),
                    contact_type=hrow.get("contact_type", "conformable"),
                    formation_above=hrow.get("formation_above", ""),
                    formation_below=hrow.get("formation_below", ""),
                )
            else:
                # Pickless entity (created but not yet drawn). HorizonPick() rejects
                # zero-length arrays, so build via empty() and restore metadata.
                hp = HorizonPick.empty(
                    name=hrow["name"],
                    color=hrow.get("color", "#2ca02c"),
                    line_width=float(hrow.get("line_width", 1.5)),
                    line_style=hrow.get("line_style", "solid"),
                )
                if hrow.get("uuid"):
                    hp.uuid = hrow["uuid"]
                hp.contact_type    = hrow.get("contact_type", "conformable")
                hp.formation_above = hrow.get("formation_above", "")
                hp.formation_below = hrow.get("formation_below", "")
                hp.seismic_tied    = bool(hrow.get("seismic_tied", 0))
            _restore_construction_rule(hp, hrow.get("construction_rule_json"))
            proj.horizon_picks.append(hp)

        # Faults + picks
        for frow in data["faults"]:
            picks = frow.get("picks", [])
            if picks:
                dists  = np.array([p["distance_along"] for p in picks], dtype=float)
                depths = np.array([p["depth"]          for p in picks], dtype=float)
                snames = np.array([p.get("slice_ref") or p["section_name"]
                                   for p in picks], dtype=object)
                skinds = np.array([p.get("slice_kind") or "section"
                                   for p in picks], dtype=object)
                map_xs = np.array([p.get("x") if p.get("x") is not None else float("nan")
                                   for p in picks], dtype=float)
                map_ys = np.array([p.get("y") if p.get("y") is not None else float("nan")
                                   for p in picks], dtype=float)
                anchors = np.array([p.get("twt_anchor") if p.get("twt_anchor") is not None
                                    else float("nan") for p in picks], dtype=float)
                fp = HorizonPick(
                    dists, depths,
                    name=frow["name"],
                    color=frow.get("color", "#d62728"),
                    line_width=float(frow.get("line_width", 1.5)),
                    line_style=frow.get("line_style", "solid"),
                    section_names=snames,
                    map_x=map_xs,
                    map_y=map_ys,
                    slice_kinds=skinds,
                    twt_anchor=anchors,
                    seismic_tied=bool(frow.get("seismic_tied", 0)),
                    uuid=frow.get("uuid"),
                )
            else:
                # Pickless entity — build via empty() (see horizons above).
                fp = HorizonPick.empty(
                    name=frow["name"],
                    color=frow.get("color", "#d62728"),
                    line_width=float(frow.get("line_width", 1.5)),
                    line_style=frow.get("line_style", "solid"),
                )
                if frow.get("uuid"):
                    fp.uuid = frow["uuid"]
                fp.seismic_tied = bool(frow.get("seismic_tied", 0))
            _restore_construction_rule(fp, frow.get("construction_rule_json"))
            proj.fault_picks.append(fp)

        # Wells
        for wrow in data["wells"]:
            _td = wrow.get("td")
            well = Well(
                name=wrow["name"],
                x=float(wrow["x"]),
                y=float(wrow["y"]),
                kb=float(wrow.get("kb_elevation", 0.0)),
                uwi=wrow.get("uwi", ""),
                td=float(_td) if _td is not None else None,
                uuid=wrow.get("uuid") or None,
            )
            well.original_x = wrow.get("original_x")
            well.original_y = wrow.get("original_y")
            well.original_crs_epsg = wrow.get("original_crs_epsg")
            for top in wrow.get("tops", []):
                well.add_formation_top(top["formation_name"], float(top["md"]))
            # Time-depth relations (checkshots / sonic TDRs)
            for trow in wrow.get("tdrs", []):
                try:
                    from section_tool.core.tdr import TimeDepthRelation
                    well.add_tdr(TimeDepthRelation.from_dict(
                        _json.loads(trow["data_json"])))
                except Exception:
                    pass
            # Restore log curves from stored data
            for lg in wrow.get("logs", []):
                try:
                    raw = _json.loads(lg["data_json"])
                    depths_list = _json.loads(raw["depths"])
                    values_list = _json.loads(raw["values"])
                    lc = LogCurve(
                        lg["mnemonic"], lg.get("unit", ""),
                        np.array(depths_list), np.array(values_list)
                    )
                    well.add_log(lc)
                except Exception:
                    pass
            proj.wells.append(well)

        # Seismic refs
        for srow in data["seismic"]:
            from section_tool.io.project import SeismicRef
            raw_path = srow.get("file_path") or ""
            resolved = self._pm.resolve_file_path(raw_path) if raw_path else None
            ref = SeismicRef(
                path=resolved,
                name=srow["name"],
                x_field=int(srow.get("x_field", 181)),
                y_field=int(srow.get("y_field", 185)),
                scalar_field=int(srow.get("scalar_field", 71)),
                apply_scalar=bool(srow.get("apply_scalar", 1)),
                domain=srow.get("domain", "twt"),
                depth_units=srow.get("depth_units", "ms"),
                max_offset=float(srow.get("max_offset", 500.0)),
                crs_epsg=int(srow.get("crs_epsg", 32632)),
                extent_x_min=float(srow.get("extent_xmin", 0.0)),
                extent_x_max=float(srow.get("extent_xmax", 0.0)),
                extent_y_min=float(srow.get("extent_ymin", 0.0)),
                extent_y_max=float(srow.get("extent_ymax", 0.0)),
                n_traces_total=int(srow.get("n_traces", 0)),
            )
            proj.seismic_refs.append(ref)

        # Polygons
        for prow in data["polygons"]:
            verts = np.array(_json.loads(prow["vertices_json"]))
            # Restore reference-based perimeter so bound polygons reload as bound
            # (and keep auto-updating when their bounding entities change).
            bounds = []
            bounds_raw = prow.get("bounds_json")
            if bounds_raw:
                try:
                    bounds = [
                        PolygonBoundary(
                            category=b["category"],
                            index=int(b["index"]),
                            reversed=bool(b.get("reversed", False)),
                        )
                        for b in _json.loads(bounds_raw)
                    ]
                except Exception:
                    bounds = []
            poly = SectionPolygon(
                vertices=verts,
                name=prow["name"],
                fill_color=prow.get("fill_color", "#9467bd"),
                fill_alpha=float(prow.get("fill_opacity", 0.6)),
                formation=prow.get("formation_name", ""),
                section_name=prow.get("section_name", ""),
                bounds=bounds,
            )
            _restore_construction_rule(poly, prow.get("construction_rule_json"))
            proj.polygons.append(poly)

        # Reference lines
        for rrow in data["reference_lines"]:
            rl = ReferenceLine(
                kind=rrow.get("line_type", "horizontal"),
                value=float(rrow.get("value", 0.0)),
                name=rrow.get("name", ""),
                color=rrow.get("color", "#999999"),
                visible=bool(rrow.get("visible", 1)),
                map_x=rrow.get("map_x"),
                map_y=rrow.get("map_y"),
            )
            proj.reference_lines.append(rl)

        # Annotations
        for arow in data["annotations"]:
            ann = Annotation(
                text=arow.get("text", ""),
                position=(float(arow.get("distance", 0.0)),
                          float(arow.get("depth", 0.0))),
                section_name=arow.get("section_name", ""),
                font_size=int(arow.get("font_size", 10)),
                rotation=float(arow.get("rotation", 0.0)),
                color=arow.get("color", "#000000"),
            )
            proj.annotations.append(ann)

        # AOI
        aoi_row = data.get("aoi")
        if aoi_row:
            from section_tool.core.aoi import AOI
            proj.aoi = AOI(
                name=aoi_row.get("name", "AOI"),
                polygon_wkt=aoi_row["polygon_wkt"],
                crs_epsg=int(aoi_row["crs_epsg"]),
            )

        # Surfaces (metadata from DB, points from .npy sidecar files)
        surfaces_dir = os.path.join(folder_path, "surfaces")
        for srow in data.get("surfaces", []):
            try:
                pf = srow.get("points_file")
                if pf:
                    npy_path = os.path.join(surfaces_dir, pf)
                    if os.path.exists(npy_path):
                        points = np.load(npy_path)
                    else:
                        continue
                else:
                    continue
                surf = _surface_from_db_row(srow, points)
                proj.surfaces.append(surf)
            except Exception:
                pass

        # Restoration sequence
        proj.restoration_sequence = db.get_restoration_sequence()

        # Velocity model (JSON blob in the velocity_model KV table; empty dict
        # → a fresh unconverted model. from_dict migrates the v1 schema.)
        from section_tool.core.velocity_model import VelocityModel
        proj.velocity_model = VelocityModel.from_dict(data.get("velocity_model", {}))
        _lvm = data.get("lateral_velocity_model", {})
        if _lvm and _lvm.get("controls"):
            from section_tool.core.lateral_velocity import LateralVelocityModel
            proj.lateral_velocity_model = LateralVelocityModel.from_dict(_lvm)

        # Vector layers (re-read features from source files on load)
        try:
            self._vector_layers = db.load_vector_layers()
        except Exception:
            self._vector_layers = []

        return proj

    def save_project(self) -> None:
        """Commit database changes or save HDF5 depending on project type."""
        if self._project_path is None:
            raise ValueError("No project path set; use save_project_as() first")
        if self._pm.is_open:
            self._pm.db.save_velocity_model(self._project.velocity_model)
            self._pm.db.save_lateral_velocity_model(self._project.lateral_velocity_model)
            self._pm.save()
        else:
            self._project.save(self._project_path)
        self._set_modified(False)

    def save_project_as(self, new_path: str | os.PathLike) -> None:
        """Copy project to *new_path* (folder for SQLite, file for HDF5)."""
        new_path = str(new_path)
        if self._pm.is_open:
            self._pm.save_as(new_path)
            self._project_path = new_path
        elif os.path.isdir(new_path) or not os.path.splitext(new_path)[1]:
            # First-time save of an in-memory project to a folder.
            # Initialize SQLite project manager and sync any existing sections.
            self._pm.new_project(
                new_path,
                self._project.name,
                self._project.crs_epsg,
                depth_units=self._project.depth_units,
                depth_domain=self._project.depth_domain,
                default_depth_min=self._project.default_depth_min,
                default_depth_max=self._project.default_depth_max,
            )
            for section in self._project.sections:
                self._pm.db.upsert_section(section)
            for hslice in self._project.horizontal_slices:
                self._pm.db.upsert_horizontal_slice(hslice)
            for hp in self._project.horizon_picks:
                self._pm.db.upsert_horizon(hp)
            for fp in self._project.fault_picks:
                self._pm.db.upsert_fault(fp)
            for w in self._project.wells:
                self._pm.db.upsert_well(w)
            for ref in self._project.seismic_refs:
                self._pm.db.upsert_seismic(ref)
            if self._project.polygons:
                self._pm.db.replace_all_polygons(self._project.polygons)
            if self._project.reference_lines:
                self._pm.db.replace_all_reference_lines(self._project.reference_lines)
            if self._project.annotations:
                self._pm.db.replace_all_annotations(self._project.annotations)
            if self._project.aoi is not None:
                self._pm.db.upsert_aoi(self._project.aoi)
            for surface in self._project.surfaces:
                self._save_surface_to_db(surface)
            for lyr in self._vector_layers:
                self._pm.db.upsert_vector_layer(lyr)
            self._pm.db.save_velocity_model(self._project.velocity_model)
            self._pm.db.save_lateral_velocity_model(self._project.lateral_velocity_model)
            self._pm.save()
            self._project_path = new_path
        else:
            self._project_path = new_path
            self._project.save(self._project_path)
        self.project_path_changed.emit(self._project_path)
        self._set_modified(False)

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def add_section(self, section: Section) -> None:
        self._project.sections.append(section)
        self._set_modified()
        self._db_write(lambda: self._pm.db.upsert_section(section))
        self.section_added.emit(section)

    def add_horizontal_slice(self, hslice) -> None:
        """Register a horizontal plan slice (fixed-elevation observation frame)."""
        self._project.horizontal_slices.append(hslice)
        self._set_modified()
        self._db_write(lambda: self._pm.db.upsert_horizontal_slice(hslice))

    def remove_section(self, section: Section) -> None:
        self._project.sections.remove(section)
        self._set_modified()
        self._db_write(lambda: self._pm.db.delete_section(section.name))
        self.section_removed.emit(section)
        if self._active_section is section:
            fallback = self._project.sections[0] if self._project.sections else None
            self.set_active_section(fallback)

    def update_section(self, index: int, section: Section) -> None:
        """Replace the section at *index* with *section* in-place."""
        old = self._project.sections[index]
        self._project.sections[index] = section
        self._set_modified()
        self._db_write(lambda: self._pm.db.upsert_section(section))
        self.section_modified.emit(index, section)
        if self._active_section is old:
            self.set_active_section(section)

    def recompute_pick_display_coords(self, section_name: str) -> None:
        """Recompute distance_along for all picks/reference-lines on *section_name*.

        Called after a section node is moved.  Picks that have map coordinates
        stored are reprojected; picks without map coords (legacy) are untouched.
        Saves updated picks to the database.
        """
        sec = next((s for s in self._project.sections if s.name == section_name), None)
        if sec is None:
            return

        changed_h, changed_f = [], []
        for i, hp in enumerate(self._project.horizon_picks):
            if any(sn == section_name for sn in hp.section_names()):
                hp.recompute_distances(sec)
                changed_h.append(i)
        for i, fp in enumerate(self._project.fault_picks):
            if any(sn == section_name for sn in fp.section_names()):
                fp.recompute_distances(sec)
                changed_f.append(i)

        # Reproject vertical reference lines
        for rl in self._project.reference_lines:
            if rl.kind == "vertical" and rl.map_x is not None and rl.map_y is not None:
                new_dist, _ = sec.project_point(rl.map_x, rl.map_y)
                rl.value = new_dist

        # Persist
        for i in changed_h:
            hp = self._project.horizon_picks[i]
            self._db_write(lambda h=hp: self._pm.db.upsert_horizon(h))
        for i in changed_f:
            fp = self._project.fault_picks[i]
            self._db_write(lambda f=fp: self._pm.db.upsert_fault(f))
        if self._project.reference_lines:
            self._db_write(lambda: self._pm.db.replace_all_reference_lines(
                self._project.reference_lines))

    def set_active_section(self, section: Section | None) -> None:
        """Set the actively-viewed section; emits only on change."""
        if self._active_section is not section:
            self._active_section = section
            self.active_section_changed.emit(section)
            self._rebuild_topology()

    @property
    def active_slice(self):
        """The active workspace slice — a Section, a HorizontalSlice, or None."""
        return self._active_slice

    def set_active_slice(self, slice_) -> None:
        """Set the active workspace slice (Model A — additive, behaviour-preserving).

        A Section (or None) is delegated to :meth:`set_active_section`, so
        ``active_section`` and the entire existing section surface behave exactly
        as today. A HorizontalSlice leaves ``active_section`` untouched and only
        updates the parallel active-slice selection. Emits ``active_slice_changed``
        on change.
        """
        from section_tool.core.slices import HorizontalSlice
        if not isinstance(slice_, HorizontalSlice):
            # Section or None — delegate; the existing surface is unchanged.
            self.set_active_section(slice_)
        # (HorizontalSlice: active_section is deliberately left as-is.)
        if self._active_slice is not slice_:
            self._active_slice = slice_
            self.active_slice_changed.emit(slice_)

    # ------------------------------------------------------------------
    # Live topology management
    # ------------------------------------------------------------------

    def _rebuild_topology(self) -> None:
        """Rebuild the topology graph for the active section from current picks."""
        section = self._active_section
        if section is None:
            self._topology = None
            return

        proj = self._project
        sec_name = section.name
        total = section.total_length()

        # Compute max depth from all picks
        import numpy as np
        candidates = [5000.0]
        for hp in list(proj.horizon_picks) + list(proj.fault_picks):
            v = hp.depths[~np.isnan(hp.depths)]
            if len(v):
                candidates.append(float(v.max()))
        max_depth = max(candidates) * 1.25 + 500.0

        if self._topology is None or self._topology.section_name != sec_name:
            self._topology = SectionTopology(sec_name, total, max_depth)
        else:
            self._topology.update_bounds(total, max_depth)
            self._topology.clear_user_lines()

        # Add horizon lines
        for i, hp in enumerate(proj.horizon_picks):
            idxs = hp.section_indices(sec_name)
            if len(idxs) >= 2:
                coords = list(zip(hp._distances[idxs].tolist(),
                                  hp._depths[idxs].tolist()))
                self._topology.update_line(f"horizon_{i}", "horizon", coords)

        # Add fault lines
        for i, fp in enumerate(proj.fault_picks):
            idxs = fp.section_indices(sec_name)
            if len(idxs) >= 2:
                coords = list(zip(fp._distances[idxs].tolist(),
                                  fp._depths[idxs].tolist()))
                self._topology.update_line(f"fault_{i}", "fault", coords)

        # Add reference lines (horizontal, vertical)
        for i, rl in enumerate(proj.reference_lines):
            if not rl.visible:
                continue
            if rl.kind == "horizontal":
                self._topology.update_line(f"ref_{i}", "reference",
                                           [(0.0, rl.value), (total, rl.value)])
            elif rl.kind == "vertical":
                self._topology.update_line(f"ref_{i}", "reference",
                                           [(rl.value, 0.0), (rl.value, max_depth)])

        self.topology_changed.emit(sec_name)

    # ------------------------------------------------------------------
    # Polygon-bounds cascade
    # ------------------------------------------------------------------

    def _recompute_polygon_bounds(self, category: str, entity_idx: int) -> None:
        """Refresh cached vertices of any polygon referencing the changed entity.

        Called automatically by update_horizon_pick / update_fault_pick so that
        reference-based polygons stay in sync with their bounding entities.
        The updated _vertices are ready before the next render triggered by the
        horizon_pick_modified / fault_pick_modified signal.
        """
        sec = self._active_section
        sec_name = sec.name if sec is not None else ""
        proj = self._project
        for poly_idx, poly in enumerate(proj.polygons):
            if not poly.bounds:
                continue
            for b in poly.bounds:
                if b.category == category and b.index == entity_idx:
                    try:
                        new_verts = poly.compute_polygon_points(proj, sec_name)
                        poly._vertices = new_verts
                        poly.free_points = new_verts
                    except Exception:
                        pass
                    break

    # ------------------------------------------------------------------
    # Surfaces
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Surface public accessors
    # ------------------------------------------------------------------

    def get_surfaces(self) -> list:
        return list(self._project.surfaces)

    def get_visible_surfaces(self) -> list:
        return [s for s in self._project.surfaces if getattr(s, "visible", True)]

    @property
    def project_crs_epsg(self) -> int:
        return self._project.crs_epsg

    # ------------------------------------------------------------------
    # Restoration sequence
    # ------------------------------------------------------------------

    @property
    def restoration_sequence(self):
        return self._project.restoration_sequence

    def set_restoration_sequence(self, sequence) -> None:
        self._project.restoration_sequence = sequence
        self._set_modified()
        self._db_write(lambda: self._pm.db.set_restoration_sequence(sequence))

    # ------------------------------------------------------------------
    # Surface CRUD
    # ------------------------------------------------------------------

    def add_surface(self, surface: Surface) -> None:
        self._project.surfaces.append(surface)
        self._set_modified()
        self._db_write(lambda: self._save_surface_to_db(surface))
        self.surface_added.emit(surface)

    def remove_surface(self, surface: Surface) -> None:
        name = surface.name
        self._project.surfaces.remove(surface)
        self._set_modified()
        self._db_write(lambda: self._pm.db.delete_surface_by_name(name))
        self.surface_removed.emit(surface)

    def update_surface(self, index: int, surface: Surface) -> None:
        self._project.surfaces[index] = surface
        self._set_modified()
        self._db_write(lambda: self._save_surface_to_db(surface))
        self.surface_modified.emit(index, surface)

    def _save_surface_to_db(self, surface: Surface) -> None:
        """Persist surface metadata to DB and points to a .npy sidecar file."""
        if not self._pm.is_open:
            return
        surfaces_dir = os.path.join(self._pm.project_path, "surfaces")
        os.makedirs(surfaces_dir, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_" else "_"
                       for c in surface.name)
        rel = f"{safe}.npy"
        np.save(os.path.join(surfaces_dir, rel), surface.points)
        self._pm.db.upsert_surface(surface, points_file=rel)

    # ------------------------------------------------------------------
    # AOI
    # ------------------------------------------------------------------

    def set_aoi(self, aoi) -> None:
        """Set or clear the project AOI. Pass None to remove."""
        self._project.aoi = aoi
        self._set_modified()
        self._db_write(lambda: self._pm.db.upsert_aoi(aoi))
        self.aoi_changed.emit(aoi)

    # ------------------------------------------------------------------
    # Horizon picks
    # ------------------------------------------------------------------

    def add_horizon_pick(self, pick: HorizonPick) -> None:
        self._project.horizon_picks.append(pick)
        self._set_modified()
        self._db_write(lambda: self._pm.db.upsert_horizon(pick))
        self.horizon_pick_added.emit(pick)
        self._rebuild_topology()

    def remove_horizon_pick(self, pick: HorizonPick) -> None:
        self._project.horizon_picks.remove(pick)
        self._set_modified()
        self._db_write(lambda: self._pm.db.delete_horizon(pick.name))
        self.horizon_pick_removed.emit(pick)
        self._rebuild_topology()

    def update_horizon_pick(self, index: int, pick: HorizonPick) -> None:
        self._project.horizon_picks[index] = pick
        self._set_modified()
        self._db_write(lambda: self._pm.db.upsert_horizon(pick))
        self._recompute_polygon_bounds("Horizons", index)
        self.horizon_pick_modified.emit(index, pick)
        self._rebuild_topology()

    # ------------------------------------------------------------------
    # Wells
    # ------------------------------------------------------------------

    def add_well(self, well: Well) -> None:
        self._project.wells.append(well)
        self._set_modified()
        self._db_write(lambda: self._pm.db.upsert_well(well))
        self.well_added.emit(well)

    def remove_well(self, well: Well) -> None:
        self._project.wells.remove(well)
        self._set_modified()
        self._db_write(lambda: self._pm.db.delete_well(well.name))
        self.well_removed.emit(well)
        if self._active_well is well:
            self.set_active_well(None)

    def update_well(self, index: int, well: Well) -> None:
        old = self._project.wells[index]
        self._project.wells[index] = well
        self._set_modified()
        self._db_write(lambda: self._pm.db.upsert_well(well))
        self.well_modified.emit(index, well)
        if self._active_well is old:
            self.set_active_well(well)

    def set_active_well(self, well: Well | None) -> None:
        """Set the selected well; emits only on change."""
        if self._active_well is not well:
            self._active_well = well
            self.active_well_changed.emit(well)

    # ------------------------------------------------------------------
    # Seismic refs
    # ------------------------------------------------------------------

    def add_seismic_ref(self, ref: SeismicRef) -> None:
        self._project.seismic_refs.append(ref)
        self._set_modified()
        self._db_write(lambda: self._pm.db.upsert_seismic(ref))
        self.seismic_ref_added.emit(ref)

    def remove_seismic_ref(self, ref: SeismicRef) -> None:
        self._project.seismic_refs.remove(ref)
        self._set_modified()
        self._db_write(lambda: self._pm.db.delete_seismic(ref.name))
        self.seismic_ref_removed.emit(ref)

    # ------------------------------------------------------------------
    # Section polygons
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Fault picks
    # ------------------------------------------------------------------

    def add_fault_pick(self, pick: HorizonPick) -> None:
        self._project.fault_picks.append(pick)
        self._set_modified()
        self._db_write(lambda: self._pm.db.upsert_fault(pick))
        self.fault_pick_added.emit(pick)
        self._rebuild_topology()

    def remove_fault_pick(self, pick: HorizonPick) -> None:
        self._project.fault_picks.remove(pick)
        self._set_modified()
        self._db_write(lambda: self._pm.db.delete_fault(pick.name))
        self.fault_pick_removed.emit(pick)
        self._rebuild_topology()

    def update_fault_pick(self, index: int, pick: HorizonPick) -> None:
        self._project.fault_picks[index] = pick
        self._set_modified()
        self._db_write(lambda: self._pm.db.upsert_fault(pick))
        self._recompute_polygon_bounds("Faults", index)
        self.fault_pick_modified.emit(index, pick)
        self._rebuild_topology()

    def add_polygon(self, polygon: SectionPolygon) -> None:
        self._project.polygons.append(polygon)
        self._set_modified()
        self._db_write(lambda: self._pm.db.replace_all_polygons(self._project.polygons))
        self.polygon_added.emit(polygon)

    def remove_polygon(self, polygon: SectionPolygon) -> None:
        self._project.polygons.remove(polygon)
        self._set_modified()
        self._db_write(lambda: self._pm.db.replace_all_polygons(self._project.polygons))
        self.polygon_removed.emit(polygon)

    def update_polygon(self, index: int, polygon: SectionPolygon) -> None:
        self._project.polygons[index] = polygon
        self._set_modified()
        self._db_write(lambda: self._pm.db.replace_all_polygons(self._project.polygons))
        self.polygon_modified.emit(index, polygon)

    # ------------------------------------------------------------------
    # Reference lines
    # ------------------------------------------------------------------

    def add_reference_line(self, rl: ReferenceLine) -> None:
        self._project.reference_lines.append(rl)
        self._set_modified()
        self._db_write(lambda: self._pm.db.replace_all_reference_lines(
            self._project.reference_lines))
        self.reference_line_added.emit(rl)
        self._rebuild_topology()

    def remove_reference_line(self, rl: ReferenceLine) -> None:
        self._project.reference_lines.remove(rl)
        self._set_modified()
        self._db_write(lambda: self._pm.db.replace_all_reference_lines(
            self._project.reference_lines))
        self.reference_line_removed.emit(rl)
        self._rebuild_topology()

    def update_reference_line(self, index: int, rl: ReferenceLine) -> None:
        self._project.reference_lines[index] = rl
        self._set_modified()
        self._db_write(lambda: self._pm.db.replace_all_reference_lines(
            self._project.reference_lines))
        self.reference_line_modified.emit(index, rl)
        self._rebuild_topology()

    # ------------------------------------------------------------------
    # Annotations (Phase 6)
    # ------------------------------------------------------------------

    def add_annotation(self, ann: Annotation) -> None:
        self._project.annotations.append(ann)
        self._set_modified()
        self._db_write(lambda: self._pm.db.replace_all_annotations(
            self._project.annotations))
        self.annotation_added.emit(ann)

    def remove_annotation(self, ann: Annotation) -> None:
        self._project.annotations.remove(ann)
        self._set_modified()
        self._db_write(lambda: self._pm.db.replace_all_annotations(
            self._project.annotations))
        self.annotation_removed.emit(ann)

    def update_annotation(self, index: int, ann: Annotation) -> None:
        self._project.annotations[index] = ann
        self._set_modified()
        self._db_write(lambda: self._pm.db.replace_all_annotations(
            self._project.annotations))
        self.annotation_modified.emit(index, ann)

    # ------------------------------------------------------------------
    # Intersections (Phase 2)
    # ------------------------------------------------------------------

    def add_intersection(self, isc: FaultHorizonIntersection) -> None:
        self._project.intersections.append(isc)
        self._set_modified()
        self.intersection_added.emit(isc)

    def compute_and_store_intersections(self, section) -> list:
        from section_tool.core.intersection import compute_intersections
        new_ints = compute_intersections(
            section,
            self._project.horizon_picks,
            self._project.fault_picks,
        )
        # Remove old intersections for this section
        self._project.intersections = [
            i for i in self._project.intersections
            if i.section_name != section.name
        ]
        self.blockSignals(True)
        try:
            for isc in new_ints:
                self.add_intersection(isc)
        finally:
            self.blockSignals(False)
            if new_ints:
                self._set_modified()
        return new_ints

    # ------------------------------------------------------------------
    # Project properties
    # ------------------------------------------------------------------

    def set_project_properties(
        self,
        name: str,
        depth_units: str,
        depth_domain: str,
        default_depth_min: float,
        default_depth_max: float,
    ) -> None:
        """Update editable project-level properties and persist to the database."""
        proj = self._project
        proj.name = name
        proj.depth_units = depth_units
        proj.depth_domain = depth_domain
        proj.default_depth_min = default_depth_min
        proj.default_depth_max = default_depth_max
        self._set_modified()
        self._db_write(
            lambda: self._pm.db.set_project_settings(
                name=proj.name,
                crs_epsg=proj.crs_epsg,
                depth_units=proj.depth_units,
                depth_domain=proj.depth_domain,
                default_depth_min=proj.default_depth_min,
                default_depth_max=proj.default_depth_max,
            )
        )
        self.project_settings_changed.emit()
        self.project_changed.emit()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_modified(self, value: bool = True) -> None:
        if self._is_modified != value:
            self._is_modified = value
            self.project_modified_changed.emit(value)

    def __repr__(self) -> str:
        return (
            f"AppState(project={self._project.name!r}, "
            f"modified={self._is_modified}, "
            f"path={self._project_path!r}, "
            f"active_section={getattr(self._active_section, 'name', None)!r}, "
            f"active_well={getattr(self._active_well, 'name', None)!r})"
        )
