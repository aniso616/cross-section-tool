from __future__ import annotations

import os

from PySide6.QtCore import QObject, Signal

from cross_section_tool.core.annotation import Annotation
from cross_section_tool.core.command_stack import Command, CommandStack
from cross_section_tool.core.intersection import FaultHorizonIntersection
from cross_section_tool.core.polygons import SectionPolygon
from cross_section_tool.core.reference_line import ReferenceLine
from cross_section_tool.core.section import Section
from cross_section_tool.core.surfaces import HorizonPick, Surface
from cross_section_tool.core.topology import SectionTopology
from cross_section_tool.core.wells import Well
from cross_section_tool.io.project import Project, SeismicRef


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
    project_path_changed = Signal(str)   # "" means no file
    project_modified_changed = Signal(bool)

    # Sections
    section_added = Signal(object)
    section_removed = Signal(object)
    section_modified = Signal(int, object)
    active_section_changed = Signal(object)

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

    # ------------------------------------------------------------------

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._project: Project = Project()
        self._project_path: str | None = None
        self._active_section: Section | None = None
        self._active_well: Well | None = None
        self._is_modified: bool = False
        self._active_tool: str = "select"
        # Phase 7: command stack
        self._cmd_stack: CommandStack = CommandStack()
        self._active_pick_category: str | None = None
        self._active_pick_index: int | None = None
        # Live topology (one per active section)
        self._topology: SectionTopology | None = None

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

    def record_command(self, description: str,
                       undo, redo=None) -> None:
        """Phase 7: record an already-applied operation for undo/redo."""
        self._cmd_stack.push(Command(description=description,
                                     undo=undo, redo=redo or (lambda: None)))

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

    def new_project(self, name: str = "", crs_epsg: int = 32632) -> None:
        """Replace the current project with a fresh empty one."""
        self._project = Project(name=name, crs_epsg=crs_epsg)
        self._project_path = None
        self._active_section = None
        self._active_well = None
        self._is_modified = False
        self._topology = None
        self._cmd_stack.clear()
        self.project_path_changed.emit("")
        self.project_changed.emit()
        self.project_modified_changed.emit(False)

    def open_project(self, path: str | os.PathLike) -> None:
        """Load a project from *path* and replace the current state."""
        self._project = Project.load(path)
        self._project_path = str(path)
        self._active_section = None
        self._active_well = None
        self._is_modified = False
        self._topology = None
        self.project_path_changed.emit(self._project_path)
        self.project_changed.emit()
        self.project_modified_changed.emit(False)

    def save_project(self) -> None:
        """Save to the current :attr:`project_path`.

        Raises
        ------
        ValueError
            If no path has been set yet; call :meth:`save_project_as` first.
        """
        if self._project_path is None:
            raise ValueError("No project path set; use save_project_as() first")
        self._project.save(self._project_path)
        self._set_modified(False)

    def save_project_as(self, path: str | os.PathLike) -> None:
        """Save to *path* and update :attr:`project_path`."""
        self._project_path = str(path)
        self._project.save(self._project_path)
        self.project_path_changed.emit(self._project_path)
        self._set_modified(False)

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def add_section(self, section: Section) -> None:
        self._project.sections.append(section)
        self._set_modified()
        self.section_added.emit(section)

    def remove_section(self, section: Section) -> None:
        self._project.sections.remove(section)
        self._set_modified()
        self.section_removed.emit(section)
        if self._active_section is section:
            fallback = self._project.sections[0] if self._project.sections else None
            self.set_active_section(fallback)

    def update_section(self, index: int, section: Section) -> None:
        """Replace the section at *index* with *section* in-place."""
        old = self._project.sections[index]
        self._project.sections[index] = section
        self._set_modified()
        self.section_modified.emit(index, section)
        if self._active_section is old:
            self.set_active_section(section)

    def set_active_section(self, section: Section | None) -> None:
        """Set the actively-viewed section; emits only on change."""
        if self._active_section is not section:
            self._active_section = section
            self.active_section_changed.emit(section)
            self._rebuild_topology()

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
    # Surfaces
    # ------------------------------------------------------------------

    def add_surface(self, surface: Surface) -> None:
        self._project.surfaces.append(surface)
        self._set_modified()
        self.surface_added.emit(surface)

    def remove_surface(self, surface: Surface) -> None:
        self._project.surfaces.remove(surface)
        self._set_modified()
        self.surface_removed.emit(surface)

    def update_surface(self, index: int, surface: Surface) -> None:
        self._project.surfaces[index] = surface
        self._set_modified()
        self.surface_modified.emit(index, surface)

    # ------------------------------------------------------------------
    # Horizon picks
    # ------------------------------------------------------------------

    def add_horizon_pick(self, pick: HorizonPick) -> None:
        self._project.horizon_picks.append(pick)
        self._set_modified()
        self.horizon_pick_added.emit(pick)
        self._rebuild_topology()

    def remove_horizon_pick(self, pick: HorizonPick) -> None:
        self._project.horizon_picks.remove(pick)
        self._set_modified()
        self.horizon_pick_removed.emit(pick)
        self._rebuild_topology()

    def update_horizon_pick(self, index: int, pick: HorizonPick) -> None:
        self._project.horizon_picks[index] = pick
        self._set_modified()
        self.horizon_pick_modified.emit(index, pick)
        self._rebuild_topology()

    # ------------------------------------------------------------------
    # Wells
    # ------------------------------------------------------------------

    def add_well(self, well: Well) -> None:
        self._project.wells.append(well)
        self._set_modified()
        self.well_added.emit(well)

    def remove_well(self, well: Well) -> None:
        self._project.wells.remove(well)
        self._set_modified()
        self.well_removed.emit(well)
        if self._active_well is well:
            self.set_active_well(None)

    def update_well(self, index: int, well: Well) -> None:
        old = self._project.wells[index]
        self._project.wells[index] = well
        self._set_modified()
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
        self.seismic_ref_added.emit(ref)

    def remove_seismic_ref(self, ref: SeismicRef) -> None:
        self._project.seismic_refs.remove(ref)
        self._set_modified()
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
        self.fault_pick_added.emit(pick)
        self._rebuild_topology()

    def remove_fault_pick(self, pick: HorizonPick) -> None:
        self._project.fault_picks.remove(pick)
        self._set_modified()
        self.fault_pick_removed.emit(pick)
        self._rebuild_topology()

    def update_fault_pick(self, index: int, pick: HorizonPick) -> None:
        self._project.fault_picks[index] = pick
        self._set_modified()
        self.fault_pick_modified.emit(index, pick)
        self._rebuild_topology()

    def add_polygon(self, polygon: SectionPolygon) -> None:
        self._project.polygons.append(polygon)
        self._set_modified()
        self.polygon_added.emit(polygon)

    def remove_polygon(self, polygon: SectionPolygon) -> None:
        self._project.polygons.remove(polygon)
        self._set_modified()
        self.polygon_removed.emit(polygon)

    def update_polygon(self, index: int, polygon: SectionPolygon) -> None:
        self._project.polygons[index] = polygon
        self._set_modified()
        self.polygon_modified.emit(index, polygon)

    # ------------------------------------------------------------------
    # Reference lines
    # ------------------------------------------------------------------

    def add_reference_line(self, rl: ReferenceLine) -> None:
        self._project.reference_lines.append(rl)
        self._set_modified()
        self.reference_line_added.emit(rl)
        self._rebuild_topology()

    def remove_reference_line(self, rl: ReferenceLine) -> None:
        self._project.reference_lines.remove(rl)
        self._set_modified()
        self.reference_line_removed.emit(rl)
        self._rebuild_topology()

    def update_reference_line(self, index: int, rl: ReferenceLine) -> None:
        self._project.reference_lines[index] = rl
        self._set_modified()
        self.reference_line_modified.emit(index, rl)
        self._rebuild_topology()

    # ------------------------------------------------------------------
    # Annotations (Phase 6)
    # ------------------------------------------------------------------

    def add_annotation(self, ann: Annotation) -> None:
        self._project.annotations.append(ann)
        self._set_modified()
        self.annotation_added.emit(ann)

    def remove_annotation(self, ann: Annotation) -> None:
        self._project.annotations.remove(ann)
        self._set_modified()
        self.annotation_removed.emit(ann)

    def update_annotation(self, index: int, ann: Annotation) -> None:
        self._project.annotations[index] = ann
        self._set_modified()
        self.annotation_modified.emit(index, ann)

    # ------------------------------------------------------------------
    # Intersections (Phase 2)
    # ------------------------------------------------------------------

    def add_intersection(self, isc: FaultHorizonIntersection) -> None:
        self._project.intersections.append(isc)
        self._set_modified()
        self.intersection_added.emit(isc)

    def compute_and_store_intersections(self, section) -> list:
        from cross_section_tool.core.intersection import compute_intersections
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
