from __future__ import annotations

import os
import sys

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction, QCloseEvent, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStyle,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Collapse-strip widget (Phase 2)
# ---------------------------------------------------------------------------

class _CollapseStrip(QWidget):
    """16px-wide dark strip with a centred ◀/▶ arrow button.

    Emits :attr:`clicked` when the arrow button is pressed.
    Call :meth:`set_collapsed` to flip the arrow direction.
    """

    from PySide6.QtCore import Signal
    clicked = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(16)
        self.setStyleSheet("background: #383838;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._btn = QPushButton("◀")
        self._btn.setFlat(True)
        self._btn.setFixedSize(16, 32)
        self._btn.setStyleSheet(
            "QPushButton { background: transparent; color: #c8c8c8; border: none;"
            " font-size: 10px; }"
            "QPushButton:hover { color: white; background: #505050; }"
        )
        self._btn.clicked.connect(self.clicked)
        layout.addStretch()
        layout.addWidget(self._btn)
        layout.addStretch()

    def set_collapsed(self, collapsed: bool) -> None:
        self._btn.setText("▶" if collapsed else "◀")

from cross_section_tool.app_state import AppState
from cross_section_tool.core.section import Section
from cross_section_tool.core.surfaces import HorizonPick
from cross_section_tool.views.map_view import MapView
from cross_section_tool.views.project_panel import ProjectPanel
from cross_section_tool.views.section_view import SectionView
from cross_section_tool.views.tool_palette import ToolPalette
from cross_section_tool.views.viewer_3d import Viewer3D


class MainWindow(QMainWindow):
    """Top-level application window for the cross-section interpretation tool.

    Hosts a :class:`MapView` on the left and a tabbed
    :class:`SectionView` / :class:`Viewer3D` on the right, all driven by a
    shared :class:`AppState`.

    Parameters
    ----------
    state:
        Optional pre-built :class:`AppState`.  Mainly useful for testing;
        in production the default ``None`` creates a fresh one.
    """

    APP_NAME = "Cross Section Tool"
    APP_VERSION = "0.1.0"

    def __init__(self, state: AppState | None = None) -> None:
        super().__init__()
        self._state = state or AppState()
        self._setup_ui()
        self._build_menus()
        self._build_toolbar()
        self._connect_signals()
        self._update_title()
        self._update_status()
        self.resize(1280, 800)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setWindowTitle(self.APP_NAME)

        # Views
        self._map_view = MapView(self._state, self)
        self._section_view = SectionView(self._state, self)
        self._viewer_3d = Viewer3D(self._state, self)

        # Right-hand tab widget
        self._tabs = QTabWidget()
        self._tabs.addTab(self._section_view, "Section")
        self._tabs.addTab(self._viewer_3d, "3D View")

        # Map panel: view + collapse strip (always visible on right edge)
        self._map_view.setMinimumWidth(0)
        self._map_collapse_strip = _CollapseStrip(self)
        self._map_collapse_strip.clicked.connect(self._toggle_map_panel)
        self._map_collapse_strip.setToolTip("Collapse / expand map panel  (Ctrl+2)")
        self._map_panel_collapsed = False
        self._map_panel_width     = 320

        map_container = QWidget()
        map_hbox = QHBoxLayout(map_container)
        map_hbox.setContentsMargins(0, 0, 0, 0)
        map_hbox.setSpacing(0)
        map_hbox.addWidget(self._map_view, stretch=1)
        map_hbox.addWidget(self._map_collapse_strip)

        # Horizontal splitter (map_container | section/3D)
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.addWidget(map_container)
        self._splitter.addWidget(self._tabs)
        self._splitter.setSizes([320, 960])
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setHandleWidth(2)

        # Tool palette + splitter as the central widget
        self._tool_palette = ToolPalette(self)
        content = QWidget()
        hbox = QHBoxLayout(content)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(0)
        hbox.addWidget(self._tool_palette)
        hbox.addWidget(self._splitter)

        self.setCentralWidget(content)

        # Dockable project panel with custom dark title bar
        self._project_panel = ProjectPanel(self._state, self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._project_panel)
        self._setup_project_panel_title_bar()

        # Status bar
        self._status_label = QLabel("New project")
        self.statusBar().addWidget(self._status_label)

    def _setup_project_panel_title_bar(self) -> None:
        """Replace the dock's default title bar with a dark custom one."""
        title_bar = QWidget()
        title_bar.setStyleSheet("background: #383838;")
        tbl = QHBoxLayout(title_bar)
        tbl.setContentsMargins(6, 3, 4, 3)
        lbl = QLabel("Project")
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        lbl.setFont(font)
        lbl.setStyleSheet("color: #c8c8c8;")
        tbl.addWidget(lbl)
        tbl.addStretch()
        close_btn = QPushButton("◀")
        close_btn.setFlat(True)
        close_btn.setFixedSize(18, 18)
        close_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #c8c8c8; border: none; font-size: 10px; }"
            "QPushButton:hover { color: white; }"
        )
        close_btn.setToolTip("Hide Project panel  (Ctrl+1)")
        close_btn.clicked.connect(self._project_panel.close)
        tbl.addWidget(close_btn)
        self._project_panel.setTitleBarWidget(title_bar)

        # Re-open button (▶) in the dock's floating/re-dock action
        self._project_panel.visibilityChanged.connect(
            lambda visible: close_btn.setText("◀" if visible else "▶")
        )

    def _build_menus(self) -> None:
        mb = self.menuBar()

        # ---- File ----
        file_menu = mb.addMenu("&File")

        self._new_action = QAction("&New Project", self)
        self._new_action.setShortcut(QKeySequence.StandardKey.New)
        self._new_action.triggered.connect(self._on_new)
        file_menu.addAction(self._new_action)

        self._open_action = QAction("&Open Project…", self)
        self._open_action.setShortcut(QKeySequence.StandardKey.Open)
        self._open_action.triggered.connect(self._on_open)
        file_menu.addAction(self._open_action)

        file_menu.addSeparator()

        self._save_action = QAction("&Save", self)
        self._save_action.setShortcut(QKeySequence.StandardKey.Save)
        self._save_action.triggered.connect(self._on_save)
        file_menu.addAction(self._save_action)

        self._save_as_action = QAction("Save &As…", self)
        self._save_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self._save_as_action.triggered.connect(self._on_save_as)
        file_menu.addAction(self._save_as_action)

        file_menu.addSeparator()

        self._import_las_action = QAction("Import &LAS Well…", self)
        self._import_las_action.triggered.connect(self._on_import_las)
        file_menu.addAction(self._import_las_action)

        self._import_segy_action = QAction("Import Se&ismic (SEG-Y)…", self)
        self._import_segy_action.triggered.connect(self._on_import_segy)
        file_menu.addAction(self._import_segy_action)

        file_menu.addSeparator()

        self._exit_action = QAction("E&xit", self)
        self._exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self._exit_action.triggered.connect(self.close)
        file_menu.addAction(self._exit_action)

        # ---- Section ----
        section_menu = mb.addMenu("&Section")

        self._new_section_action = QAction("New Section (east–west default)", self)
        self._new_section_action.triggered.connect(self._on_new_section)
        section_menu.addAction(self._new_section_action)

        # ---- View ----
        view_menu = mb.addMenu("&View")

        self._pick_action = QAction("&Horizon Pick Mode", self)
        self._pick_action.setCheckable(True)
        # Shortcut now handled by tool palette (P key → horizon_pick)
        self._pick_action.toggled.connect(self._section_view.set_picking_active)
        view_menu.addAction(self._pick_action)

        view_menu.addSeparator()

        self._vd_action = QAction("Variable &Density Display", self)
        self._vd_action.triggered.connect(
            lambda: self._section_view.set_display_mode("variable_density")
        )
        view_menu.addAction(self._vd_action)

        self._wiggle_action = QAction("&Wiggle Display", self)
        self._wiggle_action.triggered.connect(
            lambda: self._section_view.set_display_mode("wiggle")
        )
        view_menu.addAction(self._wiggle_action)

        # ---- Reference Lines ----
        ref_menu = mb.addMenu("&Reference")
        self._add_hline_action = QAction("Add &Horizontal Line…", self)
        self._add_hline_action.triggered.connect(
            lambda: self._add_reference_line_kind("horizontal"))
        ref_menu.addAction(self._add_hline_action)
        self._add_vline_action = QAction("Add &Vertical Line…", self)
        self._add_vline_action.triggered.connect(
            lambda: self._add_reference_line_kind("vertical"))
        ref_menu.addAction(self._add_vline_action)

        # ---- Help ----
        help_menu = mb.addMenu("&Help")

        self._about_action = QAction("&About", self)
        self._about_action.triggered.connect(self._on_about)
        help_menu.addAction(self._about_action)

    def _build_toolbar(self) -> None:
        style = self.style()
        self._new_action.setIcon(
            style.standardIcon(QStyle.StandardPixmap.SP_FileIcon)
        )
        self._open_action.setIcon(
            style.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
        )
        self._save_action.setIcon(
            style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)
        )

        tb: QToolBar = self.addToolBar("Main")
        tb.setObjectName("MainToolBar")
        tb.setMovable(False)
        tb.setIconSize(QSize(20, 20))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        tb.addAction(self._new_action)
        tb.addAction(self._open_action)
        tb.addAction(self._save_action)

    def _connect_signals(self) -> None:
        s = self._state
        s.project_path_changed.connect(lambda _: self._update_title())
        s.project_modified_changed.connect(lambda _: self._update_title())
        s.project_changed.connect(self._update_title)
        s.project_changed.connect(self._update_status)
        s.project_modified_changed.connect(lambda _: self._update_status())
        s.project_path_changed.connect(lambda _: self._update_status())
        # Update status counts whenever sections or wells change
        s.section_added.connect(lambda _: self._update_status())
        s.section_removed.connect(lambda _: self._update_status())
        s.well_added.connect(lambda _: self._update_status())
        s.well_removed.connect(lambda _: self._update_status())
        # horizon_pick_requested removed — section view now writes directly to AppState
        self._section_view.polygon_finished.connect(self._state.add_polygon)
        self._tool_palette.tool_changed.connect(self._on_tool_changed)
        # Keep menu pick-action in sync with palette
        self._pick_action.toggled.connect(self._on_pick_action_toggled)
        # Project panel pick-target selection → auto-switch tool
        self._project_panel.pick_target_selected.connect(self._on_pick_target_selected)
        # Project panel → AppState mutations
        self._project_panel.object_deleted.connect(self._on_panel_delete)
        self._project_panel.object_renamed.connect(self._on_panel_rename)
        self._project_panel.object_color_changed.connect(self._on_panel_color)
        self._project_panel.object_line_width_changed.connect(self._on_panel_line_width)
        self._project_panel.object_line_style_changed.connect(self._on_panel_line_style)
        self._project_panel.add_requested.connect(self._on_panel_add)
        # Status bar from map view drag
        self._map_view.status_message.connect(self._on_map_status)
        # Status bar updates when tool or active pick target changes
        s.tool_changed.connect(lambda _: self._update_status())
        s.active_pick_target_changed.connect(lambda *_: self._update_status())
        # Keyboard shortcuts for tools (application-wide)
        _tool_keys = {
            "V": "select",   "H": "pan",          "Z": "zoom",
            "S": "new_section", "E": "edit_nodes",
            "P": "horizon_pick", "F": "fault_pick",
            "G": "polygon",  "M": "measure",
        }
        for key, tool_id in _tool_keys.items():
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(
                lambda tid=tool_id: self._tool_palette.set_active_tool(tid)
            )
        # Panel toggle shortcuts
        sc1 = QShortcut(QKeySequence("Ctrl+1"), self)
        sc1.activated.connect(self._project_panel.toggleViewAction().trigger)
        sc2 = QShortcut(QKeySequence("Ctrl+2"), self)
        sc2.activated.connect(self._toggle_map_panel)
        sc3 = QShortcut(QKeySequence("Ctrl+3"), self)
        sc3.activated.connect(self._toggle_section_panel)

    # ------------------------------------------------------------------
    # Title / status helpers
    # ------------------------------------------------------------------

    def _update_title(self, *_args) -> None:
        path = self._state.project_path
        name = os.path.basename(path) if path else "Untitled"
        prefix = "* " if self._state.is_modified else ""
        self.setWindowTitle(f"{prefix}{name} — {self.APP_NAME}")

    def _update_status(self, *_args) -> None:
        path = self._state.project_path
        msg = os.path.basename(path) if path else "New project"
        if self._state.is_modified:
            msg += "  [unsaved]"
        n_sec  = len(self._state.project.sections)
        n_well = len(self._state.project.wells)
        msg += f"  |  {n_sec} section(s)  {n_well} well(s)"
        cat = self._state.active_pick_category
        idx = self._state.active_pick_index
        if cat is not None and idx is not None:
            proj = self._state.project
            picks = proj.horizon_picks if cat == "Horizons" else proj.fault_picks
            if idx < len(picks):
                obj_name = picks[idx].name or f"{cat[:-1]} {idx + 1}"
                tool_label = self._state.active_tool.replace("_", " ").title()
                msg += f"  |  Active: {obj_name}  |  Tool: {tool_label}"
        self._status_label.setText(msg)

    # ------------------------------------------------------------------
    # Unsaved-changes guard
    # ------------------------------------------------------------------

    def _check_unsaved_changes(self) -> bool:
        """Return True if it is safe to discard the current project."""
        if not self._state.is_modified:
            return True
        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            "The project has unsaved changes.\nSave before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Save:
            return self._save_project()
        elif reply == QMessageBox.StandardButton.Discard:
            return True
        return False  # Cancel

    # ------------------------------------------------------------------
    # File operations (testable, no dialogs)
    # ------------------------------------------------------------------

    def _new_project(self, crs_epsg: int = 32632) -> None:
        """Create a fresh project (no dialog)."""
        self._state.new_project(crs_epsg=crs_epsg)

    def _open_project(self, path: str) -> bool:
        """Load a project from *path*. Returns True on success, False on error (no dialog)."""
        try:
            self._state.open_project(path)
            return True
        except Exception:
            return False

    def _save_project(self) -> bool:
        """Save to the current path. Opens a dialog if no path is set."""
        if self._state.project_path is None:
            return self._save_project_as_dialog()
        try:
            self._state.save_project()
            return True
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))
            return False

    def _save_project_as(self, path: str) -> bool:
        """Save to *path* (no dialog). Returns True on success, False on error."""
        try:
            self._state.save_project_as(path)
            return True
        except Exception:
            return False

    def _save_project_as_dialog(self) -> bool:
        """Open a Save As dialog and save."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "", "HDF5 Project (*.h5);;All Files (*)"
        )
        if not path:
            return False
        ok = self._save_project_as(path)
        if not ok:
            QMessageBox.critical(self, "Save Error", f"Could not save:\n{path}")
        return ok

    # ------------------------------------------------------------------
    # Action slots (with dialogs)
    # ------------------------------------------------------------------

    def _on_new(self) -> None:
        if self._check_unsaved_changes():
            self._new_project()

    def _on_open(self) -> None:
        if not self._check_unsaved_changes():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "HDF5 Project (*.h5);;All Files (*)"
        )
        if path:
            if not self._open_project(path):
                QMessageBox.critical(self, "Open Error", f"Could not open:\n{path}")

    def _on_save(self) -> None:
        self._save_project()

    def _on_save_as(self) -> None:
        self._save_project_as_dialog()

    def _on_import_las(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import LAS Files", "", "LAS Files (*.las *.LAS);;All Files (*)"
        )
        for path in paths:
            try:
                from cross_section_tool.io.las import read_las
                well = read_las(path, crs_epsg=self._state.project.crs_epsg)
                self._state.add_well(well)
            except Exception as exc:
                QMessageBox.warning(self, "Import Warning", f"{path}:\n{exc}")

    def _on_import_segy(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import SEG-Y Files",
            "",
            "SEG-Y Files (*.segy *.sgy *.SGY);;All Files (*)",
        )
        for path in paths:
            from cross_section_tool.io.project import SeismicRef
            ref = SeismicRef(
                path=path,
                name=os.path.splitext(os.path.basename(path))[0],
                crs_epsg=self._state.project.crs_epsg,
            )
            self._state.add_seismic_ref(ref)

    def _on_new_section(self) -> None:
        """Add a simple 10-km east–west section centred on existing data."""
        existing = self._state.project.sections
        if existing:
            # Offset 1 km north of the last section's first node
            x0 = float(existing[-1].nodes[0, 0])
            y0 = float(existing[-1].nodes[0, 1]) + 1000.0
        else:
            x0, y0 = 0.0, 0.0
        sec = Section(
            [(x0, y0), (x0 + 10_000.0, y0)],
            name=f"Section {len(existing) + 1}",
            crs_epsg=self._state.project.crs_epsg,
        )
        self._state.add_section(sec)
        self._state.set_active_section(sec)

    def _on_pick_requested(self, distance: float, depth: float) -> None:
        """Append a point to the last horizon pick, or create a new one."""
        picks = self._state.project.horizon_picks
        if picks:
            last = picks[-1]
            last.insert_pick(distance, depth)
            self._state.update_horizon_pick(len(picks) - 1, last)
        else:
            pick = HorizonPick(
                [distance], [depth],
                name=f"Horizon {len(picks) + 1}",
            )
            self._state.add_horizon_pick(pick)

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            f"About {self.APP_NAME}",
            f"<b>{self.APP_NAME}</b> v{self.APP_VERSION}<br><br>"
            "A desktop geoscience cross-section interpretation tool.<br>"
            "Built with PySide6, Matplotlib, and PyVista.",
        )

    def _on_panel_delete(self, category: str, index: int) -> None:
        proj = self._state.project
        try:
            if category == "Sections" and index < len(proj.sections):
                self._state.remove_section(proj.sections[index])
            elif category == "Horizons" and index < len(proj.horizon_picks):
                self._state.remove_horizon_pick(proj.horizon_picks[index])
            elif category == "Faults" and index < len(proj.fault_picks):
                self._state.remove_fault_pick(proj.fault_picks[index])
            elif category == "Reference Lines" and index < len(proj.reference_lines):
                self._state.remove_reference_line(proj.reference_lines[index])
        except Exception:
            pass

    def _on_panel_rename(self, category: str, index: int, name: str) -> None:
        import copy
        proj = self._state.project
        try:
            if category == "Sections" and index < len(proj.sections):
                sec = copy.deepcopy(proj.sections[index])
                sec.name = name
                self._state.update_section(index, sec)
            elif category == "Horizons" and index < len(proj.horizon_picks):
                pick = copy.deepcopy(proj.horizon_picks[index])
                pick.name = name
                self._state.update_horizon_pick(index, pick)
            elif category == "Faults" and index < len(proj.fault_picks):
                pick = copy.deepcopy(proj.fault_picks[index])
                pick.name = name
                self._state.update_fault_pick(index, pick)
        except Exception:
            pass

    def _on_panel_color(self, category: str, index: int, color: str) -> None:
        import copy
        proj = self._state.project
        try:
            if category == "Horizons" and index < len(proj.horizon_picks):
                pick = copy.deepcopy(proj.horizon_picks[index])
                pick.color = color
                self._state.update_horizon_pick(index, pick)
            elif category == "Faults" and index < len(proj.fault_picks):
                pick = copy.deepcopy(proj.fault_picks[index])
                pick.color = color
                self._state.update_fault_pick(index, pick)
        except Exception:
            pass

    def _on_panel_line_width(self, category: str, index: int, width: float) -> None:
        import copy
        proj = self._state.project
        try:
            if category == "Horizons" and index < len(proj.horizon_picks):
                pick = copy.deepcopy(proj.horizon_picks[index])
                pick.line_width = width
                self._state.update_horizon_pick(index, pick)
            elif category == "Faults" and index < len(proj.fault_picks):
                pick = copy.deepcopy(proj.fault_picks[index])
                pick.line_width = width
                self._state.update_fault_pick(index, pick)
        except Exception:
            pass

    def _on_panel_line_style(self, category: str, index: int, style: str) -> None:
        import copy
        proj = self._state.project
        try:
            if category == "Horizons" and index < len(proj.horizon_picks):
                pick = copy.deepcopy(proj.horizon_picks[index])
                pick.line_style = style
                self._state.update_horizon_pick(index, pick)
            elif category == "Faults" and index < len(proj.fault_picks):
                pick = copy.deepcopy(proj.fault_picks[index])
                pick.line_style = style
                self._state.update_fault_pick(index, pick)
        except Exception:
            pass

    def _on_panel_add(self, category: str) -> None:
        if category == "Sections":
            self._on_new_section()
        elif category == "Horizons":
            self._add_new_horizon()
        elif category == "Faults":
            self._add_new_fault()
        elif category == "Reference Lines":
            self._add_reference_line_dialog()

    def _add_new_horizon(self) -> None:
        from PySide6.QtWidgets import QInputDialog, QColorDialog
        name, ok = QInputDialog.getText(self, "New Horizon", "Horizon name:",
                                        text=f"Horizon {len(self._state.project.horizon_picks)+1}")
        if not ok or not name.strip():
            return
        color = QColorDialog.getColor(parent=self)
        col = color.name() if color.isValid() else "#2ca02c"
        from cross_section_tool.core.surfaces import HorizonPick
        hp = HorizonPick.empty(name=name.strip(), color=col)
        self._state.add_horizon_pick(hp)
        idx = len(self._state.project.horizon_picks) - 1
        self._state.set_active_pick_target("Horizons", idx)
        self._tool_palette.set_active_tool("horizon_pick")

    def _add_reference_line_kind(self, kind: str) -> None:
        from PySide6.QtWidgets import QInputDialog
        from cross_section_tool.core.reference_line import ReferenceLine
        label = "Depth value:" if kind == "horizontal" else "Distance along section:"
        value, ok = QInputDialog.getDouble(self, "Reference Line", label, 0.0)
        if not ok:
            return
        rl = ReferenceLine(kind=kind, value=value)
        self._state.add_reference_line(rl)

    def _add_reference_line_dialog(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        from cross_section_tool.core.reference_line import ReferenceLine
        kinds = ["Horizontal (depth)", "Vertical (distance)"]
        kind_str, ok = QInputDialog.getItem(self, "Reference Line", "Type:", kinds, 0, False)
        if not ok:
            return
        kind = "horizontal" if "Horizontal" in kind_str else "vertical"
        label = "Depth value:" if kind == "horizontal" else "Distance along section:"
        value, ok2 = QInputDialog.getDouble(self, "Reference Line", label, 0.0)
        if not ok2:
            return
        name, ok3 = QInputDialog.getText(self, "Reference Line", "Name (optional):", text="")
        rl = ReferenceLine(kind=kind, value=value, name=name.strip())
        self._state.add_reference_line(rl)

    def _add_new_fault(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "New Fault", "Fault name:",
                                        text=f"Fault {len(self._state.project.fault_picks)+1}")
        if not ok or not name.strip():
            return
        from cross_section_tool.core.surfaces import HorizonPick
        fp = HorizonPick.empty(name=name.strip(), color="#d62728")
        self._state.add_fault_pick(fp)
        idx = len(self._state.project.fault_picks) - 1
        self._state.set_active_pick_target("Faults", idx)
        self._tool_palette.set_active_tool("fault_pick")

    def _toggle_map_panel(self) -> None:
        """Collapse / expand the map panel."""
        sizes = self._splitter.sizes()
        if self._map_panel_collapsed:
            # Restore: show map view again
            self._map_view.show()
            self._splitter.setSizes([self._map_panel_width, sizes[1]])
            self._map_collapse_strip.set_collapsed(False)
            self._map_panel_collapsed = False
        else:
            # Collapse: hide map view, container shrinks to strip width (16px)
            self._map_panel_width = sizes[0] or 320
            self._map_view.hide()
            total = sizes[0] + sizes[1]
            self._splitter.setSizes([16, total - 16])
            self._map_collapse_strip.set_collapsed(True)
            self._map_panel_collapsed = True

    def _toggle_section_panel(self) -> None:
        """Collapse / expand the section/3D tabs panel."""
        sizes = self._splitter.sizes()
        if sizes[1] == 0:
            self._splitter.setSizes([sizes[0], 960])
        else:
            self._splitter.setSizes([sizes[0] + sizes[1], 0])

    def _on_map_status(self, msg: str) -> None:
        if msg:
            self._status_label.setText(msg)
        else:
            self._update_status()

    def _on_pick_target_selected(self, cat: str, idx: int) -> None:
        """Clicking a horizon/fault in the panel also activates the matching tool."""
        self._state.set_active_pick_target(cat, idx)
        if cat == "Horizons":
            self._tool_palette.set_active_tool("horizon_pick")
        elif cat == "Faults":
            self._tool_palette.set_active_tool("fault_pick")

    def _on_tool_changed(self, tool_id: str) -> None:
        """Route palette tool activation to views and AppState."""
        self._state.set_active_tool(tool_id)
        self._section_view.set_picking_active(tool_id == "horizon_pick")
        self._section_view.set_fault_picking(tool_id == "fault_pick")
        self._section_view.set_polygon_drawing(tool_id == "polygon")
        # Keep menu action in sync without triggering a re-entry loop
        self._pick_action.blockSignals(True)
        self._pick_action.setChecked(tool_id == "horizon_pick")
        self._pick_action.blockSignals(False)

    def _on_pick_action_toggled(self, checked: bool) -> None:
        """Sync the View-menu pick action back to the tool palette."""
        target = "horizon_pick" if checked else "select"
        if self._tool_palette.active_tool != target:
            self._tool_palette.set_active_tool(target)

    # ------------------------------------------------------------------
    # Close event
    # ------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._check_unsaved_changes():
            event.accept()
        else:
            event.ignore()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Start the application and return the exit code."""
    if argv is None:
        argv = sys.argv
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName(MainWindow.APP_NAME)
    app.setApplicationVersion(MainWindow.APP_VERSION)
    app.setOrganizationName("Geoscience")
    app.setStyleSheet("QToolTip { padding: 4px; }")
    # 500ms tooltip delay
    from PySide6.QtWidgets import QToolTip
    from PySide6.QtGui import QFont
    QToolTip.setFont(QFont("Segoe UI", 9))

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
