from __future__ import annotations

import os
import sys
import traceback
from contextlib import contextmanager


@contextmanager
def _wait_cursor():
    """Show the system wait cursor while a slow operation runs."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    QApplication.processEvents()
    try:
        yield
    finally:
        QApplication.restoreOverrideCursor()


def _global_exception_handler(exc_type, exc_value, exc_tb):
    """Show an error dialog instead of crashing on unhandled exceptions."""
    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print(f"UNHANDLED EXCEPTION:\n{error_msg}", file=sys.stderr)
    try:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(
            None, "Unexpected Error",
            f"An error occurred:\n\n{exc_value}\n\n"
            "The application may be unstable. Save your work.",
        )
    except Exception:
        pass


sys.excepthook = _global_exception_handler

from PySide6.QtCore import Qt, QSize, QSettings, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
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
# Geological colour palettes — auto-assigned on new horizon/fault (Phase 3)
# ---------------------------------------------------------------------------
_HORIZON_COLORS = [
    "#4169E1", "#228B22", "#FF8C00", "#B22222", "#9467BD",
    "#17BEBB", "#FFD700", "#2CA02C", "#1F77B4", "#FF7F0E",
    "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF", "#D62728",
]
_FAULT_COLORS = [
    "#DC1414", "#B22222", "#FF4500", "#8B0000", "#CD5C5C", "#E9967A",
]


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

from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick
from section_tool.views.map_view import MapView
from section_tool.views.project_panel import ProjectPanel
from section_tool.views.section_view import SectionView
from section_tool.views.tool_palette import ToolPalette
from section_tool.views.viewer_3d import Viewer3D


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

    APP_NAME = "Section"
    APP_VERSION = "0.1.0"

    # Central shortcut registry — single source of truth for both
    # QShortcut registration and the Help → Keyboard Shortcuts dialog.
    # Space is handled by keyPressEvent/keyReleaseEvent (hold semantics),
    # so it is documented here but not in the QShortcut objects.
    SHORTCUT_REGISTRY: list[tuple[str, str, str]] = [
        # (key_sequence, description, category)
        ("V",            "Select Object",            "Navigation"),
        ("A",            "Direct Select (Nodes)",    "Navigation"),
        ("H",            "Pan",                      "Navigation"),
        ("Z",            "Zoom",                     "Navigation"),
        ("Space",        "Temporary Pan (hold)",     "Navigation"),    # keyPressEvent
        ("P",            "Horizon Pick",             "Interpretation"),
        ("F",            "Fault Pick",               "Interpretation"),
        ("G",            "Polygon",                  "Interpretation"),
        ("S",            "Section Draw",             "Interpretation"),
        ("R",            "Reference Line (cycle)",   "Construction"),
        ("M",            "Measure",                  "Construction"),
        ("Delete",       "Delete Selected",          "Editing"),
        ("Ctrl+Z",       "Undo",                     "Editing"),
        ("Ctrl+Shift+Z", "Redo",                     "Editing"),
        ("Ctrl+C",       "Copy",                     "Editing"),
        ("Ctrl+V",       "Paste",                    "Editing"),
        ("Ctrl+A",       "Select All",               "Editing"),
        ("Escape",       "Cancel / Deselect",        "Editing"),
        ("Ctrl+N",       "New Project",              "File"),
        ("Ctrl+O",       "Open Project",             "File"),
        ("Ctrl+S",       "Save Project",             "File"),
        ("Ctrl+0",       "Zoom to All Data",         "View"),
        ("Ctrl+1",       "Toggle Map Panel",         "View"),
        ("Ctrl+2",       "Toggle Section Panel",     "View"),
        ("Ctrl+3",       "Toggle 3D View",           "View"),
        ("Ctrl+4",       "Toggle Project Panel",     "View"),
        ("Ctrl+5",       "Toggle Properties Panel",  "View"),
    ]

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
        # Auto-save every 5 minutes
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(5 * 60 * 1000)
        self._autosave_timer.timeout.connect(self._on_autosave)
        self._autosave_timer.start()
        # Restore saved layout if available
        _s = QSettings("Geoscience", "CrossSectionTool")
        _geom = _s.value("window/geometry")
        _state = _s.value("window/state")
        if _geom is not None:
            self.restoreGeometry(_geom)
        if _state is not None:
            self.restoreState(_state)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setWindowTitle(self.APP_NAME)

        # Allow panels to be tiled, tabbed, and floated freely
        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
            | QMainWindow.DockOption.AllowNestedDocks
        )

        # ── Views ─────────────────────────────────────────────────────────────
        self._map_view     = MapView(self._state, self)
        self._section_view = SectionView(self._state, self)
        self._viewer_3d    = Viewer3D(self._state, self)

        # ── Minimal central widget — dock widgets fill all usable space ────────
        _central = QWidget()
        _central.setMaximumSize(0, 0)
        self.setCentralWidget(_central)

        # ── Tool palette — fixed QToolBar on left edge, never floats ──────────
        self._tool_palette = ToolPalette()
        self._tool_tb = QToolBar("Tools")
        self._tool_tb.setObjectName("ToolPaletteTB")
        self._tool_tb.setMovable(False)
        self._tool_tb.setFloatable(False)
        self._tool_tb.setOrientation(Qt.Orientation.Vertical)
        self._tool_tb.setStyleSheet(
            "QToolBar#ToolPaletteTB { background: #f0f0f0; "
            "border-right: 1px solid #c8c8c8; padding: 0; spacing: 0; }"
        )
        self._tool_tb.addWidget(self._tool_palette)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self._tool_tb)

        # ── Options bar — full-width QToolBar below main toolbar ──────────────
        self._build_options_bar()

        # ── Map dock ──────────────────────────────────────────────────────────
        self._map_dock = QDockWidget("Map", self)
        self._map_dock.setObjectName("MapDock")
        self._map_dock.setWidget(self._map_view)
        self._map_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self._map_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )

        # ── Section dock ──────────────────────────────────────────────────────
        self._section_dock = QDockWidget("Section", self)
        self._section_dock.setObjectName("SectionDock")
        self._section_dock.setWidget(self._section_view)
        self._section_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self._section_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )

        # ── 3D View dock ──────────────────────────────────────────────────────
        self._view3d_dock = QDockWidget("3D View", self)
        self._view3d_dock.setObjectName("View3DDock")
        self._view3d_dock.setWidget(self._viewer_3d)
        self._view3d_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self._view3d_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )

        # ── Project panel ─────────────────────────────────────────────────────
        self._project_panel = ProjectPanel(self._state, self)
        self._project_panel.setObjectName("ProjectDock")
        self._project_panel.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self._setup_project_panel_title_bar()

        # ── Properties panel ──────────────────────────────────────────────────
        from section_tool.views.properties_panel import PropertiesPanel
        self._properties_panel = PropertiesPanel(self._state, self)
        self._properties_panel.setObjectName("PropertiesDock")
        self._properties_panel.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)

        # ── Apply default dock layout ─────────────────────────────────────────
        self._apply_default_dock_layout()

        # ── Status bar ────────────────────────────────────────────────────────
        self.statusBar().setStyleSheet(
            "QStatusBar { background: #f8f8f8; border-top: 1px solid #ddd; "
            "font-size: 8pt; }"
        )
        self._status_label = QLabel("New project")
        self.statusBar().addWidget(self._status_label, 1)
        self._hint_label = QLabel("")
        self._hint_label.setStyleSheet("color: #888; font-style: italic;")
        self.statusBar().addPermanentWidget(self._hint_label)

    def _apply_default_dock_layout(self) -> None:
        """Establish the default dock arrangement (Map/Project/Props left, Section right)."""
        # Left column: Map on top, Project in middle, Properties at bottom
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._map_dock)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._project_panel)
        self.splitDockWidget(self._map_dock, self._project_panel,
                             Qt.Orientation.Vertical)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._properties_panel)
        self.splitDockWidget(self._project_panel, self._properties_panel,
                             Qt.Orientation.Vertical)
        # Right area: Section primary, 3D View tabbed behind it
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._section_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._view3d_dock)
        self.tabifyDockWidget(self._section_dock, self._view3d_dock)
        self._section_dock.raise_()
        # Default left-column width
        self.resizeDocks([self._map_dock], [400], Qt.Orientation.Horizontal)

    def _reset_layout(self) -> None:
        """Restore the default dock arrangement and clear saved state."""
        settings = QSettings("Geoscience", "CrossSectionTool")
        settings.remove("window/state")
        for dock in (self._map_dock, self._section_dock, self._view3d_dock,
                     self._project_panel, self._properties_panel):
            dock.setFloating(False)
        self._apply_default_dock_layout()

    def _build_options_bar(self) -> None:
        """Phase 3 — full-width context-sensitive options bar (top QToolBar)."""
        from section_tool.views.context_toolbar import ContextToolbar
        self._options_bar_tb = QToolBar("Options")
        self._options_bar_tb.setObjectName("OptionsBar")
        self._options_bar_tb.setMovable(False)
        self._options_bar_tb.setFloatable(False)
        self._options_bar_tb.setStyleSheet(
            "QToolBar#OptionsBar { background: #e8e8e8; "
            "border-bottom: 1px solid #ccc; padding: 0 4px; spacing: 4px; }"
        )
        self._options_bar_tb.setFixedHeight(32)
        self._ctx = ContextToolbar(self._state)
        self._ctx.action_requested.connect(self._on_context_toolbar_action)
        # Route picking actions to section_view; skip creation actions handled by app
        self._ctx.action_requested.connect(
            lambda a: self._section_view._on_context_action(a)
            if a not in ("new_horizon", "new_fault")
            and hasattr(self._section_view, "_on_context_action") else None
        )
        self._options_bar_tb.addWidget(self._ctx)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self._options_bar_tb)

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

        # ================================================================
        # File
        # ================================================================
        file_menu = mb.addMenu("&File")

        self._new_action = QAction("&New Project\tCtrl+N", self)
        self._new_action.triggered.connect(self._on_new)
        file_menu.addAction(self._new_action)

        self._open_action = QAction("&Open Project…\tCtrl+O", self)
        self._open_action.triggered.connect(self._on_open)
        file_menu.addAction(self._open_action)

        file_menu.addSeparator()

        self._save_action = QAction("&Save\tCtrl+S", self)
        self._save_action.triggered.connect(self._on_save)
        file_menu.addAction(self._save_action)

        self._save_as_action = QAction("Save &As…\tCtrl+Shift+S", self)
        self._save_as_action.triggered.connect(self._on_save_as)
        file_menu.addAction(self._save_as_action)

        file_menu.addSeparator()

        self._import_las_action = QAction("Import &LAS Well…", self)
        self._import_las_action.triggered.connect(self._on_import_las)
        file_menu.addAction(self._import_las_action)

        # Import submenu
        import_menu = QMenu("&Import", self)
        self._import_las_action = QAction("LAS Well Log…", self)
        self._import_las_action.triggered.connect(self._on_import_las)
        import_menu.addAction(self._import_las_action)
        self._import_well_tops_action = QAction("Well Tops CSV…", self)
        self._import_well_tops_action.triggered.connect(self._on_import_well_tops)
        import_menu.addAction(self._import_well_tops_action)
        import_menu.addSeparator()
        self._import_segy_action = QAction("SEG-Y Seismic…", self)
        self._import_segy_action.triggered.connect(self._on_import_segy)
        import_menu.addAction(self._import_segy_action)
        self._import_img_action = QAction("Section Image…", self)
        self._import_img_action.triggered.connect(self._on_import_section_image)
        import_menu.addAction(self._import_img_action)
        import_menu.addSeparator()
        self._import_topo_action = QAction("Topography Profile (CSV)…", self)
        self._import_topo_action.triggered.connect(self._on_import_topography)
        import_menu.addAction(self._import_topo_action)
        file_menu.addMenu(import_menu)

        # Export submenu
        export_menu = QMenu("&Export", self)
        self._export_img_action = QAction("Section Image (PNG/SVG/PDF)…", self)
        self._export_img_action.triggered.connect(self._on_export_section_image)
        export_menu.addAction(self._export_img_action)
        self._export_csv_action = QAction("Horizons to CSV…", self)
        self._export_csv_action.triggered.connect(self._on_export_horizons_csv)
        export_menu.addAction(self._export_csv_action)
        file_menu.addMenu(export_menu)

        file_menu.addSeparator()
        self._exit_action = QAction("E&xit", self)
        self._exit_action.triggered.connect(self.close)
        file_menu.addAction(self._exit_action)

        # ================================================================
        # Edit
        # ================================================================
        edit_menu = mb.addMenu("&Edit")
        undo_a = QAction("&Undo\tCtrl+Z", self)
        undo_a.triggered.connect(self._state.undo)
        edit_menu.addAction(undo_a)
        redo_a = QAction("&Redo\tCtrl+Shift+Z", self)
        redo_a.triggered.connect(self._state.redo)
        edit_menu.addAction(redo_a)
        edit_menu.addSeparator()
        selall_a = QAction("Select &All\tCtrl+A", self)
        edit_menu.addAction(selall_a)

        # ================================================================
        # View
        # ================================================================
        view_menu = mb.addMenu("&View")
        # Dock panel toggles — shortcuts registered as ApplicationShortcut QShortcuts
        # in _register_shortcuts(); the \t hints show them in the menu.
        _map_ta = self._map_dock.toggleViewAction()
        _map_ta.setText(_map_ta.text() + "\tCtrl+1")
        view_menu.addAction(_map_ta)
        _sec_ta = self._section_dock.toggleViewAction()
        _sec_ta.setText(_sec_ta.text() + "\tCtrl+2")
        view_menu.addAction(_sec_ta)
        _3d_ta = self._view3d_dock.toggleViewAction()
        _3d_ta.setText(_3d_ta.text() + "\tCtrl+3")
        view_menu.addAction(_3d_ta)
        view_menu.addSeparator()
        _proj_ta = self._project_panel.toggleViewAction()
        _proj_ta.setText(_proj_ta.text() + "\tCtrl+4")
        view_menu.addAction(_proj_ta)
        _props_ta = self._properties_panel.toggleViewAction()
        _props_ta.setText(_props_ta.text() + "\tCtrl+5")
        view_menu.addAction(_props_ta)
        view_menu.addSeparator()
        reset_layout_a = QAction("&Reset Layout", self)
        reset_layout_a.triggered.connect(self._reset_layout)
        view_menu.addAction(reset_layout_a)
        view_menu.addSeparator()
        zfit_a = QAction("Zoom to &Fit\tCtrl+0", self)
        zfit_a.triggered.connect(self._zoom_to_fit)
        view_menu.addAction(zfit_a)
        view_menu.addSeparator()
        self._vd_action = QAction("Variable &Density Display", self)
        self._vd_action.triggered.connect(
            lambda: self._section_view.set_display_mode("variable_density"))
        view_menu.addAction(self._vd_action)
        self._wiggle_action = QAction("&Wiggle Display", self)
        self._wiggle_action.triggered.connect(
            lambda: self._section_view.set_display_mode("wiggle"))
        view_menu.addAction(self._wiggle_action)
        view_menu.addSeparator()
        self._strat_col_action = QAction("Stratigraphic &Column", self)
        self._strat_col_action.setCheckable(True)
        self._strat_col_action.setChecked(True)
        self._strat_col_action.toggled.connect(
            lambda v: self._section_view.set_strat_column_visible(v))
        view_menu.addAction(self._strat_col_action)
        self._sea_level_action = QAction("Show &Sea Level", self)
        self._sea_level_action.setCheckable(True)
        self._sea_level_action.setChecked(True)
        self._sea_level_action.toggled.connect(
            lambda v: self._section_view.set_sea_level_visible(v))
        view_menu.addAction(self._sea_level_action)
        view_menu.addSeparator()
        self._fps_action = QAction("Show &FPS", self)
        self._fps_action.setCheckable(True)
        self._fps_action.setChecked(False)
        self._fps_action.toggled.connect(
            lambda v: self._section_view.set_fps_display(v))
        view_menu.addAction(self._fps_action)

        # ================================================================
        # Section
        # ================================================================
        section_menu = mb.addMenu("Se&ction")
        self._new_section_action = QAction("New Section (draw on map)  S", self)
        self._new_section_action.triggered.connect(
            lambda: self._tool_palette.set_active_tool("new_section"))
        section_menu.addAction(self._new_section_action)
        ew_action = QAction("New Section (east–west default)", self)
        ew_action.triggered.connect(self._on_new_section)
        section_menu.addAction(ew_action)
        ns_action = QAction("New Section (north–south)…", self)
        ns_action.triggered.connect(self._on_new_section_ns)
        section_menu.addAction(ns_action)
        ud_action = QAction("New Section (user defined)…", self)
        ud_action.triggered.connect(self._on_new_section_user_defined)
        section_menu.addAction(ud_action)
        section_menu.addSeparator()
        self._gen_polygons_action = QAction("Generate Polygons From Boundaries…", self)
        self._gen_polygons_action.triggered.connect(self._on_generate_polygons)
        section_menu.addAction(self._gen_polygons_action)
        section_menu.addSeparator()
        self._strat_column_action = QAction("Edit Stratigraphic Column…", self)
        self._strat_column_action.triggered.connect(self._on_edit_strat_column)
        section_menu.addAction(self._strat_column_action)
        section_menu.addSeparator()
        self._extract_seismic_action = QAction("Extract Seismic Along Active Section…", self)
        self._extract_seismic_action.triggered.connect(self._on_extract_seismic_for_section)
        section_menu.addAction(self._extract_seismic_action)

        # ================================================================
        # Interpret
        # ================================================================
        interp_menu = mb.addMenu("&Interpret")
        new_h_a = QAction("New &Horizon…", self)
        new_h_a.triggered.connect(self._add_new_horizon)
        interp_menu.addAction(new_h_a)
        new_f_a = QAction("New &Fault…", self)
        new_f_a.triggered.connect(self._add_new_fault)
        interp_menu.addAction(new_f_a)
        interp_menu.addSeparator()
        ref_sub = QMenu("New &Reference Line", self)
        self._add_hline_action = QAction("&Horizontal…", self)
        self._add_hline_action.triggered.connect(
            lambda: self._add_reference_line_kind("horizontal"))
        ref_sub.addAction(self._add_hline_action)
        self._add_vline_action = QAction("&Vertical…", self)
        self._add_vline_action.triggered.connect(
            lambda: self._add_reference_line_kind("vertical"))
        ref_sub.addAction(self._add_vline_action)
        interp_menu.addMenu(ref_sub)

        self._pick_action = QAction("&Horizon Pick Mode", self)
        self._pick_action.setCheckable(True)
        interp_menu.addSeparator()
        interp_menu.addAction(self._pick_action)

        # ================================================================
        # Tools
        # ================================================================
        tools_menu = mb.addMenu("&Tools")
        for tid, label, key in [
            ("select",       "Select Object",      "V"),
            ("node_edit",    "Direct Select / Nodes", "A"),
            ("pan",          "Pan",                "H"),
            ("zoom",         "Zoom",               "Z"),
            ("new_section",  "Draw Section",       "S"),
            ("horizon_pick", "Horizon Pick",       "P"),
            ("fault_pick",   "Fault Pick",         "F"),
            ("polygon",      "Polygon",            "G"),
            ("measure",      "Measure",            "M"),
        ]:
            a = QAction(f"{label}\t{key}", self)
            a.triggered.connect(
                lambda _checked, t=tid: self._tool_palette.set_active_tool(t))
            tools_menu.addAction(a)
            if tid in ("zoom", "new_section", "polygon"):
                tools_menu.addSeparator()

        tools_menu.addSeparator()
        self._view_segy_hdr_action = QAction("View SEG-Y Header…", self)
        self._view_segy_hdr_action.triggered.connect(self._on_view_segy_header)
        tools_menu.addAction(self._view_segy_hdr_action)

        # ================================================================
        # Help
        # ================================================================
        help_menu = mb.addMenu("&Help")
        kbd_action = QAction("&Keyboard Shortcuts…", self)
        kbd_action.triggered.connect(self._on_show_shortcuts_dialog)
        help_menu.addAction(kbd_action)
        help_menu.addSeparator()
        self._about_action = QAction("&About Section…", self)
        self._about_action.triggered.connect(self._on_about)
        help_menu.addAction(self._about_action)

    def _build_toolbar(self) -> None:
        """Slim icon-only main toolbar: file ops + undo/redo + export."""
        style = self.style()
        SP = QStyle.StandardPixmap
        self._new_action.setIcon(style.standardIcon(SP.SP_FileIcon))
        self._open_action.setIcon(style.standardIcon(SP.SP_DirOpenIcon))
        self._save_action.setIcon(style.standardIcon(SP.SP_DialogSaveButton))
        self._save_as_action.setIcon(style.standardIcon(SP.SP_DialogSaveButton))

        tb: QToolBar = self.addToolBar("Main")
        tb.setObjectName("MainToolBar")
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setIconSize(QSize(18, 18))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        tb.setStyleSheet(
            "QToolBar { background: #f8f8f8; border-bottom: 1px solid #ddd; }"
        )

        tb.addAction(self._new_action)
        tb.addAction(self._open_action)
        tb.addAction(self._save_action)
        tb.addSeparator()

        # Undo / Redo
        self._undo_tb_action = QAction("Undo", self)
        self._undo_tb_action.setShortcut(QKeySequence.StandardKey.Undo)
        self._undo_tb_action.setIcon(style.standardIcon(SP.SP_ArrowBack))
        self._undo_tb_action.setToolTip("Undo  (Ctrl+Z)")
        self._undo_tb_action.triggered.connect(self._state.undo)
        tb.addAction(self._undo_tb_action)

        self._redo_tb_action = QAction("Redo", self)
        self._redo_tb_action.setIcon(style.standardIcon(SP.SP_ArrowForward))
        self._redo_tb_action.setToolTip("Redo  (Ctrl+Shift+Z)")
        self._redo_tb_action.triggered.connect(self._state.redo)
        tb.addAction(self._redo_tb_action)
        tb.addSeparator()

        # Export
        self._export_tb_action = QAction("Export Image", self)
        self._export_tb_action.setIcon(style.standardIcon(SP.SP_ArrowRight))
        self._export_tb_action.setToolTip("Export Section Image…")
        self._export_tb_action.triggered.connect(self._on_export_section_image)
        tb.addAction(self._export_tb_action)

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
        # FIX 1: right-click / double-click during picking → return to select
        self._section_view.pick_ended.connect(
            lambda: self._tool_palette.set_active_tool("select")
        )
        # Options bar context toolbar already connected in _build_options_bar
        self._tool_palette.tool_changed.connect(self._on_tool_changed)
        # Sync palette when tool changes via state (e.g. after section draw)
        s.tool_changed.connect(self._on_state_tool_changed)
        # Keep menu pick-action in sync with palette
        self._pick_action.toggled.connect(self._on_pick_action_toggled)
        # Project panel pick-target selection → auto-switch tool
        self._project_panel.pick_target_selected.connect(self._on_pick_target_selected)
        # Properties dialogs (Phase A/B/E)
        self._project_panel.properties_requested.connect(self._on_panel_properties)
        # Project panel → AppState mutations
        self._project_panel.object_deleted.connect(self._on_panel_delete)
        self._project_panel.object_renamed.connect(self._on_panel_rename)
        self._project_panel.object_color_changed.connect(self._on_panel_color)
        self._project_panel.object_line_width_changed.connect(self._on_panel_line_width)
        self._project_panel.object_line_style_changed.connect(self._on_panel_line_style)
        self._project_panel.add_requested.connect(self._on_panel_add)
        self._project_panel.create_ew_section_through_well.connect(
            self._on_create_ew_section_through_well)
        self._project_panel.create_ns_section_through_well.connect(
            self._on_create_ns_section_through_well)
        # Status bar from map view drag
        self._map_view.status_message.connect(self._on_map_status)
        # Status bar updates when tool or active pick target changes
        s.tool_changed.connect(lambda _: self._update_status())
        s.active_pick_target_changed.connect(lambda *_: self._update_status())
        # Tool availability — update when section or picks change
        s.active_section_changed.connect(lambda _: self._update_tool_availability())
        s.horizon_pick_added.connect(lambda _: self._update_tool_availability())
        s.horizon_pick_removed.connect(lambda _: self._update_tool_availability())
        s.fault_pick_added.connect(lambda _: self._update_tool_availability())
        s.fault_pick_removed.connect(lambda _: self._update_tool_availability())
        s.project_changed.connect(self._update_tool_availability)
        # Phase 7: undo/redo status flashes
        s.undo_performed.connect(lambda d: self._flash_status(f"Undo: {d}"))
        s.redo_performed.connect(lambda d: self._flash_status(f"Redo: {d}"))
        # Phase 3: wire node selection → properties panel
        self._section_view.node_selected.connect(
            self._properties_panel.set_selected_node)
        # Phase 3: deselect node in props when mode changes
        s.active_pick_target_changed.connect(
            lambda *_: self._properties_panel.set_selected_node(None))
        # Phase 2: update pick status when target changes
        s.active_pick_target_changed.connect(
            lambda *_: self._update_pick_status() if s.active_tool in (
                "horizon_pick", "fault_pick") else None)
        # Phase 6: annotations
        s.annotation_added.connect(lambda _: self._section_view.request_render())
        s.annotation_removed.connect(lambda _: self._section_view.request_render())
        s.annotation_modified.connect(lambda *_: self._section_view.request_render())
        # FPS display from section view
        self._section_view.frame_time_ms.connect(self._on_frame_time)
        # All keyboard shortcuts registered centrally
        self._register_shortcuts()
        # Initial tool availability pass
        self._update_tool_availability()

    # ------------------------------------------------------------------
    # Tool availability
    # ------------------------------------------------------------------

    def _update_tool_availability(self, *_args) -> None:
        """Recompute which palette tools are enabled based on current state."""
        has_section = self._state.active_section is not None
        proj = self._state.project
        has_picks = bool(proj.horizon_picks or proj.fault_picks)
        self._tool_palette.update_tool_availability(has_section, has_picks)

    # ------------------------------------------------------------------
    # Title / status helpers
    # ------------------------------------------------------------------

    def _update_title(self, *_args) -> None:
        path = self._state.project_path
        if path:
            name = os.path.basename(path)
        else:
            name = self._state.project.name or "Untitled"
        prefix = "* " if self._state.is_modified else ""
        self.setWindowTitle(f"{prefix}{name} — {self.APP_NAME}")

    _TOOL_HINTS = {
        "select":       "Click object to select  ·  Double-click for node editing  ·  Drag to move",
        "node_edit":    "Click node to select  ·  Drag to move  ·  Delete to remove",
        "pan":          "Drag to pan  ·  Scroll to zoom",
        "zoom":         "Scroll to zoom  ·  Shift+Z to fit",
        "new_section":  "Click to place nodes  ·  Double-click or Enter to finish  ·  Escape to cancel",
        "horizon_pick": "Click to place pick  ·  Right-click or Escape to end",
        "fault_pick":   "Click to place pick  ·  Right-click or Escape to end",
        "polygon":      "Click to place vertices  ·  Right-click to close",
        "h_ref":        "Click on section to place horizontal guide",
        "v_ref":        "Click on section to place vertical guide",
        "a_ref":        "1st click: anchor  ·  2nd click: direction",
        "measure":      "Click start point  ·  Click end point",
    }

    def _update_status(self, *_args) -> None:
        path = self._state.project_path
        msg = os.path.basename(path) if path else "New project"
        if self._state.is_modified:
            msg += "  ✎"
        n_sec  = len(self._state.project.sections)
        n_well = len(self._state.project.wells)
        msg += f"  |  {n_sec}S  {n_well}W"
        cat = self._state.active_pick_category
        idx = self._state.active_pick_index
        if cat is not None and idx is not None:
            proj = self._state.project
            picks = proj.horizon_picks if cat == "Horizons" else proj.fault_picks
            if idx < len(picks):
                obj_name = picks[idx].name or f"{cat[:-1]} {idx + 1}"
                msg += f"  |  Active: {obj_name}"
        tool = self._state.active_tool
        msg += f"  |  {tool.replace('_', ' ').title()}"
        self._status_label.setText(msg)
        # Hint in permanent label
        hint = self._TOOL_HINTS.get(tool, "")
        if hasattr(self, "_hint_label"):
            self._hint_label.setText(hint)

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

    def _new_project(
        self,
        name: str = "",
        crs_epsg: int = 32632,
        depth_units: str = "m",
        depth_domain: str = "md",
        default_depth_min: float = 0.0,
        default_depth_max: float = 5000.0,
        folder_path: str | None = None,
    ) -> None:
        """Create a fresh project (no dialog)."""
        self._state.new_project(
            name=name,
            crs_epsg=crs_epsg,
            depth_units=depth_units,
            depth_domain=depth_domain,
            default_depth_min=default_depth_min,
            default_depth_max=default_depth_max,
            folder_path=folder_path,
        )

    def _open_project(self, path: str) -> bool:
        """Load a project from *path*. Returns True on success, False on error (no dialog)."""
        try:
            self._state.open_project(path)
            self._check_autosave_recovery()
            # Auto-zoom to show all loaded data
            from PySide6.QtCore import QTimer as _QT
            _QT.singleShot(100, self._map_view.zoom_to_all_data)
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
        """Open a Save As dialog — folder chooser for SQLite, file for HDF5."""
        pm = self._state.project_manager
        if pm.is_open:
            # SQLite project: choose a folder
            new_folder = QFileDialog.getExistingDirectory(
                self, "Save Project As", os.path.dirname(pm.project_path or "")
            )
            if not new_folder:
                return False
            project_name = self._state.project.name or "Untitled"
            dest = os.path.join(new_folder, project_name)
            ok = self._save_project_as(dest)
        else:
            # Legacy HDF5
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Project As", "", "HDF5 Project (*.h5);;All Files (*)"
            )
            if not path:
                return False
            ok = self._save_project_as(path)
        if not ok:
            QMessageBox.critical(self, "Save Error", "Could not save project.")
        return ok

    # ------------------------------------------------------------------
    # Action slots (with dialogs)
    # ------------------------------------------------------------------

    def _on_new(self) -> None:
        if not self._check_unsaved_changes():
            return
        from section_tool.views.new_project_dialog import NewProjectDialog
        dlg = NewProjectDialog(
            current_crs=self._state.project.crs_epsg,
            parent=self,
        )
        if dlg.exec() != NewProjectDialog.DialogCode.Accepted:
            return
        self._new_project(
            name=dlg.project_name(),
            crs_epsg=dlg.crs_epsg(),
            depth_units=dlg.depth_units(),
            depth_domain=dlg.depth_domain(),
            default_depth_min=dlg.default_depth_min(),
            default_depth_max=dlg.default_depth_max(),
            folder_path=dlg.folder_path() or None,
        )

    def _on_open(self) -> None:
        if not self._check_unsaved_changes():
            return
        # Try folder picker first (SQLite projects), fall back to file picker (HDF5)
        folder = QFileDialog.getExistingDirectory(
            self, "Open Project Folder", os.path.expanduser("~")
        )
        if folder:
            if not self._open_project(folder):
                QMessageBox.critical(self, "Open Error", f"Could not open:\n{folder}")
            return
        # User cancelled folder dialog — offer legacy HDF5 open
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Legacy Project (HDF5)", "",
            "HDF5 Project (*.h5);;All Files (*)"
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
        if not paths:
            return
        import lasio
        from section_tool.io.las import extract_header_full
        from section_tool.core.wells import LogCurve, Well
        from section_tool.views.las_import_dialog import LASImportDialog
        _MAX_LOG_SAMPLES = 10_000
        place_manually_wells: list[int] = []  # indices of wells needing manual placement

        for path in paths:
            try:
                with _wait_cursor():
                    las = lasio.read(str(path))
                    header = extract_header_full(las)
            except Exception as exc:
                QMessageBox.warning(self, "Import Error",
                                    f"Cannot read LAS file:\n{path}\n\n{exc}")
                continue

            dlg = LASImportDialog(
                las, path, header,
                project_crs_epsg=self._state.project.crs_epsg,
                parent=self,
            )
            if dlg.exec() != LASImportDialog.DialogCode.Accepted:
                continue

            try:
                well = self._build_well_from_dialog(dlg, las, header, _MAX_LOG_SAMPLES)
            except Exception as exc:
                QMessageBox.warning(self, "Import Error",
                                    f"Failed to import well:\n{path}\n\n{exc}")
                continue

            well_idx = len(self._state.project.wells)
            self._state.add_well(well)

            if dlg.place_manually():
                place_manually_wells.append(well_idx)
            else:
                self._warn_if_well_far_from_sections([well])

        # Auto-zoom to show the newly added well(s)
        self._map_view.zoom_to_all_data()

        # After all imports, enter place-well mode for the last "place manually" well
        if place_manually_wells:
            last_idx = place_manually_wells[-1]
            wells = self._state.project.wells
            if last_idx < len(wells):
                self._map_view.start_place_well(last_idx)
                self._statusbar.showMessage(
                    f"Click on the map to place well '{wells[last_idx].name}'", 0
                )

    def _build_well_from_dialog(self, dlg, las, header, max_samples: int):
        """Construct a Well from a completed LASImportDialog."""
        import numpy as np
        from section_tool.core.wells import LogCurve, Well, DeviationSurvey

        x, y, kb = dlg.x(), dlg.y(), dlg.kb()
        well_crs = dlg.well_crs_epsg()
        proj_crs = self._state.project.crs_epsg

        # CRS transformation if needed
        if well_crs is not None and well_crs != proj_crs and x != 0.0:
            try:
                from section_tool.core.crs import transform_points
                tx, ty = transform_points(
                    np.array([x]), np.array([y]),
                    from_epsg=well_crs, to_epsg=proj_crs,
                )
                orig_x, orig_y = x, y
                x, y = float(tx[0]), float(ty[0])
            except Exception as exc:
                QMessageBox.warning(
                    self, "CRS Transform Warning",
                    f"Could not transform coordinates: {exc}\n"
                    "Coordinates imported as-is."
                )
                orig_x, orig_y = None, None
        else:
            orig_x, orig_y = None, None

        well = Well(name=dlg.well_name(), x=x, y=y, kb=kb, uwi=dlg.uwi())
        if orig_x is not None:
            well.original_x = orig_x
            well.original_y = orig_y
            well.original_crs_epsg = well_crs

        # Depth index
        if not las.curves:
            return well
        depth_mnemonic = las.curves[0].mnemonic
        depths = np.asarray(las[depth_mnemonic], dtype=float)
        if len(depths) == 0:
            return well

        # Selected log curves only
        selected = set(dlg.selected_curves())
        for curve in las.curves[1:]:
            if curve.mnemonic not in selected:
                continue
            values = np.asarray(las[curve.mnemonic], dtype=float)
            if len(values) != len(depths):
                continue
            lc = LogCurve(curve.mnemonic, curve.unit or "", depths, values)
            if lc.n_samples > max_samples:
                step = lc.n_samples // max_samples
                lc = LogCurve(lc.name, lc.units,
                              lc._depths[::step], lc._values[::step])
            well.add_log(lc)

        return well

    def _warn_if_well_far_from_sections(self, wells) -> None:
        """Warn the user if a loaded well is far from all existing section lines."""
        import math
        sections = self._state.project.sections
        if not sections:
            return
        for well in wells:
            if well.x == 0.0 and well.y == 0.0:
                continue
            min_dist = float("inf")
            for sec in sections:
                for node in sec.nodes:
                    d = math.hypot(well.x - node[0], well.y - node[1])
                    if d < min_dist:
                        min_dist = d
            if min_dist > 50_000:
                QMessageBox.information(
                    self, "Well Location",
                    f"Well '{well.name}' is at ({well.x:.0f}, {well.y:.0f}).\n"
                    f"The nearest section is {min_dist/1000:.1f} km away.\n\n"
                    "Your sections may be in a different coordinate system. "
                    "Consider creating a section near the well location.",
                )

    def _on_view_segy_header(self) -> None:
        """Tools → View SEG-Y Header: pick a file and show the header inspector."""
        path, _ = QFileDialog.getOpenFileName(
            self, "View SEG-Y Header",
            "", "SEG-Y Files (*.segy *.sgy *.SGY);;All Files (*)",
        )
        if not path:
            return
        from section_tool.views.segy_header_dialog import SEGYHeaderDialog
        SEGYHeaderDialog(path, self).exec()

    def _on_create_ew_section_through_well(self, well_idx: int) -> None:
        """Project panel: Create E–W section through the selected well."""
        self._create_section_through_well(well_idx, orientation="ew")

    def _on_create_ns_section_through_well(self, well_idx: int) -> None:
        """Project panel: Create N–S section through the selected well."""
        self._create_section_through_well(well_idx, orientation="ns")

    def _create_section_through_well(self, well_idx: int, orientation: str) -> None:
        import numpy as np
        from section_tool.core.section import Section
        wells = self._state.project.wells
        if well_idx >= len(wells):
            return
        well = wells[well_idx]
        half = 5_000.0  # 5 km each side
        if orientation == "ew":
            nodes = np.array([
                [well.x - half, well.y],
                [well.x + half, well.y],
            ])
            name = f"E-W through {well.name}"
        else:
            nodes = np.array([
                [well.x, well.y - half],
                [well.x, well.y + half],
            ])
            name = f"N-S through {well.name}"
        section = Section(nodes, name=name)
        self._state.add_section(section)
        self._state.set_active_section(section)

    def _on_import_segy(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import SEG-Y Files",
            "",
            "SEG-Y Files (*.segy *.sgy *.SGY);;All Files (*)",
        )
        for path in paths:
            from section_tool.io.project import SeismicRef
            from section_tool.views.seismic_import_dialog import SeismicImportDialog
            from section_tool.views.segy_header_dialog import SEGYHeaderDialog
            fname = os.path.basename(path)
            dlg = SeismicImportDialog(
                sections=self._state.project.sections,
                filename=fname,
                parent=self,
            )
            # Offer header viewer button inside the import dialog
            from PySide6.QtWidgets import QPushButton as _QPB
            hdr_btn = _QPB("View SEG-Y Header…")
            hdr_btn.clicked.connect(lambda _checked, p=path: SEGYHeaderDialog(p, dlg).exec())
            dlg.layout().insertWidget(0, hdr_btn)
            if dlg.exec() != dlg.DialogCode.Accepted:
                continue
            # Quick header-only read for spatial extent (no amplitude data)
            try:
                from section_tool.io.segy import read_segy_header
                with _wait_cursor():
                    hdr = read_segy_header(
                        path,
                        x_field=dlg.x_field,
                        y_field=dlg.y_field,
                        apply_scalar=dlg.apply_scalar,
                    )
                xr, yr = hdr["x_range"], hdr["y_range"]
                n_tot = hdr["n_traces"]
            except Exception:
                xr, yr, n_tot = (0.0, 0.0), (0.0, 0.0), 0

            ref = SeismicRef(
                path=path,
                name=os.path.splitext(fname)[0],
                x_field=dlg.x_field,
                y_field=dlg.y_field,
                apply_scalar=dlg.apply_scalar,
                domain=dlg.domain,
                depth_units=dlg.depth_units,
                crs_epsg=self._state.project.crs_epsg,
                extent_x_min=float(xr[0]),
                extent_x_max=float(xr[1]),
                extent_y_min=float(yr[0]),
                extent_y_max=float(yr[1]),
                n_traces_total=int(n_tot),
            )
            self._state.add_seismic_ref(ref)
            # Auto-zoom to show seismic extent
            self._map_view.zoom_to_all_data()

    def _on_extract_seismic_for_section(self) -> None:
        """Extract seismic traces along the active section from a SEG-Y file."""
        section = self._state.active_section
        if section is None:
            QMessageBox.information(self, "No Active Section",
                                    "Activate a section first.")
            return
        refs = self._state.project.seismic_refs
        if not refs:
            QMessageBox.information(self, "No Seismic Data",
                                    "Import a SEG-Y file first (File → Import → SEG-Y Seismic).")
            return

        if len(refs) == 1:
            ref = refs[0]
        else:
            from PySide6.QtWidgets import QInputDialog
            names = [r.name for r in refs]
            name, ok = QInputDialog.getItem(
                self, "Select SEG-Y", "Extract from which seismic volume?",
                names, 0, False
            )
            if not ok:
                return
            ref = refs[names.index(name)]

        # Resolve output path into a cache sub-folder alongside the project
        import os, re
        _safe = lambda s: re.sub(r"[^\w\-]", "_", s)
        base_dir = (os.path.dirname(self._state.project_path)
                    if self._state.project_path else os.getcwd())
        cache_dir = os.path.join(base_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        out_npy = os.path.join(
            cache_dir,
            f"{_safe(section.name)}_{_safe(ref.name)}.extract.npy"
        )

        from PySide6.QtCore import Qt as _Qt
        from PySide6.QtWidgets import QProgressDialog, QApplication as _QApp, QCheckBox, QDialog, QVBoxLayout, QDialogButtonBox, QLabel
        dlg = QDialog(self)
        dlg.setWindowTitle("Extract Seismic Along Section")
        vl = QVBoxLayout(dlg)
        vl.addWidget(QLabel(
            f"<b>{ref.name}</b> → {section.name}<br>"
            f"Output: <code>{os.path.basename(out_npy)}</code>"
        ))
        interp_cb = QCheckBox("Interpolate traces to regular grid")
        interp_cb.setChecked(False)
        vl.addWidget(interp_cb)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        vl.addWidget(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        progress = QProgressDialog(
            f"Extracting {ref.name}…", "Cancel", 0, 100, self
        )
        progress.setWindowModality(_Qt.WindowModality.WindowModal)
        progress.show()
        _QApp.processEvents()
        cancelled = False

        def _prog(pct: int) -> None:
            nonlocal cancelled
            progress.setValue(pct)
            _QApp.processEvents()
            if progress.wasCanceled():
                cancelled = True

        try:
            import numpy as _np
            from section_tool.io.segy import extract_seismic_along_section
            meta = extract_seismic_along_section(
                ref.path, section, out_npy,
                x_field=ref.x_field,
                y_field=ref.y_field,
                apply_scalar=ref.apply_scalar,
                interpolate=interp_cb.isChecked(),
                progress_callback=_prog,
            )
            if not cancelled:
                data = _np.load(out_npy)
                self._state.set_seismic_for_section(section.name, data, meta)
        except Exception as exc:
            QMessageBox.warning(self, "Extraction Error",
                                f"Seismic extraction failed:\n\n{exc}")
        finally:
            progress.close()

    def _on_import_section_image(self) -> None:
        """Import a raster image (scanned section) as a section background."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Section Image",
            "",
            "Images (*.png *.jpg *.jpeg *.tiff *.tif *.bmp);;All Files (*)",
        )
        if not path:
            return
        section = self._state.active_section
        if section is None:
            QMessageBox.information(self, "No Active Section",
                                    "Activate a section first, then import an image.")
            return
        from PySide6.QtWidgets import QDoubleSpinBox, QDialog, QFormLayout, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle("Section Image Range")
        fl = QFormLayout(dlg)
        total = section.total_length()
        max_d = 10000.0

        def _spin(lo, hi, val, dec=0):
            s = QDoubleSpinBox()
            s.setRange(lo, hi); s.setValue(val); s.setDecimals(dec)
            return s

        d_start = _spin(0, 1e8, 0.0, 1)
        d_end   = _spin(0, 1e8, total, 1)
        z_top   = _spin(-1e6, 1e6, 0.0, 1)
        z_bot   = _spin(-1e6, 1e6, max_d, 1)
        fl.addRow("Distance start (m):", d_start)
        fl.addRow("Distance end (m):",   d_end)
        fl.addRow("Depth top (m):",      z_top)
        fl.addRow("Depth bottom (m):",   z_bot)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        fl.addRow(bb)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        self._section_view.add_image_overlay(
            path=path,
            section_name=section.name,
            dist_range=(d_start.value(), d_end.value()),
            depth_range=(z_top.value(), z_bot.value()),
        )

    def _on_import_topography(self) -> None:
        """Import topography profile from CSV (distance, elevation columns)."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Topography Profile", "",
            "CSV Files (*.csv *.txt);;All Files (*)"
        )
        if not path:
            return
        section = self._state.active_section
        if section is None:
            QMessageBox.information(self, "No Active Section",
                                    "Activate a section before importing topography.")
            return
        try:
            import csv
            import numpy as np
            distances, elevations = [], []
            with _wait_cursor():
                with open(path, newline="", encoding="utf-8-sig") as fh:
                    reader = csv.reader(fh)
                    header = next(reader, None)
                    for row in reader:
                        if len(row) >= 2:
                            try:
                                distances.append(float(row[0]))
                                elevations.append(float(row[1]))
                            except ValueError:
                                continue
            if len(distances) < 2:
                raise ValueError("Need at least 2 data rows.")
            self._section_view.set_topography(
                section_name=section.name,
                distances=np.array(distances),
                elevations=np.array(elevations),
            )
            QMessageBox.information(self, "Import OK",
                                    f"Topography loaded: {len(distances)} points.")
        except Exception as exc:
            QMessageBox.critical(self, "Import Error", str(exc))

    def _on_import_well_tops(self) -> None:
        """Import well tops from a CSV file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Well Tops CSV", "",
            "CSV Files (*.csv *.txt);;All Files (*)"
        )
        if not path:
            return
        from section_tool.views.well_tops_dialog import WellTopsDialog
        dlg = WellTopsDialog(path, crs_epsg=self._state.project.crs_epsg, parent=self)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        try:
            with _wait_cursor():
                wells = dlg.load_wells()
        except Exception as exc:
            QMessageBox.warning(self, "Import Error", str(exc))
            return
        added = 0
        for well in wells:
            # Check if well already exists
            existing = next(
                (w for w in self._state.project.wells if w.name == well.name), None
            )
            if existing is not None:
                # Merge formation tops into existing well
                import copy
                idx = self._state.project.wells.index(existing)
                updated = copy.deepcopy(existing)
                for name, md in well.formation_tops.items():
                    updated.add_formation_top(name, md)
                self._state.update_well(idx, updated)
            else:
                self._state.add_well(well)
                added += 1
        QMessageBox.information(
            self, "Import Complete",
            f"Imported {added} new well(s), tops merged into existing wells."
        )

    def _on_new_section_ns(self) -> None:
        """New north–south section via dialog."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QDoubleSpinBox
        dlg = QDialog(self); dlg.setWindowTitle("New North-South Section")
        fl = QFormLayout(dlg)
        existing = self._state.project.sections
        x0_default, y0_default = self._get_smart_center()
        easting_spin = QDoubleSpinBox(); easting_spin.setRange(-1e8, 1e8)
        easting_spin.setValue(x0_default); easting_spin.setDecimals(1)
        length_spin = QDoubleSpinBox(); length_spin.setRange(100, 1e7)
        length_spin.setValue(10_000.0); length_spin.setDecimals(0)
        center_n_spin = QDoubleSpinBox(); center_n_spin.setRange(-1e8, 1e8)
        center_n_spin.setValue(y0_default); center_n_spin.setDecimals(1)
        fl.addRow("Easting (X):", easting_spin)
        fl.addRow("Center Northing (Y):", center_n_spin)
        fl.addRow("Length (m):", length_spin)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        fl.addRow(bb)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        x = easting_spin.value(); L = length_spin.value(); cy = center_n_spin.value()
        sec = Section(
            [(x, cy - L / 2), (x, cy + L / 2)],
            name=f"Section {len(existing) + 1}",
            crs_epsg=self._state.project.crs_epsg,
        )
        self._state.add_section(sec)
        self._state.set_active_section(sec)

    def _on_new_section_user_defined(self) -> None:
        """New section with user-specified azimuth, length, and origin."""
        import math
        from PySide6.QtWidgets import (
            QDialog, QDialogButtonBox, QFormLayout, QDoubleSpinBox,
            QGroupBox, QRadioButton, QVBoxLayout,
        )
        dlg = QDialog(self); dlg.setWindowTitle("New User-Defined Section")
        main_vb = QVBoxLayout(dlg)

        # Parameters
        param_grp = QGroupBox("Section geometry")
        pfl = QFormLayout(param_grp)
        az_spin = QDoubleSpinBox(); az_spin.setRange(0, 360); az_spin.setValue(90)
        az_spin.setDecimals(1); az_spin.setSuffix("°")
        len_spin = QDoubleSpinBox(); len_spin.setRange(100, 1e7)
        len_spin.setValue(10_000.0); len_spin.setDecimals(0); len_spin.setSuffix(" m")
        pfl.addRow("Azimuth (from N):", az_spin)
        pfl.addRow("Length:", len_spin)
        main_vb.addWidget(param_grp)

        # Origin mode
        orig_grp = QGroupBox("Origin")
        ovb = QVBoxLayout(orig_grp)
        rb_center = QRadioButton("Center point (section extends half-length each way)")
        rb_start  = QRadioButton("Start point (section extends full length along azimuth)")
        rb_center.setChecked(True)
        ovb.addWidget(rb_center); ovb.addWidget(rb_start)
        ofl = QFormLayout()
        x_spin = QDoubleSpinBox(); x_spin.setRange(-1e8, 1e8); x_spin.setDecimals(1)
        y_spin = QDoubleSpinBox(); y_spin.setRange(-1e8, 1e8); y_spin.setDecimals(1)
        existing = self._state.project.sections
        cx_def, cy_def = self._get_smart_center()
        x_spin.setValue(cx_def)
        y_spin.setValue(cy_def)
        ofl.addRow("X (Easting):", x_spin)
        ofl.addRow("Y (Northing):", y_spin)
        ovb.addLayout(ofl)
        main_vb.addWidget(orig_grp)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        main_vb.addWidget(bb)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return

        az_rad = math.radians(az_spin.value())
        dx = math.sin(az_rad) * len_spin.value()
        dy = math.cos(az_rad) * len_spin.value()
        ox, oy = x_spin.value(), y_spin.value()
        if rb_center.isChecked():
            x0, y0 = ox - dx / 2, oy - dy / 2
            x1, y1 = ox + dx / 2, oy + dy / 2
        else:
            x0, y0, x1, y1 = ox, oy, ox + dx, oy + dy
        sec = Section(
            [(x0, y0), (x1, y1)],
            name=f"Section {len(existing) + 1}",
            crs_epsg=self._state.project.crs_epsg,
        )
        self._state.add_section(sec)
        self._state.set_active_section(sec)

    def _get_smart_center(self) -> tuple[float, float]:
        """Best center point: existing data → map view center → origin."""
        import numpy as _np
        # First priority: any existing data in the project
        xs, ys = [], []
        for w in self._state.project.wells:
            if w.x != 0 or w.y != 0:
                xs.append(w.x); ys.append(w.y)
        for ref in self._state.project.seismic_refs:
            if ref.extent_x_max != ref.extent_x_min:
                xs += [ref.extent_x_min, ref.extent_x_max]
                ys += [ref.extent_y_min, ref.extent_y_max]
        for sec in self._state.project.sections:
            for node in sec.nodes:
                if node[0] != 0 or node[1] != 0:
                    xs.append(float(node[0])); ys.append(float(node[1]))
        if xs:
            return float(_np.mean(xs)), float(_np.mean(ys))
        # Fallback: map view center
        return self._map_view.map_center

    def _on_new_section(self) -> None:
        """Add a simple 10-km east–west section centred on current map view."""
        cx, cy = self._get_smart_center()
        existing = self._state.project.sections
        if existing:
            cy = existing[-1].nodes[0, 1] + 1_000.0  # 1 km north of last section
        sec = Section(
            [(cx - 5_000.0, cy), (cx + 5_000.0, cy)],
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

    def _on_export_section_image(self) -> None:
        """Phase 8: render section to PNG/SVG/PDF."""
        if self._state.active_section is None:
            QMessageBox.information(self, "No Section", "Activate a section first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Section Image", "",
            "PNG Image (*.png);;SVG Vector (*.svg);;PDF Document (*.pdf)"
        )
        if not path:
            return
        try:
            fig = self._section_view.render_to_figure(12.0, 7.0, 200)
            fig.savefig(path, bbox_inches="tight")
            QMessageBox.information(self, "Export OK", f"Saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _on_export_horizons_csv(self) -> None:
        """Phase 8: export picks for the active section to CSV."""
        section = self._state.active_section
        if section is None:
            QMessageBox.information(self, "No Section", "Activate a section first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Horizons CSV", "", "CSV Files (*.csv)"
        )
        if not path:
            return
        import csv
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow([
                    "section_name", "horizon_name",
                    "distance", "depth", "x", "y", "z",
                    "confidence", "quality",
                ])
                for hp in self._state.project.horizon_picks:
                    idxs = hp.section_indices(section.name)
                    for fi in idxs:
                        d   = float(hp._distances[fi])
                        z   = float(hp._depths[fi])
                        x, y = section.section_to_map(d)
                        conf = float(hp._confidence[fi]) if len(hp._confidence) > fi else 1.0
                        qual = str(hp._quality[fi]) if len(hp._quality) > fi else "picked"
                        writer.writerow([
                            section.name, hp.name, d, z,
                            f"{x:.2f}", f"{y:.2f}", f"{z:.2f}",
                            f"{conf:.2f}", qual,
                        ])
            QMessageBox.information(self, "Export OK", f"Saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

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
            elif category == "Wells" and index < len(proj.wells):
                self._state.remove_well(proj.wells[index])
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
        from section_tool.views.horizon_dialog import HorizonDialog
        from section_tool.core.surfaces import HorizonPick
        n = len(self._state.project.horizon_picks) + 1
        default_name = f"Horizon {n}"
        # Remember current tool so we can restore / continue picking
        _prev_tool = self._state.active_tool
        dlg = HorizonDialog(self, name=default_name, color=self._next_horizon_color())
        if dlg.exec() != dlg.DialogCode.Accepted or not dlg.name:
            # Restore tool even on cancel
            self._tool_palette.set_active_tool(_prev_tool)
            return
        hp = HorizonPick.empty(name=dlg.name, color=dlg.color)
        hp.contact_type    = dlg.contact_type
        hp.formation_above = dlg.formation_above
        hp.formation_below = dlg.formation_below
        self._state.add_horizon_pick(hp)
        idx = len(self._state.project.horizon_picks) - 1
        self._state.set_active_pick_target("Horizons", idx)
        # Always switch to horizon_pick after adding (natural next step)
        self._tool_palette.set_active_tool("horizon_pick")

    def _on_generate_polygons(self) -> None:
        """Detect closed regions via live topology and import as polygons."""
        import traceback as _tb
        section = self._state.active_section
        if section is None:
            QMessageBox.information(self, "No Section",
                                    "Activate a section first.")
            return
        from section_tool.core.polygons import SectionPolygon
        import numpy as np

        # Diagnostics (printed to terminal for debugging)
        topo = self._state.topology
        print(f"[Generate Polygons] section={section.name!r}  "
              f"topology={'OK' if topo else 'None'}")
        if topo:
            user_lines = [k for k in topo._lines if not k.startswith("__")]
            print(f"  topology lines={user_lines}")
            print(f"  intersections={len(topo.intersections)}")

        polys = []
        topo_error = None

        # Try topology-based generation first
        if topo is not None and topo.section_name == section.name:
            try:
                polys = topo.get_all_faces()
                print(f"  topology faces={len(polys)}")
                for i, p in enumerate(polys):
                    print(f"    face {i}: area={p.area:.0f}  bounds={p.bounds}")
            except Exception as exc:
                topo_error = exc
                print(f"  topology.get_all_faces() FAILED: {exc}")
                _tb.print_exc()

        # Fallback to standalone polygon_detection if topology produced nothing
        if not polys:
            if topo_error is not None:
                print("  Falling back to polygon_detection module")
            from section_tool.core.polygon_detection import detect_polygons
            try:
                polys = detect_polygons(
                    self._state.project.horizon_picks,
                    self._state.project.fault_picks,
                    self._state.project.reference_lines,
                    section,
                    section_name=section.name,
                )
                print(f"  fallback found {len(polys)} polygons")
            except Exception as exc:
                print(f"  fallback FAILED: {exc}")
                _tb.print_exc()
                QMessageBox.critical(self, "Detection Error",
                                     f"Polygon detection failed:\n{exc}")
                return

        if not polys:
            detail = ""
            if topo is not None:
                user_lines = [k for k in topo._lines if not k.startswith("__")]
                detail = (f"\n\nTopology has {len(user_lines)} line(s). "
                          "Ensure each horizon has at least 2 picks on this section "
                          "and extends across the full width.")
            QMessageBox.information(self, "No Polygons Found",
                                    "No closed regions were detected." + detail)
            return

        reply = QMessageBox.question(
            self, "Import Polygons",
            f"{len(polys)} closed region(s) detected.\nImport all as polygons?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        existing = len(self._state.project.polygons)
        added_polys = []
        for i, shp in enumerate(polys):
            try:
                coords = list(shp.exterior.coords)
            except AttributeError:
                continue
            if coords[0] == coords[-1]:
                coords = coords[:-1]
            if len(coords) < 3:
                continue
            # Assign distinct colors cycling through a geological palette
        _POLY_COLORS = [
            "#4878d0", "#ee854a", "#6acc64", "#d65f5f", "#956cb4",
            "#8c613c", "#dc7ec0", "#797979", "#d5bb67", "#82c6e2",
        ]
        color = _POLY_COLORS[(existing + i) % len(_POLY_COLORS)]
        added_polys.append(SectionPolygon(
                vertices=np.array(coords),
                name=f"Region {existing + i + 1}",
                fill_color=color,
                fill_alpha=0.45,
            ))
        self._state.blockSignals(True)
        try:
            for poly in added_polys:
                self._state.add_polygon(poly)
        finally:
            self._state.blockSignals(False)
            if added_polys:
                self._state.project_changed.emit()

    def _on_edit_strat_column(self) -> None:
        """Phase 5: open stratigraphic column editor (stub)."""
        from section_tool.views.strat_column_dialog import StratColumnDialog
        dlg = StratColumnDialog(self._state, self)
        dlg.exec()

    def _add_reference_line_kind(self, kind: str) -> None:
        from PySide6.QtWidgets import QInputDialog
        from section_tool.core.reference_line import ReferenceLine
        label = "Depth value:" if kind == "horizontal" else "Distance along section:"
        value, ok = QInputDialog.getDouble(self, "Reference Line", label, 0.0)
        if not ok:
            return
        rl = ReferenceLine(kind=kind, value=value)
        self._state.add_reference_line(rl)

    def _add_reference_line_dialog(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        from section_tool.core.reference_line import ReferenceLine
        kinds = ["Horizontal (depth)", "Vertical (distance)"]
        kind_str, ok = QInputDialog.getItem(self, "Reference Line", "Type:", kinds, 0, False)
        if not ok:
            return
        kind = "horizontal" if "Horizontal" in kind_str else "vertical"
        label = "Depth value:" if kind == "horizontal" else "Distance along section:"
        value, ok2 = QInputDialog.getDouble(self, "Reference Line", label, 0.0)
        if not ok2:
            return
        from PySide6.QtWidgets import QLineEdit
        name, ok3 = QInputDialog.getText(
            self, "Reference Line", "Name (optional):",
            QLineEdit.EchoMode.Normal, "")
        rl = ReferenceLine(kind=kind, value=value, name=name.strip())
        self._state.add_reference_line(rl)

    def _add_new_polygon(self) -> None:
        from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout,
                                       QLineEdit, QColorDialog, QDoubleSpinBox,
                                       QComboBox)
        from PySide6.QtGui import QColor
        from section_tool.core.polygons import SectionPolygon

        dlg = QDialog(self)
        dlg.setWindowTitle("New Polygon")
        fl = QFormLayout(dlg)
        n = len(self._state.project.polygons) + 1
        name_edit = QLineEdit(f"Polygon {n}")
        fl.addRow("Name:", name_edit)

        # Formation dropdown
        fm_combo = QComboBox()
        fm_combo.addItem("(Unassigned)", "")
        for fm in self._state.project.strat_column.formations:
            fm_combo.addItem(fm.name, fm.name)
        fl.addRow("Formation:", fm_combo)

        # Color picker
        _color = "#9467bd"
        color_btn = QPushButton("   ")
        color_btn.setStyleSheet(f"background: {_color};")

        def _pick_color():
            nonlocal _color
            c = QColorDialog.getColor(QColor(_color), dlg)
            if c.isValid():
                _color = c.name()
                color_btn.setStyleSheet(f"background: {_color};")
        color_btn.clicked.connect(_pick_color)
        fl.addRow("Color:", color_btn)

        opacity_spin = QDoubleSpinBox()
        opacity_spin.setRange(0.1, 1.0)
        opacity_spin.setValue(0.6)
        opacity_spin.setSingleStep(0.1)
        fl.addRow("Opacity:", opacity_spin)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        fl.addRow(bb)

        if dlg.exec() != dlg.DialogCode.Accepted:
            return

        self._section_view.set_polygon_preflight(
            name=name_edit.text().strip() or f"Polygon {n}",
            formation=fm_combo.currentData() or "",
            color=_color,
            opacity=opacity_spin.value(),
        )
        # Ensure polygon drawing mode is active
        if self._state.active_tool != "polygon":
            self._tool_palette.set_active_tool("polygon")

    def _add_new_fault(self) -> None:
        from section_tool.views.fault_dialog import FaultDialog
        from section_tool.core.surfaces import HorizonPick
        n = len(self._state.project.fault_picks) + 1
        default_name = f"Fault {n}"
        _prev_tool = self._state.active_tool
        dlg = FaultDialog(self, name=default_name, color=self._next_fault_color())
        if dlg.exec() != dlg.DialogCode.Accepted or not dlg.name:
            self._tool_palette.set_active_tool(_prev_tool)
            return
        fp = HorizonPick.empty(name=dlg.name, color=dlg.color)
        fp.fault_type    = dlg.fault_type
        fp.dip_direction = dlg.dip_direction
        self._state.add_fault_pick(fp)
        idx = len(self._state.project.fault_picks) - 1
        self._state.set_active_pick_target("Faults", idx)
        self._tool_palette.set_active_tool("fault_pick")

    def _horizon_properties(self, index: int) -> None:
        """Phase A: edit horizon attributes."""
        import copy
        from section_tool.views.horizon_dialog import HorizonDialog
        picks = self._state.project.horizon_picks
        if index >= len(picks):
            return
        hp = picks[index]
        dlg = HorizonDialog(self, name=hp.name, contact_type=hp.contact_type,
                            color=hp.color, formation_above=hp.formation_above,
                            formation_below=hp.formation_below)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        hp2 = copy.deepcopy(hp)
        hp2.name            = dlg.name
        hp2.contact_type    = dlg.contact_type
        hp2.color           = dlg.color
        hp2.formation_above = dlg.formation_above
        hp2.formation_below = dlg.formation_below
        self._state.update_horizon_pick(index, hp2)

    def _fault_properties(self, index: int) -> None:
        """Phase B: edit fault attributes."""
        import copy
        from section_tool.views.fault_dialog import FaultDialog
        picks = self._state.project.fault_picks
        if index >= len(picks):
            return
        fp = picks[index]
        dlg = FaultDialog(self, name=fp.name, fault_type=fp.fault_type,
                          color=fp.color, dip_direction=fp.dip_direction)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        fp2 = copy.deepcopy(fp)
        fp2.name          = dlg.name
        fp2.fault_type    = dlg.fault_type
        fp2.color         = dlg.color
        fp2.dip_direction = dlg.dip_direction
        self._state.update_fault_pick(index, fp2)

    def _toggle_map_panel(self) -> None:
        """Toggle map dock visibility (legacy; now delegates to dock action)."""
        self._map_dock.toggleViewAction().trigger()

    def _toggle_section_panel(self) -> None:
        """Toggle section dock visibility (legacy; now delegates to dock action)."""
        self._section_dock.toggleViewAction().trigger()

    # ------------------------------------------------------------------
    # Phase 6 helpers
    # ------------------------------------------------------------------

    def _zoom_to_fit(self) -> None:
        """Ctrl+0 / Shift+Z: zoom map to all data, reset section view."""
        self._map_view.zoom_to_all_data()
        self._map_view.render()
        self._section_view._ax_limits_set = False
        self._section_view.render()

    # ------------------------------------------------------------------
    # Auto-colour helpers (Phase 3)
    # ------------------------------------------------------------------

    def _next_horizon_color(self) -> str:
        used = {hp.color for hp in self._state.project.horizon_picks}
        for c in _HORIZON_COLORS:
            if c not in used:
                return c
        n = len(self._state.project.horizon_picks)
        return _HORIZON_COLORS[n % len(_HORIZON_COLORS)]

    def _next_fault_color(self) -> str:
        used = {fp.color for fp in self._state.project.fault_picks}
        for c in _FAULT_COLORS:
            if c not in used:
                return c
        n = len(self._state.project.fault_picks)
        return _FAULT_COLORS[n % len(_FAULT_COLORS)]

    def _cycle_ref_line_tool(self) -> None:
        """R key: cycle H-Ref → V-Ref → A-Ref."""
        tools = ["h_ref", "v_ref", "a_ref"]
        cur = self._state.active_tool
        if cur in tools:
            self._ref_cycle_idx = (tools.index(cur) + 1) % len(tools)
        else:
            self._ref_cycle_idx = 0
        self._tool_palette.set_active_tool(tools[self._ref_cycle_idx])

    def _register_shortcuts(self) -> None:
        """Register all keyboard shortcuts as application-level QShortcut objects.

        ApplicationShortcut context ensures shortcuts fire regardless of which
        panel currently has keyboard focus (canvas, tree, property widget, etc.).

        QAction.setShortcut is intentionally NOT used for these keys to avoid
        duplicate-fire ambiguity.  Menu items show hints via \\t in their text.

        Space-bar temporary-pan is handled by keyPressEvent/keyReleaseEvent
        (requires hold semantics) and therefore does not appear here.
        """
        _ctx = Qt.ShortcutContext.ApplicationShortcut
        self._ref_cycle_idx = 0

        def _sc(key: str, slot) -> QShortcut:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(_ctx)
            sc.activated.connect(slot)
            return sc

        def _tool(tid: str):
            return lambda: self._tool_palette.set_active_tool(tid)

        # ── Navigation / tool selection ───────────────────────────────────
        _sc("V", _tool("select"))
        _sc("A", _tool("node_edit"))
        _sc("H", _tool("pan"))
        _sc("Z", _tool("zoom"))

        # ── Interpretation ────────────────────────────────────────────────
        _sc("P", _tool("horizon_pick"))
        _sc("F", _tool("fault_pick"))
        _sc("G", _tool("polygon"))
        _sc("S", _tool("new_section"))

        # ── Construction ──────────────────────────────────────────────────
        _sc("R", self._cycle_ref_line_tool)
        _sc("M", _tool("measure"))

        # ── Editing ───────────────────────────────────────────────────────
        _sc("Delete",       self._on_delete_shortcut)
        _sc("Ctrl+Z",       self._state.undo)
        _sc("Ctrl+Shift+Z", self._state.redo)
        _sc("Ctrl+C",       lambda: None)   # stub — future copy
        _sc("Ctrl+V",       lambda: None)   # stub — future paste
        _sc("Ctrl+A",       lambda: None)   # stub — future select all
        _sc("Escape",       self._on_escape_shortcut)

        # ── File ──────────────────────────────────────────────────────────
        _sc("Ctrl+N", self._on_new)
        _sc("Ctrl+O", self._on_open)
        _sc("Ctrl+S", self._on_save)

        # ── View ──────────────────────────────────────────────────────────
        _sc("Ctrl+0", self._zoom_to_fit)
        _sc("Ctrl+1", self._map_dock.toggleViewAction().trigger)
        _sc("Ctrl+2", self._section_dock.toggleViewAction().trigger)
        _sc("Ctrl+3", self._view3d_dock.toggleViewAction().trigger)
        _sc("Ctrl+4", self._project_panel.toggleViewAction().trigger)
        _sc("Ctrl+5", self._properties_panel.toggleViewAction().trigger)

    def _on_escape_shortcut(self) -> None:
        """Escape: cancel current operation and return to select tool."""
        self._tool_palette.set_active_tool("select")

    def _on_delete_shortcut(self) -> None:
        """Delete: cancel active pick or remove the last-added pick point."""
        tool = self._state.active_tool
        if tool in ("horizon_pick", "fault_pick"):
            self._tool_palette.set_active_tool("select")

    def _on_show_shortcuts_dialog(self) -> None:
        """Help → Keyboard Shortcuts: show the shortcuts reference dialog."""
        from section_tool.views.shortcuts_dialog import KeyboardShortcutsDialog
        dlg = KeyboardShortcutsDialog(self.SHORTCUT_REGISTRY, self)
        dlg.exec()

    # Space-bar temporary pan
    def keyPressEvent(self, event) -> None:
        from PySide6.QtCore import Qt as _Qt
        if (event.key() == _Qt.Key.Key_Space
                and not event.isAutoRepeat()
                and self._state.active_tool != "pan"):
            self._space_prev_tool = self._state.active_tool
            self._tool_palette.set_active_tool("pan")
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        from PySide6.QtCore import Qt as _Qt
        if (event.key() == _Qt.Key.Key_Space
                and not event.isAutoRepeat()
                and hasattr(self, "_space_prev_tool")):
            self._tool_palette.set_active_tool(self._space_prev_tool)
            del self._space_prev_tool
        super().keyReleaseEvent(event)

    def _on_frame_time(self, ms: float) -> None:
        """Show render frame time in the permanent hint label."""
        fps = 1000.0 / max(ms, 0.1)
        self._hint_label.setText(f"{ms:.0f} ms  |  {fps:.0f} fps")

    def _on_autosave(self) -> None:
        """Auto-save every 5 minutes if modified."""
        if not self._state.is_modified or not self._state.project_path:
            return
        pm = self._state.project_manager
        try:
            if pm.is_open:
                pm.autosave()
            else:
                autosave_path = self._state.project_path + ".autosave.h5"
                self._state.project.save(autosave_path)
            self._flash_status("Auto-saved")
        except Exception:
            pass

    def _check_autosave_recovery(self) -> None:
        """On startup, offer to recover from an autosave if one is newer than the project."""
        path = self._state.project_path
        if not path:
            return
        pm = self._state.project_manager
        if pm.is_open:
            if not pm.autosave_is_newer():
                return
        else:
            autosave_path = path + ".autosave.h5"
            if not os.path.exists(autosave_path):
                return
            import os.path as _osp
            if _osp.getmtime(autosave_path) <= _osp.getmtime(path):
                return
        autosave_path = path + ".autosave.h5"
        if not os.path.exists(autosave_path):
            return
        try:
            import os.path as _osp
            if _osp.getmtime(autosave_path) <= _osp.getmtime(path):
                return
        except OSError:
            return
        reply = QMessageBox.question(
            self, "Recover Auto-save",
            "An auto-saved version from a previous session was found and is newer "
            "than the saved project.\nRecover it?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self._state.open_project(autosave_path)
                # Restore the original path so next Save goes to the right file
                self._state._project_path = str(path)
                self._state.project_path_changed.emit(str(path))
            except Exception as exc:
                QMessageBox.warning(self, "Recovery Failed", str(exc))

    def _flash_status(self, msg: str) -> None:
        """Briefly show *msg* in the status bar, then restore."""
        self._status_label.setText(msg)
        QTimer.singleShot(2000, self._update_status)

    def _on_map_status(self, msg: str) -> None:
        if msg:
            self._status_label.setText(msg)
        else:
            self._update_status()

    def _on_context_toolbar_action(self, action: str) -> None:
        """Route actions from context toolbar that need app-level handling."""
        if action == "new_horizon":
            self._add_new_horizon()
        elif action == "new_fault":
            self._add_new_fault()
        elif action == "new_polygon":
            self._add_new_polygon()
        elif action == "end_pick":
            self._tool_palette.set_active_tool("select")

    def _on_panel_properties(self, cat: str, idx: int) -> None:
        if cat == "Horizons":
            self._horizon_properties(idx)
        elif cat == "Faults":
            self._fault_properties(idx)

    def _ensure_pick_target(self, tool_id: str) -> None:
        """Phase 2: auto-select a pick target if none is set (non-blocking)."""
        cat = "Horizons" if tool_id == "horizon_pick" else "Faults"
        cur_cat = self._state.active_pick_category
        cur_idx = self._state.active_pick_index
        proj = self._state.project
        picks = proj.horizon_picks if cat == "Horizons" else proj.fault_picks

        if cur_cat == cat and cur_idx is not None and cur_idx < len(picks):
            return  # already valid

        if picks:
            # Auto-select the first available object
            self._state.set_active_pick_target(cat, 0)
        else:
            # No objects yet — prompt via status bar, don't open a blocking dialog
            kind = "horizon" if cat == "Horizons" else "fault"
            self._status_label.setText(
                f"No {kind}s yet.  Right-click '{cat}' in Project panel → "
                f"Add {kind.title()}…  then activate this tool."
            )
            # Return to select so the user isn't stuck in pick mode with no target
            self._tool_palette.set_active_tool("select")

    def _update_pick_status(self) -> None:
        """Phase 2: show picking target + existing pick count in status bar."""
        cat = self._state.active_pick_category
        idx = self._state.active_pick_index
        if cat is None or idx is None:
            return
        proj = self._state.project
        picks = proj.horizon_picks if cat == "Horizons" else proj.fault_picks
        if idx >= len(picks):
            return
        hp = picks[idx]
        sec = self._state.active_section
        n = hp.n_picks_for_section(sec.name) if sec else hp.n_picks
        self._status_label.setText(
            f"Picking: {hp.name}  ({n} existing picks)  |  Right-click or Escape to end"
        )

    def _on_pick_target_selected(self, cat: str, idx: int) -> None:
        """Clicking a horizon/fault in the panel also activates the matching tool."""
        self._state.set_active_pick_target(cat, idx)
        if cat == "Horizons":
            self._tool_palette.set_active_tool("horizon_pick")
        elif cat == "Faults":
            self._tool_palette.set_active_tool("fault_pick")

    def _on_state_tool_changed(self, tool_id: str) -> None:
        """Sync palette when tool changes via state (e.g. from map view draw finish)."""
        if self._tool_palette.active_tool != tool_id:
            self._tool_palette.set_active_tool(tool_id)

    def _on_tool_changed(self, tool_id: str) -> None:
        """Route palette tool activation to views and AppState."""
        self._state.set_active_tool(tool_id)
        self._section_view.set_picking_active(tool_id == "horizon_pick")
        self._section_view.set_fault_picking(tool_id == "fault_pick")
        self._section_view.set_polygon_drawing(tool_id == "polygon")
        self._section_view.set_ref_line_tool(tool_id)
        self._section_view.apply_tool_cursor(tool_id)
        self._map_view.apply_tool_cursor(tool_id)
        # Phase 2: update status bar with picking info
        if tool_id in ("horizon_pick", "fault_pick"):
            self._ensure_pick_target(tool_id)
            self._update_pick_status()
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
            settings = QSettings("Geoscience", "CrossSectionTool")
            settings.setValue("window/geometry", self.saveGeometry())
            settings.setValue("window/state", self.saveState())
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
    # Detect dark / light system theme
    _lum = app.palette().window().color().lightness()
    _dark = _lum < 128
    if _dark:
        _bg      = "#2B2B2B"
        _bg2     = "#333333"
        _bg3     = "#3C3C3C"
        _border  = "#555555"
        _text    = "#E0E0E0"
        _dim     = "#999999"
        _menu_bg = "#2D2D2D"
        _tab_sel = "#3C3C3C"
        _handle  = "#4A4A4A"
    else:
        _bg      = "#F5F5F5"
        _bg2     = "#EBEBEB"
        _bg3     = "#E0E0E0"
        _border  = "#C8C8C8"
        _text    = "#1A1A1A"
        _dim     = "#888888"
        _menu_bg = "#FFFFFF"
        _tab_sel = "#FFFFFF"
        _handle  = "#CCCCCC"
    app.setStyleSheet(f"""
        QMainWindow                     {{ background: {_bg}; }}
        QWidget                         {{ font-size: 9pt; color: {_text}; }}
        QToolTip                        {{ padding: 4px 6px; background: {_menu_bg};
                                          color: {_text}; border: 1px solid {_border};
                                          font-size: 8pt; }}
        QMenuBar                        {{ background: {_bg2}; border-bottom: 1px solid {_border}; font-size: 9pt; }}
        QMenuBar::item:selected         {{ background: #3B82F6; color: white; }}
        QMenu                           {{ background: {_menu_bg}; border: 1px solid {_border}; font-size: 9pt; }}
        QMenu::item:selected            {{ background: #3B82F6; color: white; }}
        QTabWidget::pane                {{ border: 1px solid {_border}; margin: 0; }}
        QTabBar::tab                    {{ padding: 5px 14px; font-size: 9pt; background: {_bg2};
                                          border: 1px solid {_border}; border-bottom: none;
                                          min-width: 60px; }}
        QTabBar::tab:selected           {{ background: {_tab_sel}; border-bottom: 2px solid #3B82F6; }}
        QDockWidget::title              {{ background: #383838; color: #c8c8c8;
                                          padding: 4px 6px; font-size: 9pt; font-weight: bold; }}
        QSplitter::handle               {{ background: {_handle}; }}
        QSplitter::handle:horizontal    {{ width: 5px; }}
        QTreeWidget                     {{ font-size: 9pt; border: none; background: {_bg}; }}
        QTreeWidget::item               {{ min-height: 22px; padding: 1px 2px; }}
        QTreeWidget::item:selected      {{ background: #3B82F6; color: white; }}
        QStatusBar                      {{ font-size: 8pt; background: {_bg2}; border-top: 1px solid {_border}; }}
        QScrollBar:vertical             {{ width: 10px; background: {_bg}; }}
        QScrollBar::handle:vertical     {{ background: {_border}; border-radius: 4px; min-height: 20px; }}
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical   {{ height: 0; }}
        QSpinBox, QDoubleSpinBox        {{ font-size: 9pt; min-width: 60px; padding: 2px 4px;
                                          background: {_bg2}; border: 1px solid {_border};
                                          border-radius: 3px; }}
        QComboBox                       {{ font-size: 9pt; min-width: 60px; padding: 2px 4px;
                                          background: {_bg2}; border: 1px solid {_border};
                                          border-radius: 3px; }}
        QComboBox::drop-down            {{ width: 16px; }}
        QLineEdit                       {{ font-size: 9pt; padding: 2px 4px;
                                          background: {_bg2}; border: 1px solid {_border};
                                          border-radius: 3px; }}
        QPushButton                     {{ font-size: 8pt; padding: 3px 8px;
                                          background: {_bg3}; border: 1px solid {_border};
                                          border-radius: 3px; }}
        QPushButton:hover               {{ background: #4A90D9; color: white; border-color: #3B82F6; }}
        QPushButton:pressed             {{ background: #3B82F6; color: white; }}
        QLabel                          {{ font-size: 9pt; background: transparent; }}
        QCheckBox                       {{ font-size: 9pt; spacing: 5px; }}
        QGroupBox                       {{ font-size: 9pt; font-weight: bold;
                                          border: 1px solid {_border}; border-radius: 4px;
                                          margin-top: 8px; padding-top: 4px; }}
        QGroupBox::title                {{ subcontrol-origin: margin; left: 8px;
                                          padding: 0 3px; }}
    """)
    # 500ms tooltip delay
    from PySide6.QtWidgets import QToolTip
    from PySide6.QtGui import QFont
    QToolTip.setFont(QFont("Segoe UI", 9))

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
