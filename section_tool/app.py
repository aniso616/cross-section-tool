from __future__ import annotations

import logging
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

from PySide6.QtCore import QObject, Qt, QSize, QSettings, QTimer, QPoint
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
    QStackedLayout,
    QStackedWidget,
    QStyle,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import QSize as _QSize   # toolbar icon size

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
# CRS reprojection helper for vector features
# ---------------------------------------------------------------------------

def _reproject_features(features, src_epsg: int, dst_epsg: int) -> list:
    """Return *features* with all geometry coordinates reprojected from
    *src_epsg* to *dst_epsg*.  Supports Point, LineString, Polygon,
    MultiPoint, MultiLineString, MultiPolygon geometry types."""
    from section_tool.core.crs import reproject_xy

    def _reproject_coord(xy):
        x, y = reproject_xy(xy[0], xy[1], src_epsg, dst_epsg)
        return (x, y) + tuple(xy[2:])  # preserve Z if present

    def _reproject_ring(ring):
        return [_reproject_coord(c) for c in ring]

    def _reproject_geometry(geom):
        if geom is None:
            return geom
        gtype = geom.get("type", "")
        coords = geom.get("coordinates")
        if coords is None:
            return geom
        if gtype == "Point":
            new_coords = _reproject_coord(coords)
        elif gtype in ("LineString", "MultiPoint"):
            new_coords = _reproject_ring(coords)
        elif gtype in ("Polygon", "MultiLineString"):
            new_coords = [_reproject_ring(ring) for ring in coords]
        elif gtype == "MultiPolygon":
            new_coords = [[_reproject_ring(ring) for ring in poly]
                          for poly in coords]
        else:
            return geom  # unknown type — leave unchanged
        result = dict(geom)
        result["coordinates"] = new_coords
        return result

    out = []
    for feat in features:
        f2 = dict(feat)
        if "geometry" in f2 and f2["geometry"] is not None:
            f2["geometry"] = _reproject_geometry(dict(f2["geometry"]))
        out.append(f2)
    return out


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
        ("W",            "Well Top Pick",            "Interpretation"),
        ("R",            "Reference Line (cycle)",   "Construction"),
        ("M",            "Measure",                  "Construction"),
        ("E",            "Extend Pick",              "Construction"),
        ("T",            "Trim Pick",                "Construction"),
        ("D",            "Dip-Constrained Pick",     "Construction"),
        ("K",            "Kink Band",                "Construction"),
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
        self._sync_theme_actions()
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
        # Restore saved theme
        from section_tool.style import set_theme
        _saved_theme = _s.value("view/theme", "dark")
        try:
            set_theme(_saved_theme)
        except Exception:
            set_theme("dark")

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

        # ── Restoration panel ─────────────────────────────────────────────────
        from section_tool.views.restoration_panel import RestorationPanel
        self._restoration_panel = QDockWidget("Restoration", self)
        self._restoration_panel.setObjectName("RestorationDock")
        self._restoration_panel.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self._restoration_panel.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self._restoration_widget = RestorationPanel(self._state, self)
        self._restoration_panel.setWidget(self._restoration_widget)

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
        # Restoration panel: tabbed behind Properties in left column
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._restoration_panel)
        self.tabifyDockWidget(self._properties_panel, self._restoration_panel)
        # Default left-column width
        self.resizeDocks([self._map_dock], [400], Qt.Orientation.Horizontal)

    def _reset_layout(self) -> None:
        """Restore the default dock arrangement and clear saved state."""
        settings = QSettings("Geoscience", "CrossSectionTool")
        settings.remove("window/state")
        for dock in (self._map_dock, self._section_dock, self._view3d_dock,
                     self._project_panel, self._properties_panel,
                     self._restoration_panel):
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
        import_menu.addSeparator()
        self._import_vector_action = QAction("Shapefile / GeoPackage…", self)
        self._import_vector_action.triggered.connect(self._on_import_vector)
        import_menu.addAction(self._import_vector_action)
        import_menu.addSeparator()
        self._import_surface_action = QAction("Surface / Horizon Grid (XYZ)…", self)
        self._import_surface_action.triggered.connect(self._on_import_surface)
        import_menu.addAction(self._import_surface_action)
        self._gen_surface_action = QAction("Generate Surface from Horizon…", self)
        self._gen_surface_action.triggered.connect(self._on_generate_surface)
        import_menu.addAction(self._gen_surface_action)
        file_menu.addMenu(import_menu)

        # Export submenu
        export_menu = QMenu("&Export", self)
        self._export_section_action = QAction("&Export Section…\tCtrl+E", self)
        self._export_section_action.triggered.connect(self._on_export_section_dialog)
        export_menu.addAction(self._export_section_action)
        export_menu.addSeparator()
        self._export_img_action = QAction("Quick Image (PNG/SVG/PDF)…", self)
        self._export_img_action.triggered.connect(self._on_export_section_image)
        export_menu.addAction(self._export_img_action)
        self._export_csv_action = QAction("Horizons to CSV…", self)
        self._export_csv_action.triggered.connect(self._on_export_horizons_csv)
        export_menu.addAction(self._export_csv_action)
        file_menu.addMenu(export_menu)

        file_menu.addSeparator()
        self._props_action = QAction("Properties…", self)
        self._props_action.triggered.connect(self._on_project_properties)
        file_menu.addAction(self._props_action)

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
        _rest_ta = self._restoration_panel.toggleViewAction()
        _rest_ta.setText("&Restoration Panel\tCtrl+6")
        view_menu.addAction(_rest_ta)
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
        self._cross_sec_ghosts_action = QAction("Show &Cross-Section Picks", self)
        self._cross_sec_ghosts_action.setCheckable(True)
        self._cross_sec_ghosts_action.setChecked(True)
        self._cross_sec_ghosts_action.toggled.connect(
            lambda v: self._section_view.set_cross_section_ghosts_visible(v))
        view_menu.addAction(self._cross_sec_ghosts_action)
        view_menu.addSeparator()
        from PySide6.QtGui import QActionGroup
        from PySide6.QtWidgets import QMenu as _QMenu
        theme_menu = _QMenu("&Theme", self)
        self._theme_action_group = QActionGroup(self)
        self._theme_action_group.setExclusive(True)
        for _tid, _tlabel in [("dark", "Dark (on-screen)"), ("print", "Print (white)")]:
            _ta = QAction(_tlabel, self)
            _ta.setCheckable(True)
            _ta.setData(_tid)
            self._theme_action_group.addAction(_ta)
            theme_menu.addAction(_ta)
        self._theme_action_group.triggered.connect(self._on_theme_changed)
        view_menu.addMenu(theme_menu)
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
        # Grouped by the workflow pipeline (separators between groups).  The basic
        # interaction tools (Select/Pan/Zoom/Pick…) live in the tool palette and
        # keyboard shortcuts, not here — the menu is for dialogs/actions.

        # --- Interpretation: pick-construction helpers ---
        construct_sub = QMenu("&Construction Tools", self)
        for tid, label, key in [
            ("extend",          "Extend Pick",          "E"),
            ("trim",            "Trim Pick",            "T"),
            ("parallel",        "Parallel Offset",      ""),
            ("dip_constrained", "Dip-Constrained Pick", "D"),
            ("kink_band",       "Kink Band",            "K"),
        ]:
            label_str = f"{label}\t{key}" if key else label
            a = QAction(label_str, self)
            a.triggered.connect(
                lambda _checked, t=tid: self._tool_palette.set_active_tool(t))
            construct_sub.addAction(a)
        tools_menu.addMenu(construct_sub)

        # --- Time–Depth / Conversion ---
        tools_menu.addSeparator()
        self._depth_stretch_action = QAction("&Depth Stretch (Time→Depth)…", self)
        self._depth_stretch_action.triggered.connect(self._on_depth_stretch)
        tools_menu.addAction(self._depth_stretch_action)   # well calibration folds in here
        self._view_segy_hdr_action = QAction("View SEG-Y Header…", self)
        self._view_segy_hdr_action.triggered.connect(self._on_view_segy_header)
        tools_menu.addAction(self._view_segy_hdr_action)

        # --- Structural / Restoration ---
        tools_menu.addSeparator()
        self._balance_check_action = QAction("Check Section &Balance…", self)
        self._balance_check_action.triggered.connect(self._on_balance_check)
        tools_menu.addAction(self._balance_check_action)
        self._restoration_stack_action = QAction("Restoration &Stack…", self)
        self._restoration_stack_action.triggered.connect(self._on_restoration_stack)
        tools_menu.addAction(self._restoration_stack_action)
        self._topology_audit_action = QAction("&Topology Audit…", self)
        self._topology_audit_action.triggered.connect(self._on_topology_audit)
        tools_menu.addAction(self._topology_audit_action)

        # --- Thermal ---
        tools_menu.addSeparator()
        self._thermal_action = QAction("&Thermal Modeling…", self)
        self._thermal_action.triggered.connect(self._on_thermal_modeling)
        tools_menu.addAction(self._thermal_action)

        # --- Analysis / Utilities ---
        tools_menu.addSeparator()
        self._attribute_table_action = QAction("&Attribute Table…", self)
        self._attribute_table_action.triggered.connect(self._on_attribute_table)
        tools_menu.addAction(self._attribute_table_action)
        self._set_aoi_action = QAction("Set Area of Interest (AOI)…", self)
        self._set_aoi_action.triggered.connect(self._on_set_aoi)
        tools_menu.addAction(self._set_aoi_action)
        self._clear_aoi_action = QAction("Clear AOI", self)
        self._clear_aoi_action.triggered.connect(lambda: self._state.set_aoi(None))
        tools_menu.addAction(self._clear_aoi_action)

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
        self._project_panel.visibility_changed.connect(self._on_panel_visibility)
        self._project_panel.object_selected.connect(self._on_panel_object_selected)
        self._project_panel.add_requested.connect(self._on_panel_add)
        # Bidirectional selection sync: section view → project panel
        s.selected_entity_changed.connect(self._project_panel.select_entity)
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
        s.active_slice_changed.connect(lambda _: self._update_tool_availability())
        s.horizon_pick_added.connect(lambda _: self._update_tool_availability())
        s.horizon_pick_removed.connect(lambda _: self._update_tool_availability())
        s.fault_pick_added.connect(lambda _: self._update_tool_availability())
        s.fault_pick_removed.connect(lambda _: self._update_tool_availability())
        s.project_changed.connect(self._update_tool_availability)
        # Phase 7: undo/redo status flashes
        s.undo_performed.connect(lambda d: self._flash_status(f"Undo: {d}"))
        s.redo_performed.connect(lambda d: self._flash_status(f"Redo: {d}"))
        s.theme_changed.connect(self._section_view.on_theme_changed)
        # Phase 3: wire node selection → properties panel
        # node_selected emits (str, int, int) separately; pack into tuple for set_selected_node
        self._section_view.node_selected.connect(
            lambda cat, oi, pi: self._properties_panel.set_selected_node((cat, oi, pi)))
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
        # Restoration panel: step_changed → rebuild panel and request render
        self._restoration_widget.step_changed.connect(self._on_restoration_step_changed)
        # Rebuild restoration panel when project changes
        s.project_changed.connect(self._restoration_widget.rebuild)
        # FPS display from section view
        self._section_view.frame_time_ms.connect(self._on_frame_time)
        # All keyboard shortcuts registered centrally
        self._register_shortcuts()
        # Initial tool availability pass
        self._update_tool_availability()

    # ------------------------------------------------------------------
    # Restoration
    # ------------------------------------------------------------------

    def _on_restoration_step_changed(self, step: int) -> None:
        """Restoration panel advanced to *step* — re-render section."""
        self._section_view.request_render()

    # ------------------------------------------------------------------
    # Tool availability
    # ------------------------------------------------------------------

    def _zslice_active(self) -> bool:
        """True when a horizontal z-slice is the active workspace (Model A guard)."""
        return getattr(self._state.active_slice, "kind", None) == "horizontal"

    def _update_tool_availability(self, *_args) -> None:
        """Recompute which palette tools are enabled based on current state."""
        has_section = self._state.active_section is not None
        proj = self._state.project
        has_picks = bool(proj.horizon_picks or proj.fault_picks)
        self._tool_palette.update_tool_availability(
            has_section, has_picks, section_workspace=not self._zslice_active())

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
        depth_domain: str = "depth",
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

    def _on_project_properties(self) -> None:
        if not self._state.project:
            return
        from section_tool.views.project_properties_dialog import ProjectPropertiesDialog
        ProjectPropertiesDialog(self._state, self).exec()

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

    def _on_import_surface(self) -> None:
        """Import → Surface: load an XYZ surface file."""
        from section_tool.io.surface_readers import read_surface, supported_extensions
        exts = " ".join(f"*{e}" for e in supported_extensions())
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Surface",
            "", f"Surface Files ({exts});;All Files (*)",
        )
        if not path:
            return
        project_crs = self._state.project.crs_epsg
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            # Load with crs_epsg=0 so we can detect/ask about source CRS
            surf = read_surface(path, crs_epsg=0)
        except Exception as exc:
            QMessageBox.critical(self, "Import Failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        # Ask user about source CRS if unknown, then reproject if needed
        src_epsg = surf.crs_epsg  # 0 means unknown
        if not src_epsg:
            from section_tool.views.source_crs_dialog import SourceCRSDialog
            from PySide6.QtWidgets import QDialog as _QDialog
            dlg = SourceCRSDialog(project_crs, os.path.basename(path), self)
            if dlg.exec() == _QDialog.Accepted:
                src_epsg = dlg.source_epsg()
            else:
                return  # user cancelled

        if src_epsg and src_epsg != project_crs:
            from section_tool.core.crs import reproject_points_xy
            import numpy as np
            xs, ys = reproject_points_xy(
                surf.points[:, 0], surf.points[:, 1], src_epsg, project_crs
            )
            surf.points[:, 0] = xs
            surf.points[:, 1] = ys

        surf.crs_epsg = project_crs
        surf._interpolator = None  # invalidate any cached interpolator

        try:
            self._state.add_surface(surf)
            b = surf.bounds()
            zr = surf.z_range()
        except Exception as exc:
            QMessageBox.critical(self, "Import Failed", str(exc))
            return

        # Show info THEN render — QMessageBox blocks the event loop, so draw_idle()
        # calls inside render() would never paint until after the dialog anyway.
        QMessageBox.information(
            self, "Surface Loaded",
            f"'{surf.name}'\nFormat: {surf.source_format}\n"
            f"Points: {surf.n_points:,}\n"
            f"Z domain: {surf.z_domain}  range: {zr[0]:.1f}–{zr[1]:.1f}\n"
            f"Bounds: {b[0]:.0f}–{b[2]:.0f} E, {b[1]:.0f}–{b[3]:.0f} N",
        )
        self._section_view.render()
        self._map_view.render()

    def _on_generate_surface(self) -> None:
        """Import → Generate Surface from Horizon: interpolate picks to a grid."""
        from PySide6.QtWidgets import QMessageBox as _QMB
        from section_tool.views.generate_surface_dialog import GenerateSurfaceDialog
        if not self._state.project.horizon_picks:
            _QMB.information(
                self, "Generate Surface",
                "No horizon picks available. Pick a horizon on at least one "
                "section first.",
            )
            return
        GenerateSurfaceDialog(self._state, self).exec()

    def _on_set_aoi(self) -> None:
        """Tools → Set AOI: define a rectangular area of interest."""
        from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout,
                                         QDoubleSpinBox, QLineEdit)
        from section_tool.core.aoi import AOI

        dlg = QDialog(self)
        dlg.setWindowTitle("Set Area of Interest")
        layout = QFormLayout(dlg)

        def _spin(val=0.0, lo=-1e9, hi=1e9):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(1)
            s.setValue(val)
            s.setSingleStep(1000.0)
            return s

        # Pre-fill from current map limits if available
        try:
            xl = self._map_view._ax.get_xlim()
            yl = self._map_view._ax.get_ylim()
        except Exception:
            xl = yl = (0.0, 10000.0)

        xmin_s = _spin(xl[0]); xmax_s = _spin(xl[1])
        ymin_s = _spin(min(yl)); ymax_s = _spin(max(yl))
        name_ed = QLineEdit("AOI")

        layout.addRow("Name:", name_ed)
        layout.addRow("X min (E):", xmin_s)
        layout.addRow("X max (E):", xmax_s)
        layout.addRow("Y min (N):", ymin_s)
        layout.addRow("Y max (N):", ymax_s)

        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        layout.addRow(bb)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        aoi = AOI.from_rectangle(
            x_min=xmin_s.value(), x_max=xmax_s.value(),
            y_min=ymin_s.value(), y_max=ymax_s.value(),
            crs_epsg=self._state.project.crs_epsg,
            name=name_ed.text().strip() or "AOI",
        )
        self._state.set_aoi(aoi)
        self._map_view.render()
        self._section_view.render()

    def _on_topology_audit(self) -> None:
        """Tools → Topology Audit: check interpretation hygiene for the active section."""
        from section_tool.views.topology_audit_dialog import TopologyAuditDialog
        dlg = TopologyAuditDialog(self._state, parent=self)
        dlg.exec()

    def _on_depth_stretch(self) -> None:
        """Tools → Depth Stretch: configure and apply a time→depth conversion.

        Applying installs the velocity model and re-derives seismic-tied geometry
        from its TWT anchors (depth-native stays fixed); re-render shows it.
        """
        from section_tool.views.depth_stretch_dialog import DepthStretchDialog

        def _applied():
            try:
                self._state._set_modified(True)
            except Exception:
                pass
            try:
                self._section_view.render()
            except Exception:
                pass
            try:
                self._map_view.request_render()
            except Exception:
                pass

        dlg = DepthStretchDialog(self._state, on_apply=_applied, parent=self)
        dlg.exec()

    def _on_thermal_modeling(self) -> None:
        """Tools → Thermal Modeling: open the thermal modeling dialog."""
        from section_tool.views.thermal_modeling_dialog import ThermalModelingDialog
        section = self._state.active_section
        if section is None:
            QMessageBox.information(
                self, "No active section",
                "Select or create a section before running thermal modeling.",
            )
            return
        dlg = ThermalModelingDialog(self._state, section, parent=self)
        dlg.exec()

    def _on_balance_check(self) -> None:
        """Tools → Check Section Balance: open the balance check dialog."""
        from section_tool.views.balance_check_dialog import BalanceCheckDialog
        section = self._state.active_section
        if section is None:
            QMessageBox.information(
                self, "No active section",
                "Select a section before running the balance check.",
            )
            return
        dlg = BalanceCheckDialog(self._state, section, parent=self)
        dlg.exec()

    def _on_restoration_stack(self) -> None:
        """Tools → Restoration Stack: show the full restoration sequence timeline."""
        from section_tool.views.restoration_stack_dialog import RestorationStackDialog
        dlg = RestorationStackDialog(self._state, parent=self)
        dlg.exec()

    def _on_attribute_table(self) -> None:
        """Tools → Attribute Table: tabular view of all geological element attributes."""
        from section_tool.views.attribute_table_dialog import AttributeTableDialog
        dlg = AttributeTableDialog(self._state, parent=self)
        dlg.exec()


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
        imported_twt = False
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
                max_offset=dlg.max_offset,
                crs_epsg=self._state.project.crs_epsg,
                extent_x_min=float(xr[0]),
                extent_x_max=float(xr[1]),
                extent_y_min=float(yr[0]),
                extent_y_max=float(yr[1]),
                n_traces_total=int(n_tot),
            )
            self._state.add_seismic_ref(ref)
            if ref.domain == "twt":
                imported_twt = True
            # Auto-zoom to show seismic extent
            self._map_view.zoom_to_all_data()

        # A time volume can't display directly in a depth section — bridge into
        # the Depth Stretch tool (bulk/average bootstrap pre-filled) so the
        # conversion is set up explicitly, the front door for time→depth.
        if imported_twt:
            self._on_depth_stretch()

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

        # Resolve output path INSIDE the project's own cache/ folder, keyed by
        # section geometry + SEG-Y path.  Previously this used
        # os.path.dirname(project_path) (the project's PARENT) with a name-only
        # key, so every project under the same parent shared one global cache
        # file — a "Section 1" in project A would overwrite/serve project B's
        # "Section 1", rendering a wrong-geometry extract.  seismic_extract_npy_path
        # is project-scoped and geometry-hashed, so collisions cannot happen and
        # editing a section's nodes re-extracts automatically.
        import os
        if not self._state.project_path:
            QMessageBox.information(
                self, "No Project",
                "Open or save a project before extracting seismic.")
            return
        out_npy = self._state.project_manager.seismic_extract_npy_path(
            section, ref.name, ref.path)
        os.makedirs(os.path.dirname(out_npy), exist_ok=True)

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
                max_offset=float(getattr(ref, "max_offset", 500.0)),
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

    def _on_import_vector(self) -> None:
        """Import Shapefile / GeoPackage / GeoJSON as a map overlay."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Vector Data", "",
            "Vector Files (*.shp *.gpkg *.geojson *.GeoJSON);;All Files (*)"
        )
        if not path:
            return
        try:
            import fiona
            from PySide6.QtWidgets import QDialog as _QDialog
            with _wait_cursor():
                with fiona.open(path) as src:
                    geom_type = src.schema["geometry"]
                    crs       = src.crs
                    features  = [dict(f) for f in src]

            from section_tool.core.crs import epsg_from_fiona_crs
            project_crs = self._state.project.crs_epsg
            src_epsg = epsg_from_fiona_crs(crs)
            if src_epsg is None:
                from section_tool.views.source_crs_dialog import SourceCRSDialog
                dlg = SourceCRSDialog(project_crs, os.path.basename(path), self)
                if dlg.exec() == _QDialog.Accepted:
                    src_epsg = dlg.source_epsg()
                else:
                    return  # user cancelled
            if src_epsg and src_epsg != project_crs:
                features = _reproject_features(features, src_epsg, project_crs)

            self._state.add_vector_layer(path, features, crs, geom_type)
            n = len(features)
            if hasattr(self, "status_strip"):
                self.status_strip.set_hint(
                    f"Loaded {n} features from {os.path.basename(path)}"
                )
        except ImportError:
            QMessageBox.critical(self, "Missing library",
                                 "fiona is required for vector import.\n"
                                 "Install with: pip install fiona")
        except Exception as exc:
            QMessageBox.warning(self, "Import Error", str(exc))

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
            QButtonGroup, QDialog, QDialogButtonBox, QFormLayout, QDoubleSpinBox,
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
        origin_grp = QButtonGroup(dlg)
        origin_grp.setExclusive(True)
        origin_grp.addButton(rb_center, 0)
        origin_grp.addButton(rb_start, 1)
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

    def _sync_theme_actions(self) -> None:
        from section_tool.style import get_theme
        current = get_theme().name
        for action in self._theme_action_group.actions():
            action.setChecked(action.data() == current)

    def _on_theme_changed(self, action) -> None:
        from section_tool.style import set_theme
        theme_id = action.data()
        try:
            set_theme(theme_id)
        except Exception:
            return
        QSettings("Geoscience", "CrossSectionTool").setValue("view/theme", theme_id)
        self._state.theme_changed.emit(theme_id)

    def _on_export_section_dialog(self) -> None:
        """Open the Export Section dialog with live preview and print parameters."""
        section = self._state.active_section
        if section is None:
            QMessageBox.information(self, "Export", "Activate a section first.")
            return
        from section_tool.export.print_dialog import PrintExportDialog
        dlg = PrintExportDialog(self._state, section, self)
        dlg.exec()

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
            elif category == "Polygons" and index < len(proj.polygons):
                self._state.remove_polygon(proj.polygons[index])
            elif category == "Surfaces" and index < len(proj.surfaces if hasattr(proj, "surfaces") else []):
                self._state.remove_surface(proj.surfaces[index])
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

    def _on_panel_visibility(self, category: str, index: int, visible: bool) -> None:
        import copy
        proj = self._state.project
        try:
            if category == "Horizons" and index < len(proj.horizon_picks):
                pick = copy.deepcopy(proj.horizon_picks[index])
                pick.visible = visible
                self._state.update_horizon_pick(index, pick)
            elif category == "Faults" and index < len(proj.fault_picks):
                pick = copy.deepcopy(proj.fault_picks[index])
                pick.visible = visible
                self._state.update_fault_pick(index, pick)
            elif category == "Polygons" and index < len(proj.polygons):
                poly = copy.deepcopy(proj.polygons[index])
                poly.visible = visible
                self._state.update_polygon(index, poly)
            elif category == "Wells" and index < len(proj.wells):
                well = copy.deepcopy(proj.wells[index])
                well.visible = visible
                self._state.update_well(index, well)
        except Exception:
            pass

    def _on_panel_object_selected(self, category: str, index: int) -> None:
        self._properties_panel.set_selected_object(category, index)
        # Also set selection in section view (panel → section sync)
        self._section_view.set_selected_from_panel(category, index)

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

    @staticmethod
    def _formation_for_face(face, horizon_lines_info: list) -> tuple[str, str]:
        """Return (formation_name, fill_color_hex) for a Shapely polygon face.

        Uses the centroid depth to find which two horizons bracket it (depth-sorted).
        All polygons between the same pair of horizons share the same name+color,
        so fault-split blocks within one stratigraphic unit stay visually consistent.
        """
        from shapely.geometry import LineString as _LS
        cx, cy = face.centroid.x, face.centroid.y

        # Interpolate each horizon's depth at the centroid's x position
        hz_depths: list[tuple[float, str, str]] = []  # (depth, name, color)
        for h_name, h_line, h_color, h_form_above, h_form_below in horizon_lines_info:
            try:
                vert = _LS([(cx, -1e9), (cx, 1e9)])
                inter = h_line.intersection(vert)
                if inter.is_empty:
                    continue
                if inter.geom_type == "Point":
                    hz_depths.append((inter.y, h_name, h_color))
                elif inter.geom_type == "MultiPoint":
                    pt = min(inter.geoms, key=lambda p: abs(p.y - cy))
                    hz_depths.append((pt.y, h_name, h_color))
            except Exception:
                continue

        hz_depths.sort(key=lambda t: t[0])  # shallowest first

        above_name, above_color = "Surface", "#87CEEB"
        below_name, below_color = "Base", "#8B7765"

        for depth, h_name, h_color in hz_depths:
            if depth > cy:
                below_name, below_color = h_name, h_color
                break
            above_name, above_color = h_name, h_color

        if above_name == "Surface":
            form_name = f"Above {below_name}"
            fill_color = below_color
        elif below_name == "Base":
            form_name = f"Below {above_name}"
            fill_color = above_color
        else:
            form_name = f"{above_name} – {below_name}"
            fill_color = above_color   # convention: color by upper contact

        return form_name, fill_color

    def _on_generate_polygons(self) -> None:
        """Generate filled polygons from horizon/fault boundaries using Shapely directly."""
        from shapely.geometry import LineString
        from shapely.ops import unary_union, polygonize
        from section_tool.core.polygons import SectionPolygon
        import numpy as np

        section = self._state.active_section
        if section is None:
            QMessageBox.warning(self, "No Section", "Activate a section first.")
            return

        total = section.total_length()
        lines = []
        horizon_lines_info = []  # (name, LineString, color, form_above, form_below)

        # Collect horizon lines, extended to section edges
        for hp in self._state.project.horizon_picks:
            idxs = hp.section_indices(section.name)
            if len(idxs) < 2:
                continue
            d = hp._distances[idxs]
            z = hp._depths[idxs]
            order = np.argsort(d)
            d, z = d[order], z[order]
            coords = list(zip(d.tolist(), z.tolist()))
            if d[0] > 0:
                dd = d[1] - d[0]
                slope = (z[1] - z[0]) / dd if abs(dd) > 1e-6 else 0.0
                coords.insert(0, (0.0, float(z[0] - slope * d[0])))
            if d[-1] < total:
                dd = d[-1] - d[-2]
                slope = (z[-1] - z[-2]) / dd if abs(dd) > 1e-6 else 0.0
                coords.append((float(total), float(z[-1] + slope * (total - d[-1]))))
            ls = LineString(coords)
            lines.append(ls)
            horizon_lines_info.append((
                hp.name, ls, hp.color,
                getattr(hp, "formation_above", ""),
                getattr(hp, "formation_below", ""),
            ))

        # Collect fault lines (no extension)
        for fp in self._state.project.fault_picks:
            idxs = fp.section_indices(section.name)
            if len(idxs) < 2:
                continue
            d = fp._distances[idxs]
            z = fp._depths[idxs]
            order = np.argsort(d)
            lines.append(LineString(list(zip(d[order].tolist(), z[order].tolist()))))

        if not lines:
            QMessageBox.information(self, "No Polygons Found",
                                    "No horizons or faults found on this section.\n"
                                    "Pick at least one horizon spanning the full width.")
            return

        max_d = self._section_view._compute_max_depth(section)
        boundary = LineString([(0,0),(total,0),(total,max_d),(0,max_d),(0,0)])
        lines.append(boundary)

        try:
            merged = unary_union(lines)
            faces = list(polygonize(merged))
        except Exception as exc:
            QMessageBox.critical(self, "Generation Error",
                                 f"Polygon generation failed:\n{exc}")
            return

        min_area = total * max_d * 0.001
        faces = [f for f in faces if f.area >= min_area]

        if not faces:
            QMessageBox.information(self, "No Polygons Found",
                                    "No closed regions detected.\n"
                                    "Ensure horizons extend across the full section width.")
            return

        reply = QMessageBox.question(
            self, "Import Polygons",
            f"{len(faces)} region(s) detected. Import as polygons?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Formation counter for fallback numbering when no horizons bound a face
        _FALLBACK = ["#4878d0","#ee854a","#6acc64","#d65f5f","#956cb4",
                     "#8c613c","#dc7ec0","#797979","#d5bb67","#82c6e2"]
        existing = len(self._state.project.polygons)
        # Track formation→color for consistency across fault-split blocks
        form_color_map: dict[str, str] = {}
        added = []
        for i, face in enumerate(faces):
            coords = list(face.exterior.coords)
            if coords[0] == coords[-1]:
                coords = coords[:-1]
            if len(coords) < 3:
                continue

            form_name, fill_color = self._formation_for_face(face, horizon_lines_info)

            # Check if the strat catalog has a formation with this name — use its color
            try:
                strat_col = self._state.project.strat_column
                fm = strat_col.get_formation(form_name)
                if fm is None:
                    # Try formation_above/formation_below labels on bounding horizons
                    pass
                if fm is not None:
                    r, g, b = fm.color
                    fill_color = "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))
            except Exception:
                pass

            # Keep color consistent for same formation across fault blocks
            if form_name in form_color_map:
                fill_color = form_color_map[form_name]
            else:
                form_color_map[form_name] = fill_color

            # Ordinal suffix if multiple blocks of the same formation
            count_so_far = sum(
                1 for p in added if p.name == form_name or p.name.startswith(form_name + " (")
            )
            poly_name = form_name if count_so_far == 0 else f"{form_name} ({count_so_far + 1})"

            added.append(SectionPolygon(
                vertices=np.array(coords),
                name=poly_name,
                fill_color=fill_color,
                fill_alpha=0.45,
                section_name=section.name,
            ))

        # Remove any existing polygons already generated for this section
        existing_sec = [p for p in self._state.project.polygons
                        if getattr(p, "section_name", "") == section.name]
        if existing_sec:
            replace = QMessageBox.question(
                self, "Replace Existing Polygons",
                f"{len(existing_sec)} polygon(s) already exist for this section.\n"
                "Replace them with the newly generated set?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if replace == QMessageBox.StandardButton.Yes:
                self._state.blockSignals(True)
                try:
                    for p in list(existing_sec):
                        self._state.remove_polygon(p)
                finally:
                    self._state.blockSignals(False)
            # If No, fall through and append alongside existing

        self._state.blockSignals(True)
        try:
            for poly in added:
                self._state.add_polygon(poly)
        finally:
            self._state.blockSignals(False)
            if added:
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
        """Toggle map dock visibility (legacy; dock may be retired in game UI)."""
        if getattr(self, "_map_dock", None) is not None:
            self._map_dock.toggleViewAction().trigger()

    def _toggle_section_panel(self) -> None:
        """Toggle section dock visibility (legacy; dock may be retired in game UI)."""
        if getattr(self, "_section_dock", None) is not None:
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
        _sc("E", _tool("extend"))
        _sc("T", _tool("trim"))
        _sc("D", _tool("dip_constrained"))
        _sc("K", _tool("kink_band"))

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
        _sc("Ctrl+E", self._on_export_section_dialog)

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
        """Delete: remove the selected node (Nodes tool); otherwise the legacy
        pick-tool → Select fallback. Does not delete whole objects."""
        sv = self._section_view
        if sv.has_node_selected():
            msg = sv.delete_selected_node()
            if msg:
                self._flash_status(msg)        # refused (≥2-node floor)
            else:
                self._properties_panel.set_selected_node(None)
            return
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
        # In the game UI the old _status_label is an orphaned, invisible stub;
        # route map status (node-drag coords, nearest-well) to the visible
        # bottom status strip instead. The persistent E/N + lat/long cursor
        # readout lives on the MapHUD itself.
        if hasattr(self, "status_strip"):
            self.status_strip.set_hint(msg or "")
            return
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
        """Set active pick target from selected entity; create new if none selected."""
        from section_tool.core.surfaces import HorizonPick as _HP
        cat = "Horizons" if tool_id == "horizon_pick" else "Faults"
        proj = self._state.project
        picks = proj.horizon_picks if cat == "Horizons" else proj.fault_picks

        # Selected entity takes priority as the pick target
        sel_cat = self._state.selected_entity_category
        sel_idx = self._state.selected_entity_index
        if sel_cat == cat and 0 <= sel_idx < len(picks):
            self._state.set_active_pick_target(cat, sel_idx)
            return

        # No matching selection — create a new entity and start picking on it
        kind = "Horizon" if cat == "Horizons" else "Fault"
        default_color = self._next_horizon_color() if cat == "Horizons" else self._next_fault_color()
        new_pick = _HP.empty(name=f"{kind} {len(picks) + 1}", color=default_color)
        if cat == "Horizons":
            self._state.add_horizon_pick(new_pick)
        else:
            self._state.add_fault_pick(new_pick)
        new_idx = len(picks) - 1
        self._state.set_active_pick_target(cat, new_idx)
        self._state.set_selected_entity(cat, new_idx)

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
        node_word = "node" if n == 1 else "nodes"
        self._status_label.setText(
            f"Picking {hp.name}  ·  {n} {node_word}  ·  Right-click or Escape to finish"
        )

    def _on_pick_target_selected(self, cat: str, idx: int) -> None:
        """Selecting a horizon/fault in the panel selects it and arms it as the
        pick target — WITHOUT changing the active tool. Selection and active-tool
        are independent: the tool stays Select. Pressing H/F afterwards picks
        onto the selected target (see _ensure_pick_target)."""
        self._state.set_selected_entity(cat, idx)
        self._state.set_active_pick_target(cat, idx)

    def _on_state_tool_changed(self, tool_id: str) -> None:
        """Sync palette when tool changes via state (e.g. from map view draw finish)."""
        if self._tool_palette.active_tool != tool_id:
            self._tool_palette.set_active_tool(tool_id)

    def _on_tool_changed(self, tool_id: str) -> None:
        """Route palette tool activation to views and AppState."""
        # Guard: block Edit Nodes on bound polygons — their shape is controlled
        # by horizon/fault picks, not by dragging nodes directly.
        if tool_id == "node_edit":
            sel_cat = self._state.selected_entity_category
            sel_idx = self._state.selected_entity_index
            if sel_cat == "Polygons" and sel_idx >= 0:
                proj = self._state.project
                if sel_idx < len(proj.polygons):
                    poly = proj.polygons[sel_idx]
                    if hasattr(poly, "is_bound") and poly.is_bound():
                        self._flash_status(
                            "Bound polygon — edit its bounding horizons/faults to change shape. "
                            "(Select a horizon and press A instead.)"
                        )
                        # Revert to select tool
                        self._tool_palette.set_active_tool("select")
                        return

        self._state.set_active_tool(tool_id)
        self._section_view.set_picking_active(tool_id == "horizon_pick")
        self._section_view.set_fault_picking(tool_id == "fault_pick")
        self._section_view.set_polygon_drawing(tool_id == "polygon")
        self._section_view.set_ref_line_tool(tool_id)
        # Plan fault-draw tool lives on the z-slice view (tiled layout only) and
        # is only meaningful when a horizontal slice is the active workspace.
        if hasattr(self, "_zslice_view"):
            self._zslice_view.set_fault_drawing(
                tool_id == "plan_fault" and self._zslice_active())
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
# Game UI — canvas-first window
# ---------------------------------------------------------------------------

class _CanvasMouseFilter(QObject):
    """Event filter: updates smart cursor on mouse move.  Never consumes events."""

    def __init__(self, window: "SectionMainWindow", parent=None):
        super().__init__(parent)
        self._win = window

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.Type.MouseMove:
            self._win._update_smart_cursor(event.pos())
        return False  # never consume — let canvas handle everything


class SectionMainWindow(MainWindow):
    """Full-screen game-style UI wrapping all existing MainWindow functionality.

    The section canvas fills the entire window.  A transparent HUD layer floats
    above it.  Navigation is WASD + scroll; tools are keyboard-activated.
    """

    # Maps new ToolManager IDs → existing AppState tool IDs
    _NEW_TO_OLD: dict[str | None, str] = {
        "select":     "select",
        "node_edit":  "node_edit",
        "horizon":    "horizon_pick",
        "fault":      "fault_pick",
        "pick":       "horizon_pick",
        "polygon":    "polygon",
        "measure":    "measure",
        None:         "select",
    }

    def __init__(self, state=None):
        super().__init__(state)
        self._convert_to_game_ui()

    # ------------------------------------------------------------------
    # Override: shortcuts — no single-letter conflicts with WASD / tools
    # ------------------------------------------------------------------

    def _register_shortcuts(self) -> None:
        _ctx = Qt.ShortcutContext.ApplicationShortcut
        self._ref_cycle_idx = 0

        def _sc(key: str, slot):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(_ctx)
            sc.activated.connect(slot)
            return sc

        # Escape: always fires regardless of which panel has focus
        _sc("Escape",       self._on_game_escape)
        _sc("Delete",       self._on_delete_shortcut)
        _sc("Ctrl+Z",       self._state.undo)
        _sc("Ctrl+Shift+Z", self._state.redo)
        _sc("Ctrl+C",       lambda: None)
        _sc("Ctrl+V",       lambda: None)
        _sc("Ctrl+A",       lambda: None)
        _sc("Ctrl+N",       self._on_new)
        _sc("Ctrl+O",       self._on_open)
        _sc("Ctrl+S",       self._on_save)
        _sc("Ctrl+0",       self._zoom_to_fit)

        # Tool shortcuts — ApplicationShortcut so they fire even when the canvas
        # doesn't have focus. Letters come from TOOL_HOTKEYS (the single source
        # the tooltips also read), so a binding and its tooltip can never drift.
        # tool_mgr-routed tools go through handle_key (Qt.Key mapped by TOOL_KEYS,
        # reconciled by a test); the rest activate the palette tool directly.
        from section_tool.views.tool_palette import TOOL_HOTKEYS
        _TOOLMGR_IDS = {"select", "node_edit", "horizon_pick",
                        "fault_pick", "polygon", "measure"}
        _DIRECT_IDS  = {"trim", "dip_constrained", "extend", "kink_band",
                        "parallel", "zoom", "new_section", "plan_fault"}

        def _mk_tk(qt_key):
            return lambda: hasattr(self, "_tool_mgr") and self._tool_mgr.handle_key(qt_key)

        def _mk_direct(tool_id):
            return lambda: self._tool_palette.set_active_tool(tool_id)

        for _tid, _letter in TOOL_HOTKEYS.items():
            if _tid in _TOOLMGR_IDS:
                _sc(_letter, _mk_tk(getattr(Qt.Key, f"Key_{_letter}")))
            elif _tid in _DIRECT_IDS:
                _sc(_letter, _mk_direct(_tid))
            # h_ref/v_ref/a_ref ("R") are display-only here; bound via the cycle.

        # Secondary / non-single-tool bindings kept explicit:
        _sc("W", lambda: hasattr(self, "_tool_mgr") and self._tool_mgr.handle_key(Qt.Key.Key_W))
        _sc("R", self._cycle_ref_line_tool)   # cycles h_ref / v_ref / a_ref

    # ------------------------------------------------------------------
    # Override: remove Space-bar temporary pan
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        super(QMainWindow, self).keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        super(QMainWindow, self).keyReleaseEvent(event)

    # ------------------------------------------------------------------
    # Game UI construction
    # ------------------------------------------------------------------

    def _convert_to_game_ui(self) -> None:
        # 1. Remove all inherited toolbars — replaced by a single new one.
        for tb in list(self.findChildren(QToolBar)):
            tb.setVisible(False)
            self.removeToolBar(tb)
            tb.setParent(None)

        # 2. Remove all dock widgets from dock areas — panels move to splitter.
        for dock in (self._section_dock, self._view3d_dock, self._map_dock,
                     self._project_panel, self._properties_panel):
            self.removeDockWidget(dock)
            dock.setVisible(False)

        # 3. Section view: hide header rows, suppress pick banner.
        self._section_view.set_game_mode(True)

        # 4. Build tiled toolbar (replaces old icon toolbar).
        self._build_tiled_toolbar()

        # 5. Build splitter layout.
        self._build_tiled_layout()

        # 6. Status strip at bottom.
        from section_tool.views.status_strip import StatusStrip
        self.status_strip = StatusStrip(self)
        self.setStatusBar(self.status_strip)
        # Redirect old MainWindow label refs so _update_status/_on_map_status
        # don't crash after the old status bar is replaced.
        self._status_label = QLabel()
        self._hint_label   = QLabel()

        # 7. WASD navigator for the section view.
        from section_tool.navigation.wasd_navigator import WASDNavigator
        self._wasd_nav = WASDNavigator(
            self._section_view.canvas,
            self._section_view.view_state,
        )

        # 8. Tool manager + key filter.
        from section_tool.interaction.tool_manager import ToolManager, ToolKeyFilter
        self._tool_mgr = ToolManager()
        self._tool_key_filter = ToolKeyFilter(self._tool_mgr, self)
        self._tool_key_filter.palette_invoke_requested.connect(
            self.section_tile.command_palette.invoke
        )
        self._tool_mgr.tool_changed.connect(self._on_game_tool_changed)
        # All on-screen tool indicators are driven by AppState.tool_changed so
        # they reflect construction tools too (those bypass the ToolManager).
        self._state.tool_changed.connect(self.hud.tool_indicator.set_tool)
        self._state.tool_changed.connect(self.status_strip.set_tool)
        self._state.tool_changed.connect(self.section_tile.tool_hud.set_tool)

        # Install on canvas — navigator AFTER key_filter (LIFO: navigator runs first).
        self._section_view.canvas.installEventFilter(self._tool_key_filter)
        self._section_view.canvas.installEventFilter(self._wasd_nav)

        # 9. Smart cursor (scene=None → stub).
        from section_tool.interaction.smart_cursor import SmartCursor
        self._smart_cursor = SmartCursor(self._section_view.canvas, scene=None)
        self._tool_mgr.tool_changed.connect(self._smart_cursor.set_active_tool)
        self._mouse_filter = _CanvasMouseFilter(self, self)
        self._section_view.canvas.installEventFilter(self._mouse_filter)

        # 10. Command palette actions.
        self._cmd_palette = self.section_tile.command_palette  # alias
        self._cmd_palette.command_selected.connect(self._on_command)
        _ck = QShortcut(QKeySequence("Ctrl+K"), self)
        _ck.setContext(Qt.ShortcutContext.ApplicationShortcut)
        _ck.activated.connect(self._cmd_palette.invoke)

        # 11. Wire HUD and status strip to section view signals.
        self._section_view.view_changed.connect(self._on_hud_update)
        self._section_view.coords_updated.connect(self._on_section_coords)
        self._section_view.cursor_map_pos.connect(self._on_section_cursor_map)

        # 12. Wire project and properties panels.
        self._project_panel.pick_target_selected.connect(
            self._on_pick_target_selected)
        self.status_strip.set_hint(
            "New Section → draw line on map  |  Edit section nodes → click in map panel")

        # 13. Panel toggle shortcuts.
        QShortcut(QKeySequence("Ctrl+1"), self, self._toggle_map_tile)
        QShortcut(QKeySequence("Ctrl+2"), self, self._toggle_section_tile)
        QShortcut(QKeySequence("Ctrl+3"), self, self._toggle_view3d_tile)
        QShortcut(QKeySequence("Ctrl+4"), self, self._toggle_project_panel)
        QShortcut(QKeySequence("Ctrl+5"), self, self._toggle_properties_panel)
        QShortcut(QKeySequence("F11"), self, self._toggle_fullscreen)

        # 16. Re-wire inherited view-menu panel actions to tiled widgets.
        self._rewire_view_menu_for_tiles()

        # 14. Belt-and-suspenders: any state change re-renders all views.
        # Individual object signals already connect to view renders, but
        # project_changed as a catch-all ensures nothing is missed.
        for _sig in (
            self._state.project_changed,
            self._state.surface_added, self._state.surface_removed,
            self._state.surface_modified,
        ):
            _sig.connect(self._force_render_all)

        # 15. Show maximized; apply proportions after event loop starts.
        self.showMaximized()
        QTimer.singleShot(80, self._apply_default_proportions)

    def _force_render_all(self, *_args) -> None:
        """Ensure every view re-renders after any project state change."""
        self._section_view.request_render()
        self._map_view.request_render()
        # Mark minimap dirty so it repaints on its next 1-second tick
        if hasattr(self, "hud") and hasattr(self.hud, "map_inset"):
            mi = self.hud.map_inset
            if mi is not None:
                mi._dirty = True
                mi._timer.start()

    def _build_tiled_toolbar(self) -> None:
        """Minimal toolbar: Save, Undo, Redo + section info on the right."""
        tb = QToolBar("Main", self)
        tb.setObjectName("TiledMainToolBar")
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setIconSize(_QSize(16, 16))
        tb.setStyleSheet("""
            QToolBar {
                background: #1a1c20;
                border-bottom: 1px solid #2a2d33;
                spacing: 2px;
                padding: 0px 6px;
            }
            QToolButton {
                color: #b0b8c0;
                background: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 3px 8px;
                font-size: 11px;
            }
            QToolButton:hover {
                background: #252830;
                border-color: #3a3f48;
                color: #d0d8e0;
            }
            QToolButton:pressed { background: #2e3340; }
            QToolBar::separator {
                width: 1px;
                background: #2a2d33;
                margin: 5px 3px;
            }
        """)

        style = self.style()
        SP = QStyle.StandardPixmap

        save_a = QAction(self)
        save_a.setIcon(style.standardIcon(SP.SP_DialogSaveButton))
        save_a.setToolTip("Save  (Ctrl+S)")
        save_a.triggered.connect(self._on_save)
        tb.addAction(save_a)

        tb.addSeparator()

        undo_a = QAction(self)
        undo_a.setIcon(style.standardIcon(SP.SP_ArrowBack))
        undo_a.setToolTip("Undo  (Ctrl+Z)")
        undo_a.triggered.connect(self._state.undo)
        tb.addAction(undo_a)

        redo_a = QAction(self)
        redo_a.setIcon(style.standardIcon(SP.SP_ArrowForward))
        redo_a.setToolTip("Redo  (Ctrl+Shift+Z)")
        redo_a.triggered.connect(self._state.redo)
        tb.addAction(redo_a)

        # Spacer pushes section info to the right
        spacer = QWidget()
        from PySide6.QtWidgets import QSizePolicy as _QSP
        spacer.setSizePolicy(_QSP.Policy.Expanding, _QSP.Policy.Preferred)
        tb.addWidget(spacer)

        self._section_info_label = QLabel("")
        self._section_info_label.setStyleSheet("color: #666; font-size: 8pt; padding-right: 8px;")
        tb.addWidget(self._section_info_label)

        self.addToolBar(tb)
        self.tiled_toolbar = tb

        # Update section info when the active section / workspace slice changes
        self._state.active_section_changed.connect(self._update_section_info)
        self._state.active_slice_changed.connect(lambda _: self._update_section_info())
        self._state.section_modified.connect(lambda *_: self._update_section_info(
            self._state.active_section))

    def _update_section_info(self, section=None) -> None:
        lbl = getattr(self, "_section_info_label", None)
        if lbl is None:
            return
        # A horizontal z-slice is the active workspace → show slice info, not the
        # (stale) section's.
        if self._zslice_active():
            zs = self._state.active_slice
            lbl.setText(f"{zs.name}  ·  z = {zs.elevation:,.0f} m  ·  EPSG:{zs.crs_epsg}")
            return
        if section is None:
            section = self._state.active_section
        if section is None:
            lbl.setText("")
            return
        try:
            azs = section.segment_azimuths()
            az = f"{azs[0]:.0f}°" if len(azs) == 1 else f"{azs[0]:.0f}°–{azs[-1]:.0f}°"
        except Exception:
            az = "—"
        lbl.setText(f"{section.name}  ·  {section.total_length()/1000:.2f} km  ·  Az {az}")

    def _build_tiled_layout(self) -> None:
        """Build three-column splitter: [project] | [section/map] | [properties]."""
        from section_tool.views.section_tile import SectionTile
        from section_tool.views.map_tile     import MapTile

        _splitter_style = """
            QSplitter::handle          { background: #2a2d33; }
            QSplitter::handle:hover    { background: #3a4050; }
        """

        # Outer: horizontal [tool-rail | project | center | properties]
        self.h_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.h_splitter.setHandleWidth(3)
        self.h_splitter.setStyleSheet(_splitter_style)

        # Leftmost: the existing ToolPalette as a thin, fixed-width tool rail.
        # The widget at __init__ is alive and fully wired; its old toolbar host
        # was removed in _convert_to_game_ui, so we simply re-parent it here.
        # NB: re-parenting hides the widget, so show() must come AFTER addWidget.
        self.h_splitter.addWidget(self._tool_palette)
        self._tool_palette.setVisible(True)

        # Left panel: vertical splitter — project tree on top, properties below.
        # Both reuse the existing live docks (detached in _convert_to_game_ui);
        # properties used to sit on the far right, moved here so section/map
        # reclaim that width. Moving the live widgets, not rebuilding them.
        self._project_panel.setTitleBarWidget(QWidget(self._project_panel))
        self._project_panel.setVisible(True)
        self._properties_panel.setTitleBarWidget(QWidget(self._properties_panel))
        self._properties_panel.setMinimumWidth(0)
        self._properties_panel.setVisible(True)

        self.left_panel = QSplitter(Qt.Orientation.Vertical, self)
        self.left_panel.setHandleWidth(3)
        self.left_panel.setStyleSheet(_splitter_style)
        self.left_panel.addWidget(self._project_panel)
        self.left_panel.addWidget(self._properties_panel)
        self.left_panel.setStretchFactor(0, 3)   # project tree ~60%
        self.left_panel.setStretchFactor(1, 2)   # properties ~40%
        self.h_splitter.addWidget(self.left_panel)

        # Center: vertical [section tile | map tile]
        self.v_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.v_splitter.setHandleWidth(3)
        self.v_splitter.setStyleSheet(_splitter_style)

        self.section_tile = SectionTile(self._section_view, self._state, self)
        self.hud = self.section_tile.hud   # expose for existing HUD wiring code

        self.map_tile = MapTile(self._map_view, self)

        # Z-slice plan workspace — a sibling plan view sharing the map's canvas
        # chrome (MapTile + MapHUDLayer). Occupies the section column when a
        # horizontal slice is the active workspace; hidden otherwise.
        from section_tool.views.zslice_view import ZSliceView
        self._zslice_view = ZSliceView(self._state, self)
        self.zslice_tile = MapTile(self._zslice_view, self)
        self._zslice_view.cursor_map_pos.connect(self._on_map_cursor_pos)
        for _sig in (self._state.project_changed, self._state.section_added,
                     self._state.section_removed, self._state.section_modified):
            _sig.connect(lambda *_a: self._zslice_view.request_render())
        # Plan fault-draw: right-click/Esc ends the trace → revert to select;
        # transient hints (e.g. "select a fault first") go to the status strip.
        self._zslice_view.draw_ended.connect(
            lambda: self._tool_palette.set_active_tool("select"))
        self._zslice_view.status_message.connect(self._flash_status)

        # 3D viewer tile — re-homed from the retired dock world. Wraps the live
        # Viewer3D (lazy OpenGL: no GL context until the user clicks Enable, so
        # reparenting here is safe). PROVISIONAL placement: its own slot below
        # the map. TBD whether 3D keeps a dedicated slot, shares the section
        # slot via a router (like z-slice), or becomes a detachable window.
        from section_tool.views.view3d_tile import View3DTile
        self.view3d_tile = View3DTile(self._viewer_3d, self)

        self.v_splitter.addWidget(self.section_tile)
        self.v_splitter.addWidget(self.zslice_tile)
        self.v_splitter.addWidget(self.map_tile)
        self.v_splitter.addWidget(self.view3d_tile)
        self.zslice_tile.setVisible(False)       # shown by the active-slice router
        self.view3d_tile.setVisible(False)       # shown by the View ▸ 3D toggle
        self.v_splitter.setStretchFactor(0, 3)   # section: 60%
        self.v_splitter.setStretchFactor(1, 3)   # z-slice (same slot as section)
        self.v_splitter.setStretchFactor(2, 2)   # map: 40%
        self.v_splitter.setStretchFactor(3, 2)   # 3D (provisional, own slot)
        # Active-slice router: section ⇒ section tile, horizontal ⇒ z-slice tile.
        self._state.active_slice_changed.connect(self._on_active_slice_route)
        # Bidirectional cursor: map tile hover → section vertical indicator
        self._map_view.cursor_map_pos.connect(self._on_map_cursor_pos)
        # Reproject picks when section geometry changes
        self._map_view.section_node_moved.connect(self._on_section_node_moved)
        # Sync tool manager when pick sequence ends (e.g. right-click, Escape)
        self._section_view.pick_ended.connect(
            lambda: self._tool_mgr.handle_key(Qt.Key.Key_Escape)
        )

        # Center column = thin context/options bar (per-tool parameters) on top
        # of the section/map splitter. The ContextToolbar built in __init__ is
        # alive and wired (its action_requested already routes cst_param edits
        # to the construction tools via section_view._on_context_action); its
        # old options-bar QToolBar host was removed in _convert_to_game_ui, so
        # re-parent it here and re-theme it for the dark canvas chrome.
        center = QWidget(self)
        cv = QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        self._ctx.set_dark_theme()
        cv.addWidget(self._ctx)
        cv.addWidget(self.v_splitter, 1)
        self.h_splitter.addWidget(center)
        # show() AFTER re-parenting (re-parent hides the widget).
        self._ctx.setVisible(True)

        # Keep the tool rail a fixed thin strip: never let the splitter
        # collapse or stretch it.
        self.h_splitter.setCollapsible(0, False)
        self.h_splitter.setStretchFactor(0, 0)

        self.setCentralWidget(self.h_splitter)

        # Floating building-tool palette — a frameless overlay on the section
        # canvas. It owns no activation logic: clicks route through the one true
        # path (_tool_palette.set_active_tool), and its highlight tracks
        # AppState.tool_changed so it agrees with rail + hotkeys.
        from section_tool.views.floating_tool_palette import FloatingToolPalette
        self._floating_palette = FloatingToolPalette(self.section_tile)
        self._floating_palette.tool_activation_requested.connect(
            self._tool_palette.set_active_tool)
        self._state.tool_changed.connect(self._floating_palette.set_active)
        self._floating_palette.set_active(self._state.active_tool)
        self._floating_palette.move(12, 40)      # default inset; persisted in Phase 2
        self._floating_palette.show()
        self._floating_palette.raise_()

    def _apply_default_proportions(self) -> None:
        """Set default panel proportions after the window has real geometry."""
        w        = self.width()
        rail_w   = self._tool_palette.width()   # fixed thin tool rail (56 px)
        left_w   = 280                          # project tree + properties stack
        center_w = max(w - left_w - rail_w, 400)
        # h_splitter = [tool_palette, left_panel, central]; properties moved into
        # left_panel, so the old right slot is gone and center reclaims it.
        self.h_splitter.setSizes([rail_w, left_w, center_w])
        # Left panel: project tree ~60% over properties ~40% (vertical).
        h = max(self.height(), 600)
        self.left_panel.setSizes([int(h * 0.60), int(h * 0.40)])
        # Section | Map side by side — section gets 60%, map gets 40%
        section_w = int(center_w * 0.60)
        map_w     = int(center_w * 0.40)
        # v_splitter = [section_tile, zslice_tile, map_tile, view3d_tile];
        # section & z-slice share a slot (router toggles visibility), so both get
        # section width; map keeps 40%; the (hidden) 3D tile gets a map-sized
        # slot for when it's toggled on.
        self.v_splitter.setSizes([section_w, section_w, map_w, map_w])

        # Regression guard: a tile added without growing the size list collapsed
        # the map once already (0067856). Warn-and-repair any visible pane left
        # at 0 width — loud in the console, but the user still gets a working
        # layout. Covers both splitters now that left_panel stacks two docks.
        self._repair_collapsed_panes(self.h_splitter, "h_splitter")
        self._repair_collapsed_panes(self.v_splitter, "v_splitter")

        # Floating palette: restore its saved spot now the section tile has real
        # geometry (validates on-canvas; default = inset from the top-left).
        if hasattr(self, "_floating_palette"):
            self._floating_palette.restore_position(QPoint(12, 40))
            self._floating_palette.raise_()

    def _repair_collapsed_panes(self, splitter, label: str) -> None:
        """Warn + repair any visible pane in *splitter* that got 0 extent."""
        sizes = splitter.sizes()
        for i in range(splitter.count()):
            wdg = splitter.widget(i)
            if wdg.isVisible() and i < len(sizes) and sizes[i] == 0:
                logging.getLogger(__name__).warning(
                    "%s: visible pane %r got 0 extent (%d sizes / %d widgets). "
                    "Repairing.", label, wdg.objectName() or wdg,
                    len(sizes), splitter.count())
                sizes[i] = max(int(self.width() * 0.20), 200)
                splitter.setSizes(sizes)

    # ------------------------------------------------------------------
    # Tiled toolbar action stubs
    # ------------------------------------------------------------------

    def _on_new_section_tiled(self) -> None:
        if hasattr(self, "status_strip"):
            self.status_strip.set_hint(
                "Click on the map below to place section endpoints")
        self._tool_palette.set_active_tool("new_section")

    def _on_add_well_tiled(self) -> None:
        self._on_import_las()

    def _on_import_data_tiled(self) -> None:
        self._on_import_las()

    def _on_import_seismic_tiled(self) -> None:
        self._on_import_segy()

    # ------------------------------------------------------------------
    # Mode / view (tiled layout has persistent panels, no mode switching)
    # ------------------------------------------------------------------

    def set_mode(self, mode) -> None:
        pass   # Tiled layout: all views always visible

    def _toggle_map_dock(self) -> None:
        self.map_tile.setVisible(not self.map_tile.isVisible())

    def _toggle_map_tile(self) -> None:
        self.map_tile.setVisible(not self.map_tile.isVisible())

    def _toggle_section_tile(self) -> None:
        self.section_tile.setVisible(not self.section_tile.isVisible())

    def _toggle_view3d_tile(self) -> None:
        # Route through the View-menu action so the checkmark stays in sync and
        # _set_tile_visible restores the tile's width when it's re-shown.
        act = getattr(self, "_view3d_view_action", None)
        if act is not None:
            act.toggle()
        else:
            self._set_tile_visible(self.view3d_tile, self.view3d_tile.isHidden())

    def _on_active_slice_route(self, slice_) -> None:
        """Route the active workspace slice to the correct view.

        A horizontal slice shows the z-slice plan tile (bound + rendered) in the
        section column; a Section (or None) restores the section tile. The map
        tile (surface plan) is unaffected.
        """
        is_horizontal = getattr(slice_, "kind", None) == "horizontal"
        if is_horizontal:
            self._zslice_view.set_slice(slice_)
        self.zslice_tile.setVisible(is_horizontal)
        self.section_tile.setVisible(not is_horizontal)
        # Revert the active tool if it no longer belongs to the new workspace:
        # plan-draw is meaningless on a section, section-plane tools on a z-slice.
        active = self._tool_palette.active_tool
        if not is_horizontal and active in self._tool_palette._ZSLICE_WORKSPACE:
            self._tool_palette.set_active_tool("select")
        elif is_horizontal and active in self._tool_palette._SECTION_WORKSPACE:
            self._tool_palette.set_active_tool("select")

    def _toggle_project_panel(self) -> None:
        self._project_panel.setVisible(not self._project_panel.isVisible())

    def _toggle_properties_panel(self) -> None:
        self._properties_panel.setVisible(not self._properties_panel.isVisible())

    def _set_tile_visible(self, tile, visible: bool) -> None:
        """Show/hide a v_splitter tile, restoring width on show.

        A splitter pane that was hidden comes back at 0 width, so when showing
        a tile that returned collapsed, give it a usable share again.
        """
        tile.setVisible(visible)
        if visible:
            sizes = self.v_splitter.sizes()
            idx = self.v_splitter.indexOf(tile)
            if 0 <= idx < len(sizes) and sizes[idx] < 20:        # came back collapsed
                sizes[idx] = max(int(self.width() * 0.20), 200)
                self.v_splitter.setSizes(sizes)

    def _retire_dock(self, attr: str, live_view, tile) -> None:
        """Detach and delete an orphaned dock once its view lives in a tile.

        The live view was reparented into the tile (MapTile/SectionTile do
        setParent(self)), so the dock owns nothing; guard anyway — if the dock
        still parents the view, rescue it to the tile before deleting.
        """
        dock = getattr(self, attr, None)
        if dock is None:
            return
        if live_view is not None and live_view.parent() is dock:
            live_view.setParent(tile)
        try:
            self.removeDockWidget(dock)
        except RuntimeError:
            pass
        dock.deleteLater()
        setattr(self, attr, None)

    def _rewire_view_menu_for_tiles(self) -> None:
        """Swap dock-based View toggles for clean tile-bound actions, then
        retire the orphaned Map/Section docks.

        Order matters: the menu items are currently the docks' toggleViewActions,
        so we replace the menu actions first, then delete the docks.
        """
        from PySide6.QtGui import QAction
        from PySide6.QtWidgets import QMenu

        map_old   = self._map_dock.toggleViewAction()
        sec_old   = self._section_dock.toggleViewAction()
        d3_action = self._view3d_dock.toggleViewAction()

        # The base class built the View menu as a local; rather than re-find it
        # by title (the menubar hands back transient action wrappers whose
        # .menu() shiboken may invalidate), take the live menu straight from an
        # action that already lives in it — that wrapper is tied to map_old,
        # which we hold, so it stays valid.
        menus = [o for o in map_old.associatedObjects() if isinstance(o, QMenu)]
        view_menu = menus[0] if menus else None

        # Fresh checkable actions bound ONLY to the live tiles — no surviving
        # binding to the orphaned docks (which caused empty floating windows).
        # NB: use isHidden(), not isVisible() — _rewire runs before the window
        # is shown, so isVisible() is False for every tile; isHidden() reflects
        # the intended default (only zslice_tile is explicitly hidden).
        map_new = QAction(map_old.text(), self)
        map_new.setCheckable(True)
        map_new.setChecked(not self.map_tile.isHidden())
        map_new.toggled.connect(
            lambda checked, t=self.map_tile: self._set_tile_visible(t, checked))

        sec_new = QAction(sec_old.text(), self)
        sec_new.setCheckable(True)
        sec_new.setChecked(not self.section_tile.isHidden())
        sec_new.toggled.connect(
            lambda checked, t=self.section_tile: self._set_tile_visible(t, checked))

        # 3D: re-homed into view3d_tile (own provisional slot). Fresh checkable
        # action bound to the tile, default unchecked — the tile shows the lazy
        # "Enable 3D View" button until the user activates the PyVista context.
        d3_new = QAction(d3_action.text(), self)
        d3_new.setCheckable(True)
        d3_new.setChecked(not self.view3d_tile.isHidden())
        d3_new.toggled.connect(
            lambda checked, t=self.view3d_tile: self._set_tile_visible(t, checked))

        # View ▸ Tool Palette — toggle the floating building palette. No bare
        # hotkey: bare letters are the tool-activation namespace (and would fire
        # while typing in a field), so this is menu-only by default.
        pal_act = QAction("Tool &Palette", self)
        pal_act.setCheckable(True)
        pal_act.setChecked(not self._floating_palette.isHidden())
        pal_act.toggled.connect(
            lambda checked: self._floating_palette.setVisible(checked))

        if view_menu is not None:
            view_menu.insertAction(map_old, map_new)
            view_menu.removeAction(map_old)
            view_menu.insertAction(sec_old, sec_new)
            view_menu.removeAction(sec_old)
            view_menu.insertAction(d3_action, d3_new)
            view_menu.removeAction(d3_action)
            view_menu.addSeparator()
            view_menu.addAction(pal_act)
        self._map_view_action     = map_new
        self._section_view_action = sec_new
        self._view3d_view_action  = d3_new
        self._palette_view_action = pal_act

        # Retire the now-orphaned docks (their views live in tiles now).
        self._retire_dock("_map_dock", self._map_view, self.map_tile)
        self._retire_dock("_section_dock", self._section_view, self.section_tile)
        self._retire_dock("_view3d_dock", self._viewer_3d, self.view3d_tile)

    def _reset_layout(self) -> None:
        """Game UI has no docks — reset splitter proportions and default tile
        visibility instead of the (now-removed) dock arrangement.

        Overrides the base dock-based reset, which referenced the retired
        Map/Section docks and never touched the splitter (see layout diagnosis).
        """
        QSettings("Geoscience", "CrossSectionTool").remove("window/state")
        # Default workspace: section + map visible, z-slice hidden.
        self.section_tile.setVisible(True)
        self.zslice_tile.setVisible(False)
        self.map_tile.setVisible(True)
        if getattr(self, "_map_view_action", None) is not None:
            self._map_view_action.setChecked(True)
        if getattr(self, "_section_view_action", None) is not None:
            self._section_view_action.setChecked(True)
        self._apply_default_proportions()

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showMaximized()
        else:
            self.showFullScreen()

    # ------------------------------------------------------------------
    # Tool routing
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Save / Open (SQLite project folders, overrides HDF5 MainWindow logic)
    # ------------------------------------------------------------------

    def _on_save(self) -> None:
        pm = self._state.project_manager
        if pm.is_open and pm.project_path:
            try:
                self._state.save_project()
                self._update_title()
            except Exception as exc:
                QMessageBox.critical(self, "Save Error", str(exc))
        else:
            self._on_save_as()

    def _on_save_as(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        # Step 1: pick parent directory
        parent = QFileDialog.getExistingDirectory(
            self, "Choose Location for New Project", os.path.expanduser("~")
        )
        if not parent:
            return
        # Step 2: ask for project name
        default_name = self._state.project.name or "Untitled"
        name, ok = QInputDialog.getText(
            self, "Project Name", "Enter project name:", text=default_name
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        dest = os.path.join(parent, name)
        # Step 3: confirm overwrite if folder exists
        if os.path.exists(dest):
            reply = QMessageBox.question(
                self, "Folder Already Exists",
                f"'{name}' already exists at that location. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        try:
            os.makedirs(dest, exist_ok=True)
            self._state.save_project_as(dest)
            self._update_title()
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    def _on_open(self) -> None:
        if not self._check_unsaved_changes():
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Open Project Folder", os.path.expanduser("~")
        )
        if not folder:
            return
        db_path = os.path.join(folder, "project.sqlite")
        if not os.path.exists(db_path):
            QMessageBox.warning(
                self, "Invalid Project",
                "No project.sqlite found in this folder.\n"
                f"({folder})"
            )
            return
        if not self._open_project(folder):
            QMessageBox.critical(self, "Open Error", f"Could not open:\n{folder}")
        else:
            self._update_title()
            if hasattr(self, "status_strip"):
                self.status_strip.set_hint("")

    def _set_grid(self, visible: bool) -> None:
        self._section_view.set_grid_visible(visible)
        self._map_view.set_grid_visible(visible)

    def _on_game_tool_changed(self, tool) -> None:
        """Map new ToolManager ID → existing AppState tool and activate it."""
        old_tool = self._NEW_TO_OLD.get(tool, "select")
        self._on_tool_changed(old_tool)

    def _on_game_escape(self) -> None:
        """Escape, focus-independent (window-level shortcut). One action per
        press, in strict priority:

        1. A focused text editor owns Escape — blur it, leave everything else.
        2. An uncommitted pick draft → discard it, stay in the tool.
        3. (Select + object selected) or (Nodes + node selected) → clear that
           selection and empty the properties panel, stay in the tool. A pick
           tool's armed target is NOT cleared here.
        4. Otherwise → return to Select.
        """
        from PySide6.QtWidgets import (QApplication, QLineEdit, QAbstractSpinBox,
                                        QComboBox, QTextEdit, QPlainTextEdit)
        fw = QApplication.focusWidget()
        if isinstance(fw, (QLineEdit, QAbstractSpinBox, QTextEdit, QPlainTextEdit)) \
                or (isinstance(fw, QComboBox) and fw.isEditable()):
            fw.clearFocus()          # let the field's Escape blur it; selection untouched
            return

        if self._section_view.discard_pick_draft():
            return

        sv = self._section_view
        tool = self._tool_palette.active_tool
        if tool == "node_edit" and sv.has_node_selected():
            sv.clear_node_selection()
            self._properties_panel.set_selected_node(None)
            return
        if tool == "select" and (sv.has_object_selected()
                                 or self._state.selected_entity_index >= 0):
            sv.clear_object_selection()
            self._state.set_selected_entity("", -1)
            self._properties_panel.set_selected_object("", -1)
            self._properties_panel.set_selected_node(None)
            return

        self._tool_mgr.reset()
        self._tool_palette.set_active_tool("select")

    # ------------------------------------------------------------------
    # HUD feeds
    # ------------------------------------------------------------------

    def _on_hud_update(self) -> None:
        """Refresh all HUD elements from current section view extent."""
        vs = self._section_view.view_state
        x_min, x_max = vs.x_min, vs.x_max
        z_min, z_max = vs.z_min, vs.z_max
        self.hud.depth_ruler.set_view_range(z_min, z_max)
        self.hud.scale_bar.set_range(x_min, x_max)
        bands = self._compute_formation_bands(z_min, z_max)
        self.hud.formation_strip.set_stratigraphy(bands, z_min, z_max)
        # Also feed bands to depth ruler for the formation chaser strip
        self.hud.depth_ruler.set_formations(bands)

    def _compute_formation_bands(self, z_min: float, z_max: float) -> list:
        """Build FormationBand list from horizon picks on the active section."""
        from section_tool.hud.formation_strip import FormationBand
        section = self._state.active_section
        if section is None:
            return []
        horizon_picks = self._state.project.horizon_picks
        strat_col = self._state.project.strat_column
        import numpy as np
        section_horizons: list[tuple[float, object]] = []
        for hp in horizon_picks:
            idxs = hp.section_indices(section.name)
            if len(idxs) > 0:
                depths = hp._depths[idxs]
                section_horizons.append((float(np.median(depths)), hp))
        section_horizons.sort(key=lambda t: t[0])
        if not section_horizons:
            return []
        bands = []
        def _color(fm, fallback_hex):
            if fm and hasattr(fm, "color"):
                try:
                    from matplotlib.colors import to_rgb
                    r, g, b = to_rgb(fallback_hex)
                    r2, g2, b2 = fm.color[0]/255, fm.color[1]/255, fm.color[2]/255
                    return (int(r2*255), int(g2*255), int(b2*255))
                except Exception:
                    pass
            try:
                from matplotlib.colors import to_rgb
                r, g, b = to_rgb(fallback_hex)
                return (int(r*255), int(g*255), int(b*255))
            except Exception:
                return (120, 130, 140)
        # Band above shallowest horizon
        top_d, top_hp = section_horizons[0]
        fm_name = getattr(top_hp, "formation_above", "")
        if fm_name:
            fm = strat_col.get_formation(fm_name)
            bands.append(FormationBand(z_min, top_d, fm_name,
                                       _color(fm, top_hp.color)))
        for i in range(len(section_horizons) - 1):
            d_top, hp_top = section_horizons[i]
            d_bot, hp_bot = section_horizons[i + 1]
            nm = getattr(hp_top, "formation_below", "") or \
                 getattr(hp_bot, "formation_above", "")
            fm = strat_col.get_formation(nm) if nm else None
            bands.append(FormationBand(d_top, d_bot, nm or f"Unit {i+1}",
                                       _color(fm, hp_top.color)))
        # Band below deepest horizon
        bot_d, bot_hp = section_horizons[-1]
        fm_name = getattr(bot_hp, "formation_below", "")
        if fm_name:
            fm = strat_col.get_formation(fm_name)
            bands.append(FormationBand(bot_d, z_max, fm_name,
                                       _color(fm, bot_hp.color)))
        return bands

    def _on_section_coords(self, x_m: float, depth_m: float) -> None:
        elev_m = self._section_view._surface_elev_at(x_m) - depth_m
        if hasattr(self, "hud") and hasattr(self.hud, "nav_readout"):
            self.hud.nav_readout.update_coords(x_m, depth_m, elev_m)
        if hasattr(self, "hud"):
            self.hud.depth_ruler.set_cursor_depth(depth_m)
            self.hud.formation_strip.set_cursor_depth(depth_m)
        if hasattr(self, "status_strip"):
            self.status_strip.update_coords(x_m, depth_m, elev_m)

    def _on_section_cursor_map(self, map_x: float, map_y: float) -> None:
        """Section cursor → crosshair on map inset and map tile."""
        if hasattr(self, "hud") and self.hud.map_inset:
            self.hud.map_inset.update_crosshair(map_x, map_y)
        if hasattr(self, "map_tile"):
            self._map_view.show_cursor_crosshair(map_x, map_y)

    def _on_map_cursor_pos(self, map_x: float, map_y: float) -> None:
        """Map tile cursor → vertical indicator on section canvas."""
        if hasattr(self, "section_tile"):
            self._section_view.show_map_cursor_on_section(map_x, map_y)

    def _on_section_node_moved(self, sec_idx: int, node_idx: int,
                                new_x: float, new_y: float) -> None:
        """After a section node drag, reproject all picks to the new geometry."""
        sections = self._state.project.sections
        if sec_idx < len(sections):
            sec_name = sections[sec_idx].name
            self._state.recompute_pick_display_coords(sec_name)
            self._section_view.render()

    def _update_smart_cursor(self, canvas_pos) -> None:
        self._smart_cursor.update(canvas_pos, self._section_view.view_state)

    # ------------------------------------------------------------------
    # Right-click context menu
    # ------------------------------------------------------------------

    def _build_context_menu(self) -> QMenu | None:
        hit = self._smart_cursor.current_hit
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background: rgba(20, 20, 26, 240);
                border: 1px solid rgba(75, 75, 100, 180);
                border-radius: 6px;
                padding: 4px 0;
                color: rgba(210, 210, 210, 255);
                font-size: 13px;
            }
            QMenu::item          { padding: 6px 24px 6px 16px; border-radius: 4px; }
            QMenu::item:selected { background: rgba(70, 110, 170, 210); }
            QMenu::item:disabled { color: rgba(120, 120, 130, 180); }
            QMenu::separator     { height: 1px;
                                   background: rgba(75, 75, 100, 120);
                                   margin: 3px 8px; }
        """)

        if hit is None:
            act = self._tool_mgr.active
            if act == "horizon":
                menu.addAction("Start Horizon Here")
            elif act == "fault":
                menu.addAction("Start Fault Here")
            menu.addAction("Add Annotation…")
            menu.addSeparator()
            tip = menu.addAction("Measure Distance  [hold M]")
            tip.setEnabled(False)
            return menu

        if hit.type == "horizon":
            hdr = menu.addAction(f"Horizon: {hit.object.name}")
            hdr.setEnabled(False)
            menu.addSeparator()
            menu.addAction("Edit Name…",
                lambda: self._on_panel_rename("horizon", hit.object))
            menu.addSeparator()
            menu.addAction("Delete Horizon",
                lambda: self._on_panel_delete(("horizon", hit.object)))
            return menu

        if hit.type == "fault":
            hdr = menu.addAction(f"Fault: {hit.object.name}")
            hdr.setEnabled(False)
            menu.addSeparator()
            menu.addAction("Edit Name…",
                lambda: self._on_panel_rename("fault", hit.object))
            menu.addSeparator()
            menu.addAction("Delete Fault",
                lambda: self._on_panel_delete(("fault", hit.object)))
            return menu

        if hit.type == "well_log":
            hdr = menu.addAction(f"Well: {hit.object.name}")
            hdr.setEnabled(False)
            menu.addSeparator()
            return menu

        return None

    # ------------------------------------------------------------------
    # Command palette → action routing
    # ------------------------------------------------------------------

    def _on_command(self, command_id: str) -> None:
        from section_tool.modes import Mode
        dispatch = {
            "tool_horizon":  lambda: self._tool_mgr.handle_key(Qt.Key.Key_H),
            "tool_fault":    lambda: self._tool_mgr.handle_key(Qt.Key.Key_F),
            "tool_pick":     lambda: self._tool_mgr.handle_key(Qt.Key.Key_W),
            "tool_annotate": lambda: self._tool_mgr.handle_key(Qt.Key.Key_A),
            # Construction tools bypass the ToolManager; activate via the palette.
            "tool_extend":   lambda: self._tool_palette.set_active_tool("extend"),
            "tool_trim":     lambda: self._tool_palette.set_active_tool("trim"),
            "tool_parallel": lambda: self._tool_palette.set_active_tool("parallel"),
            "tool_dip":      lambda: self._tool_palette.set_active_tool("dip_constrained"),
            "tool_kink":     lambda: self._tool_palette.set_active_tool("kink_band"),
            "mode_section":  lambda: self.set_mode(Mode.SECTION),
            "mode_map":      self._toggle_map_dock,
            "mode_3d":       self._toggle_view3d_tile,
            "view_fit":      self._zoom_to_fit,
            "export_image":  self._on_export_section_image,
            "export_svg":    self._on_export_section_image,
            "project_open":   self._on_open,
            "project_save":   self._on_save,
            "view_grid_on":   lambda: self._set_grid(True),
            "view_grid_off":  lambda: self._set_grid(False),
        }
        action = dispatch.get(command_id)
        if action:
            action()


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
        /* Styling the box switches the spinbox to QStyleSheetStyle, which only
           hit-tests the steppers if the sub-controls are styled explicitly.
           Without these rules the arrows draw but clicks land on nothing — the
           app-wide "steppers do nothing" bug.  Size + position them, give a
           hover/pressed cue, and draw a CSS-triangle arrow (no image asset). */
        QSpinBox::up-button, QDoubleSpinBox::up-button {{
            subcontrol-origin: border; subcontrol-position: top right;
            width: 18px; border-left: 1px solid {_border};
            border-top-right-radius: 3px; background: {_bg3}; }}
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            subcontrol-origin: border; subcontrol-position: bottom right;
            width: 18px; border-left: 1px solid {_border};
            border-bottom-right-radius: 3px; background: {_bg3}; }}
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
            background: #4A90D9; }}
        QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
        QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{
            background: #3B82F6; }}
        /* Clear triangular arrows (not the default stacked-box glyphs): suppress
           the native arrow image and draw a CSS triangle via borders. */
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
            image: none; width: 0; height: 0;
            border-left: 5px solid transparent; border-right: 5px solid transparent;
            border-bottom: 7px solid #E0E0E0; }}
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
            image: none; width: 0; height: 0;
            border-left: 5px solid transparent; border-right: 5px solid transparent;
            border-top: 7px solid #E0E0E0; }}
        QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover {{
            border-bottom-color: #FFFFFF; }}
        QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover {{
            border-top-color: #FFFFFF; }}
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

    window = SectionMainWindow()
    # showMaximized() called in _convert_to_game_ui(); show() is a no-op here
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
