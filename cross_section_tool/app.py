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

        # ── Views ─────────────────────────────────────────────────────────────
        self._map_view     = MapView(self._state, self)
        self._section_view = SectionView(self._state, self)
        self._viewer_3d    = Viewer3D(self._state, self)

        # ── Section / 3D tab panel ────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.addTab(self._section_view, "Section")
        self._tabs.addTab(self._viewer_3d,    "3D View")

        # ── Map panel with collapse strip ─────────────────────────────────────
        self._map_view.setMinimumWidth(0)
        self._map_view.setMinimumHeight(300)
        self._map_collapse_strip = _CollapseStrip(self)
        self._map_collapse_strip.clicked.connect(self._toggle_map_panel)
        self._map_collapse_strip.setToolTip("Collapse / expand map  (Ctrl+2)")
        self._map_panel_collapsed = False
        self._map_panel_width     = 360

        map_container = QWidget()
        _mh = QHBoxLayout(map_container)
        _mh.setContentsMargins(0, 0, 0, 0)
        _mh.setSpacing(0)
        _mh.addWidget(self._map_view, stretch=1)
        _mh.addWidget(self._map_collapse_strip)

        # ── Central widget: horizontal splitter (map | section/3D) ───────────
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.addWidget(map_container)
        self._splitter.addWidget(self._tabs)
        self._splitter.setSizes([360, 960])
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setHandleWidth(5)
        self.setCentralWidget(self._splitter)

        # ── Tool palette — QToolBar docked LEFT, not movable ─────────────────
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

        # ── Project panel (RIGHT dock, top half) ──────────────────────────────
        self._project_panel = ProjectPanel(self._state, self)
        self._project_panel.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,
                           self._project_panel)
        self._setup_project_panel_title_bar()

        # ── Properties panel (RIGHT dock, bottom half) ────────────────────────
        from cross_section_tool.views.properties_panel import PropertiesPanel
        self._properties_panel = PropertiesPanel(self._state, self)
        self._properties_panel.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,
                           self._properties_panel)
        # Stack them vertically in the right dock area
        self.splitDockWidget(self._project_panel, self._properties_panel,
                             Qt.Orientation.Vertical)

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

    def _build_options_bar(self) -> None:
        """Phase 3 — full-width context-sensitive options bar (top QToolBar)."""
        from cross_section_tool.views.context_toolbar import ContextToolbar
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
        self._exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self._exit_action.triggered.connect(self.close)
        file_menu.addAction(self._exit_action)

        # ================================================================
        # Edit
        # ================================================================
        edit_menu = mb.addMenu("&Edit")
        undo_a = QAction("&Undo", self)
        undo_a.setShortcut(QKeySequence.StandardKey.Undo)
        undo_a.triggered.connect(self._state.undo)
        edit_menu.addAction(undo_a)
        redo_a = QAction("&Redo", self)
        redo_a.setShortcut(QKeySequence("Ctrl+Shift+Z"))
        redo_a.triggered.connect(self._state.redo)
        edit_menu.addAction(redo_a)
        edit_menu.addSeparator()
        selall_a = QAction("Select &All", self)
        selall_a.setShortcut(QKeySequence("Ctrl+A"))
        edit_menu.addAction(selall_a)

        # ================================================================
        # View
        # ================================================================
        view_menu = mb.addMenu("&View")
        # Panel toggles
        self._view_project_action = QAction("Project Panel", self)
        self._view_project_action.setCheckable(True)
        self._view_project_action.setChecked(True)
        self._view_project_action.setShortcut(QKeySequence("Ctrl+4"))
        self._view_project_action.toggled.connect(
            lambda v: self._project_panel.setVisible(v))
        view_menu.addAction(self._view_project_action)
        self._view_props_action = QAction("Properties Panel", self)
        self._view_props_action.setCheckable(True)
        self._view_props_action.setChecked(True)
        self._view_props_action.setShortcut(QKeySequence("Ctrl+5"))
        self._view_props_action.toggled.connect(
            lambda v: self._properties_panel.setVisible(v))
        view_menu.addAction(self._view_props_action)
        view_menu.addSeparator()
        zfit_a = QAction("Zoom to &Fit", self)
        zfit_a.setShortcut(QKeySequence("Ctrl+0"))
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
        section_menu.addSeparator()
        self._gen_polygons_action = QAction("Generate Polygons From Boundaries…", self)
        self._gen_polygons_action.triggered.connect(self._on_generate_polygons)
        section_menu.addAction(self._gen_polygons_action)
        section_menu.addSeparator()
        self._strat_column_action = QAction("Edit Stratigraphic Column…", self)
        self._strat_column_action.triggered.connect(self._on_edit_strat_column)
        section_menu.addAction(self._strat_column_action)

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

        # ================================================================
        # Help
        # ================================================================
        help_menu = mb.addMenu("&Help")
        self._about_action = QAction("&About Cross Section Tool…", self)
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
        s.annotation_added.connect(lambda _: self._section_view.render())
        s.annotation_removed.connect(lambda _: self._section_view.render())
        s.annotation_modified.connect(lambda *_: self._section_view.render())
        # FPS display from section view
        self._section_view.frame_time_ms.connect(self._on_frame_time)
        # Keyboard shortcuts for tools (application-wide)
        _tool_keys = {
            "V": "select",    "A": "node_edit",
            "H": "pan",       "Z": "zoom",
            "S": "new_section",
            "P": "horizon_pick", "F": "fault_pick",
            "G": "polygon",   "M": "measure",
        }
        for key, tool_id in _tool_keys.items():
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(
                lambda tid=tool_id: self._tool_palette.set_active_tool(tid)
            )

        # Shift+Z → zoom to fit
        sc_fit = QShortcut(QKeySequence("Shift+Z"), self)
        sc_fit.setContext(Qt.ShortcutContext.ApplicationShortcut)
        sc_fit.activated.connect(self._zoom_to_fit)

        # R → cycle through reference line tools
        self._ref_cycle_idx = 0
        sc_ref = QShortcut(QKeySequence("R"), self)
        sc_ref.setContext(Qt.ShortcutContext.ApplicationShortcut)
        sc_ref.activated.connect(self._cycle_ref_line_tool)
        # Global undo/redo shortcuts (Phase 7)
        sc_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        sc_undo.setContext(Qt.ShortcutContext.ApplicationShortcut)
        sc_undo.activated.connect(self._state.undo)
        sc_redo = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
        sc_redo.setContext(Qt.ShortcutContext.ApplicationShortcut)
        sc_redo.activated.connect(self._state.redo)

        # Panel toggle shortcuts
        sc1 = QShortcut(QKeySequence("Ctrl+1"), self)
        sc1.activated.connect(self._project_panel.toggleViewAction().trigger)
        sc2 = QShortcut(QKeySequence("Ctrl+2"), self)
        sc2.activated.connect(self._toggle_map_panel)
        sc3 = QShortcut(QKeySequence("Ctrl+3"), self)
        sc3.activated.connect(self._toggle_section_panel)
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
        name = os.path.basename(path) if path else "Untitled"
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
        if not self._check_unsaved_changes():
            return
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(
            self, "New Project", "Project name:", text="Untitled"
        )
        if not ok:
            return
        crs, ok2 = QInputDialog.getInt(
            self, "New Project",
            "Coordinate Reference System (EPSG code):",
            value=32632, min=1, max=999999,
        )
        if not ok2:
            return
        self._new_project(name=name.strip() or "Untitled", crs_epsg=crs)

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
            from cross_section_tool.views.seismic_import_dialog import SeismicImportDialog
            fname = os.path.basename(path)
            dlg = SeismicImportDialog(
                sections=self._state.project.sections,
                filename=fname,
                parent=self,
            )
            if dlg.exec() != dlg.DialogCode.Accepted:
                continue
            ref = SeismicRef(
                path=path,
                name=os.path.splitext(fname)[0],
                x_field=dlg.x_field,
                y_field=dlg.y_field,
                apply_scalar=dlg.apply_scalar,
                domain=dlg.domain,
                depth_units=dlg.depth_units,
                crs_epsg=self._state.project.crs_epsg,
            )
            self._state.add_seismic_ref(ref)

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

    def _on_import_well_tops(self) -> None:
        """Import well tops from a CSV file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Well Tops CSV", "",
            "CSV Files (*.csv *.txt);;All Files (*)"
        )
        if not path:
            return
        from cross_section_tool.views.well_tops_dialog import WellTopsDialog
        dlg = WellTopsDialog(path, crs_epsg=self._state.project.crs_epsg, parent=self)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        try:
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
        from cross_section_tool.views.horizon_dialog import HorizonDialog
        from cross_section_tool.core.surfaces import HorizonPick
        default_name = f"Horizon {len(self._state.project.horizon_picks) + 1}"
        dlg = HorizonDialog(self, name=default_name)
        if dlg.exec() != dlg.DialogCode.Accepted or not dlg.name:
            return
        hp = HorizonPick.empty(name=dlg.name, color=dlg.color)
        hp.contact_type    = dlg.contact_type
        hp.formation_above = dlg.formation_above
        hp.formation_below = dlg.formation_below
        self._state.add_horizon_pick(hp)
        idx = len(self._state.project.horizon_picks) - 1
        self._state.set_active_pick_target("Horizons", idx)
        self._tool_palette.set_active_tool("horizon_pick")

    def _on_generate_polygons(self) -> None:
        """Phase 4: detect and import closed regions as polygons."""
        section = self._state.active_section
        if section is None:
            QMessageBox.information(self, "No Section",
                                    "Activate a section first.")
            return
        from cross_section_tool.core.polygon_detection import detect_polygons
        from cross_section_tool.core.polygons import SectionPolygon
        try:
            polys = detect_polygons(
                self._state.project.horizon_picks,
                self._state.project.fault_picks,
                self._state.project.reference_lines,
                section,
                section_name=section.name,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Detection Error", str(exc))
            return
        if not polys:
            QMessageBox.information(self, "No Polygons Found",
                                    "No closed regions were detected.")
            return
        # Simple dialog: ask user to confirm import
        reply = QMessageBox.question(
            self, "Import Polygons",
            f"{len(polys)} closed region(s) detected.\nImport all as polygons?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        existing = len(self._state.project.polygons)
        import numpy as np
        for i, shp in enumerate(polys):
            coords = list(shp.exterior.coords)
            if coords[0] == coords[-1]:
                coords = coords[:-1]  # drop duplicate closing vertex
            if len(coords) < 3:
                continue
            poly = SectionPolygon(
                vertices=np.array(coords),
                name=f"Region {existing + i + 1}",
            )
            self._state.add_polygon(poly)

    def _on_edit_strat_column(self) -> None:
        """Phase 5: open stratigraphic column editor (stub)."""
        from cross_section_tool.views.strat_column_dialog import StratColumnDialog
        dlg = StratColumnDialog(self._state, self)
        dlg.exec()

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

    def _add_new_polygon(self) -> None:
        from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout,
                                       QLineEdit, QColorDialog, QDoubleSpinBox,
                                       QComboBox)
        from PySide6.QtGui import QColor
        from cross_section_tool.core.polygons import SectionPolygon

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
        from cross_section_tool.views.fault_dialog import FaultDialog
        from cross_section_tool.core.surfaces import HorizonPick
        default_name = f"Fault {len(self._state.project.fault_picks) + 1}"
        dlg = FaultDialog(self, name=default_name)
        if dlg.exec() != dlg.DialogCode.Accepted or not dlg.name:
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
        from cross_section_tool.views.horizon_dialog import HorizonDialog
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
        from cross_section_tool.views.fault_dialog import FaultDialog
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

    # ------------------------------------------------------------------
    # Phase 6 helpers
    # ------------------------------------------------------------------

    def _zoom_to_fit(self) -> None:
        """Shift+Z: reset both views to full data extent."""
        self._map_view.render()
        self._section_view._ax_limits_set = False
        self._section_view.render()

    def _cycle_ref_line_tool(self) -> None:
        """R key: cycle H-Ref → V-Ref → A-Ref."""
        tools = ["h_ref", "v_ref", "a_ref"]
        cur = self._state.active_tool
        if cur in tools:
            self._ref_cycle_idx = (tools.index(cur) + 1) % len(tools)
        else:
            self._ref_cycle_idx = 0
        self._tool_palette.set_active_tool(tools[self._ref_cycle_idx])

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

    def _flash_status(self, msg: str) -> None:
        """Phase 7: briefly show *msg* in the status bar, then restore."""
        from PySide6.QtCore import QTimer
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
