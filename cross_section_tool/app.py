from __future__ import annotations

import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QToolBar,
    QWidget,
)

from cross_section_tool.app_state import AppState
from cross_section_tool.core.section import Section
from cross_section_tool.core.surfaces import HorizonPick
from cross_section_tool.views.map_view import MapView
from cross_section_tool.views.section_view import SectionView
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

        # Horizontal splitter
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.addWidget(self._map_view)
        self._splitter.addWidget(self._tabs)
        self._splitter.setSizes([320, 960])
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 3)

        self.setCentralWidget(self._splitter)

        # Status bar
        self._status_label = QLabel("New project")
        self.statusBar().addWidget(self._status_label)

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
        self._pick_action.setShortcut(QKeySequence("P"))
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

        # ---- Help ----
        help_menu = mb.addMenu("&Help")

        self._about_action = QAction("&About", self)
        self._about_action.triggered.connect(self._on_about)
        help_menu.addAction(self._about_action)

    def _build_toolbar(self) -> None:
        tb: QToolBar = self.addToolBar("Main")
        tb.setObjectName("MainToolBar")
        tb.setMovable(False)
        tb.addAction(self._new_action)
        tb.addAction(self._open_action)
        tb.addAction(self._save_action)
        tb.addSeparator()
        tb.addAction(self._new_section_action)
        tb.addSeparator()
        tb.addAction(self._pick_action)

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
        self._section_view.horizon_pick_requested.connect(self._on_pick_requested)

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
        if path:
            msg = os.path.basename(path)
        else:
            msg = "New project"
        if self._state.is_modified:
            msg += "  [unsaved changes]"
        n_sec = len(self._state.project.sections)
        n_well = len(self._state.project.wells)
        msg += f"  |  {n_sec} section(s)  {n_well} well(s)"
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
        """Load a project from *path*. Returns True on success."""
        try:
            self._state.open_project(path)
            return True
        except Exception as exc:
            QMessageBox.critical(self, "Open Error", str(exc))
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
        """Save to *path* (no dialog)."""
        try:
            self._state.save_project_as(path)
            return True
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))
            return False

    def _save_project_as_dialog(self) -> bool:
        """Open a Save As dialog and save."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "", "HDF5 Project (*.h5);;All Files (*)"
        )
        if not path:
            return False
        return self._save_project_as(path)

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
            self._open_project(path)

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

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
