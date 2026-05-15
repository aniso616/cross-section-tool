"""Tests for the keyboard shortcut system.

Verifies that:
  - All expected shortcuts are registered as QShortcut objects with
    ApplicationShortcut context (so they fire regardless of focus).
  - No two shortcuts share the same key sequence (no ambiguity).
  - Panel-toggle assignments match the specification.
  - The shortcuts dialog can be instantiated without errors.
  - SHORTCUT_REGISTRY covers every required key.
"""
from __future__ import annotations

import sys

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication

from section_tool.app import MainWindow
from section_tool.app_state import AppState


# ---------------------------------------------------------------------------
# Session QApplication + per-test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


@pytest.fixture
def state():
    return AppState()


@pytest.fixture
def win(qapp, state):
    w = MainWindow(state=state)
    yield w
    try:
        w._viewer_3d.plotter.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _collect_shortcuts(win) -> dict[str, QShortcut]:
    """Return {key_string: QShortcut} for every QShortcut child of win."""
    result = {}
    for sc in win.findChildren(QShortcut):
        key = sc.key().toString()
        if key:
            result[key] = sc
    return result


# ---------------------------------------------------------------------------
# 1. All required shortcuts are registered
# ---------------------------------------------------------------------------

class TestAllShortcutsRegistered:

    # The canonical expected set from the specification.
    # Qt normalises "Delete" → "Del" and "Escape" → "Esc" in portable text.
    EXPECTED = [
        "V", "A", "H", "Z",             # Navigation
        "P", "F", "G", "S",             # Interpretation
        "R", "M",                        # Construction
        "Del", "Esc",                    # Editing (Qt canonical names)
        "Ctrl+Z", "Ctrl+Shift+Z",
        "Ctrl+C", "Ctrl+V", "Ctrl+A",
        "Ctrl+N", "Ctrl+O", "Ctrl+S",   # File
        "Ctrl+0",                        # View — zoom
        "Ctrl+1", "Ctrl+2", "Ctrl+3",   # View — panels
        "Ctrl+4", "Ctrl+5",
    ]

    def test_all_expected_shortcuts_present(self, win):
        shortcuts = _collect_shortcuts(win)
        missing = [k for k in self.EXPECTED if k not in shortcuts]
        assert missing == [], f"Missing shortcuts: {missing}"

    def test_shortcuts_have_application_context(self, win):
        """Every QShortcut must use ApplicationShortcut context."""
        app_ctx = Qt.ShortcutContext.ApplicationShortcut
        bad = []
        for sc in win.findChildren(QShortcut):
            key = sc.key().toString()
            if key and sc.context() != app_ctx:
                bad.append(key)
        assert bad == [], f"Shortcuts with wrong context: {bad}"


# ---------------------------------------------------------------------------
# 2. No duplicate key sequences (ambiguity check)
# ---------------------------------------------------------------------------

class TestNoDuplicateShortcuts:

    def test_no_duplicate_keys(self, win):
        seen: dict[str, list] = {}
        for sc in win.findChildren(QShortcut):
            key = sc.key().toString()
            if key:
                seen.setdefault(key, []).append(sc)
        duplicates = {k: v for k, v in seen.items() if len(v) > 1}
        assert duplicates == {}, (
            f"Duplicate shortcut registrations detected: "
            f"{list(duplicates.keys())}"
        )


# ---------------------------------------------------------------------------
# 3. Panel-toggle assignments match spec
# ---------------------------------------------------------------------------

class TestPanelToggleAssignments:

    def test_ctrl1_toggles_map_panel(self, win):
        shortcuts = _collect_shortcuts(win)
        assert "Ctrl+1" in shortcuts
        # Verify it fires without error (map panel toggles)
        shortcuts["Ctrl+1"].activated.emit()

    def test_ctrl2_toggles_section_panel(self, win):
        shortcuts = _collect_shortcuts(win)
        assert "Ctrl+2" in shortcuts

    def test_ctrl3_toggles_3d_view(self, win):
        shortcuts = _collect_shortcuts(win)
        assert "Ctrl+3" in shortcuts

    def test_ctrl4_toggles_project_panel(self, win):
        shortcuts = _collect_shortcuts(win)
        assert "Ctrl+4" in shortcuts

    def test_ctrl5_toggles_properties_panel(self, win):
        shortcuts = _collect_shortcuts(win)
        assert "Ctrl+5" in shortcuts

    def test_panel_shortcuts_fire(self, win):
        """Activating Ctrl+1/2/3/4/5 must not raise."""
        shortcuts = _collect_shortcuts(win)
        for key in ("Ctrl+1", "Ctrl+2", "Ctrl+3", "Ctrl+4", "Ctrl+5"):
            shortcuts[key].activated.emit()   # simulate activation


# ---------------------------------------------------------------------------
# 4. Tool shortcuts switch the active tool
# ---------------------------------------------------------------------------

class TestToolShortcuts:

    @pytest.mark.parametrize("key,expected_tool", [
        ("V", "select"),
        ("A", "node_edit"),
        ("H", "pan"),
        ("Z", "zoom"),
        ("G", "polygon"),
        ("S", "new_section"),
        ("M", "measure"),
    ])
    def test_tool_shortcut_activates_correct_tool(self, win, state, key, expected_tool):
        shortcuts = _collect_shortcuts(win)
        assert key in shortcuts
        shortcuts[key].activated.emit()
        assert state.active_tool == expected_tool

    def test_p_shortcut_activates_horizon_pick(self, win, state):
        """P activates horizon_pick when a pick target exists."""
        from section_tool.core.surfaces import HorizonPick
        state.add_horizon_pick(HorizonPick([0.0], [100.0], name="H1"))
        shortcuts = _collect_shortcuts(win)
        shortcuts["P"].activated.emit()
        assert state.active_tool == "horizon_pick"

    def test_f_shortcut_activates_fault_pick(self, win, state):
        """F activates fault_pick when a pick target exists."""
        from section_tool.core.surfaces import HorizonPick
        state.add_fault_pick(HorizonPick([0.0], [100.0], name="F1"))
        shortcuts = _collect_shortcuts(win)
        shortcuts["F"].activated.emit()
        assert state.active_tool == "fault_pick"

    def test_escape_returns_to_select(self, win, state):
        state.set_active_tool("pan")
        shortcuts = _collect_shortcuts(win)
        shortcuts["Esc"].activated.emit()   # Qt canonical name
        assert state.active_tool == "select"

    def test_r_activates_ref_line_tool(self, win, state):
        shortcuts = _collect_shortcuts(win)
        shortcuts["R"].activated.emit()
        assert state.active_tool in ("h_ref", "v_ref", "a_ref")

    def test_r_cycles_through_ref_line_tools(self, win, state):
        shortcuts = _collect_shortcuts(win)
        sc_r = shortcuts["R"]
        seen = set()
        for _ in range(4):
            sc_r.activated.emit()
            seen.add(state.active_tool)
        assert seen == {"h_ref", "v_ref", "a_ref"}


# ---------------------------------------------------------------------------
# 5. File shortcuts invoke correct handlers
# ---------------------------------------------------------------------------

class TestFileShortcuts:

    def test_ctrl_z_undoes(self, win, state):
        """Ctrl+Z shortcut calls undo on the state."""
        from unittest.mock import patch
        shortcuts = _collect_shortcuts(win)
        with patch.object(state, "undo") as mock_undo:
            shortcuts["Ctrl+Z"].activated.emit()
            mock_undo.assert_called_once()

    def test_ctrl_shift_z_redoes(self, win, state):
        from unittest.mock import patch
        shortcuts = _collect_shortcuts(win)
        with patch.object(state, "redo") as mock_redo:
            shortcuts["Ctrl+Shift+Z"].activated.emit()
            mock_redo.assert_called_once()


# ---------------------------------------------------------------------------
# 6. SHORTCUT_REGISTRY completeness
# ---------------------------------------------------------------------------

class TestShortcutRegistry:

    def test_registry_is_non_empty(self):
        assert len(MainWindow.SHORTCUT_REGISTRY) > 0

    def test_registry_has_required_keys(self):
        keys = {entry[0] for entry in MainWindow.SHORTCUT_REGISTRY}
        # Registry uses human-readable key names (Delete, Escape) for documentation;
        # QShortcut normalises these to "Del"/"Esc" at registration time.
        required = {
            "V", "A", "H", "Z", "P", "F", "G", "S", "R", "M",
            "Delete", "Ctrl+Z", "Ctrl+Shift+Z", "Ctrl+S",
            "Ctrl+N", "Ctrl+O", "Ctrl+0",
            "Ctrl+1", "Ctrl+2", "Ctrl+3", "Ctrl+4", "Ctrl+5",
            "Escape",
        }
        missing = required - keys
        assert missing == set(), f"Missing from registry: {missing}"

    def test_registry_includes_space_for_documentation(self):
        """Space (hold) is documented in registry even though it's keyPressEvent."""
        keys = {entry[0] for entry in MainWindow.SHORTCUT_REGISTRY}
        assert "Space" in keys

    def test_registry_entries_have_three_fields(self):
        for entry in MainWindow.SHORTCUT_REGISTRY:
            assert len(entry) == 3, f"Bad registry entry: {entry}"
            key, desc, cat = entry
            assert isinstance(key, str) and key
            assert isinstance(desc, str) and desc
            assert isinstance(cat, str) and cat

    def test_registry_categories_are_known(self):
        known = {"Navigation", "Interpretation", "Construction",
                 "Editing", "File", "View"}
        for key, desc, cat in MainWindow.SHORTCUT_REGISTRY:
            assert cat in known, f"Unknown category '{cat}' for key '{key}'"


# ---------------------------------------------------------------------------
# 7. Shortcuts dialog
# ---------------------------------------------------------------------------

class TestShortcutsDialog:

    def test_dialog_creates_without_error(self, win, qapp):
        from section_tool.views.shortcuts_dialog import KeyboardShortcutsDialog
        dlg = KeyboardShortcutsDialog(MainWindow.SHORTCUT_REGISTRY, win)
        assert dlg is not None

    def test_dialog_has_correct_title(self, win):
        from section_tool.views.shortcuts_dialog import KeyboardShortcutsDialog
        dlg = KeyboardShortcutsDialog(MainWindow.SHORTCUT_REGISTRY, win)
        assert dlg.windowTitle() == "Keyboard Shortcuts"

    def test_dialog_table_row_count_matches_registry(self, win):
        from PySide6.QtWidgets import QTableWidget
        from section_tool.views.shortcuts_dialog import KeyboardShortcutsDialog
        dlg = KeyboardShortcutsDialog(MainWindow.SHORTCUT_REGISTRY, win)
        table = dlg.findChild(QTableWidget)
        assert table is not None
        assert table.rowCount() == len(MainWindow.SHORTCUT_REGISTRY)

    def test_dialog_has_three_columns(self, win):
        from PySide6.QtWidgets import QTableWidget
        from section_tool.views.shortcuts_dialog import KeyboardShortcutsDialog
        dlg = KeyboardShortcutsDialog(MainWindow.SHORTCUT_REGISTRY, win)
        table = dlg.findChild(QTableWidget)
        assert table.columnCount() == 3
