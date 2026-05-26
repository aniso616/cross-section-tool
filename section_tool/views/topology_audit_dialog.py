"""Topology Audit dialog — shows detected interpretation hygiene issues."""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

if TYPE_CHECKING:
    from section_tool.app_state import AppState


_SEVERITY_ICON = {"error": "✖", "warning": "▲", "info": "ℹ"}
_SEVERITY_COLOR = {"error": "#ff6060", "warning": "#ffcc44", "info": "#88aacc"}


class TopologyAuditDialog(QDialog):
    """Modal dialog listing topology issues for the active section.

    Parameters
    ----------
    state:
        Application state used to resolve the active section and run fixes.
    parent:
        Parent widget.
    """

    def __init__(self, state: "AppState", parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._issues: list = []
        self.setWindowTitle("Topology Audit")
        self.resize(800, 480)
        self._setup_ui()
        self._run_audit()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Summary label
        self._summary_label = QLabel("Running…")
        self._summary_label.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(self._summary_label)

        # Table
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(
            ["", "Entity", "Category", "Description", ""]
        )
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.setColumnWidth(0, 28)
        self._table.setColumnWidth(1, 160)
        self._table.setColumnWidth(2, 90)
        self._table.horizontalHeader().setSectionResizeMode(
            3, self._table.horizontalHeader().ResizeMode.Stretch
        )
        self._table.setColumnWidth(4, 70)
        self._table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._table)

        # Action bar
        bar = QHBoxLayout()
        self._fix_all_btn = QPushButton("Fix All Auto-fixable")
        self._fix_all_btn.clicked.connect(self._on_fix_all)
        bar.addWidget(self._fix_all_btn)
        bar.addStretch()
        rerun_btn = QPushButton("Re-run Audit")
        rerun_btn.clicked.connect(self._run_audit)
        bar.addWidget(rerun_btn)
        layout.addLayout(bar)

        # Close button
        bbox = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bbox.rejected.connect(self.reject)
        layout.addWidget(bbox)

    # ------------------------------------------------------------------
    # Audit logic
    # ------------------------------------------------------------------

    def _run_audit(self) -> None:
        from section_tool.core.topology_audit import audit_section
        section = self._state.active_section
        if section is None:
            self._summary_label.setText("No active section.")
            self._issues = []
            self._populate_table()
            return
        self._issues = audit_section(section, self._state.project)
        self._populate_table()
        self._update_summary()

    def _populate_table(self) -> None:
        self._table.setRowCount(0)
        for row_idx, issue in enumerate(self._issues):
            self._table.insertRow(row_idx)

            # Severity icon
            icon_item = QTableWidgetItem(_SEVERITY_ICON.get(issue.severity, "?"))
            icon_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            icon_item.setForeground(
                Qt.GlobalColor.white
                if True
                else Qt.GlobalColor.black
            )
            # Background color per severity
            from PySide6.QtGui import QColor
            icon_item.setBackground(QColor(_SEVERITY_COLOR.get(issue.severity, "#888")))
            self._table.setItem(row_idx, 0, icon_item)

            self._table.setItem(row_idx, 1, QTableWidgetItem(issue.entity_name))
            self._table.setItem(row_idx, 2, QTableWidgetItem(issue.category))
            self._table.setItem(row_idx, 3, QTableWidgetItem(issue.description))

            if issue.auto_fixable:
                fix_btn = QPushButton("Fix")
                fix_btn.setSizePolicy(
                    QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
                )
                fix_btn.clicked.connect(lambda _checked, i=issue: self._apply_fix(i))
                self._table.setCellWidget(row_idx, 4, fix_btn)

    def _update_summary(self) -> None:
        if not self._issues:
            self._summary_label.setText("No issues found.")
            self._fix_all_btn.setEnabled(False)
            return
        errors   = sum(1 for i in self._issues if i.severity == "error")
        warnings = sum(1 for i in self._issues if i.severity == "warning")
        infos    = sum(1 for i in self._issues if i.severity == "info")
        fixable  = sum(1 for i in self._issues if i.auto_fixable)
        parts = []
        if errors:
            parts.append(f"{errors} error(s)")
        if warnings:
            parts.append(f"{warnings} warning(s)")
        if infos:
            parts.append(f"{infos} info")
        self._summary_label.setText(
            "  ".join(parts) + (f"  ({fixable} auto-fixable)" if fixable else "")
        )
        self._fix_all_btn.setEnabled(fixable > 0)

    def _apply_fix(self, issue) -> None:
        if issue.fix_action is None:
            return
        try:
            issue.fix_action(self._state)
        except Exception as exc:
            QMessageBox.warning(self, "Fix Failed", str(exc))
        self._run_audit()

    def _on_fix_all(self) -> None:
        fixable = [i for i in self._issues if i.auto_fixable and i.fix_action]
        if not fixable:
            return
        for issue in fixable:
            try:
                issue.fix_action(self._state)
            except Exception:
                pass
        self._run_audit()
