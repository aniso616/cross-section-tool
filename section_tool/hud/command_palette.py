from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)


class CommandPalette(QWidget):
    command_selected = Signal(str)

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._commands = []
        self._build_ui()
        self.register_defaults()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.frame = QFrame(self)
        self.frame.setStyleSheet("""
            QFrame {
                background: rgba(18, 18, 24, 238);
                border: 1px solid rgba(75, 75, 105, 200);
                border-radius: 10px;
            }
        """)
        fl = QVBoxLayout(self.frame)
        fl.setContentsMargins(0, 0, 0, 8)
        fl.setSpacing(0)

        self.input = QLineEdit(self.frame)
        self.input.setPlaceholderText("Type a command…")
        self.input.setStyleSheet("""
            QLineEdit {
                background: transparent;
                border: none;
                border-bottom: 1px solid rgba(75, 75, 105, 140);
                color: rgba(220, 220, 220, 255);
                font-size: 15px;
                padding: 14px 16px;
            }
        """)
        self.input.textChanged.connect(self._filter)
        fl.addWidget(self.input)

        self.results = QListWidget(self.frame)
        self.results.setStyleSheet("""
            QListWidget {
                background: transparent;
                border: none;
                color: rgba(200, 200, 200, 255);
                font-size: 13px;
                padding: 4px 0;
            }
            QListWidget::item          { padding: 7px 16px; border-radius: 4px; }
            QListWidget::item:selected { background: rgba(65, 105, 165, 210); }
            QListWidget::item:disabled { color: rgba(110, 110, 120, 180); }
        """)
        self.results.setFixedHeight(280)
        self.results.itemActivated.connect(self._on_select)
        fl.addWidget(self.results)

        layout.addWidget(self.frame)

    def register_command(self, id, label, shortcut="", description=""):
        self._commands.append(
            {"id": id, "label": label,
             "shortcut": shortcut, "desc": description}
        )

    def register_defaults(self):
        for args in [
            ("tool_horizon",   "Draw Horizon",          "H"),
            ("tool_fault",     "Draw Fault",             "F"),
            ("tool_pick",      "Pick Well Top",          "T"),
            ("tool_annotate",  "Add Annotation",         "A"),
            ("mode_section",   "Switch to Section View", "1"),
            ("mode_map",       "Switch to Map View",     "2"),
            ("mode_3d",        "Switch to 3D View",      "3"),
            ("view_fit",       "Fit View to Data",       "Ctrl+0"),
            ("export_image",   "Export Section Image…",  ""),
            ("export_svg",     "Export as SVG…",         ""),
            ("project_open",    "Open Project…",          "Ctrl+O"),
            ("project_save",    "Save Project",           "Ctrl+S"),
            ("view_grid_on",    "Show Grid Lines",        ""),
            ("view_grid_off",   "Hide Grid Lines",        ""),
        ]:
            self.register_command(*args)

    def invoke(self):
        self._filter("")
        self.input.clear()
        self._reposition()
        self.show()
        self.raise_()
        self.input.setFocus()

    def _reposition(self):
        pr = self.parent().rect()
        w  = min(580, pr.width() - 80)
        self.setFixedWidth(w)
        self.move((pr.width() - w) // 2, int(pr.height() * 0.18))
        self.adjustSize()

    def _filter(self, text):
        q = text.lower()
        filtered = [c for c in self._commands
                    if q in c["label"].lower() or q in c["desc"].lower()]
        self.results.clear()
        for c in filtered:
            display = c["label"]
            if c["shortcut"]:
                display += f"  [{c['shortcut']}]"
            item = QListWidgetItem(display)
            item.setData(Qt.ItemDataRole.UserRole, c["id"])
            self.results.addItem(item)
        if filtered:
            self.results.setCurrentRow(0)

    def _on_select(self, item):
        self.command_selected.emit(item.data(Qt.ItemDataRole.UserRole))
        self.hide()

    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key.Key_Escape:
            self.hide()
        elif k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.results.currentItem():
                self._on_select(self.results.currentItem())
        elif k == Qt.Key.Key_Down:
            self.results.setCurrentRow(
                min(self.results.currentRow() + 1, self.results.count() - 1))
        elif k == Qt.Key.Key_Up:
            self.results.setCurrentRow(
                max(self.results.currentRow() - 1, 0))
        else:
            self.input.setFocus()
            super().keyPressEvent(event)
