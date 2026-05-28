from PySide6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QLabel,
                                QComboBox, QLineEdit, QDialogButtonBox)


class SourceCRSDialog(QDialog):
    """Ask the user what CRS an imported file is in. Defaults to project CRS."""

    _COMMON = [
        ("WGS 84 / UTM 29N", 32629),
        ("WGS 84 / UTM 30N", 32630),
        ("WGS 84 / UTM 31N", 32631),
        ("WGS 84 / UTM 32N", 32632),
        ("WGS 84 / UTM 33N", 32633),
        ("WGS 84 lat/lon (geographic)", 4326),
        ("NAD83 / UTM 13N", 26913),
        ("NAD83 / UTM 14N", 26914),
        ("Custom…", None),
    ]

    def __init__(self, project_crs: int, filename: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Source CRS")
        self._project_crs = project_crs
        layout = QVBoxLayout(self)

        msg = QLabel(
            f"What CRS is <b>{filename}</b> in?\n\n"
            f"If the data is already in the project CRS (EPSG:{project_crs}), "
            f"just click OK."
        )
        msg.setWordWrap(True)
        layout.addWidget(msg)

        form = QFormLayout()
        self._combo = QComboBox()
        self._combo.addItem(f"Same as project (EPSG:{project_crs})", project_crs)
        for label, epsg in self._COMMON:
            # Don't duplicate the project CRS
            if epsg != project_crs:
                self._combo.addItem(label, epsg)
        self._combo.setCurrentIndex(0)

        self._custom = QLineEdit()
        self._custom.setPlaceholderText("e.g. 32631")
        self._custom.setEnabled(False)

        self._combo.currentIndexChanged.connect(self._on_combo_changed)
        form.addRow("CRS:", self._combo)
        form.addRow("EPSG code:", self._custom)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_combo_changed(self):
        self._custom.setEnabled(self._combo.currentData() is None)

    def source_epsg(self) -> int:
        d = self._combo.currentData()
        if d is not None:
            return d
        text = self._custom.text().strip()
        return int(text) if text.isdigit() else self._project_crs
