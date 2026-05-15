"""Import File dialog — copy into project vs. external reference."""
from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QGroupBox, QLabel,
    QRadioButton, QVBoxLayout,
)


def _format_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.0f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


class ImportFileDialog(QDialog):
    """Ask the user whether to copy a file into the project or reference it externally.

    Parameters
    ----------
    source_path:
        Absolute path to the file being imported.
    file_type_label:
        Human-readable type, e.g. ``"SEG-Y"`` or ``"LAS"``.
    dest_dir:
        The project sub-folder the file would be copied to.
    """

    def __init__(
        self,
        source_path: str,
        file_type_label: str = "File",
        dest_dir: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Import {file_type_label}")
        self.setMinimumWidth(480)

        fname = os.path.basename(source_path)
        try:
            size_str = _format_size(os.path.getsize(source_path))
        except OSError:
            size_str = "unknown size"

        layout = QVBoxLayout(self)

        header = QLabel(f"<b>Import: {fname}</b>  ({size_str})")
        layout.addWidget(header)

        grp = QGroupBox("How to store this file")
        vl = QVBoxLayout(grp)

        self._copy_rb = QRadioButton(
            "Copy into project folder  (self-contained, portable)"
        )
        self._copy_rb.setChecked(True)
        vl.addWidget(self._copy_rb)

        if dest_dir:
            dest_lbl = QLabel(f"   Destination: {os.path.join(dest_dir, fname)}")
            dest_lbl.setStyleSheet("color: grey; font-size: 8pt;")
            vl.addWidget(dest_lbl)

        self._ref_rb = QRadioButton(
            "Reference original location  (lighter, breaks if file moves)"
        )
        vl.addWidget(self._ref_rb)

        orig_lbl = QLabel(f"   Original: {source_path}")
        orig_lbl.setStyleSheet("color: grey; font-size: 8pt;")
        orig_lbl.setWordWrap(True)
        vl.addWidget(orig_lbl)

        layout.addWidget(grp)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.button(QDialogButtonBox.StandardButton.Ok).setText("Import")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    # ------------------------------------------------------------------

    def copy_into_project(self) -> bool:
        """True if the user chose to copy the file into the project folder."""
        return self._copy_rb.isChecked()
