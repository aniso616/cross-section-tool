"""Import Time–Depth Data dialog — one door, classify by evidence.

Replaces the two separate "Well Checkshot" / "Well Sonic TDR" menu items. The
file is parsed first; this dialog then states what was found (point count, depth
range, spacing regularity, detected TWT units) and pre-selects a *kind* by
heuristic — checkshot for a sparse irregular table, sonic-integrated for a dense
regular grid. The user can override, but the evidence is on screen, so data can
never silently wear a grade the numbers contradict.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QFrame, QLabel,
    QVBoxLayout,
)

from section_tool.io.tdr_io import TdrClassification
from section_tool.core.tdr import KIND_LABEL, DEPTH_REFERENCE_LABEL
from section_tool.core.zdomain import ZDomain

# One-line format hint shown up front — answers "what should this file look like?"
_FORMAT_HINT = (
    "Expected: two whitespace-separated columns — <b>depth</b> (m) and "
    "<b>TWT</b> (s or ms). Header/comment lines are skipped.<br>"
    "Example row: <tt>553.6&nbsp;&nbsp;&nbsp;0.544</tt>"
)

_KIND_ORDER = ("checkshot", "sonic_integrated", "imported")
_REF_ORDER = ("MD", "TVDSS", "TVD_KB")


class TdrImportDialog(QDialog):
    """Confirm the kind + depth reference for a parsed time–depth file.

    Construct with the filename and a :class:`TdrClassification`; on accept,
    :meth:`result_kind`, :meth:`result_depth_reference` and
    :meth:`result_twt_domain` give the (possibly user-overridden) choices.
    """

    def __init__(self, filename: str, cls: TdrClassification, well_name: str = "",
                 parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Time–Depth Data")
        self.setMinimumWidth(500)
        self._cls = cls

        layout = QVBoxLayout(self)

        head = QLabel(f"<b>{filename}</b>" + (f" → {well_name}" if well_name else ""))
        layout.addWidget(head)

        hint = QLabel(_FORMAT_HINT)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #999999;")
        layout.addWidget(hint)

        line = QFrame(); line.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(line)

        unit = "ms" if cls.twt_domain == ZDomain.TWT_MS else "s"
        spacing = (f"regular ~{cls.median_spacing:.0f} m grid"
                   if cls.spacing_regular
                   else f"irregular (median {cls.median_spacing:.0f} m)")
        found = QLabel(
            f"<b>Found:</b> {cls.n_points} points · depth "
            f"{cls.depth_min:.0f}–{cls.depth_max:.0f} m · spacing {spacing} · "
            f"TWT {cls.twt_min:.3g}–{cls.twt_max:.3g} {unit}")
        found.setWordWrap(True)
        layout.addWidget(found)

        evidence = QLabel(cls.evidence)
        evidence.setWordWrap(True)
        evidence.setStyleSheet("color: #4A90D9;")
        layout.addWidget(evidence)

        form = QFormLayout()
        self._kind_combo = QComboBox()
        for k in _KIND_ORDER:
            self._kind_combo.addItem(KIND_LABEL.get(k, k), k)
        self._kind_combo.setCurrentIndex(_KIND_ORDER.index(cls.suggested_kind))
        form.addRow("Kind:", self._kind_combo)

        self._ref_combo = QComboBox()
        for r in _REF_ORDER:
            self._ref_combo.addItem(DEPTH_REFERENCE_LABEL.get(r, r), r)
        self._ref_combo.setCurrentIndex(_REF_ORDER.index(cls.suggested_depth_reference))
        form.addRow("Depth reference:", self._ref_combo)

        self._twt_combo = QComboBox()
        self._twt_combo.addItem("Auto-detect", None)
        self._twt_combo.addItem("Seconds (s)", ZDomain.TWT_S)
        self._twt_combo.addItem("Milliseconds (ms)", ZDomain.TWT_MS)
        form.addRow("TWT units:", self._twt_combo)
        layout.addLayout(form)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def result_kind(self) -> str:
        return self._kind_combo.currentData()

    def result_depth_reference(self) -> str:
        return self._ref_combo.currentData()

    def result_twt_domain(self) -> ZDomain | None:
        return self._twt_combo.currentData()
