"""SEG-Y header viewer — text header, binary header, and trace header sample."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QHeaderView, QLabel,
    QMessageBox, QPushButton, QTabWidget, QTableWidget,
    QTableWidgetItem, QTextEdit, QVBoxLayout,
)


# ---------------------------------------------------------------------------
# Binary header field catalogue
# ---------------------------------------------------------------------------

_BIN_FIELDS = [
    # (label, segyio BinField name, description)
    ("Job ID",                      "JobID",          "Job identification number"),
    ("Line number",                  "LineNumber",      "Line number"),
    ("Reel number",                  "ReelNumber",      "Reel number"),
    ("Traces per ensemble",          "EnsembleFold",    "Traces per CDP ensemble"),
    ("Aux. traces per ensemble",     "AuxEnsembleFold", "Auxiliary traces per ensemble"),
    ("Sample interval (µs)",         "Interval",        "Sample interval in microseconds"),
    ("Sample interval (orig., µs)",  "IntervalOriginal","Original sample interval"),
    ("Samples per trace",            "Samples",         "Number of samples per data trace"),
    ("Samples per trace (orig.)",    "SamplesOriginal", "Original samples per trace"),
    ("Data sample format",           "Format",          "1=IBM float, 2=int32, 3=int16, 5=IEEE float, 8=int8"),
    ("Ensemble fold",                "EnsembleFold",    "CDP fold"),
    ("Trace sorting code",           "SortingCode",     "1=as recorded, 2=CDP, 4=shot"),
    ("Vertical sum code",            "VerticalSum",     ""),
    ("Sweep freq. at start",         "SweepFrequencyStart", "Hz"),
    ("Sweep freq. at end",           "SweepFrequencyEnd",   "Hz"),
    ("Sweep length",                 "SweepLength",     "ms"),
    ("Sweep type code",              "SweepType",       "1=linear, 2=parabolic, 3=exponential, 4=other"),
    ("Measurement system",           "MeasurementSystem", "1=meters, 2=feet"),
    ("Impulse signal polarity",      "ImpulseSignalPolarity", ""),
    ("SEG-Y format revision",        "SEGYRevision",    ""),
    ("Fixed trace length flag",      "FixedLengthTraces", "1=fixed length"),
    ("Ext. text headers count",      "ExtendedHeaders", ""),
]

_COORD_FIELDS = [
    ("CDP_X",            181, "CDP X coordinate"),
    ("CDP_Y",            185, "CDP Y coordinate"),
    ("SourceX",           73, "Source X coordinate"),
    ("SourceY",           77, "Source Y coordinate"),
    ("GroupX",            81, "Receiver group X"),
    ("GroupY",            85, "Receiver group Y"),
    ("INLINE_3D",        189, "In-line number"),
    ("CROSSLINE_3D",     193, "Cross-line number"),
    ("SourceGroupScalar",71, "Coord. scalar (neg=divisor, pos=multiplier)"),
    ("SampleCount",      115, "Number of samples in trace"),
    ("DelayRecordingTime",109, "Delay recording time (ms)"),
]

_GEOMETRY_FIELDS = [
    ("FieldRecord",        9, "Original field record number"),
    ("TraceNumber",       13, "Trace sequence number within SEG-Y file"),
    ("CDP",               21, "CDP ensemble number"),
    ("ShotPointScalar",   69, "Scalar for shot point"),
    ("ShotPoint",         17, "Energy source point number"),
    ("ElevationScalar",   69, "Scalar for elevations and depths"),
    ("ReceiverGroupElevation", 41, "Receiver group elevation"),
    ("SourceDepth",       49, "Source depth below surface"),
    ("SourceMeasurement", 225, "Source measurement"),
]


class SEGYHeaderDialog(QDialog):
    """Three-tab SEG-Y header inspector: text header, binary header, trace headers.

    Can be used as a standalone viewer (Tools → View SEG-Y Header) or embedded
    in the import workflow.
    """

    def __init__(self, path: str, parent=None) -> None:
        super().__init__(parent)
        self._path = path
        fname = os.path.basename(path)
        self.setWindowTitle(f"SEG-Y Header: {fname}")
        self.setMinimumSize(740, 580)

        layout = QVBoxLayout(self)
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs, 1)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        try:
            self._load(path)
        except ImportError:
            self._show_error("segyio is not installed.\n\npip install segyio")
        except Exception as exc:
            self._show_error(f"Failed to read SEG-Y headers:\n\n{exc}")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, path: str) -> None:
        import segyio
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            f_ctx = segyio.open(path, ignore_geometry=True)
        finally:
            QApplication.restoreOverrideCursor()

        with f_ctx as f:
            text_hdr = f.text[0].decode("cp500", errors="replace")
            bin_hdr  = {k: f.bin[k] for k in f.bin.keys()}
            n_show   = min(10, f.tracecount)
            trace_hdrs = [dict(f.header[i]) for i in range(n_show)]
            n_traces = f.tracecount

        self._tabs.addTab(self._build_text_tab(text_hdr), "Text Header")
        self._tabs.addTab(self._build_binary_tab(bin_hdr, n_traces), "Binary Header")
        self._tabs.addTab(self._build_trace_tab(trace_hdrs), "Trace Headers")

    def _show_error(self, msg: str) -> None:
        lbl = QLabel(msg)
        lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        lbl.setStyleSheet("padding: 12px; color: red;")
        lbl.setWordWrap(True)
        self._tabs.addTab(lbl, "Error")

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _build_text_tab(self, text: str) -> QTextEdit:
        te = QTextEdit()
        te.setReadOnly(True)
        mono = QFont("Courier New", 9)
        mono.setStyleHint(QFont.StyleHint.TypeWriter)
        te.setFont(mono)
        # Standard text header: 40 lines × 80 chars
        lines = [text[i*80:(i+1)*80] for i in range(40)]
        te.setPlainText("\n".join(lines))
        return te

    def _build_binary_tab(self, bin_hdr: dict, n_traces: int) -> QTableWidget:
        try:
            import segyio
        except ImportError:
            return QTableWidget()

        tbl = QTableWidget()
        tbl.setColumnCount(3)
        tbl.setHorizontalHeaderLabels(["Field", "Value", "Description"])
        tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        tbl.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        tbl.verticalHeader().setVisible(False)
        tbl.setAlternatingRowColors(True)
        tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        rows: list[tuple[str, str, str]] = [
            ("Total traces", str(n_traces), "Number of data traces in file"),
        ]
        seen: set[str] = set()
        for label, field_name, desc in _BIN_FIELDS:
            if field_name in seen:
                continue
            seen.add(field_name)
            try:
                field = getattr(segyio.BinField, field_name, None)
                if field is None:
                    continue
                val = bin_hdr.get(field, "—")
                rows.append((label, str(val), desc))
            except Exception:
                pass

        tbl.setRowCount(len(rows))
        for r, (label, val, desc) in enumerate(rows):
            tbl.setItem(r, 0, QTableWidgetItem(label))
            tbl.setItem(r, 1, QTableWidgetItem(val))
            tbl.setItem(r, 2, QTableWidgetItem(desc))

        return tbl

    def _build_trace_tab(self, trace_hdrs: list[dict]) -> QWidget:
        try:
            import segyio
        except ImportError:
            return QLabel("segyio not installed")

        container = QWidget()
        vl = QVBoxLayout(container)
        vl.setContentsMargins(4, 4, 4, 4)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Field set:"))
        self._trace_set_combo = QComboBox()
        self._trace_set_combo.addItem("Coordinates", _COORD_FIELDS)
        self._trace_set_combo.addItem("Geometry",    _GEOMETRY_FIELDS)
        top_row.addWidget(self._trace_set_combo)
        top_row.addStretch()
        vl.addLayout(top_row)

        self._trace_tbl = QTableWidget()
        self._trace_tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._trace_tbl.setAlternatingRowColors(True)
        self._trace_tbl.verticalHeader().setVisible(False)
        vl.addWidget(self._trace_tbl)

        self._trace_hdrs = trace_hdrs
        self._populate_trace_table(_COORD_FIELDS)
        self._trace_set_combo.currentIndexChanged.connect(
            lambda _: self._populate_trace_table(
                self._trace_set_combo.currentData()
            )
        )

        return container

    def _populate_trace_table(self, fields: list[tuple]) -> None:
        try:
            import segyio
        except ImportError:
            return

        tbl = self._trace_tbl
        tbl.clear()
        n = len(self._trace_hdrs)
        tbl.setColumnCount(1 + n)
        tbl.setHorizontalHeaderLabels(
            ["Field (byte)"] + [f"Trace {i+1}" for i in range(n)]
        )
        tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for c in range(1, n + 1):
            tbl.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)

        rows = []
        for label, byte_pos, _desc in fields:
            try:
                field = segyio.TraceField(byte_pos)
            except ValueError:
                continue
            row_vals = []
            for hdr in self._trace_hdrs:
                row_vals.append(str(hdr.get(field, "—")))
            rows.append((f"{label} ({byte_pos})", row_vals))

        tbl.setRowCount(len(rows))
        for r, (label, vals) in enumerate(rows):
            tbl.setItem(r, 0, QTableWidgetItem(label))
            for c, v in enumerate(vals):
                tbl.setItem(r, c + 1, QTableWidgetItem(v))
