"""Print/Export dialog with live preview and per-export parameter controls."""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from section_tool.export.print_params import PrintExportParams
from section_tool.export.print_renderer import PALETTES, PAPER_SIZES


class PrintExportDialog(QDialog):
    """Modal dialog: left = live preview, right = parameter controls."""

    def __init__(self, state, section, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Export Section — {section.name}")
        self.resize(1400, 860)
        self._state = state
        self._section = section

        self._params = PrintExportParams(title=section.name)

        # Debounce timer: refresh preview 200 ms after last control change
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._refresh_preview)

        # --- Build layout ---
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Preview pane (left)
        preview_wrap = QWidget()
        pv_layout = QVBoxLayout(preview_wrap)
        pv_layout.setContentsMargins(6, 6, 6, 6)
        pv_label = QLabel("Preview")
        pv_label.setStyleSheet("color: #888; font-size: 11px;")
        pv_layout.addWidget(pv_label)
        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumSize(700, 480)
        self._preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._preview_label.setStyleSheet(
            "background: #555; border: 1px solid #333; border-radius: 3px;")
        pv_layout.addWidget(self._preview_label, stretch=1)
        splitter.addWidget(preview_wrap)

        # Controls pane (right)
        controls = self._build_controls()
        controls.setMaximumWidth(440)
        splitter.addWidget(controls)
        splitter.setSizes([960, 440])

        # Bottom buttons
        btn_bar = QHBoxLayout()
        self._btn_png = QPushButton("Save PNG…")
        self._btn_pdf = QPushButton("Save PDF…")
        self._btn_svg = QPushButton("Save SVG…")
        self._btn_close = QPushButton("Close")
        for btn in (self._btn_png, self._btn_pdf, self._btn_svg):
            btn.setMinimumWidth(110)
        btn_bar.addStretch()
        btn_bar.addWidget(self._btn_png)
        btn_bar.addWidget(self._btn_pdf)
        btn_bar.addWidget(self._btn_svg)
        btn_bar.addSpacing(16)
        btn_bar.addWidget(self._btn_close)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.addWidget(splitter, stretch=1)
        root.addLayout(btn_bar)

        self._btn_png.clicked.connect(self._save_png)
        self._btn_pdf.clicked.connect(self._save_pdf)
        self._btn_svg.clicked.connect(self._save_svg)
        self._btn_close.clicked.connect(self.reject)

        self._refresh_preview()

    # ------------------------------------------------------------------
    # Controls builder
    # ------------------------------------------------------------------

    def _build_controls(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(8, 8, 8, 8)

        tabs = QTabWidget()
        tabs.addTab(self._build_page_tab(),     "Page")
        tabs.addTab(self._build_palette_tab(),  "Palette")
        tabs.addTab(self._build_type_tab(),     "Type")
        tabs.addTab(self._build_content_tab(),  "Content")
        tabs.addTab(self._build_titleblock_tab(), "Title block")

        layout.addWidget(tabs)
        layout.addStretch()
        scroll.setWidget(inner)

        # Wire all inputs to the debounce timer
        for w in self._all_inputs():
            self._connect_change(w)

        return scroll

    def _build_page_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._cb_paper = QComboBox()
        self._cb_paper.addItems(list(PAPER_SIZES.keys()) + ['Custom'])
        self._cb_paper.setCurrentText(self._params.paper_size)
        f.addRow("Paper:", self._cb_paper)

        self._sb_dpi = QSpinBox()
        self._sb_dpi.setRange(72, 600)
        self._sb_dpi.setValue(self._params.dpi)
        self._sb_dpi.setSuffix(" dpi")
        f.addRow("Resolution:", self._sb_dpi)

        self._sb_margin = QDoubleSpinBox()
        self._sb_margin.setRange(0.0, 2.0)
        self._sb_margin.setSingleStep(0.125)
        self._sb_margin.setValue(self._params.margin_in)
        self._sb_margin.setSuffix(" in")
        f.addRow("Margin:", self._sb_margin)

        self._sb_ve = QDoubleSpinBox()
        self._sb_ve.setRange(0.0, 50.0)
        self._sb_ve.setSingleStep(0.5)
        self._sb_ve.setValue(0.0)
        self._sb_ve.setSpecialValueText("Use section VE")
        f.addRow("Vert. exaggeration:", self._sb_ve)

        return w

    def _build_palette_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._cb_palette = QComboBox()
        self._cb_palette.addItems(list(PALETTES.keys()))
        self._cb_palette.setCurrentText(self._params.color_palette)
        f.addRow("Palette preset:", self._cb_palette)

        self._sb_horizon_w = self._make_weight_spin(self._params.horizon_line_weight)
        f.addRow("Horizon weight:", self._sb_horizon_w)

        self._sb_fault_w = self._make_weight_spin(self._params.fault_line_weight)
        f.addRow("Fault weight:", self._sb_fault_w)

        self._sb_poly_outline = self._make_weight_spin(self._params.polygon_outline_weight)
        f.addRow("Polygon outline:", self._sb_poly_outline)

        self._sb_poly_alpha = QDoubleSpinBox()
        self._sb_poly_alpha.setRange(0.0, 1.0)
        self._sb_poly_alpha.setSingleStep(0.02)
        self._sb_poly_alpha.setDecimals(2)
        self._sb_poly_alpha.setValue(self._params.polygon_fill_opacity)
        f.addRow("Polygon fill:", self._sb_poly_alpha)

        return w

    def _build_type_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._cb_font = QComboBox()
        self._cb_font.addItems([
            'Georgia', 'Times New Roman', 'DejaVu Serif',
            'Helvetica', 'Arial', 'DejaVu Sans',
        ])
        self._cb_font.setCurrentText(self._params.font_family)
        f.addRow("Font:", self._cb_font)

        self._sb_label_pt = QSpinBox()
        self._sb_label_pt.setRange(5, 24)
        self._sb_label_pt.setValue(self._params.label_size_pt)
        self._sb_label_pt.setSuffix(" pt")
        f.addRow("Label size:", self._sb_label_pt)

        self._sb_title_pt = QSpinBox()
        self._sb_title_pt.setRange(8, 36)
        self._sb_title_pt.setValue(self._params.title_size_pt)
        self._sb_title_pt.setSuffix(" pt")
        f.addRow("Title size:", self._sb_title_pt)

        self._sb_annot_pt = QSpinBox()
        self._sb_annot_pt.setRange(4, 16)
        self._sb_annot_pt.setValue(self._params.annotation_size_pt)
        self._sb_annot_pt.setSuffix(" pt")
        f.addRow("Annotation size:", self._sb_annot_pt)

        return w

    def _build_content_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)

        self._chk_strat    = QCheckBox("Show stratigraphic column")
        self._chk_strat.setChecked(self._params.show_strat_column)
        f.addRow(self._chk_strat)

        self._chk_grid     = QCheckBox("Show grid")
        self._chk_grid.setChecked(self._params.show_grid)
        f.addRow(self._chk_grid)

        self._chk_sealevel = QCheckBox("Show sea level")
        self._chk_sealevel.setChecked(self._params.show_sea_level)
        f.addRow(self._chk_sealevel)

        self._chk_axlabels = QCheckBox("Show axis labels")
        self._chk_axlabels.setChecked(self._params.show_axis_labels)
        f.addRow(self._chk_axlabels)

        self._chk_scale    = QCheckBox("Show scale bar")
        self._chk_scale.setChecked(self._params.show_scale_bar)
        f.addRow(self._chk_scale)

        self._cb_seismic = QComboBox()
        self._cb_seismic.addItems(['Omit', 'Grayscale', 'Faded grayscale'])
        f.addRow("Seismic:", self._cb_seismic)

        return w

    def _build_titleblock_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._chk_title_block = QCheckBox("Show title block")
        self._chk_title_block.setChecked(self._params.show_title_block)
        f.addRow(self._chk_title_block)

        self._le_title = QLineEdit(self._section.name)
        f.addRow("Title:", self._le_title)

        self._le_subtitle = QLineEdit()
        f.addRow("Subtitle:", self._le_subtitle)

        self._le_author = QLineEdit()
        f.addRow("Author:", self._le_author)

        self._le_date = QLineEdit()
        f.addRow("Date:", self._le_date)

        return w

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_weight_spin(value: float) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(0.1, 5.0)
        sb.setSingleStep(0.1)
        sb.setDecimals(1)
        sb.setValue(value)
        return sb

    def _all_inputs(self) -> list:
        return [
            self._cb_paper, self._sb_dpi, self._sb_margin, self._sb_ve,
            self._cb_palette, self._sb_horizon_w, self._sb_fault_w,
            self._sb_poly_outline, self._sb_poly_alpha,
            self._cb_font, self._sb_label_pt, self._sb_title_pt, self._sb_annot_pt,
            self._chk_strat, self._chk_grid, self._chk_sealevel,
            self._chk_axlabels, self._chk_scale, self._cb_seismic,
            self._chk_title_block, self._le_title, self._le_subtitle,
            self._le_author, self._le_date,
        ]

    def _connect_change(self, w) -> None:
        if isinstance(w, QComboBox):
            w.currentIndexChanged.connect(self._schedule_preview)
        elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
            w.valueChanged.connect(self._schedule_preview)
        elif isinstance(w, QCheckBox):
            w.toggled.connect(self._schedule_preview)
        elif isinstance(w, QLineEdit):
            w.textChanged.connect(self._schedule_preview)

    def _schedule_preview(self, *_) -> None:
        self._timer.start()

    def _sync_params_from_ui(self) -> None:
        p = self._params
        p.paper_size           = self._cb_paper.currentText()
        p.dpi                  = self._sb_dpi.value()
        p.margin_in            = self._sb_margin.value()
        p.vertical_exaggeration = self._sb_ve.value()
        p.color_palette        = self._cb_palette.currentText()
        p.horizon_line_weight  = self._sb_horizon_w.value()
        p.fault_line_weight    = self._sb_fault_w.value()
        p.polygon_outline_weight = self._sb_poly_outline.value()
        p.polygon_fill_opacity = self._sb_poly_alpha.value()
        p.font_family          = self._cb_font.currentText()
        p.label_size_pt        = self._sb_label_pt.value()
        p.title_size_pt        = self._sb_title_pt.value()
        p.annotation_size_pt   = self._sb_annot_pt.value()
        p.show_strat_column    = self._chk_strat.isChecked()
        p.show_grid            = self._chk_grid.isChecked()
        p.show_sea_level       = self._chk_sealevel.isChecked()
        p.show_axis_labels     = self._chk_axlabels.isChecked()
        p.show_scale_bar       = self._chk_scale.isChecked()
        seismic_map = {'Omit': 'omit', 'Grayscale': 'grayscale', 'Faded grayscale': 'faded'}
        p.seismic_inclusion    = seismic_map.get(self._cb_seismic.currentText(), 'omit')
        p.show_title_block     = self._chk_title_block.isChecked()
        p.title                = self._le_title.text()
        p.subtitle             = self._le_subtitle.text()
        p.author               = self._le_author.text()
        p.date_text            = self._le_date.text()

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _refresh_preview(self) -> None:
        self._sync_params_from_ui()
        try:
            from section_tool.export.print_renderer import render_section_to_pixmap
            pm = render_section_to_pixmap(
                self._state, self._section, self._params,
                target_width_px=self._preview_label.width() or 900,
            )
            self._preview_label.setPixmap(
                pm.scaled(self._preview_label.size(),
                          Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)
            )
        except Exception as exc:
            self._preview_label.setText(f"Preview error:\n{exc}")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._schedule_preview()

    # ------------------------------------------------------------------
    # Save handlers
    # ------------------------------------------------------------------

    def _save_png(self) -> None:
        self._sync_params_from_ui()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PNG", self._section.name + ".png", "PNG (*.png)")
        if not path:
            return
        self._params.output_format = 'png'
        self._do_save(path)

    def _save_pdf(self) -> None:
        self._sync_params_from_ui()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF", self._section.name + ".pdf", "PDF (*.pdf)")
        if not path:
            return
        self._params.output_format = 'pdf'
        self._do_save(path)

    def _save_svg(self) -> None:
        self._sync_params_from_ui()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save SVG", self._section.name + ".svg", "SVG (*.svg)")
        if not path:
            return
        self._params.output_format = 'svg'
        self._do_save(path)

    def _do_save(self, path: str) -> None:
        from PySide6.QtWidgets import QMessageBox
        from section_tool.export.print_renderer import render_section_to_file
        try:
            render_section_to_file(self._state, self._section, self._params, path)
            QMessageBox.information(self, "Export", f"Saved:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))
