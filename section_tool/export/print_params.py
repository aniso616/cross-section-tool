from __future__ import annotations
from dataclasses import dataclass


@dataclass
class PrintExportParams:
    """All parameters that control how a section renders for print/export."""

    # --- Page / canvas ---
    paper_size: str = 'A3 landscape'
    custom_width_in: float = 17.0
    custom_height_in: float = 11.0
    dpi: int = 300
    margin_in: float = 0.5

    # --- Palette ---
    color_palette: str = 'Ink (muted)'
    background: str = '#FFFFFF'

    # --- Line weights ---
    horizon_line_weight: float = 0.8
    fault_line_weight: float = 1.0
    polygon_outline_weight: float = 0.4
    polygon_fill_opacity: float = 0.12
    section_line_weight: float = 0.6

    # --- Typography ---
    font_family: str = 'Georgia'
    label_size_pt: int = 9
    title_size_pt: int = 14
    annotation_size_pt: int = 7
    label_color: str = '#222222'

    # --- Content toggles ---
    show_strat_column: bool = True
    show_grid: bool = False
    show_sea_level: bool = True
    show_axis_labels: bool = True
    show_scale_bar: bool = True
    show_title_block: bool = True
    title: str = ''
    subtitle: str = ''
    author: str = ''
    date_text: str = ''

    # --- Seismic ---
    seismic_inclusion: str = 'omit'

    # --- VE (0.0 = use section's working VE) ---
    vertical_exaggeration: float = 0.0

    # --- Output format ---
    output_format: str = 'pdf'
