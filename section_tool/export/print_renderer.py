"""Renders a section to PDF/PNG/SVG using a fresh matplotlib figure.

Completely independent of the working section view — no shared state,
no theme switching.  Every call builds a new Figure from the project data.
"""
from __future__ import annotations

import math
from io import BytesIO

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon, Rectangle as MplRect

from section_tool.export.print_params import PrintExportParams

# ---------------------------------------------------------------------------
# Paper sizes (width × height, inches, landscape orientation)
# ---------------------------------------------------------------------------

PAPER_SIZES: dict[str, tuple[float, float]] = {
    'A3 landscape':     (16.54, 11.69),
    'A4 landscape':     (11.69,  8.27),
    'Letter landscape': (11.00,  8.50),
    'Tabloid (11×17)':  (17.00, 11.00),
}

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

PALETTES: dict[str, dict[str, str]] = {
    'Ink (muted)': {
        'horizon':         '#1A4F8F',
        'fault':           '#A02020',
        'polygon_outline': '#444444',
        'section_end':     '#888888',
        'sea_level':       '#1A4F8F',
        'background':      '#FFFFFF',
        'text':            '#222222',
        'grid':            '#EEEEEE',
    },
    'Classic (USGS)': {
        'horizon':         '#5C4033',
        'fault':           '#8B0000',
        'polygon_outline': '#3D2817',
        'section_end':     '#9E7B50',
        'sea_level':       '#003366',
        'background':      '#FBFAF5',
        'text':            '#1A1208',
        'grid':            '#E8E0D0',
    },
    'High contrast': {
        'horizon':         '#000000',
        'fault':           '#C8102E',
        'polygon_outline': '#000000',
        'section_end':     '#000000',
        'sea_level':       '#000000',
        'background':      '#FFFFFF',
        'text':            '#000000',
        'grid':            '#CCCCCC',
    },
    'Monochrome': {
        'horizon':         '#202020',
        'fault':           '#404040',
        'polygon_outline': '#606060',
        'section_end':     '#606060',
        'sea_level':       '#404040',
        'background':      '#FFFFFF',
        'text':            '#202020',
        'grid':            '#DDDDDD',
    },
}

_DEFAULT_PALETTE = 'Ink (muted)'

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_section_to_pixmap(state, section, params: PrintExportParams,
                              target_width_px: int = 900):
    """Render to a QPixmap for dialog preview (low-DPI, in memory)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PySide6.QtGui import QImage, QPixmap

    w_in, h_in = _paper_size_in(params)
    preview_dpi = max(72, int(target_width_px / w_in))
    fig = _build_figure(state, section, params, dpi=preview_dpi)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    w, h = canvas.get_width_height()
    image = QImage(buf, w, h, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(image.copy())


def render_section_to_file(state, section, params: PrintExportParams,
                           output_path: str) -> None:
    """Render section to PDF/PNG/SVG at full DPI."""
    fig = _build_figure(state, section, params, dpi=params.dpi)
    fmt = params.output_format.lower()
    if fmt == 'pdf':
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(output_path) as pdf:
            pdf.savefig(fig, facecolor=params.background, bbox_inches='tight')
    elif fmt == 'svg':
        fig.savefig(output_path, format='svg',
                    facecolor=params.background, bbox_inches='tight')
    else:
        fig.savefig(output_path, format='png', dpi=params.dpi,
                    facecolor=params.background, bbox_inches='tight')


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------


def _build_figure(state, section, params: PrintExportParams,
                  dpi: int | None = None) -> Figure:
    """Construct a fresh matplotlib Figure with print styles."""
    w_in, h_in = _paper_size_in(params)
    use_dpi = dpi if dpi is not None else params.dpi
    fig = Figure(figsize=(w_in, h_in), dpi=use_dpi, facecolor=params.background)

    palette = _get_palette(params)

    # --- Layout fractions ---
    mx = params.margin_in / w_in    # horizontal margin as fraction of width
    my = params.margin_in / h_in    # vertical margin as fraction of height

    title_h  = 0.08 if params.show_title_block else 0.0
    strat_w  = 0.06 if params.show_strat_column else 0.0
    scale_h  = 0.04 if params.show_scale_bar    else 0.0

    left   = mx + strat_w
    bottom = my + scale_h
    width  = 1.0 - left - mx
    height = 1.0 - bottom - my - title_h

    section_ax = fig.add_axes([left, bottom, width, height])

    # --- View limits ---
    total_length = section.total_length()
    ve = params.vertical_exaggeration
    if ve <= 0.0:
        ve = getattr(section, 'vertical_exaggeration', 1.0)
    max_depth = _compute_depth_range(state, section)
    y_range = max_depth / max(ve, 0.01)

    section_ax.set_xlim(0.0, total_length)
    section_ax.set_ylim(y_range, 0.0)   # depth increases downward (inverted y)

    _style_section_axes(section_ax, params, palette)

    # --- Content ---
    if params.seismic_inclusion != 'omit':
        _render_seismic(section_ax, state, section, params)

    if params.show_grid:
        _render_grid(section_ax, palette)

    _render_polygons(section_ax, state, section, params, palette)
    _render_horizons(section_ax, state, section, params, palette)
    _render_faults(section_ax, state, section, params, palette)
    _render_section_ends(section_ax, total_length, y_range, params, palette)

    if params.show_sea_level:
        _render_sea_level(section_ax, total_length, y_range, params, palette)

    if params.show_scale_bar:
        _render_scale_bar(section_ax, total_length, y_range, params, palette)

    # --- Title block ---
    if params.show_title_block:
        title_ax = fig.add_axes([mx, 1.0 - my - title_h + 0.005,
                                  1.0 - 2 * mx, title_h - 0.01])
        _render_title_block(title_ax, params, palette)

    # --- Strat column ---
    if params.show_strat_column:
        strat_ax = fig.add_axes([mx, bottom, strat_w - 0.01, height])
        _render_strat_column(strat_ax, state, section, params, palette, y_range)

    return fig


# ---------------------------------------------------------------------------
# Axes styling
# ---------------------------------------------------------------------------


def _style_section_axes(ax, params: PrintExportParams, palette: dict) -> None:
    ax.set_facecolor(palette['background'])
    for spine in ax.spines.values():
        spine.set_color(palette['text'])
        spine.set_linewidth(0.5)
    ax.tick_params(colors=palette['text'], labelsize=params.label_size_pt - 1,
                   width=0.5, length=3)
    try:
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily(params.font_family)
    except Exception:
        pass
    if params.show_axis_labels:
        ax.set_xlabel('Distance (m)', fontsize=params.label_size_pt,
                      color=palette['text'])
        ax.set_ylabel('Depth (m)', fontsize=params.label_size_pt,
                      color=palette['text'])


# ---------------------------------------------------------------------------
# Content renderers
# ---------------------------------------------------------------------------


def _render_seismic(ax, state, section, params: PrintExportParams) -> None:
    """Render extracted seismic for *section* as a grayscale imshow backdrop.

    Sources the in-memory extracted amplitude array (same data the on-screen
    view shows) and positions it with the section's seismic-display datum logic
    so the print matches the screen.  No-op when no seismic has been extracted
    for the section (e.g. a freshly reopened project — see the extracted-seismic
    restore limitation).  'grayscale' draws opaque; 'faded' draws at low alpha.
    """
    try:
        ex_data, ex_meta = state.get_seismic_for_section(section.name)
    except Exception:
        return
    if ex_data is None or ex_meta is None or getattr(ex_data, "size", 0) == 0:
        return
    samples = np.asarray(ex_meta.get("samples", []), dtype=float)
    if samples.size < 2:
        return

    domain  = ex_meta.get("domain", "twt")
    sds     = getattr(section, "seismic_display", None)
    stretch = getattr(sds, "stretch_mode", "linear") if sds else "linear"
    vel     = getattr(sds, "constant_velocity", 2000.0) if sds else 2000.0
    if stretch == "linear" and domain == "twt":
        scale = vel / 2000.0
        y_top, y_bot = float(samples[0]) * scale, float(samples[-1]) * scale
    else:
        y_top, y_bot = float(samples[0]), float(samples[-1])

    dist0 = float(ex_meta.get("dist_min", 0.0))
    dist1 = float(ex_meta.get("dist_max", section.total_length()))
    vmax  = float(np.percentile(np.abs(ex_data), 99.0) or 1.0)
    alpha = 0.35 if params.seismic_inclusion == "faded" else 1.0

    ax.imshow(
        ex_data, aspect="auto",
        extent=[dist0, dist1, y_bot, y_top],
        origin="upper", cmap="gray_r",
        vmin=-vmax, vmax=vmax,
        interpolation="bilinear", alpha=alpha, zorder=-1,
    )


def _render_grid(ax, palette: dict) -> None:
    ax.grid(True, color=palette['grid'], linewidth=0.3, linestyle='-', zorder=0)


def _render_section_ends(ax, total_length: float, y_range: float,
                          params: PrintExportParams, palette: dict) -> None:
    kw = dict(color=palette['section_end'],
              linewidth=params.section_line_weight * 0.7,
              linestyle='--', alpha=0.6, zorder=2)
    ax.plot([0,            0],            [0, y_range], **kw)
    ax.plot([total_length, total_length], [0, y_range], **kw)


def _render_polygons(ax, state, section, params: PrintExportParams,
                     palette: dict) -> None:
    for poly in state.project.polygons:
        if not getattr(poly, 'visible', True):
            continue
        poly_sec = getattr(poly, 'section_name', '')
        if poly_sec and poly_sec != section.name:
            continue
        verts = poly.vertices
        if len(verts) < 3:
            continue
        patch = MplPolygon(
            verts, closed=True,
            facecolor=poly.fill_color,
            alpha=params.polygon_fill_opacity,
            edgecolor=palette['polygon_outline'],
            linewidth=params.polygon_outline_weight,
            zorder=4,
        )
        ax.add_patch(patch)


def _render_horizons(ax, state, section, params: PrintExportParams,
                     palette: dict) -> None:
    for hp in state.project.horizon_picks:
        if not getattr(hp, 'visible', True):
            continue
        dists, depths = hp.picks_for_section(section.name)
        if len(dists) < 2:
            continue
        valid = ~np.isnan(depths)
        if valid.sum() < 2:
            continue
        d_v, z_v = dists[valid], depths[valid]
        ax.plot(d_v, z_v,
                color=palette['horizon'],
                linewidth=params.horizon_line_weight,
                solid_capstyle='round',
                zorder=5)
        # Horizon name label at midpoint
        if hp.name:
            mid = len(d_v) // 2
            ax.text(d_v[mid], z_v[mid], f' {hp.name}',
                    fontsize=params.annotation_size_pt,
                    fontfamily=params.font_family,
                    color=palette['text'],
                    va='bottom', zorder=6)


def _render_faults(ax, state, section, params: PrintExportParams,
                   palette: dict) -> None:
    for fp in state.project.fault_picks:
        if not getattr(fp, 'visible', True):
            continue
        dists, depths = fp.picks_for_section(section.name)
        if len(dists) < 2:
            continue
        valid = ~np.isnan(depths)
        if valid.sum() < 2:
            continue
        ax.plot(dists[valid], depths[valid],
                color=palette['fault'],
                linewidth=params.fault_line_weight,
                linestyle='--',
                solid_capstyle='round',
                zorder=6)
        if fp.name:
            mid = len(dists[valid]) // 2
            ax.text(dists[valid][mid], depths[valid][mid], f' {fp.name}',
                    fontsize=params.annotation_size_pt,
                    fontfamily=params.font_family,
                    color=palette['fault'],
                    va='bottom', zorder=7)


def _render_sea_level(ax, total_length: float, y_range: float,
                      params: PrintExportParams, palette: dict) -> None:
    ax.axhline(0.0, color=palette['sea_level'], linewidth=0.5,
               linestyle=(0, (5, 4)), zorder=2, alpha=0.7)
    ax.text(total_length * 0.97, 0.0, 'Sea Level',
            fontsize=params.annotation_size_pt,
            fontfamily=params.font_family,
            color=palette['sea_level'],
            ha='right', va='bottom', zorder=2, alpha=0.7)


def _render_scale_bar(ax, total_length: float, y_range: float,
                      params: PrintExportParams, palette: dict) -> None:
    bar_m = _nice_scale_length(total_length)
    xlim = ax.get_xlim()
    # y_range is the deepest value; the axis is inverted so y_range is at the bottom.
    # Place scale bar 7% above the bottom (shallower = smaller y value).
    y_bar = y_range - y_range * 0.07
    tick_h = y_range * 0.009
    x0 = xlim[0] + (xlim[1] - xlim[0]) * 0.025
    x1 = x0 + bar_m

    ax.plot([x0, x1], [y_bar, y_bar],
            color=palette['text'], lw=1.0, zorder=10, clip_on=False)
    ax.plot([x0, x0], [y_bar - tick_h, y_bar + tick_h],
            color=palette['text'], lw=0.7, zorder=10, clip_on=False)
    ax.plot([x1, x1], [y_bar - tick_h, y_bar + tick_h],
            color=palette['text'], lw=0.7, zorder=10, clip_on=False)

    label = f'{bar_m / 1000:.0f} km' if bar_m >= 1000 else f'{bar_m:.0f} m'
    # Text appears below the bar: in inverted-y space, "below" = larger y = deeper
    ax.text((x0 + x1) / 2, y_bar + tick_h + y_range * 0.008, label,
            ha='center', va='top',
            fontsize=params.annotation_size_pt,
            fontfamily=params.font_family,
            color=palette['text'], zorder=10, clip_on=False)


def _render_title_block(ax, params: PrintExportParams, palette: dict) -> None:
    ax.axis('off')
    ax.set_facecolor('none')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # Separator line at the bottom of the title block
    ax.plot([0, 1], [0.05, 0.05], color=palette['text'],
            linewidth=0.5, transform=ax.transAxes, clip_on=False)

    title = params.title or '(untitled)'
    ax.text(0.5, 0.75, title,
            ha='center', va='center',
            fontsize=params.title_size_pt,
            fontfamily=params.font_family,
            fontweight='bold',
            color=palette['text'],
            transform=ax.transAxes)

    if params.subtitle:
        ax.text(0.5, 0.35, params.subtitle,
                ha='center', va='center',
                fontsize=params.label_size_pt + 1,
                fontfamily=params.font_family,
                fontstyle='italic',
                color=palette['text'],
                transform=ax.transAxes)

    credit_parts = []
    if params.author:
        credit_parts.append(params.author)
    if params.date_text:
        credit_parts.append(params.date_text)
    if credit_parts:
        ax.text(0.99, 0.12, ' — '.join(credit_parts),
                ha='right', va='bottom',
                fontsize=params.annotation_size_pt,
                fontfamily=params.font_family,
                color=palette['text'],
                transform=ax.transAxes)


def _render_strat_column(ax, state, section, params: PrintExportParams,
                          palette: dict, y_range: float) -> None:
    ax.set_facecolor(palette['background'])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(y_range, 0.0)

    polygons = [p for p in state.project.polygons
                if not getattr(p, 'section_name', '') or p.section_name == section.name]
    if not polygons:
        return

    fm_depths: dict[str, tuple[float, float]] = {}
    fm_color:  dict[str, str] = {}
    for poly in polygons:
        if not getattr(poly, 'visible', True):
            continue
        fm_name = getattr(poly, 'formation', '') or poly.name or ''
        base = fm_name.rsplit(' (', 1)[0]
        verts = poly.vertices
        if len(verts) == 0:
            continue
        depths = verts[:, 1]
        d_top, d_bot = float(depths.min()), float(depths.max())
        if base in fm_depths:
            old_top, old_bot = fm_depths[base]
            fm_depths[base] = (min(old_top, d_top), max(old_bot, d_bot))
        else:
            fm_depths[base] = (d_top, d_bot)
            fm_color[base] = poly.fill_color

    for fm_name, (d_top, d_bot) in sorted(fm_depths.items(), key=lambda kv: kv[1][0]):
        rect = MplRect(
            (0.05, d_top), 0.9, d_bot - d_top,
            facecolor=fm_color.get(fm_name, '#777777'),
            alpha=0.85,
            edgecolor=palette['polygon_outline'],
            linewidth=0.3,
            zorder=5, clip_on=True,
        )
        ax.add_patch(rect)
        band_h = abs(d_bot - d_top)
        if band_h > y_range * 0.06 and fm_name:
            ax.text(0.5, (d_top + d_bot) / 2, fm_name,
                    ha='center', va='center',
                    fontsize=params.annotation_size_pt,
                    fontfamily=params.font_family,
                    color=palette['text'],
                    rotation=90, clip_on=True, zorder=6)

    # Column label
    ax.text(0.5, -y_range * 0.03, 'Stratigraphy',
            ha='center', va='bottom',
            fontsize=params.annotation_size_pt,
            fontfamily=params.font_family,
            color=palette['text'], clip_on=False)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _paper_size_in(params: PrintExportParams) -> tuple[float, float]:
    if params.paper_size == 'Custom':
        return params.custom_width_in, params.custom_height_in
    return PAPER_SIZES.get(params.paper_size, (17.0, 11.0))


def _get_palette(params: PrintExportParams) -> dict[str, str]:
    pal = dict(PALETTES.get(params.color_palette, PALETTES[_DEFAULT_PALETTE]))
    pal['background'] = params.background
    pal['text'] = params.label_color
    return pal


def _compute_depth_range(state, section) -> float:
    candidates = [5000.0]
    for hp in state.project.horizon_picks:
        _, depths = hp.picks_for_section(section.name)
        v = depths[~np.isnan(depths)]
        if len(v):
            candidates.append(float(v.max()))
    for fp in state.project.fault_picks:
        _, depths = fp.picks_for_section(section.name)
        v = depths[~np.isnan(depths)]
        if len(v):
            candidates.append(float(v.max()))
    for poly in state.project.polygons:
        if not getattr(poly, 'section_name', '') or poly.section_name == section.name:
            verts = poly.vertices
            if len(verts) >= 3:
                candidates.append(float(verts[:, 1].max()))
    return max(candidates)


def _nice_scale_length(total_m: float) -> float:
    target = total_m / 6.0
    magnitude = 10 ** math.floor(math.log10(max(target, 1.0)))
    for factor in (1, 2, 5, 10):
        candidate = factor * magnitude
        if candidate >= target * 0.5:
            return candidate
    return magnitude * 10
