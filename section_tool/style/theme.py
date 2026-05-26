"""Visual theme system for the cross-section tool.

Themes map semantic roles (horizon, fault, snap-indicator, …) to concrete
matplotlib style values.  All views call ``get_theme()`` at render time;
switching themes and calling ``request_render()`` is enough to restyle
the entire application without touching rendering logic.

Usage::

    from section_tool.style import get_theme, set_theme

    theme = get_theme()
    ax.plot(d, z, color=theme.horizon.color, lw=theme.horizon.width)

    set_theme("print")   # switches globally; views re-render on next frame
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class EntityStyle:
    color: str
    width: float
    dash: tuple[float, ...] = ()   # () = solid; (4, 3) etc = dash pattern
    alpha: float = 1.0


@dataclass(frozen=True)
class MarkerStyle:
    shape: Literal["square", "diamond", "x", "circle", "plus", "triangle"]
    size_px: float
    fill: str | None   # None = unfilled (markerfacecolor="none")
    edge: str
    edge_width: float = 1.0


@dataclass(frozen=True)
class Theme:
    name: str
    background: str
    grid: str
    axis_label: str
    axis_tick: str

    # Entity strokes (per-entity user color overrides these defaults;
    # theme controls line weight, dash pattern, alpha, and default color)
    horizon: EntityStyle
    fault: EntityStyle
    polygon_outline: EntityStyle
    polygon_fill_alpha: float   # multiplier applied on top of per-entity alpha
    well_track: EntityStyle
    section_line: EntityStyle
    sea_level: EntityStyle
    reference_line: EntityStyle

    # Markers (geometric — not blobs)
    node: MarkerStyle            # interior node in edit mode
    endpoint: MarkerStyle        # polyline first/last node in edit mode
    intersection: MarkerStyle    # topology crossing indicator
    pick_active: MarkerStyle     # active-pick cursor indicator
    cross_section_ghost: MarkerStyle

    # Snap feedback
    snap_node: MarkerStyle
    snap_endpoint: MarkerStyle
    snap_intersection: MarkerStyle
    snap_midpoint: MarkerStyle

    # Selection
    selection_halo_color: str
    selection_halo_width: float
    selection_halo_alpha: float
    edit_node_handle: MarkerStyle

    # Text
    font_family: str
    label_size: int
    annotation_size: int
    label_color: str
    label_background: str
    label_background_alpha: float


# ---------------------------------------------------------------------------
# Dark theme — on-screen work, dark canvas
# ---------------------------------------------------------------------------

DARK = Theme(
    name="dark",
    background="#0e1014",
    grid="#1c2028",
    axis_label="#c4c8d0",
    axis_tick="#8a8f99",

    horizon=EntityStyle(color="#7AB8FF", width=1.0),
    fault=EntityStyle(color="#FF6464", width=1.2),
    polygon_outline=EntityStyle(color="#888888", width=0.5, alpha=0.6),
    polygon_fill_alpha=0.18,
    well_track=EntityStyle(color="#FFD27A", width=1.2),
    section_line=EntityStyle(color="#5588DD", width=0.8),
    sea_level=EntityStyle(color="#88BBFF", width=0.5, dash=(4.0, 3.0), alpha=0.6),
    reference_line=EntityStyle(color="#888888", width=0.5, dash=(2.0, 4.0), alpha=0.5),

    node=MarkerStyle("square", 4.0, fill=None, edge="#c4c8d0", edge_width=0.8),
    endpoint=MarkerStyle("diamond", 5.5, fill="#0e1014", edge="#e0e4ec", edge_width=1.0),
    intersection=MarkerStyle("x", 6.0, fill=None, edge="#FFAA44", edge_width=1.2),
    pick_active=MarkerStyle("plus", 7.0, fill=None, edge="#44FFCC", edge_width=1.2),
    cross_section_ghost=MarkerStyle("circle", 7.0, fill=None, edge="inherit", edge_width=1.0),

    snap_node=MarkerStyle("square", 8.0, fill=None, edge="#66FF99", edge_width=1.0),
    snap_endpoint=MarkerStyle("diamond", 9.0, fill=None, edge="#66FF99", edge_width=1.0),
    snap_intersection=MarkerStyle("x", 9.0, fill=None, edge="#66FF99", edge_width=1.2),
    snap_midpoint=MarkerStyle("triangle", 7.0, fill=None, edge="#66FF99", edge_width=1.0),

    selection_halo_color="#FFFFFF",
    selection_halo_width=2.5,
    selection_halo_alpha=0.35,
    edit_node_handle=MarkerStyle("square", 6.0, fill="#FFFFFF", edge="#4488FF", edge_width=1.0),

    font_family="Inter, system-ui, sans-serif",
    label_size=9,
    annotation_size=8,
    label_color="#c4c8d0",
    label_background="#0e1014",
    label_background_alpha=0.85,
)


# ---------------------------------------------------------------------------
# Print theme — white background, restrained ink palette for reports
# ---------------------------------------------------------------------------

PRINT = Theme(
    name="print",
    background="#FFFFFF",
    grid="#EEEEEE",
    axis_label="#222222",
    axis_tick="#666666",

    horizon=EntityStyle(color="#1A4F8F", width=0.8),
    fault=EntityStyle(color="#A02020", width=1.0),
    polygon_outline=EntityStyle(color="#444444", width=0.4, alpha=0.7),
    polygon_fill_alpha=0.12,
    well_track=EntityStyle(color="#806020", width=1.0),
    section_line=EntityStyle(color="#1A4F8F", width=0.6),
    sea_level=EntityStyle(color="#1A4F8F", width=0.4, dash=(4.0, 3.0), alpha=0.5),
    reference_line=EntityStyle(color="#888888", width=0.4, dash=(2.0, 4.0), alpha=0.5),

    node=MarkerStyle("square", 3.0, fill=None, edge="#444444", edge_width=0.6),
    endpoint=MarkerStyle("diamond", 4.5, fill="#FFFFFF", edge="#222222", edge_width=0.8),
    intersection=MarkerStyle("x", 5.0, fill=None, edge="#A02020", edge_width=1.0),
    pick_active=MarkerStyle("plus", 6.0, fill=None, edge="#208030", edge_width=1.0),
    cross_section_ghost=MarkerStyle("circle", 6.0, fill=None, edge="inherit", edge_width=0.8),

    snap_node=MarkerStyle("square", 7.0, fill=None, edge="#208030", edge_width=0.8),
    snap_endpoint=MarkerStyle("diamond", 8.0, fill=None, edge="#208030", edge_width=0.8),
    snap_intersection=MarkerStyle("x", 8.0, fill=None, edge="#208030", edge_width=1.0),
    snap_midpoint=MarkerStyle("triangle", 6.0, fill=None, edge="#208030", edge_width=0.8),

    selection_halo_color="#FF8800",
    selection_halo_width=1.5,
    selection_halo_alpha=0.6,
    edit_node_handle=MarkerStyle("square", 5.0, fill="#FFFFFF", edge="#208030", edge_width=0.8),

    font_family="Georgia, Times New Roman, serif",
    label_size=8,
    annotation_size=7,
    label_color="#222222",
    label_background="#FFFFFF",
    label_background_alpha=1.0,
)


# ---------------------------------------------------------------------------
# Registry and global accessor
# ---------------------------------------------------------------------------

THEMES: dict[str, Theme] = {"dark": DARK, "print": PRINT}

_current_theme: Theme = DARK


def get_theme() -> Theme:
    """Return the currently-active theme."""
    return _current_theme


def set_theme(name: str) -> None:
    """Switch to a named theme.  Valid names: ``'dark'``, ``'print'``."""
    global _current_theme
    if name not in THEMES:
        raise ValueError(f"Unknown theme {name!r}; valid: {list(THEMES)}")
    _current_theme = THEMES[name]


# ---------------------------------------------------------------------------
# Per-entity color adaptation
# ---------------------------------------------------------------------------

def _hex_to_hls(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h, l, s


def _hls_to_hex(h: float, l: float, s: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#{:02x}{:02x}{:02x}".format(
        int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
    )


def adapt_entity_color(hex_color: str, theme: Theme | None = None) -> str:
    """Adjust a user-chosen entity color to suit the current (or given) theme.

    In the print theme, saturation and lightness are scaled so user-picked
    fluorescent colors don't blow out on white paper.  In the dark theme the
    color is returned unchanged.
    """
    if theme is None:
        theme = get_theme()
    if theme.name == "print":
        try:
            h, l, s = _hex_to_hls(hex_color)
            s = min(s * 0.70, 1.0)
            l = min(l * 0.85, 0.90)
            return _hls_to_hex(h, l, s)
        except Exception:
            return hex_color
    return hex_color


# ---------------------------------------------------------------------------
# MarkerStyle → matplotlib kwargs
# ---------------------------------------------------------------------------

_SHAPE_TO_MARKER: dict[str, str] = {
    "square":   "s",
    "diamond":  "D",
    "x":        "x",
    "circle":   "o",
    "plus":     "+",
    "triangle": "^",
}


def marker_kwargs(style: MarkerStyle, entity_color: str | None = None) -> dict:
    """Convert a :class:`MarkerStyle` to matplotlib ``plot`` keyword arguments."""
    edge = entity_color if style.edge == "inherit" else style.edge
    return {
        "marker":          _SHAPE_TO_MARKER[style.shape],
        "markersize":      style.size_px,
        "markerfacecolor": "none" if style.fill is None else style.fill,
        "markeredgecolor": edge,
        "markeredgewidth": style.edge_width,
        "linestyle":       "none",
    }
