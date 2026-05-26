"""Single source of truth for all HUD and canvas colors, plus the theme system.

Legacy color constants are preserved here for backward compatibility.
New code should use the theme system: ``from section_tool.style import get_theme``.
"""

# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------

BG_CANVAS = "#0e1014"       # matplotlib figure and axes face

# ---------------------------------------------------------------------------
# HUD color family — (R, G, B, A) tuples; unpack with QColor(*value)
# ---------------------------------------------------------------------------

C_RULE  = (90,  100, 110, 160)   # rule lines, tick marks, borders
C_LABEL = (120, 135, 145, 175)   # all labels and readout text
C_DIM   = (80,   90, 100, 140)   # secondary labels, empty states
C_READ  = (160, 185, 205, 220)   # cursor line, active readout values
C_BG    = (14,   16,  20, 195)   # label backgrounds, inset fill

# ---------------------------------------------------------------------------
# Seismic colormaps available in UI
# ---------------------------------------------------------------------------

SEISMIC_COLORMAPS = {
    "Grayscale":        "gray_r",
    "Red-Blue":         "seismic",
    "Red-White-Blue":   "RdBu_r",
    "Variable Density": "RdYlBu_r",
    "Black-White":      "gray",
    "Bone":             "bone",
}

# ---------------------------------------------------------------------------
# Uncertainty visual grammar
# ---------------------------------------------------------------------------

HORIZON_INTERPRETED = dict(linewidth=1.5, linestyle="-",  alpha=1.00, zorder=4)
HORIZON_MODELED     = dict(linewidth=1.0, linestyle="--", alpha=0.72, zorder=3)
HORIZON_PROJECTED   = dict(linewidth=0.8, linestyle=":",  alpha=0.50, zorder=2)
FAULT_CONSTRAINED   = dict(linewidth=2.0, linestyle="-",  alpha=1.00, zorder=5)
FAULT_PROJECTED     = dict(linewidth=1.2, linestyle="-.", alpha=0.62, zorder=4)

# ---------------------------------------------------------------------------
# Theme system
# ---------------------------------------------------------------------------

from section_tool.style.theme import (  # noqa: E402
    Theme, EntityStyle, MarkerStyle,
    DARK, PRINT, THEMES,
    get_theme, set_theme,
    adapt_entity_color, marker_kwargs,
)

__all__ = [
    "BG_CANVAS", "C_RULE", "C_LABEL", "C_DIM", "C_READ", "C_BG",
    "SEISMIC_COLORMAPS",
    "HORIZON_INTERPRETED", "HORIZON_MODELED", "HORIZON_PROJECTED",
    "FAULT_CONSTRAINED", "FAULT_PROJECTED",
    "Theme", "EntityStyle", "MarkerStyle",
    "DARK", "PRINT", "THEMES",
    "get_theme", "set_theme",
    "adapt_entity_color", "marker_kwargs",
]
