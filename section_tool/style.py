# Window
BG_APP          = "#111113"
BG_CANVAS       = "#141418"

# Matplotlib canvas dark theme
CANVAS_BG       = "#1A1A1E"     # axes face
CANVAS_GRID     = "#2E2E38"     # grid lines
CANVAS_TEXT     = "#B8B8C0"     # axis labels, tick labels
CANVAS_TICK     = "#666674"     # tick marks
CANVAS_BORDER   = "#3A3A48"     # spines

# Seismic colormaps available in UI
SEISMIC_COLORMAPS = {
    "Grayscale":       "gray_r",
    "Red-Blue":        "seismic",
    "Red-White-Blue":  "RdBu_r",
    "Variable Density":"RdYlBu_r",
    "Black-White":     "gray",
    "Bone":            "bone",
}

# HUD
HUD_SURFACE     = "rgba(18, 18, 24, 210)"
HUD_BORDER      = "rgba(70, 70, 98, 170)"
HUD_TEXT        = "rgba(185, 185, 185, 165)"
HUD_TEXT_BRIGHT = "rgba(220, 220, 220, 255)"
HUD_HIGHLIGHT   = "rgba(65, 105, 165, 210)"
ACCENT          = "#4a7fc1"

# Active tool pill
TOOL_TEXT       = "rgba(120, 180, 255, 230)"
TOOL_BG         = "rgba(18, 18, 30, 170)"
TOOL_BORDER     = "rgba(80, 130, 200, 140)"

# Uncertainty visual grammar — used by canvas renderers
HORIZON_INTERPRETED = dict(linewidth=1.5, linestyle="-",   alpha=1.00, zorder=4)
HORIZON_MODELED     = dict(linewidth=1.0, linestyle="--",  alpha=0.72, zorder=3)
HORIZON_PROJECTED   = dict(linewidth=0.8, linestyle=":",   alpha=0.50, zorder=2)
FAULT_CONSTRAINED   = dict(linewidth=2.0, linestyle="-",   alpha=1.00, zorder=5)
FAULT_PROJECTED     = dict(linewidth=1.2, linestyle="-.",  alpha=0.62, zorder=4)
