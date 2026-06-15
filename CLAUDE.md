## Running the app

Claude Code must **not** launch or babysit the GUI across tool calls. A
`Start-Process` (or any spawned GUI process) started inside one tool call is
reaped when that tool call returns — the child PID dies with the call's
process group, which then reads as a false "the app crashed." There is no way
to keep a window alive between tool calls from here.

So: **the user launches the app in their own terminal**, where the window is
tied to their interactive session and stays up. Claude's job is to make code
changes, run tests, commit, and push — then stop and let the user run it.

Activate the env first: `conda activate cross-section-tool` (env lives at
`C:\Users\prkbr\miniconda3\envs\cross-section-tool`).

## Common confusion: "blank" section view

When a new section is created with no horizons/polygons/wells, or when
the default view is loaded, the section view can appear blank. This is
not a rendering bug — it is a visual alignment issue:

- Sea level draws at y=0 — the **top edge** of the canvas with the default
  ylim (max_depth, 0.0) where depth is inverted. It is a 1px blue line
  at the very top, easy to miss.
- Section endpoints draw at x=0 and x=section_length — the **left and
  right edges** of the canvas, also easy to miss.
- The matplotlib canvas has `WA_TranslucentBackground` and composites
  against the dark Qt parent background, not the pyqtgraph seismic layer
  underneath. So all overlay artists (well tracks, sea level, horizon picks)
  render as light-coloured lines against a dark background — readable with
  seismic loaded, nearly invisible against an empty dark canvas.

To confirm rendering is working:
1. Scroll-wheel zoom **out** — sea level and section endpoints move away
   from the canvas edges and become visible.
2. Press **H** and click in the canvas to add a horizon pick node — if a
   coloured dot appears, rendering is fine.
3. Use `fig.savefig(path, facecolor="black")` to inspect the matplotlib
   figure directly — it will show all overlay artists regardless of the
   Qt compositing.

Verified at commit d5b0cf3 (and every subsequent commit): `fig.savefig`
shows well track, sea level label, section endpoints, depth/distance axes
all correctly rendered on the F3 test project.
