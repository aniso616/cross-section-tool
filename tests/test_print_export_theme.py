"""Phase 2 scenario 9 — theme/print export end-to-end (test-only).

Existing tests cover PrintExportParams, paper sizes, palette dicts, and a
render-no-crash smoke. This harness adds the end-to-end assertions that were
missing: that the selected theme actually reaches the rendered output, that
content toggles change the figure structure, and that each output format is
written.

NOTE: the print renderer composes seismic via the matplotlib imshow path (not
the on-screen pyqtgraph layer), so these tests validate that seismic is
included in the print figure but cannot validate the pyqtgraph render. Seismic
inclusion was wired into _build_figure in the fix for finding #1.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from matplotlib.colors import to_rgba

from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick
from section_tool.io.project import Project
from section_tool.export.print_params import PrintExportParams
from section_tool.export.print_renderer import (
    PALETTES,
    _build_figure,
    render_section_to_file,
)


class _State:
    def __init__(self, project):
        self.project = project


def _section(name="EW", length=10_000.0):
    return Section([(0.0, 0.0), (length, 0.0)], name=name)


def _state_with_horizon(section_name="EW"):
    proj = Project()
    hp = HorizonPick(np.array([0.0, 5000.0, 10000.0]),
                     np.array([1000.0, 1200.0, 1100.0]),
                     name="TopSand", color="#2ca02c",
                     section_names=[section_name] * 3)
    proj.horizon_picks.append(hp)
    return _State(proj)


def _base_params(**over):
    """Params with all decorative axes off so structure is easy to assert."""
    defaults = dict(paper_size="A4 landscape", dpi=72,
                    show_strat_column=False, show_title_block=False,
                    show_scale_bar=False, show_grid=False, show_sea_level=False)
    defaults.update(over)
    return PrintExportParams(**defaults)


# ---------------------------------------------------------------------------
# Theme actually reaches the rendered output
# ---------------------------------------------------------------------------

class TestThemeApplied:

    @pytest.mark.parametrize("bg", ["#FFFFFF", "#101820", "#FBFAF5"])
    def test_figure_background_matches_params(self, bg):
        fig = _build_figure(_state_with_horizon(), _section(),
                            _base_params(background=bg), dpi=72)
        assert to_rgba(fig.get_facecolor()) == pytest.approx(to_rgba(bg))

    @pytest.mark.parametrize("palette_name", list(PALETTES))
    def test_horizon_line_uses_palette_color(self, palette_name):
        params = _base_params(color_palette=palette_name)
        fig = _build_figure(_state_with_horizon(), _section(), params, dpi=72)
        section_ax = fig.axes[0]
        want = to_rgba(PALETTES[palette_name]["horizon"])
        line_colors = [to_rgba(ln.get_color()) for ln in section_ax.get_lines()]
        assert any(c == pytest.approx(want) for c in line_colors), (
            f"no horizon-coloured line for palette {palette_name!r}")

    def test_distinct_palettes_give_distinct_horizon_colors(self):
        s, sec = _state_with_horizon(), _section()
        c_ink = _build_figure(s, sec, _base_params(color_palette="Ink (muted)"), 72)\
            .axes[0].get_lines()[0].get_color()
        c_usgs = _build_figure(s, sec, _base_params(color_palette="Classic (USGS)"), 72)\
            .axes[0].get_lines()[0].get_color()
        assert to_rgba(c_ink) != to_rgba(c_usgs)


# ---------------------------------------------------------------------------
# Content toggles change the figure structure
# ---------------------------------------------------------------------------

class TestToggleStructure:

    def test_strat_column_adds_an_axes(self):
        s, sec = _state_with_horizon(), _section()
        off = len(_build_figure(s, sec, _base_params(show_strat_column=False), 72).axes)
        on = len(_build_figure(s, sec, _base_params(show_strat_column=True), 72).axes)
        assert on == off + 1

    def test_title_block_adds_an_axes(self):
        s, sec = _state_with_horizon(), _section()
        off = len(_build_figure(s, sec, _base_params(show_title_block=False), 72).axes)
        on = len(_build_figure(s, sec, _base_params(show_title_block=True), 72).axes)
        assert on == off + 1


# ---------------------------------------------------------------------------
# Each output format is written to disk
# ---------------------------------------------------------------------------

class TestOutputFormats:

    @pytest.mark.parametrize("fmt", ["png", "pdf", "svg"])
    def test_render_to_file_writes_format(self, tmp_path, fmt):
        out = str(tmp_path / f"section.{fmt}")
        params = _base_params(output_format=fmt)
        render_section_to_file(_state_with_horizon(), _section(), params, out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 500


# ---------------------------------------------------------------------------
# Seismic inclusion (finding #1 fix) — print figure actually contains seismic
# ---------------------------------------------------------------------------

def _state_with_seismic(with_seismic=True):
    from section_tool.app_state import AppState
    state = AppState()
    sec = _section(name="EW")
    state.add_section(sec)
    state.set_active_section(sec)
    if with_seismic:
        data = np.tile(np.linspace(-1, 1, 50, dtype=np.float32)[:, None], (1, 40))
        meta = {"dist_min": 0.0, "dist_max": 10000.0,
                "samples": list(range(0, 100, 2)), "domain": "twt"}   # 50 samples
        state.set_seismic_for_section("EW", data, meta)
    return state, sec


class TestSeismicInclusion:

    def test_omit_renders_no_image(self):
        state, sec = _state_with_seismic()
        fig = _build_figure(state, sec, _base_params(seismic_inclusion="omit"), 72)
        assert len(fig.axes[0].images) == 0

    def test_grayscale_renders_an_image(self):
        state, sec = _state_with_seismic()
        fig = _build_figure(state, sec, _base_params(seismic_inclusion="grayscale"), 72)
        assert len(fig.axes[0].images) == 1

    def test_faded_uses_low_alpha(self):
        state, sec = _state_with_seismic()
        fig = _build_figure(state, sec, _base_params(seismic_inclusion="faded"), 72)
        img = fig.axes[0].images[0]
        assert img.get_alpha() == pytest.approx(0.35)

    def test_no_extracted_seismic_is_noop(self):
        # section present but no seismic extracted -> grayscale must not crash/add
        state, sec = _state_with_seismic(with_seismic=False)
        fig = _build_figure(state, sec,
                            _base_params(seismic_inclusion="grayscale"), 72)
        assert len(fig.axes[0].images) == 0
