"""Tests for section_tool.export.print_params and section_tool.export.print_renderer.

All tests are headless (no display required). Qt imports are guarded with
pytest.importorskip where necessary so the suite can run without a display.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers — minimal fake state / section objects for smoke tests
# ---------------------------------------------------------------------------


def _make_section(name="EW", length=10_000.0):
    from section_tool.core.section import Section
    return Section([(0.0, 0.0), (length, 0.0)], name=name)


def _make_project_with_picks(section_name="EW"):
    """Return a minimal Project that has one horizon pick on *section_name*."""
    from section_tool.io.project import Project
    from section_tool.core.surfaces import HorizonPick
    from section_tool.core.polygons import SectionPolygon

    proj = Project()
    dists = np.array([0.0, 5000.0, 10000.0])
    depths = np.array([1000.0, 1200.0, 1100.0])
    hp = HorizonPick(
        dists, depths,
        name="TopSand",
        color="#2ca02c",
        section_names=[section_name] * 3,
    )
    proj.horizon_picks.append(hp)
    return proj


class _MinimalState:
    """Minimal duck-type for state accepted by print_renderer._build_figure."""

    def __init__(self, project):
        self.project = project


# ---------------------------------------------------------------------------
# 1. PrintExportParams defaults
# ---------------------------------------------------------------------------

class TestPrintExportParamsDefaults:

    def test_instantiate_no_args(self):
        from section_tool.export.print_params import PrintExportParams
        p = PrintExportParams()
        assert p is not None

    def test_paper_size_is_nonempty_string(self):
        from section_tool.export.print_params import PrintExportParams
        p = PrintExportParams()
        assert isinstance(p.paper_size, str)
        assert len(p.paper_size) > 0

    def test_dpi_at_least_72(self):
        from section_tool.export.print_params import PrintExportParams
        p = PrintExportParams()
        assert p.dpi >= 72

    def test_polygon_fill_opacity_in_range(self):
        from section_tool.export.print_params import PrintExportParams
        p = PrintExportParams()
        assert hasattr(p, "polygon_fill_opacity")
        assert 0.0 < p.polygon_fill_opacity <= 1.0

    def test_horizon_line_weight_positive(self):
        from section_tool.export.print_params import PrintExportParams
        p = PrintExportParams()
        assert hasattr(p, "horizon_line_weight")
        assert p.horizon_line_weight > 0

    def test_all_string_fields_are_strings(self):
        from section_tool.export.print_params import PrintExportParams
        p = PrintExportParams()
        for field in ("paper_size", "color_palette", "background",
                      "font_family", "label_color", "output_format"):
            assert isinstance(getattr(p, field), str), f"{field} should be a str"

    def test_all_numeric_fields_are_numeric(self):
        from section_tool.export.print_params import PrintExportParams
        p = PrintExportParams()
        for field in ("dpi", "margin_in", "horizon_line_weight", "fault_line_weight",
                      "polygon_outline_weight", "polygon_fill_opacity",
                      "label_size_pt", "title_size_pt", "annotation_size_pt",
                      "vertical_exaggeration"):
            val = getattr(p, field)
            assert isinstance(val, (int, float)), f"{field} should be numeric"


# ---------------------------------------------------------------------------
# 2. PAPER_SIZES — known sizes with reasonable dimensions
# ---------------------------------------------------------------------------

class TestPaperDimensions:

    def test_paper_sizes_dict_exists(self):
        from section_tool.export.print_renderer import PAPER_SIZES
        assert isinstance(PAPER_SIZES, dict)
        assert len(PAPER_SIZES) > 0

    def test_a3_landscape_dimensions(self):
        from section_tool.export.print_renderer import PAPER_SIZES
        assert "A3 landscape" in PAPER_SIZES
        w, h = PAPER_SIZES["A3 landscape"]
        # landscape: width > height
        assert w > h
        # sanity: both in reasonable inch range
        assert 5.0 < w < 30.0
        assert 5.0 < h < 25.0

    def test_a4_landscape_dimensions(self):
        from section_tool.export.print_renderer import PAPER_SIZES
        assert "A4 landscape" in PAPER_SIZES
        w, h = PAPER_SIZES["A4 landscape"]
        assert w > h

    def test_letter_landscape_dimensions(self):
        from section_tool.export.print_renderer import PAPER_SIZES
        assert "Letter landscape" in PAPER_SIZES
        w, h = PAPER_SIZES["Letter landscape"]
        assert w > h

    def test_paper_size_in_function_known(self):
        """_paper_size_in should return valid dimensions for known size strings."""
        from section_tool.export.print_renderer import _paper_size_in, PAPER_SIZES
        from section_tool.export.print_params import PrintExportParams
        for size_name in PAPER_SIZES:
            p = PrintExportParams(paper_size=size_name)
            w, h = _paper_size_in(p)
            assert w > 0 and h > 0, f"paper size '{size_name}' returned non-positive dims"

    def test_paper_size_in_custom(self):
        from section_tool.export.print_renderer import _paper_size_in
        from section_tool.export.print_params import PrintExportParams
        p = PrintExportParams(paper_size="Custom", custom_width_in=20.0, custom_height_in=14.0)
        w, h = _paper_size_in(p)
        assert w == pytest.approx(20.0)
        assert h == pytest.approx(14.0)


# ---------------------------------------------------------------------------
# 3. PALETTES exist and have consistent keys
# ---------------------------------------------------------------------------

class TestPalettesExist:

    def test_palettes_dict_nonempty(self):
        from section_tool.export.print_renderer import PALETTES
        assert isinstance(PALETTES, dict)
        assert len(PALETTES) > 0

    def test_each_palette_is_dict(self):
        from section_tool.export.print_renderer import PALETTES
        for name, pal in PALETTES.items():
            assert isinstance(pal, dict), f"Palette '{name}' should be a dict"
            assert len(pal) > 0, f"Palette '{name}' should not be empty"

    def test_all_palettes_have_same_keys(self):
        """All palettes should define the same colour roles."""
        from section_tool.export.print_renderer import PALETTES
        palette_list = list(PALETTES.values())
        if len(palette_list) < 2:
            pytest.skip("Only one palette defined — can't compare keys")
        reference_keys = set(palette_list[0].keys())
        for name, pal in PALETTES.items():
            assert set(pal.keys()) == reference_keys, (
                f"Palette '{name}' has different keys than the first palette"
            )

    def test_palette_colour_values_are_strings(self):
        from section_tool.export.print_renderer import PALETTES
        for name, pal in PALETTES.items():
            for key, val in pal.items():
                assert isinstance(val, str), (
                    f"Palette '{name}' key '{key}' should be a colour string, got {type(val)}"
                )

    def test_default_palette_name_in_palettes(self):
        from section_tool.export.print_renderer import PALETTES, _DEFAULT_PALETTE
        assert _DEFAULT_PALETTE in PALETTES

    def test_default_params_palette_in_palettes(self):
        from section_tool.export.print_params import PrintExportParams
        from section_tool.export.print_renderer import PALETTES
        p = PrintExportParams()
        assert p.color_palette in PALETTES


# ---------------------------------------------------------------------------
# 4. PrintExportParams round-trip
#    (No to_dict/from_dict in this class; verified by reading source)
# ---------------------------------------------------------------------------

class TestPrintExportParamsRoundtrip:

    def test_no_serialization_methods(self):
        """PrintExportParams is a plain dataclass with no to_dict/from_dict.

        Manual round-trip via dataclasses.asdict instead.
        """
        import dataclasses
        from section_tool.export.print_params import PrintExportParams

        p1 = PrintExportParams(title="My Section", dpi=150, show_grid=True)
        d = dataclasses.asdict(p1)
        p2 = PrintExportParams(**d)

        assert p2.title == "My Section"
        assert p2.dpi == 150
        assert p2.show_grid is True
        assert p2.paper_size == p1.paper_size
        assert p2.color_palette == p1.color_palette

    def test_dataclass_fields_accessible(self):
        import dataclasses
        from section_tool.export.print_params import PrintExportParams
        fields = {f.name for f in dataclasses.fields(PrintExportParams)}
        expected = {
            "paper_size", "dpi", "margin_in", "color_palette", "background",
            "horizon_line_weight", "fault_line_weight", "polygon_outline_weight",
            "polygon_fill_opacity", "font_family", "label_size_pt", "title_size_pt",
            "annotation_size_pt", "label_color", "show_strat_column", "show_grid",
            "show_sea_level", "show_axis_labels", "show_scale_bar", "show_title_block",
            "title", "subtitle", "author", "date_text", "seismic_inclusion",
            "vertical_exaggeration", "output_format",
            "custom_width_in", "custom_height_in",
        }
        missing = expected - fields
        assert not missing, f"PrintExportParams is missing expected fields: {missing}"


# ---------------------------------------------------------------------------
# 5. Smoke test — render_section_to_file (PNG, headless matplotlib)
# ---------------------------------------------------------------------------

class TestPrintRendererRenderSmoke:

    def test_render_to_png_no_crash(self, tmp_path):
        """Build a figure from a minimal state+section and save to PNG.

        This exercises _build_figure end-to-end without Qt.
        """
        from section_tool.export.print_renderer import render_section_to_file
        from section_tool.export.print_params import PrintExportParams

        section = _make_section()
        proj = _make_project_with_picks()
        state = _MinimalState(proj)

        params = PrintExportParams(
            paper_size="A4 landscape",
            dpi=72,
            show_strat_column=False,
            show_title_block=False,
            show_scale_bar=False,
            output_format="png",
        )
        out = str(tmp_path / "section.png")
        render_section_to_file(state, section, params, out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 500

    def test_render_returns_figure_from_build_figure(self):
        """_build_figure should return a non-None Figure."""
        from matplotlib.figure import Figure
        from section_tool.export.print_renderer import _build_figure
        from section_tool.export.print_params import PrintExportParams

        section = _make_section()
        proj = _make_project_with_picks()
        state = _MinimalState(proj)

        params = PrintExportParams(
            paper_size="A4 landscape",
            dpi=72,
            show_strat_column=False,
            show_title_block=False,
        )
        fig = _build_figure(state, section, params, dpi=72)
        assert fig is not None
        assert isinstance(fig, Figure)

    def test_render_empty_project_no_crash(self, tmp_path):
        """Rendering a section with no picks or polygons should not raise."""
        from section_tool.export.print_renderer import _build_figure
        from section_tool.export.print_params import PrintExportParams
        from section_tool.io.project import Project

        section = _make_section()
        proj = Project()
        state = _MinimalState(proj)

        params = PrintExportParams(
            paper_size="A4 landscape",
            dpi=72,
            show_strat_column=False,
            show_title_block=False,
        )
        fig = _build_figure(state, section, params, dpi=72)
        assert fig is not None

    def test_render_all_toggles_on(self, tmp_path):
        """Enable all content options and confirm no crash."""
        from section_tool.export.print_renderer import render_section_to_file
        from section_tool.export.print_params import PrintExportParams

        section = _make_section()
        proj = _make_project_with_picks()
        state = _MinimalState(proj)

        params = PrintExportParams(
            paper_size="A4 landscape",
            dpi=72,
            show_strat_column=True,
            show_title_block=True,
            show_scale_bar=True,
            show_sea_level=True,
            show_grid=True,
            show_axis_labels=True,
            output_format="png",
        )
        out = str(tmp_path / "all_on.png")
        render_section_to_file(state, section, params, out)
        assert os.path.exists(out)


# ---------------------------------------------------------------------------
# 6. Paper size options
# ---------------------------------------------------------------------------

class TestPrintExportParamsPaperSizes:

    def test_paper_sizes_all_nonempty_strings(self):
        from section_tool.export.print_renderer import PAPER_SIZES
        for key in PAPER_SIZES:
            assert isinstance(key, str) and len(key) > 0

    def test_default_paper_size_in_known_set(self):
        from section_tool.export.print_params import PrintExportParams
        from section_tool.export.print_renderer import PAPER_SIZES
        p = PrintExportParams()
        # Default paper size must be in PAPER_SIZES or be "Custom"
        assert p.paper_size in PAPER_SIZES or p.paper_size == "Custom"

    def test_paper_sizes_count(self):
        from section_tool.export.print_renderer import PAPER_SIZES
        # At minimum the four sizes in the source
        assert len(PAPER_SIZES) >= 4


# ---------------------------------------------------------------------------
# 7. show_strat_column field
# ---------------------------------------------------------------------------

class TestStratColumnToggleParam:

    def test_show_strat_column_exists(self):
        from section_tool.export.print_params import PrintExportParams
        p = PrintExportParams()
        assert hasattr(p, "show_strat_column")

    def test_show_strat_column_default_is_bool(self):
        from section_tool.export.print_params import PrintExportParams
        p = PrintExportParams()
        assert isinstance(p.show_strat_column, bool)

    def test_show_strat_column_can_be_false(self):
        from section_tool.export.print_params import PrintExportParams
        p = PrintExportParams(show_strat_column=False)
        assert p.show_strat_column is False

    def test_show_strat_column_can_be_true(self):
        from section_tool.export.print_params import PrintExportParams
        p = PrintExportParams(show_strat_column=True)
        assert p.show_strat_column is True


# ---------------------------------------------------------------------------
# 8. seismic_inclusion field
# ---------------------------------------------------------------------------

class TestSeismicInclusionParam:

    def test_seismic_inclusion_exists(self):
        from section_tool.export.print_params import PrintExportParams
        p = PrintExportParams()
        assert hasattr(p, "seismic_inclusion")

    def test_seismic_inclusion_has_default(self):
        from section_tool.export.print_params import PrintExportParams
        p = PrintExportParams()
        # Source defines default as 'omit'
        assert p.seismic_inclusion == "omit"

    def test_seismic_inclusion_can_be_set(self):
        from section_tool.export.print_params import PrintExportParams
        p = PrintExportParams(seismic_inclusion="grayscale")
        assert p.seismic_inclusion == "grayscale"


# ---------------------------------------------------------------------------
# 9. _get_palette helper
# ---------------------------------------------------------------------------

class TestGetPaletteHelper:

    def test_get_palette_returns_dict(self):
        from section_tool.export.print_renderer import _get_palette
        from section_tool.export.print_params import PrintExportParams
        p = PrintExportParams()
        pal = _get_palette(p)
        assert isinstance(pal, dict)
        assert len(pal) > 0

    def test_get_palette_applies_background_override(self):
        from section_tool.export.print_renderer import _get_palette
        from section_tool.export.print_params import PrintExportParams
        p = PrintExportParams(background="#112233")
        pal = _get_palette(p)
        assert pal["background"] == "#112233"

    def test_get_palette_applies_label_color_override(self):
        from section_tool.export.print_renderer import _get_palette
        from section_tool.export.print_params import PrintExportParams
        p = PrintExportParams(label_color="#FF0000")
        pal = _get_palette(p)
        assert pal["text"] == "#FF0000"

    def test_get_palette_unknown_name_falls_back_to_default(self):
        from section_tool.export.print_renderer import _get_palette, PALETTES, _DEFAULT_PALETTE
        from section_tool.export.print_params import PrintExportParams
        p = PrintExportParams(color_palette="NonExistentPalette")
        pal = _get_palette(p)
        # Should fall back to default; check a key from the default palette
        default_pal = PALETTES[_DEFAULT_PALETTE]
        assert "horizon" in pal
