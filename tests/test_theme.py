"""Tests for the theme system."""
from __future__ import annotations

import pytest
from section_tool.style.theme import (
    DARK, PRINT, THEMES,
    get_theme, set_theme,
    adapt_entity_color, marker_kwargs,
    EntityStyle, MarkerStyle,
)


class TestThemeStructure:
    def test_dark_has_correct_name(self):
        assert DARK.name == "dark"

    def test_print_has_correct_name(self):
        assert PRINT.name == "print"

    def test_themes_dict_contains_both(self):
        assert "dark" in THEMES and "print" in THEMES

    def test_dark_background_is_dark(self):
        # background should be a dark hex color
        assert DARK.background.startswith("#")
        r = int(DARK.background[1:3], 16)
        assert r < 64   # dark red channel

    def test_print_background_is_white(self):
        assert PRINT.background == "#FFFFFF"

    def test_entity_style_is_frozen(self):
        with pytest.raises(Exception):
            DARK.horizon.color = "#000000"   # type: ignore[misc]

    def test_theme_is_frozen(self):
        with pytest.raises(Exception):
            DARK.background = "#FFFFFF"   # type: ignore[misc]

    def test_polygon_fill_alpha_is_smaller_in_print(self):
        assert PRINT.polygon_fill_alpha < DARK.polygon_fill_alpha

    def test_marker_sizes_smaller_in_print_than_dark(self):
        assert PRINT.node.size_px < DARK.node.size_px
        assert PRINT.endpoint.size_px < DARK.endpoint.size_px


class TestGetSetTheme:
    def setup_method(self):
        set_theme("dark")  # reset to default before each test

    def teardown_method(self):
        set_theme("dark")  # reset after each test

    def test_get_theme_returns_dark_by_default(self):
        assert get_theme().name == "dark"

    def test_set_theme_print(self):
        set_theme("print")
        assert get_theme().name == "print"

    def test_set_theme_dark(self):
        set_theme("print")
        set_theme("dark")
        assert get_theme().name == "dark"

    def test_set_theme_invalid_raises(self):
        with pytest.raises(ValueError):
            set_theme("neon")

    def test_get_theme_returns_same_object_as_constant(self):
        set_theme("dark")
        assert get_theme() is DARK
        set_theme("print")
        assert get_theme() is PRINT


class TestAdaptEntityColor:
    def setup_method(self):
        set_theme("dark")

    def teardown_method(self):
        set_theme("dark")

    def test_dark_theme_returns_unchanged(self):
        assert adapt_entity_color("#FF0000") == "#FF0000"

    def test_print_theme_reduces_saturation(self):
        import colorsys
        original = "#FF0000"
        set_theme("print")
        adapted = adapt_entity_color(original)
        # Saturation should be lower
        def sat(h):
            r, g, b = int(h[1:3], 16)/255, int(h[3:5], 16)/255, int(h[5:7], 16)/255
            _, _, s = colorsys.rgb_to_hls(r, g, b)
            return s
        assert sat(adapted) < sat(original)

    def test_print_theme_accepts_explicit_theme_arg(self):
        adapted = adapt_entity_color("#FF0000", PRINT)
        original = "#FF0000"
        assert adapted != original   # should be muted

    def test_invalid_hex_returns_unchanged(self):
        assert adapt_entity_color("not-a-color", PRINT) == "not-a-color"


class TestMarkerKwargs:
    def test_square_maps_to_s(self):
        style = MarkerStyle("square", 4.0, fill=None, edge="#FFFFFF")
        kw = marker_kwargs(style)
        assert kw["marker"] == "s"

    def test_diamond_maps_to_D(self):
        style = MarkerStyle("diamond", 5.5, fill="#0e1014", edge="#e0e4ec")
        kw = marker_kwargs(style)
        assert kw["marker"] == "D"

    def test_x_maps_to_x(self):
        kw = marker_kwargs(MarkerStyle("x", 6.0, fill=None, edge="#FF0000"))
        assert kw["marker"] == "x"

    def test_unfilled_gives_none_facecolor(self):
        kw = marker_kwargs(MarkerStyle("circle", 7.0, fill=None, edge="#FFFFFF"))
        assert kw["markerfacecolor"] == "none"

    def test_filled_gives_fill_color(self):
        kw = marker_kwargs(MarkerStyle("square", 4.0, fill="#112233", edge="#FFFFFF"))
        assert kw["markerfacecolor"] == "#112233"

    def test_inherit_edge_uses_entity_color(self):
        style = MarkerStyle("circle", 7.0, fill=None, edge="inherit")
        kw = marker_kwargs(style, entity_color="#AABBCC")
        assert kw["markeredgecolor"] == "#AABBCC"

    def test_inherit_edge_without_entity_color_falls_back(self):
        style = MarkerStyle("circle", 7.0, fill=None, edge="inherit")
        kw = marker_kwargs(style, entity_color=None)
        assert kw["markeredgecolor"] is None

    def test_linestyle_is_none(self):
        kw = marker_kwargs(MarkerStyle("square", 4.0, fill=None, edge="#FFF"))
        assert kw["linestyle"] == "none"
