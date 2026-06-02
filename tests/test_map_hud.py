"""Tests for the map's full-bleed config, HUD reflow, and readout reproject."""
from __future__ import annotations

import sys

import pytest
from PySide6.QtWidgets import QApplication, QWidget

from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.hud.map_hud_layer import MapHUDLayer
from section_tool.hud.nav_readout import NavReadout
from section_tool.views.map_view import MapView

DEG = "°"   # degree sign — encoding-safe in assertions


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


@pytest.fixture
def host(qapp):
    """A persistent parent widget — keeps child HUD widgets alive for the test."""
    w = QWidget()
    yield w
    w.deleteLater()


@pytest.fixture
def map_view(qapp):
    state = AppState()
    state.project.crs_epsg = 32630          # UTM 30N (projected)
    state.add_section(Section([(600_000, 6_080_000), (612_000, 6_092_000)], name="L1"))
    return MapView(state)


# ---------------------------------------------------------------------------
# Full-bleed: no chrome, equal aspect via datalim
# ---------------------------------------------------------------------------

class TestFullBleedConfig:
    def test_spines_hidden(self, map_view):
        ax = map_view.axes
        MapView._configure_axes(ax)
        assert all(not s.get_visible() for s in ax.spines.values())

    def test_no_ticks(self, map_view):
        ax = map_view.axes
        MapView._configure_axes(ax)
        assert list(ax.get_xticks()) == []
        assert list(ax.get_yticks()) == []

    def test_equal_aspect_datalim(self, map_view):
        ax = map_view.axes
        MapView._configure_axes(ax)
        assert ax.get_aspect() == 1.0          # 'equal'
        assert ax.get_adjustable() == "datalim"


# ---------------------------------------------------------------------------
# HUD reflow: refresh pulls live limits into rulers + scale bar
# ---------------------------------------------------------------------------

class TestHUDReflow:
    def test_refresh_pushes_extent_into_rulers(self, host, map_view):
        hud = MapHUDLayer(host, map_view)
        ax = map_view.axes
        ax.set_xlim(600_000, 620_000)
        ax.set_ylim(6_080_000, 6_100_000)
        hud.refresh()
        assert (round(hud.e_ruler._lo), round(hud.e_ruler._hi)) == (600_000, 620_000)
        assert (round(hud.n_ruler._lo), round(hud.n_ruler._hi)) == (6_080_000, 6_100_000)

    def test_scale_bar_reflows_on_zoom(self, host, map_view):
        hud = MapHUDLayer(host, map_view)
        map_view.canvas.resize(800, 600)
        ax = map_view.axes
        ax.set_xlim(600_000, 620_000); hud.refresh()
        wide = hud.scale_bar._mpp
        ax.set_xlim(605_000, 607_000); hud.refresh()
        narrow = hud.scale_bar._mpp
        assert narrow is not None and wide is not None
        assert narrow < wide                    # zoomed in -> fewer metres/pixel

    def test_n_ruler_is_inverted(self, host, map_view):
        # Northing increases upward: lo (south) maps to the bottom pixel.
        hud = MapHUDLayer(host, map_view)
        assert hud.n_ruler._inverted is True
        assert hud.e_ruler._inverted is False


# ---------------------------------------------------------------------------
# Readout: E/N + lat/long, with clean CRS short-circuit
# ---------------------------------------------------------------------------

class TestReadout:
    def test_en_only_without_latlong(self, host):
        nr = NavReadout(host)
        nr.update_map_coords(606_000, 6_086_000)
        assert nr.text() == "E: 606,000  |  N: 6,086,000"

    def test_latlong_appended_with_hemisphere(self, host):
        nr = NavReadout(host)
        nr.update_map_coords(606_000, 6_086_000, -1.35, 54.91)
        assert f"54.91{DEG}N" in nr.text()
        assert f"1.35{DEG}W" in nr.text()

    def test_cursor_projected_crs_shows_latlong(self, host, map_view):
        hud = MapHUDLayer(host, map_view)
        hud._on_cursor(606_000, 6_086_000)        # crs 32630 -> reprojects
        assert DEG in hud.nav_readout.text()

    def test_cursor_missing_crs_omits_latlong(self, host, map_view):
        map_view._state.project.crs_epsg = None
        hud = MapHUDLayer(host, map_view)
        hud._on_cursor(606_000, 6_086_000)
        assert DEG not in hud.nav_readout.text()
        assert "E: 606,000" in hud.nav_readout.text()
