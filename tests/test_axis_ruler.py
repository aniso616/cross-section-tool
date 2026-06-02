"""Tests for the shared AxisRuler primitive and the DepthRuler shim."""
from __future__ import annotations

import sys

import pytest
from PySide6.QtWidgets import QApplication

from section_tool.hud.axis_ruler import AxisRuler, nice_interval
from section_tool.hud.depth_ruler import DepthRuler


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


# ---------------------------------------------------------------------------
# nice_interval
# ---------------------------------------------------------------------------

class TestNiceInterval:
    def test_zero_and_negative_safe(self):
        assert nice_interval(0) == 1.0
        assert nice_interval(-5) == 1.0

    def test_rounds_to_1_2_5_decade(self):
        # span/5 is the raw target
        assert nice_interval(5000, 5) == 1000      # raw 1000 -> 1000
        assert nice_interval(7000, 5) == 2000      # raw 1400 -> 2000
        assert nice_interval(2000, 5) == 500       # raw 400  -> 500

    def test_monotonic_with_span(self):
        a = nice_interval(1000)
        b = nice_interval(100000)
        assert b > a


# ---------------------------------------------------------------------------
# Geometry: _value_to_px honors orientation direction via `inverted`
# ---------------------------------------------------------------------------

class TestValueToPixel:
    def test_not_inverted_lo_at_start(self, qapp):
        r = AxisRuler(None, major_interval=100)
        r.set_view_range(0, 1000)
        assert r._value_to_px(0, 100) == 0       # lo -> pixel 0 (top/left)
        assert r._value_to_px(1000, 100) == 100  # hi -> pixel length

    def test_inverted_lo_at_end(self, qapp):
        r = AxisRuler(None, inverted=True, major_interval=100)
        r.set_view_range(0, 1000)
        assert r._value_to_px(0, 100) == 100     # lo -> pixel length (bottom)
        assert r._value_to_px(1000, 100) == 0    # hi -> pixel 0 (top)

    def test_midpoint(self, qapp):
        r = AxisRuler(None)
        r.set_view_range(0, 1000)
        assert r._value_to_px(500, 200) == 100


# ---------------------------------------------------------------------------
# Intervals: fixed vs auto
# ---------------------------------------------------------------------------

class TestIntervals:
    def test_fixed_intervals(self, qapp):
        r = AxisRuler(None, major_interval=500, minor_interval=100)
        r.set_view_range(0, 5000)
        assert r._intervals() == (500, 100)

    def test_auto_interval_reflows_with_range(self, qapp):
        r = AxisRuler(None)               # no fixed interval -> auto
        r.set_view_range(0, 5000)
        major_wide, _ = r._intervals()
        r.set_view_range(0, 200)          # zoom in
        major_narrow, _ = r._intervals()
        assert major_narrow < major_wide


# ---------------------------------------------------------------------------
# Both orientations + formatters paint without error and produce labels
# ---------------------------------------------------------------------------

class TestPaintAndFormatters:
    def _grab(self, ruler):
        ruler.resize(60, 400) if ruler._orient == "vertical" else ruler.resize(400, 60)
        return ruler.grab()   # forces paintEvent

    def test_vertical_depth_formatter_paints(self, qapp):
        r = AxisRuler(None, orientation="vertical",
                      formatter=lambda d: f"{int(d)}", major_interval=500)
        r.set_view_range(0, 5000)
        r.set_cursor_value(1234)
        pix = self._grab(r)
        assert pix.width() > 0 and pix.height() > 0

    def test_horizontal_coord_formatter_paints(self, qapp):
        r = AxisRuler(None, orientation="horizontal",
                      formatter=lambda v: f"{v:,.0f}")
        r.set_view_range(600000, 620000)
        r.set_cursor_value(610000)
        pix = self._grab(r)
        assert pix.width() > 0 and pix.height() > 0

    def test_formatter_is_used(self, qapp):
        calls = []
        r = AxisRuler(None, orientation="horizontal",
                      formatter=lambda v: (calls.append(v), f"{v:,.0f}")[1],
                      major_interval=5000)
        r.set_view_range(0, 20000)
        self._grab(r)
        assert calls   # formatter invoked during paint


# ---------------------------------------------------------------------------
# DepthRuler shim: API preserved, formation chaser still paints
# ---------------------------------------------------------------------------

class TestDepthRulerShim:
    def test_is_axis_ruler(self, qapp):
        assert issubclass(DepthRuler, AxisRuler)

    def test_legacy_api_present(self, qapp):
        dr = DepthRuler(None)
        dr.set_view_range(0, 5000)
        dr.set_cursor_depth(1500)
        assert dr._cursor == 1500
        assert dr._intervals() == (500, 100)

    def test_formation_chaser_paints(self, qapp):
        from collections import namedtuple
        Band = namedtuple("Band", "top_m base_m color")
        dr = DepthRuler(None)
        dr.set_view_range(0, 5000)
        dr.set_formations([Band(0, 1000, (200, 120, 60)),
                           Band(1000, 3000, (60, 120, 200))])
        dr.resize(52, 400)
        pix = dr.grab()
        assert pix.height() > 0
