"""Depth Stretch panel + migrated common-helper tests.

The old Method×Setting dialog is gone; its pure helpers moved to
depth_stretch_common and its behavioural coverage migrated to the controller
(test_depth_stretch_controller.py) + these panel smoke tests.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QPushButton

from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick
from section_tool.core.wells import Well, LogCurve
from section_tool.core.tdr import TimeDepthRelation
from section_tool.core.conversion import set_anchors, build_bulk
from section_tool.core.velocity_model import VelocityModel, conversion_caption
from section_tool.core.grounded_velocity import _US_FT_TO_S_PER_M
from section_tool.views.depth_stretch_common import (
    method_availability, format_model_summary_html, band_color_hex)
from section_tool.views.depth_stretch_panel import DepthStretchPanel


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


# ---------------------------------------------------------------------------
# Migrated pure-helper tests (no Qt)
# ---------------------------------------------------------------------------

def test_method_availability_gating():
    a = method_availability(zone_tops=[], wells=[])
    assert a["bulk"][0] and a["average_vz"][0]
    assert not a["layered_from_formations"][0] and a["layered_from_formations"][1]
    assert not a["well_calibrated"][0] and "well" in a["well_calibrated"][1].lower()
    b = method_availability(zone_tops=[(0.5, "A")], wells=["w"])
    assert b["layered_from_formations"][0] and b["well_calibrated"][0]


def test_summary_html_neutral_with_chips():
    html = format_model_summary_html(VelocityModel.average_vz(1800.0, 0.6))
    assert "monospace" in html and "background-color" in html
    assert "font-style:italic" in html
    assert "1800" in html and "V(z)" in html
    layer = VelocityModel.average_vz(1800.0, 0.6).layers[0]
    assert band_color_hex(layer) in html
    assert "unconverted" in format_model_summary_html(VelocityModel())


# ---------------------------------------------------------------------------
# Panel construction across inventory permutations
# ---------------------------------------------------------------------------

def _state(section=True):
    st = AppState()
    if section:
        st.add_section(Section([(0, 0), (3000, 0)], name="L1"))
        st.set_active_section(st.project.sections[0])
    return st


def _sonic_checkshot_well():
    w = Well("F02-01", 0.0, 0.0, kb=0.0, td=3200.0)
    md = np.arange(0.0, 3001.0, 5.0)
    slow = (1.0 / 2000.0) / _US_FT_TO_S_PER_M
    w.add_log(LogCurve("DT:2", "us/ft", md, np.full_like(md, slow)))
    w.add_tdr(TimeDepthRelation([0.0, 1000.0, 2000.0, 3000.0],
                                [0.0, 0.95, 1.90, 2.85],
                                kind="checkshot", depth_reference="TVDSS"))
    return w


def test_panel_builds_for_empty_project(qapp):
    panel = DepthStretchPanel(_state())
    assert panel._selected == "bulk"          # empty → bulk recommended/selected


def test_panel_builds_for_full_f3(qapp):
    st = _state()
    st.add_well(_sonic_checkshot_well())
    hp = HorizonPick(np.array([0.0, 1500.0]), np.array([800.0, 800.0]), name="H1",
                     section_names=np.array(["L1", "L1"], dtype=object),
                     formation_below="FS8")
    set_anchors(hp, build_bulk(2000.0))
    st.project.horizon_picks.append(hp)
    panel = DepthStretchPanel(st)
    assert panel._selected == "sonic_checkshot"


# ---------------------------------------------------------------------------
# Apply through the panel + caption single source
# ---------------------------------------------------------------------------

def test_panel_apply_installs_and_calls_back(qapp):
    st = _state()
    fired = []
    panel = DepthStretchPanel(st, on_apply=lambda: fired.append(True))
    panel._apply()                             # selected = bulk
    assert fired == [True]
    m = st.project.velocity_model
    assert m is not None and not m.is_empty
    # Footer caption equals the single-source caption.
    assert conversion_caption(m) is not None


def test_panel_apply_restretches_tied(qapp):
    st = _state()
    w = _sonic_checkshot_well(); st.add_well(w)
    hp = HorizonPick(np.array([0.0, 1500.0]), np.array([1000.0, 1000.0]), name="H1",
                     section_names=np.array(["L1", "L1"], dtype=object))
    set_anchors(hp, build_bulk(2000.0))
    st.project.horizon_picks.append(hp)
    anchors = hp._twt_anchor.copy()
    panel = DepthStretchPanel(st)
    panel._select("checkshot")
    panel._apply()
    m = st.project.velocity_model
    assert np.allclose(hp._twt_anchor, anchors)                      # invariant
    assert np.allclose(hp._depths, [m.twt_to_depth(a) for a in anchors])


def test_panel_select_swaps_card(qapp):
    st = _state()
    panel = DepthStretchPanel(st)
    assert panel._selected == "bulk"
    panel._select("average_vz")
    assert panel._selected == "average_vz"


# ---------------------------------------------------------------------------
# Card knob area is rebuilt atomically on rung swap — no orphaned widgets
# (the screenshotted overlap: bulk's velocity spinner survived a swap to
# checkshot-tied and stacked under the new rung's knobs).
# ---------------------------------------------------------------------------

_EXPECTED_KNOBS = {
    "bulk":            {"knob_setting", "knob_bulk_v"},
    "average_vz":      {"knob_setting", "knob_v0", "knob_k"},
    "checkshot":       {"knob_setting"},
    "sonic_checkshot": {"knob_setting"},
    "sonic_anchors":   {"knob_setting"},
    "layered":         {"knob_setting"},
    "marker_tied":     {"knob_setting"},
}


def _card_knob_names(panel):
    """objectNames of the spin/combo knobs currently parented under the card."""
    from PySide6.QtWidgets import QDoubleSpinBox, QComboBox
    knobs = (panel._card.findChildren(QDoubleSpinBox)
             + panel._card.findChildren(QComboBox))
    return sorted(w.objectName() for w in knobs)


def test_card_knob_area_rebuilt_atomically_on_swap(qapp):
    st = _state()
    panel = DepthStretchPanel(st)
    rungs = list(_EXPECTED_KNOBS)
    # Several full cycles across every rung: widgets must never accumulate.
    for _ in range(3):
        for rung in rungs:
            panel._select(rung)
            names = _card_knob_names(panel)
            assert set(names) == _EXPECTED_KNOBS[rung], (
                f"after selecting {rung}: card knobs were {names}")
            # Exactly the active rung's widgets — no duplicates, no leaks.
            assert len(names) == len(_EXPECTED_KNOBS[rung])
            assert panel._knob_container.objectName() == "knob_container"


def test_bulk_to_checkshot_swap_drops_bulk_spinner(qapp):
    # The exact regression: bulk's 2400 m/s spinner must not survive a swap to
    # checkshot-tied and overlap the "Source: well" row.
    st = _state()
    panel = DepthStretchPanel(st)
    panel._select("bulk")
    assert "knob_bulk_v" in _card_knob_names(panel)
    panel._select("checkshot")
    assert "knob_bulk_v" not in _card_knob_names(panel)
    assert _card_knob_names(panel) == ["knob_setting"]


# ---------------------------------------------------------------------------
# Locked-rung Import action triggers the importer callback + refreshes
# ---------------------------------------------------------------------------

def test_locked_import_action_invokes_callback(qapp):
    st = _state()                              # empty → checkshot locked
    captured = []

    def on_import(token):
        captured.append(token)
        # Simulate the importer adding a checkshot well, then the panel refreshes.
        st.add_well(_sonic_checkshot_well())

    panel = DepthStretchPanel(st, on_import=on_import)
    panel._do_import("checkshot")
    assert captured == ["checkshot"]
    # After import the panel re-selects the now-higher recommendation.
    assert panel._selected in ("sonic_checkshot", "checkshot")


# ---------------------------------------------------------------------------
# Upgrade banner appears / clears, never restretches
# ---------------------------------------------------------------------------

def test_upgrade_banner_appears_and_keep_clears(qapp):
    st = _state()
    panel = DepthStretchPanel(st)
    panel._select("bulk"); panel._apply()
    applied_model = st.project.velocity_model
    # isHidden() reflects the explicit flag regardless of the (unshown) parent.
    assert panel._upgrade_bar.isHidden()           # nothing better yet

    # A checkshot well arrives → a more grounded rung is available.
    st.add_well(_sonic_checkshot_well())
    panel._render()
    assert not panel._upgrade_bar.isHidden()
    assert st.project.velocity_model is applied_model   # tag never restretched

    panel._keep_current()
    assert panel._upgrade_bar.isHidden()                # acknowledged → cleared
    assert st.project.velocity_model is applied_model   # still no restretch
