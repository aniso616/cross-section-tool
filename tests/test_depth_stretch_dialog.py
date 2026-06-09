"""Depth Stretch dialog — method ladder/gating, progressive disclosure, summary."""
from __future__ import annotations

import sys

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick
from section_tool.core.wells import Well
from section_tool.core.velocity_model import VelocityModel
from section_tool.core.conversion import set_anchors, build_bulk
from section_tool.views.depth_stretch_dialog import (
    DepthStretchDialog, method_availability, format_model_summary_html, band_color_hex)


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


def _state_with_tied_horizon():
    st = AppState()
    st.add_section(Section([(0, 0), (1000, 0)], name="L1"))
    hp = HorizonPick(np.array([0.0, 500.0]), np.array([1000.0, 1200.0]),
                     name="H1", section_names=np.array(["L1", "L1"], dtype=object))
    set_anchors(hp, build_bulk(2000.0))
    st.project.horizon_picks.append(hp)
    return st, hp


# ---- pure helpers (no Qt) -------------------------------------------------

def test_method_availability_gating():
    # bulk/average always; layered needs tops; well-calibrated needs wells
    a = method_availability(zone_tops=[], wells=[])
    assert a["bulk"][0] and a["average_vz"][0]
    assert not a["layered_from_formations"][0] and a["layered_from_formations"][1]
    assert not a["well_calibrated"][0] and "well" in a["well_calibrated"][1].lower()
    b = method_availability(zone_tops=[(0.5, "A")], wells=["w"])
    assert b["layered_from_formations"][0]
    assert b["well_calibrated"][0]


def test_summary_html_neutral_with_chips():
    html = format_model_summary_html(VelocityModel.average_vz(1800.0, 0.6))
    assert "monospace" in html                     # aligned columns
    assert "background-color" in html              # per-layer color chip
    assert "font-style:italic" in html             # provenance muted-italic
    assert "1800" in html and "V(z)" in html       # neutral numbers/method text
    # the chip color matches the schematic band color (one visual language)
    layer = VelocityModel.average_vz(1800.0, 0.6).layers[0]
    assert band_color_hex(layer) in html
    # empty model reads as provenance, not a crash
    assert "unconverted" in format_model_summary_html(VelocityModel())


# ---- dialog ---------------------------------------------------------------

def test_apply_installs_model_and_restretches_tied(qapp):
    st, hp = _state_with_tied_horizon()
    anchors = hp._twt_anchor.copy()
    fired = []
    dlg = DepthStretchDialog(st, on_apply=lambda: fired.append(True))
    dlg.method.setCurrentIndex(dlg.method.findData("average_vz"))
    dlg.v0.setValue(1800.0); dlg.k.setValue(0.5)
    dlg._apply()
    m = st.project.velocity_model
    assert fired == [True]
    assert m.construction["params"]["method"] == "average_vz"
    assert np.allclose(hp._twt_anchor, anchors)
    assert np.allclose(hp._depths, [m.twt_to_depth(a) for a in anchors])


def test_layered_and_well_rungs_gated(qapp):
    bare = AppState(); bare.add_section(Section([(0, 0), (1000, 0)], name="L1"))
    dlg = DepthStretchDialog(bare)
    li = dlg.method.findData("layered_from_formations")
    wi = dlg.method.findData("well_calibrated")
    assert not dlg.method.model().item(li).isEnabled()
    assert not dlg.method.model().item(wi).isEnabled()

    st, _ = _state_with_tied_horizon()
    st.project.wells.append(Well("W1", 0.0, 0.0))
    dlg2 = DepthStretchDialog(st)
    assert dlg2.method.model().item(dlg2.method.findData("layered_from_formations")).isEnabled()
    assert dlg2.method.model().item(dlg2.method.findData("well_calibrated")).isEnabled()


def test_progressive_disclosure(qapp):
    st, _ = _state_with_tied_horizon()
    dlg = DepthStretchDialog(st)
    dlg.method.setCurrentIndex(dlg.method.findData("bulk"))
    dlg.setting.setCurrentIndex(dlg.setting.findData("onshore"))
    assert not dlg.bulk_v.isHidden()                 # bulk → bulk velocity shown
    assert dlg.v0.isHidden() and dlg.k.isHidden()    # v0/k hidden
    assert dlg.water_v.isHidden()                    # land → no water
    dlg.method.setCurrentIndex(dlg.method.findData("average_vz"))
    assert dlg.bulk_v.isHidden()
    assert not dlg.v0.isHidden() and not dlg.k.isHidden()
    dlg.setting.setCurrentIndex(dlg.setting.findData("marine"))
    assert not dlg.water_v.isHidden() and not dlg.seafloor_ms.isHidden()


def test_marine_summary_lists_water_layer(qapp):
    st, _ = _state_with_tied_horizon()
    dlg = DepthStretchDialog(st)
    dlg.setting.setCurrentIndex(dlg.setting.findData("marine"))
    dlg.method.setCurrentIndex(dlg.method.findData("bulk"))
    dlg.seafloor_ms.setValue(400.0)
    dlg._on_changed()
    assert "water" in dlg._summary.toPlainText().lower()


def test_well_calibrated_apply_installs_calibrated(qapp):
    from section_tool.core.velocity_model import VelocityFunction
    st, _ = _state_with_tied_horizon()
    w = Well("W1", 0.0, 0.0)
    for name, md in [("A", 300.0), ("B", 800.0), ("C", 1500.0)]:
        w.add_formation_top(name, md)
    st.project.wells.append(w)
    dlg = DepthStretchDialog(st)
    dlg.method.setCurrentIndex(dlg.method.findData("well_calibrated"))
    dlg._on_well_changed()
    fn = VelocityFunction("linear_v0k", v0=1850.0, k=0.55)
    for r in range(dlg.markers.rowCount()):
        depth = float(dlg.markers.item(r, 1).text())
        dlg.markers.item(r, 2).setText(f"{fn.depth_to_twt(depth) * 1000:.3f}")
    dlg._apply()
    assert st.project.velocity_model.provenance == "well_calibrated"


def _state_with_zone_horizons():
    """A project with a strat column + two anchored zone-bounding horizons."""
    from section_tool.core.formation import StratigraphicColumn, Formation
    st = AppState(); st.add_section(Section([(0, 0), (1000, 0)], name="L1"))
    col = StratigraphicColumn()
    for nm, v in [("Over", 2500.0), ("Res", 4000.0), ("Base", 5500.0)]:
        f = Formation(nm); f.matrix_velocity = v; col.add_formation(f)
    st.project.strat_column = col
    for depth, fa, fb, nm in [(800.0, "Over", "Res", "TopRes"),
                              (1200.0, "Res", "Base", "BaseRes")]:
        hp = HorizonPick(np.array([0.0, 500.0]), np.array([depth, depth]), name=nm,
                         section_names=np.array(["L1", "L1"], dtype=object),
                         formation_above=fa, formation_below=fb)
        set_anchors(hp, build_bulk(2000.0))   # anchors at 0.8 / 1.2 s
        st.project.horizon_picks.append(hp)
    return st


def test_layered_apply_builds_formation_layers_land(qapp):
    st = _state_with_zone_horizons()
    dlg = DepthStretchDialog(st)
    dlg.setting.setCurrentIndex(dlg.setting.findData("onshore"))
    dlg.method.setCurrentIndex(dlg.method.findData("layered_from_formations"))
    dlg.datum_ms.setValue(0.0)
    dlg._apply()
    m = st.project.velocity_model
    assert m.construction["params"]["method"] == "layered_from_formations"
    assert m.method_label == "layered — formation matrix velocities"
    # datum cap (Over) + the two zones, tops at the anchors
    assert [round(l.top_twt_s, 3) for l in m.layers] == [0.0, 0.8, 1.2]
    assert [l.function.v0 for l in m.layers] == [2500.0, 4000.0, 5500.0]


def test_layered_apply_marine_prepends_water(qapp):
    st = _state_with_zone_horizons()
    dlg = DepthStretchDialog(st)
    dlg.setting.setCurrentIndex(dlg.setting.findData("marine"))
    dlg.method.setCurrentIndex(dlg.method.findData("layered_from_formations"))
    dlg.datum_ms.setValue(0.0); dlg.seafloor_ms.setValue(300.0)   # seafloor 0.3 s
    dlg._apply()
    m = st.project.velocity_model
    assert m.layers[0].name == "Water" and m.layers[0].top_twt_s == pytest.approx(0.0)
    # then the formation cap at the seafloor, then the zones
    assert [round(l.top_twt_s, 3) for l in m.layers] == [0.0, 0.3, 0.8, 1.2]
