"""M3 — DepthStretchDialog: build/preview/apply + interpretation gating."""
from __future__ import annotations

import sys

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick
from section_tool.core.conversion import set_anchors, build_bulk
from section_tool.views.depth_stretch_dialog import DepthStretchDialog


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


def _state_with_tied_horizon():
    st = AppState()
    st.add_section(Section([(0, 0), (1000, 0)], name="L1"))
    hp = HorizonPick(np.array([0.0, 500.0]), np.array([1000.0, 1200.0]),
                     name="H1", section_names=np.array(["L1", "L1"], dtype=object))
    set_anchors(hp, build_bulk(2000.0))     # seismic-tied via the bootstrap model
    st.project.horizon_picks.append(hp)
    return st, hp


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
    assert np.allclose(hp._twt_anchor, anchors)                         # invariant
    assert np.allclose(hp._depths, [m.twt_to_depth(a) for a in anchors])  # re-derived


def test_layered_gated_by_anchored_horizons(qapp):
    # No anchored horizon → layered disabled; with one → enabled.
    bare = AppState(); bare.add_section(Section([(0, 0), (1000, 0)], name="L1"))
    dlg = DepthStretchDialog(bare)
    idx = dlg.method.findData("layered_from_formations")
    assert not dlg.method.model().item(idx).isEnabled()

    st, _ = _state_with_tied_horizon()
    dlg2 = DepthStretchDialog(st)
    idx2 = dlg2.method.findData("layered_from_formations")
    assert dlg2.method.model().item(idx2).isEnabled()


def test_marine_preview_lists_water_layer(qapp):
    st, _ = _state_with_tied_horizon()
    dlg = DepthStretchDialog(st)
    dlg.setting.setCurrentText("marine")
    dlg.method.setCurrentIndex(dlg.method.findData("bulk"))
    dlg.seafloor_ms.setValue(400.0)
    dlg._update_preview()
    assert "water" in dlg._preview.text().lower()
