"""M5 UI — WellCalibrationDialog: seed from tops, compute fit, residuals, apply."""
from __future__ import annotations

import sys

import pytest
from PySide6.QtWidgets import QApplication

from section_tool.app_state import AppState
from section_tool.core.wells import Well
from section_tool.core.velocity_model import VelocityModel, VelocityFunction
from section_tool.views.well_calibration_dialog import WellCalibrationDialog


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


def _state_with_well():
    st = AppState()
    w = Well("W1", 0.0, 0.0)
    for name, md in [("TopA", 300.0), ("TopB", 800.0),
                     ("TopC", 1400.0), ("TopD", 2100.0)]:
        w.add_formation_top(name, md)
    st.project.wells.append(w)
    st.project.velocity_model = VelocityModel.average_vz(1500.0, 0.2)  # assumed/wrong
    return st, w


def _fill_twt_from_model(dlg, v0, k):
    """Set each row's TWT (ms) to the true model's value for its depth."""
    fn = VelocityFunction("linear_v0k", v0=v0, k=k)
    for r in range(dlg.table.rowCount()):
        depth = float(dlg.table.item(r, 1).text())
        dlg.table.item(r, 2).setText(f"{fn.depth_to_twt(depth) * 1000:.3f}")


def test_table_seeded_from_formation_tops(qapp):
    st, _ = _state_with_well()
    dlg = WellCalibrationDialog(st)
    assert dlg.table.rowCount() == 4
    assert dlg.table.item(0, 0).text() == "TopA"
    assert float(dlg.table.item(0, 1).text()) == pytest.approx(300.0)


def test_compute_calibrates_and_reports_residuals(qapp):
    st, _ = _state_with_well()
    dlg = WellCalibrationDialog(st)
    _fill_twt_from_model(dlg, 1850.0, 0.55)
    dlg._compute()
    assert dlg._calibrated is not None
    assert dlg._calibrated.provenance == "well_calibrated"
    assert dlg._calibrated.layers[0].function.v0 == pytest.approx(1850.0, rel=0.03)
    assert dlg._apply_btn.isEnabled()
    assert "marker" in dlg._report.text().lower()


def test_apply_installs_calibrated_model(qapp):
    st, _ = _state_with_well()
    dlg = WellCalibrationDialog(st)
    _fill_twt_from_model(dlg, 1850.0, 0.55)
    dlg._compute()
    fired = []
    dlg._on_apply = lambda: fired.append(True)
    dlg._apply()
    assert st.project.velocity_model is dlg._calibrated
    assert st.project.velocity_model.provenance == "well_calibrated"
    assert fired == [True]


def test_too_few_markers_disables_apply(qapp):
    st, _ = _state_with_well()
    dlg = WellCalibrationDialog(st)
    # leave all TWT at 0 → no usable markers
    dlg._compute()
    assert dlg._calibrated is None
    assert not dlg._apply_btn.isEnabled()
    assert "at least 2" in dlg._report.text().lower()
