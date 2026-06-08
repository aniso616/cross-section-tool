"""SEG-Y import dialog — vertical-domain declaration, conditional units, corridor."""
from __future__ import annotations

import sys

import pytest
from PySide6.QtWidgets import QApplication

from section_tool.core.section import Section
from section_tool.views.seismic_import_dialog import SeismicImportDialog


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


def _dlg(qapp):
    return SeismicImportDialog([Section([(0, 0), (1000, 0)], name="L1")], "vol.segy")


def test_default_is_twt_with_time_units_and_stretch_note(qapp):
    dlg = _dlg(qapp)
    assert dlg.domain == "twt"
    units = [dlg._units_combo.itemData(i) for i in range(dlg._units_combo.count())]
    assert units == ["ms", "s"]
    assert "depth stretch" in dlg._domain_note.text().lower()


def test_depth_domain_switches_units_to_length(qapp):
    dlg = _dlg(qapp)
    dlg._domain_combo.setCurrentIndex(dlg._domain_combo.findData("depth"))
    assert dlg.domain == "depth"
    units = [dlg._units_combo.itemData(i) for i in range(dlg._units_combo.count())]
    assert units == ["m", "ft"]
    assert "directly" in dlg._domain_note.text().lower()


def test_max_offset_default_and_readback(qapp):
    dlg = _dlg(qapp)
    assert dlg.max_offset == pytest.approx(500.0)     # legacy default
    dlg._max_offset_spin.setValue(250.0)
    assert dlg.max_offset == pytest.approx(250.0)


def test_target_section_defaults_to_auto_then_selectable(qapp):
    dlg = _dlg(qapp)
    assert dlg.target_section is None                 # "all / auto-detect"
    dlg._section_combo.setCurrentIndex(1)             # the one real section
    assert dlg.target_section.name == "L1"


def test_geometry_headers_default_cdp(qapp):
    dlg = _dlg(qapp)
    assert dlg.x_field == 181 and dlg.y_field == 185
    assert dlg.apply_scalar is True
