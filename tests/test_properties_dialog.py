"""Tests for the properties dialog framework (no Qt widget tests — no pytest-qt)."""
import sys
import os


def _make_app():
    """Create or return the QApplication singleton."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_base_dialog_collects_values():
    _make_app()
    from section_tool.views.properties_dialog import PropertiesDialog
    dlg = PropertiesDialog("horizon", {
        "name": "Test Horizon",
        "color": "#ff0000",
        "line_width": 2.0,
        "line_style": "dashed",
        "opacity": 0.8,
    })
    vals = dlg._collect_values()
    assert vals["name"] == "Test Horizon"
    assert vals["line_width"] == 2.0
    assert vals["line_style"] == "dashed"
    assert abs(vals["opacity"] - 0.8) < 0.01
    dlg.destroy()


def test_horizon_dialog_tabs():
    _make_app()
    from section_tool.views.horizon_properties_dialog import HorizonPropertiesDialog
    dlg = HorizonPropertiesDialog("Horizons", {
        "name": "H1", "color": "#2ca02c",
        "contact_type": "unconformity",
        "formation_above": "Fm A", "formation_below": "Fm B",
        "line_width": 1.5, "line_style": "solid",
    })
    assert dlg.tabs.count() == 3
    vals = dlg._collect_values()
    assert vals["contact_type"] == "unconformity"
    assert vals["formation_above"] == "Fm A"
    dlg.destroy()


def test_fault_dialog_collects():
    _make_app()
    from section_tool.views.fault_properties_dialog import FaultPropertiesDialog
    dlg = FaultPropertiesDialog("Faults", {
        "name": "F1", "color": "#d62728",
        "fault_type": "normal", "dip_direction": "right",
        "line_width": 1.5, "line_style": "solid",
    })
    vals = dlg._collect_values()
    assert vals["fault_type"] == "normal"
    assert vals["dip_direction"] == "right"
    dlg.destroy()


def test_polygon_dialog_collects():
    _make_app()
    from section_tool.views.polygon_properties_dialog import PolygonPropertiesDialog
    dlg = PolygonPropertiesDialog("Polygons", {
        "name": "P1", "color": "#9467bd",
        "fill_color": "#9467bd", "fill_opacity": 0.6,
        "formation_name": "Sand A",
        "line_width": 1.5, "line_style": "solid",
    })
    vals = dlg._collect_values()
    assert vals["formation_name"] == "Sand A"
    dlg.destroy()


def test_surface_dialog_tabs():
    _make_app()
    from section_tool.views.surface_properties_dialog import SurfacePropertiesDialog
    dlg = SurfacePropertiesDialog("Surfaces", {
        "name": "FS4", "color": "#E87722",
        "n_points": 1000, "z_domain": "twt_ms", "z_units": "ms",
        "z_range": (500.0, 1200.0),
        "bounds": (600000, 6070000, 630000, 6090000),
        "crs_epsg": 32631, "interpolation": "linear",
        "line_width": 1.5,
    })
    assert dlg.tabs.count() == 3
    dlg.destroy()


def test_properties_changed_signal():
    _make_app()
    from section_tool.views.properties_dialog import PropertiesDialog
    dlg = PropertiesDialog("horizon", {
        "name": "H", "color": "#ffffff",
        "line_width": 1.5, "line_style": "solid",
    })
    received = []
    dlg.properties_changed.connect(received.append)
    dlg._on_apply()
    assert len(received) == 1
    assert "name" in received[0]
    dlg.destroy()


def test_reference_line_dialog():
    _make_app()
    from section_tool.views.reference_line_properties_dialog import (
        ReferenceLinePropertiesDialog)
    dlg = ReferenceLinePropertiesDialog("Reference Lines", {
        "name": "Top Mancos", "color": "#999999",
        "kind": "horizontal", "value": 1500.0, "visible": True,
        "line_width": 0.8, "line_style": "dashed",
    })
    vals = dlg._collect_values()
    assert abs(vals["value"] - 1500.0) < 0.1
    dlg.destroy()
