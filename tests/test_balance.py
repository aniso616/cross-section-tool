"""core/balance.py (Restoration Step 4): Dahlstrom balance primitives against
hand-calculated fixtures, plus the deformed-vs-restored comparison in the
Balance Check dialog (and its graceful degradation with no snapshot)."""
from __future__ import annotations

import sys

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from section_tool.core import balance as B
from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick
from section_tool.core.polygons import SectionPolygon
from section_tool.core.restoration_snapshot import snapshot_interpretation
from section_tool.views.balance_check_dialog import BalanceCheckDialog


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


# ── pure primitives ─────────────────────────────────────────────────────────

def test_polygon_area_rectangle_sign_and_magnitude():
    W, H = 100.0, 40.0
    rect = [(0, 0), (W, 0), (W, H), (0, H)]              # clockwise on screen (y down)
    assert B.polygon_area(rect) == pytest.approx(W * H)         # documented: positive
    assert B.polygon_area(rect[::-1]) == pytest.approx(-W * H)  # reversed → sign flips
    assert B.polygon_area([(0, 0), (1, 1)]) == 0.0             # < 3 vertices


def test_polygon_area_magnitude_matches_sectionpolygon():
    verts = [(0, 0), (100, 0), (100, 40)]
    assert abs(B.polygon_area(verts)) == pytest.approx(SectionPolygon(verts).area)


def test_sign_convention_consistent_across_shapes():
    # Two clockwise-on-screen shapes both give positive area (one convention).
    tri = [(0, 0), (10, 0), (10, 10)]
    rect = [(0, 0), (5, 0), (5, 3), (0, 3)]
    assert B.polygon_area(tri) > 0 and B.polygon_area(rect) > 0


def test_horizon_line_length_known():
    assert B.horizon_line_length([(0, 0), (3, 0), (3, 4)]) == pytest.approx(7.0)  # 3+4
    assert B.horizon_line_length([(0, 0), (1000, 0)]) == pytest.approx(1000.0)
    assert B.horizon_line_length([(0, 0)]) == 0.0                                 # degenerate


def test_depth_to_detachment_known():
    dd = B.depth_to_detachment(excess_area=2000.0, shortening=100.0)
    assert dd.depth == pytest.approx(20.0)               # 2000 / 100
    assert dd.formula == "d = excess_area / shortening"
    assert "20" in dd.explain()
    assert np.isnan(B.depth_to_detachment(2000.0, 0.0).depth)   # no div-by-zero


def test_area_balance_discrepancy_and_threshold():
    deformed = [(0, 0), (100, 0), (100, 10.5), (0, 10.5)]      # 1050
    restored = [(0, 0), (100, 0), (100, 10.0), (0, 10.0)]      # 1000
    ab = B.area_balance(deformed, restored, name="P")
    assert ab.deformed_area == pytest.approx(1050.0)
    assert ab.restored_area == pytest.approx(1000.0)
    assert ab.discrepancy == pytest.approx(0.05)
    assert ab.is_balanced(0.05) and not ab.is_balanced(0.049)


def test_line_length_balance_pairs_by_uuid_key():
    deformed = {"a": [(0, 0), (10, 0)], "b": [(0, 0), (8, 0)]}
    restored = {"a": [(0, 0), (12, 0)], "c": [(0, 0), (5, 0)]}
    res = B.line_length_balance(deformed, restored, names={"a": "Top"})
    assert len(res) == 1 and res[0].name == "Top"        # only 'a' in both
    assert res[0].deformed_length == pytest.approx(10.0)
    assert res[0].restored_length == pytest.approx(12.0)
    assert res[0].shortening == pytest.approx(2.0)       # restored − deformed
    assert res[0].discrepancy == pytest.approx(2.0 / 12.0)


# ── dialog: comparison + graceful degradation ───────────────────────────────

def _state_with_poly_and_horizon():
    state = AppState()
    state.add_section(Section([(0.0, 0.0), (1000.0, 0.0)], name="L1", crs_epsg=32631))
    state.set_active_section(state.project.sections[0])
    hp = HorizonPick([0.0, 1000.0], [100.0, 100.0], name="Top",
                     section_names=["L1", "L1"])
    state.project.horizon_picks.append(hp)
    poly = SectionPolygon([(0, 0), (100, 0), (100, 10), (0, 10)], name="Block",
                          section_name="L1")
    state.project.polygons.append(poly)
    return state, hp, poly


def test_dialog_degrades_gracefully_without_snapshot(qapp):
    state, hp, poly = _state_with_poly_and_horizon()
    dlg = BalanceCheckDialog(state, state.active_section, snapshot=None)
    assert getattr(dlg, "_cmp_table", None) is None      # no comparison built
    assert dlg._cmp_rows == []                            # but no crash


def test_dialog_comparison_shows_known_discrepancy(qapp):
    state, hp, poly = _state_with_poly_and_horizon()
    snap = snapshot_interpretation(state.active_section, state.project)  # restored = 1000 m²
    poly._vertices[:] = [[0, 0], [100, 0], [100, 9.5], [0, 9.5]]         # deform → 950 m²

    dlg = BalanceCheckDialog(state, state.active_section, snapshot=snap)
    areas = [r for r, _ in dlg._cmp_rows if hasattr(r, "restored_area")]
    assert len(areas) == 1
    ab = areas[0]
    assert ab.restored_area == pytest.approx(1000.0)
    assert ab.deformed_area == pytest.approx(950.0)
    assert ab.discrepancy == pytest.approx(0.05)
    # threshold drives the flag (default 5%): balanced at 5%, flagged below it
    dlg._tol_spin.setValue(4.0)
    dlg._refresh_balance_flags()
    assert "over threshold" in dlg._cmp_table.item(0, 5).text()
    dlg._tol_spin.setValue(6.0)
    assert "balanced" in dlg._cmp_table.item(0, 5).text()
