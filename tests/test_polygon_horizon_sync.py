"""Phase 2 scenario 7 — bound polygons auto-update when a bounding horizon is edited.

A reference-based SectionPolygon stores PolygonBoundary refs to the picks that
form its perimeter. Editing one of those picks (update_horizon_pick /
update_fault_pick) must cascade through AppState._recompute_polygon_bounds so
the polygon's vertices follow the new geometry — before the modified signal
fires, so the next render is already correct.

Pure model-level harness (no Qt): drives AppState directly.
"""
from __future__ import annotations

import numpy as np
import pytest

from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick
from section_tool.core.polygons import PolygonBoundary, SectionPolygon


def _setup(with_bounds=True, free=False):
    """A section L1 with Top (idx 0) + Bot (idx 1) horizons and one rectangle polygon.

    The polygon is bound to Top (forward) + Bot (reversed) -> a closed rectangle,
    unless free=True (an explicit free-form polygon with no bounds).
    """
    state = AppState()
    sec = Section([(0.0, 0.0), (1000.0, 0.0)], name="L1")
    state.add_section(sec)
    state.set_active_section(sec)

    top = HorizonPick([0.0, 1000.0], [100.0, 100.0], name="Top", section_names=["L1", "L1"])
    bot = HorizonPick([0.0, 1000.0], [300.0, 300.0], name="Bot", section_names=["L1", "L1"])
    state.add_horizon_pick(top)   # index 0
    state.add_horizon_pick(bot)   # index 1

    seed = [[0.0, 100.0], [1000.0, 100.0], [1000.0, 300.0], [0.0, 300.0]]
    if free:
        poly = SectionPolygon(seed, name="Free", section_name="L1")
    else:
        bounds = [PolygonBoundary("Horizons", 0, reversed=False),
                  PolygonBoundary("Horizons", 1, reversed=True)]
        poly = SectionPolygon(seed, name="Layer", section_name="L1", bounds=bounds)
    state.add_polygon(poly)
    return state, poly


def _new_top(depth):
    return HorizonPick([0.0, 1000.0], [depth, depth], name="Top", section_names=["L1", "L1"])


# ---------------------------------------------------------------------------
# Positive: editing a bounding horizon reshapes the polygon
# ---------------------------------------------------------------------------

class TestBoundPolygonFollowsEdit:

    def test_top_depths_track_edited_horizon(self):
        state, poly = _setup()
        # baseline: top edge at depth 100
        assert poly.vertices[:, 1].min() == pytest.approx(100.0)

        state.update_horizon_pick(0, _new_top(50.0))   # raise the Top horizon

        # polygon's shallowest vertices now follow the new Top depth
        assert poly.vertices[:, 1].min() == pytest.approx(50.0)

    def test_full_rectangle_geometry_after_edit(self):
        state, poly = _setup()
        state.update_horizon_pick(0, _new_top(50.0))
        # bounds order: Top forward [(0,50),(1000,50)] + Bot reversed [(1000,300),(0,300)]
        expected = np.array([[0, 50], [1000, 50], [1000, 300], [0, 300]], dtype=float)
        assert poly.vertices == pytest.approx(expected)

    def test_recompute_runs_before_modified_signal(self):
        """When horizon_pick_modified fires, the polygon is already updated."""
        state, poly = _setup()
        seen_min = []
        state.horizon_pick_modified.connect(
            lambda idx, pick: seen_min.append(poly.vertices[:, 1].min()))
        state.update_horizon_pick(0, _new_top(50.0))
        assert seen_min == [pytest.approx(50.0)]


# ---------------------------------------------------------------------------
# Negative: free polygons and non-referenced edits are untouched
# ---------------------------------------------------------------------------

class TestNoSpuriousRecompute:

    def test_free_polygon_unchanged_by_horizon_edit(self):
        state, poly = _setup(free=True)
        before = poly.vertices.copy()
        state.update_horizon_pick(0, _new_top(50.0))
        assert poly.vertices == pytest.approx(before)

    def test_edit_of_unreferenced_horizon_leaves_polygon(self):
        state, poly = _setup()
        before = poly.vertices.copy()
        # add a third, unrelated horizon and edit it
        extra = HorizonPick([0.0, 1000.0], [800.0, 800.0], name="Deep",
                            section_names=["L1", "L1"])
        state.add_horizon_pick(extra)   # index 2 — not referenced by the polygon
        state.update_horizon_pick(2, HorizonPick([0.0, 1000.0], [900.0, 900.0],
                                                 name="Deep", section_names=["L1", "L1"]))
        assert poly.vertices == pytest.approx(before)


# ---------------------------------------------------------------------------
# Fault-bound polygons cascade the same way (update_fault_pick path)
# ---------------------------------------------------------------------------

class TestFaultBoundPolygon:

    def test_polygon_follows_fault_edit(self):
        state = AppState()
        sec = Section([(0.0, 0.0), (1000.0, 0.0)], name="L1")
        state.add_section(sec)
        state.set_active_section(sec)
        top = HorizonPick([0.0, 1000.0], [100.0, 100.0], name="Top", section_names=["L1", "L1"])
        fault = HorizonPick([0.0, 1000.0], [300.0, 300.0], name="F1", section_names=["L1", "L1"])
        state.add_horizon_pick(top)        # Horizons[0]
        state.add_fault_pick(fault)        # Faults[0]
        poly = SectionPolygon(
            [[0, 100], [1000, 100], [1000, 300], [0, 300]],
            name="Block", section_name="L1",
            bounds=[PolygonBoundary("Horizons", 0, reversed=False),
                    PolygonBoundary("Faults", 0, reversed=True)])
        state.add_polygon(poly)

        state.update_fault_pick(0, HorizonPick([0.0, 1000.0], [500.0, 500.0],
                                               name="F1", section_names=["L1", "L1"]))
        assert poly.vertices[:, 1].max() == pytest.approx(500.0)
