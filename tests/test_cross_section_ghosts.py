"""Phase 2 scenario 8 — cross-section ghost picks (headless harness).

When two sections intersect in map space and a horizon has picks on the
*other* section, the active section should render a ghost marker + label at
the intersection distance, at the depth the other section's pick has there.

These drive the real SectionView renderer (no GUI interaction) and assert
the artists it produces, plus the negative cases and the visibility toggle.
"""
from __future__ import annotations

import sys

import pytest
from PySide6.QtWidgets import QApplication

from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick
from section_tool.views.section_view import SectionView


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


# ---------------------------------------------------------------------------
# Geometry: L1 (E–W along y=0) and L2 (N–S at x=5000) cross at map (5000, 0).
#   intersection: s_along_L1 = 5000, s_along_L2 = 5000
# ---------------------------------------------------------------------------

def _build(qapp, *, pick_on="L2", visible=True, l2_parallel=False):
    """Return (view, L1, L2) with L1 active and an optional horizon pick.

    pick_on: "L2" (default, produces a ghost on L1), "L1" (no ghost), or None.
    l2_parallel: make L2 NOT intersect L1 (parallel offset section).
    """
    state = AppState()
    l1 = Section([(0.0, 0.0), (10000.0, 0.0)], name="L1")
    if l2_parallel:
        l2 = Section([(0.0, 1000.0), (10000.0, 1000.0)], name="L2")   # parallel, never crosses
    else:
        l2 = Section([(5000.0, -5000.0), (5000.0, 5000.0)], name="L2")  # crosses L1 at (5000,0)
    state.add_section(l1)
    state.add_section(l2)

    if pick_on is not None:
        # pick spanning distance 0..10000 on the chosen section; depth 100..200
        hp = HorizonPick([0.0, 10000.0], [100.0, 200.0], name="H1",
                         section_names=[pick_on, pick_on])
        hp.visible = visible
        state.add_horizon_pick(hp)

    state.set_active_section(l1)
    view = SectionView(state)
    return view, l1, l2


def _ghost_artists(view, section):
    """Run only the ghost renderer against a clean artist list; return the new artists."""
    view._overlay_artists.clear()
    view._render_cross_section_ghost_picks(section)
    return list(view._overlay_artists)


# ---------------------------------------------------------------------------
# Positive case
# ---------------------------------------------------------------------------

class TestGhostRendered:

    def test_ghost_marker_and_label_added(self, qapp):
        view, l1, _ = _build(qapp, pick_on="L2")
        artists = _ghost_artists(view, l1)
        # one marker (Line2D from ax.plot) + one annotation per crossing pick
        assert len(artists) == 2

    def test_ghost_marker_at_intersection_distance_and_depth(self, qapp):
        view, l1, _ = _build(qapp, pick_on="L2")
        artists = _ghost_artists(view, l1)
        # the Line2D marker carries the (x, y) data: x = s_along_L1 (5000),
        # y = depth of L2's pick at s_along_L2 (interp of [100,200] at 5000 = 150)
        line = next(a for a in artists if hasattr(a, "get_xdata"))
        xs, ys = list(line.get_xdata()), list(line.get_ydata())
        assert xs == pytest.approx([5000.0])
        assert ys == pytest.approx([150.0])

    def test_ghost_label_names_other_section(self, qapp):
        view, l1, _ = _build(qapp, pick_on="L2")
        artists = _ghost_artists(view, l1)
        texts = [a.get_text() for a in artists if hasattr(a, "get_text")]
        assert any("L2" in t for t in texts)


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------

class TestNoGhost:

    def test_no_ghost_when_sections_do_not_intersect(self, qapp):
        view, l1, _ = _build(qapp, pick_on="L2", l2_parallel=True)
        assert _ghost_artists(view, l1) == []

    def test_no_ghost_when_pick_only_on_active_section(self, qapp):
        # pick lives on L1 (the active section); nothing on L2 to project across
        view, l1, _ = _build(qapp, pick_on="L1")
        assert _ghost_artists(view, l1) == []

    def test_no_ghost_when_no_picks(self, qapp):
        view, l1, _ = _build(qapp, pick_on=None)
        assert _ghost_artists(view, l1) == []

    def test_invisible_pick_skipped(self, qapp):
        view, l1, _ = _build(qapp, pick_on="L2", visible=False)
        assert _ghost_artists(view, l1) == []


# ---------------------------------------------------------------------------
# Visibility toggle gates the ghost renderer in the full overlay pass
# ---------------------------------------------------------------------------

class TestGhostToggle:

    def test_toggle_off_skips_ghost_render(self, qapp, monkeypatch):
        view, l1, _ = _build(qapp, pick_on="L2")
        calls = []
        monkeypatch.setattr(view, "_render_cross_section_ghost_picks",
                            lambda section: calls.append(section))
        view._state.show_cross_section_ghosts = False
        view._render_overlays(l1)
        assert calls == []

    def test_toggle_on_invokes_ghost_render(self, qapp, monkeypatch):
        view, l1, _ = _build(qapp, pick_on="L2")
        calls = []
        monkeypatch.setattr(view, "_render_cross_section_ghost_picks",
                            lambda section: calls.append(section))
        view._state.show_cross_section_ghosts = True
        view._render_overlays(l1)
        assert calls == [l1]
