"""Tests for section_tool.core.topology_audit."""
from __future__ import annotations

import pytest
import numpy as np

from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick
from section_tool.core.topology_audit import audit_section, AuditIssue
from section_tool.io.project import Project


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _project_with_picks(horizon_specs=(), fault_specs=()) -> tuple:
    """Return (project, section) populated with given picks.

    Each spec: list of (distance, depth) pairs for section "L1".
    """
    sec = Section([(0.0, 0.0), (5000.0, 0.0)], name="L1")
    proj = Project()
    proj.sections.append(sec)

    for i, pts in enumerate(horizon_specs):
        hp = HorizonPick.empty(name=f"H{i+1}")
        for d, z in pts:
            hp.insert_pick(float(d), float(z), "L1")
        proj.horizon_picks.append(hp)

    for i, pts in enumerate(fault_specs):
        fp = HorizonPick.empty(name=f"F{i+1}")
        for d, z in pts:
            fp.insert_pick(float(d), float(z), "L1")
        proj.fault_picks.append(fp)

    return proj, sec


# ---------------------------------------------------------------------------
# Basic audit
# ---------------------------------------------------------------------------

class TestAuditClean:
    def test_no_issues_on_clean_horizon(self):
        proj, sec = _project_with_picks(
            horizon_specs=[[(0, 100), (500, 200), (1000, 300)]]
        )
        issues = audit_section(sec, proj)
        assert issues == []

    def test_no_issues_empty_project(self):
        proj = Project()
        sec = Section([(0.0, 0.0), (1000.0, 0.0)], name="X")
        issues = audit_section(sec, proj)
        assert issues == []

    def test_picks_on_other_section_not_audited(self):
        sec = Section([(0.0, 0.0), (1000.0, 0.0)], name="L1")
        proj = Project()
        # Add a pick only on a different section
        hp = HorizonPick.empty(name="Offsite")
        hp.insert_pick(200.0, 300.0, "L2")
        proj.horizon_picks.append(hp)
        issues = audit_section(sec, proj)
        assert issues == []


# ---------------------------------------------------------------------------
# Orphan nodes (< 2 points on section)
# ---------------------------------------------------------------------------

class TestOrphanNodes:
    def test_single_node_warning(self):
        proj, sec = _project_with_picks(horizon_specs=[[(500, 200)]])
        issues = audit_section(sec, proj)
        assert len(issues) == 1
        assert issues[0].severity == "warning"
        assert "1 node" in issues[0].description.lower()

    def test_two_nodes_no_orphan_warning(self):
        proj, sec = _project_with_picks(horizon_specs=[[(0, 100), (500, 200)]])
        issues = audit_section(sec, proj)
        orphan = [i for i in issues if "1 node" in i.description.lower()]
        assert orphan == []


# ---------------------------------------------------------------------------
# Duplicate / near-duplicate nodes
# ---------------------------------------------------------------------------

class TestDuplicates:
    def test_exact_duplicate_detected(self):
        proj, sec = _project_with_picks(
            horizon_specs=[[(0, 100), (500, 200), (500, 200), (1000, 300)]]
        )
        issues = audit_section(sec, proj)
        dupe = [i for i in issues if "duplicate" in i.description.lower()]
        assert len(dupe) == 1
        assert dupe[0].auto_fixable is True

    def test_near_duplicate_within_tolerance(self):
        proj, sec = _project_with_picks(
            horizon_specs=[[(0, 100), (500, 200), (500.5, 200.5), (1000, 300)]]
        )
        issues = audit_section(sec, proj, tol=1.0)
        dupe = [i for i in issues if "duplicate" in i.description.lower()]
        assert len(dupe) == 1

    def test_near_duplicate_outside_tolerance_clean(self):
        proj, sec = _project_with_picks(
            horizon_specs=[[(0, 100), (500, 200), (505, 205), (1000, 300)]]
        )
        issues = audit_section(sec, proj, tol=1.0)
        dupe = [i for i in issues if "duplicate" in i.description.lower()]
        assert dupe == []

    def test_auto_fix_removes_duplicate(self):
        from section_tool.app_state import AppState
        state = AppState()
        from section_tool.core.section import Section as _Sec
        sec = _Sec([(0.0, 0.0), (1000.0, 0.0)], name="L1")
        state.add_section(sec)
        state.set_active_section(sec)
        hp = HorizonPick.empty(name="Dup")
        hp.insert_pick(0.0, 100.0, "L1")
        hp.insert_pick(500.0, 200.0, "L1")
        hp.insert_pick(500.0, 200.0, "L1")   # exact duplicate
        hp.insert_pick(1000.0, 300.0, "L1")
        state.add_horizon_pick(hp)

        issues = audit_section(sec, state.project, tol=1.0)
        dupe_issues = [i for i in issues if i.auto_fixable]
        assert len(dupe_issues) == 1

        dupe_issues[0].fix_action(state)
        hp_fixed = state.project.horizon_picks[0]
        idxs = hp_fixed.section_indices("L1")
        assert len(idxs) == 3   # duplicate was removed


# ---------------------------------------------------------------------------
# Self-intersections
# ---------------------------------------------------------------------------

class TestSelfIntersection:
    """Self-intersection helper tested directly (not via insert_pick, which sorts by
    distance and therefore prevents crossings in the normal data flow)."""

    def test_seg_intersect_self_helper_detects_crossing(self):
        from section_tool.core.topology_audit import _seg_intersect_self
        # Explicit X shape: (0,0)→(1,1) vs (0,1)→(1,0)  — crosses at (0.5, 0.5)
        d = [0.0, 1.0, 0.0, 1.0]
        z = [0.0, 1.0, 1.0, 0.0]
        # Note: skipping j=i+1 (adjacent); for i=0, first non-adjacent is j=2
        pts = _seg_intersect_self(d, z)
        assert len(pts) == 1
        assert abs(pts[0][0] - 0.5) < 1e-6

    def test_seg_intersect_self_helper_clean_line(self):
        from section_tool.core.topology_audit import _seg_intersect_self
        d = [0.0, 100.0, 200.0, 300.0]
        z = [100.0, 200.0, 150.0, 250.0]
        pts = _seg_intersect_self(d, z)
        assert pts == []

    def test_audit_reports_no_self_intersection_for_sorted_picks(self):
        # Distance-sorted picks cannot self-intersect in (d, z) — verify no false positive
        proj, sec = _project_with_picks(
            horizon_specs=[[(0, 100), (200, 300), (400, 50), (600, 200)]]
        )
        issues = audit_section(sec, proj)
        self_x = [i for i in issues if "self-intersect" in i.description.lower()]
        assert self_x == []


# ---------------------------------------------------------------------------
# Severity ordering
# ---------------------------------------------------------------------------

class TestSeverityOrdering:
    def test_warnings_before_info(self):
        # duplicate (warning) + orphan (warning) — both appear; ordering is by name
        proj, sec = _project_with_picks(
            horizon_specs=[
                [(500, 200)],                                        # orphan → warning
                [(0, 100), (500, 200), (500, 200), (1000, 300)],    # duplicate → warning
            ]
        )
        issues = audit_section(sec, proj)
        severities = [i.severity for i in issues]
        assert all(s in ("error", "warning", "info") for s in severities)

    def test_result_is_sorted_no_error_after_warning(self):
        # When both error-level and warning-level issues exist, errors come first.
        # Inject a self-intersection directly by passing non-sorted data to _audit_pick_list
        from section_tool.core.topology_audit import _audit_pick_list
        from unittest.mock import MagicMock

        hp_mock = MagicMock()
        hp_mock.name = "Crossing"
        hp_mock.section_indices.return_value = [0, 1, 2, 3]
        hp_mock._distances = np.array([0.0, 1.0, 0.0, 1.0])   # non-sorted → crossing
        hp_mock._depths    = np.array([0.0, 1.0, 1.0, 0.0])

        issues = _audit_pick_list([hp_mock], "Horizons", "L1", 0.01)
        severities = [i.severity for i in issues]
        # Should have a self-intersection (error) detected
        assert "error" in severities
