"""Tests for polygon deduplication behavior and SectionPolygon attributes.

Key findings from source review:
- There is NO standalone _detect_duplicate_polygons or polygon_similarity function
  in the codebase. The "dedup" logic in app.py._on_generate_polygons (around line 2266)
  is simply a QMessageBox.question asking the user whether to replace existing polygons,
  followed by a loop that removes them if yes. There is no algorithmic similarity check.
- Polygon regeneration is entirely UI-driven (requires an active QApplication and
  user interaction via dialogs), so it cannot be tested as a pure unit test.
- SectionPolygon.is_bound() and is_free() are plain instance methods.
"""
from __future__ import annotations

import numpy as np
import pytest

from section_tool.core.polygons import SectionPolygon, PolygonBoundary
from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rect_polygon(**kw) -> SectionPolygon:
    """Return a simple 4-vertex rectangle polygon."""
    verts = [(0.0, 100.0), (500.0, 100.0), (500.0, 800.0), (0.0, 800.0)]
    return SectionPolygon(verts, **kw)


def _triangle_polygon(**kw) -> SectionPolygon:
    verts = [(0.0, 0.0), (1000.0, 0.0), (500.0, 500.0)]
    return SectionPolygon(verts, **kw)


def _section(name="EW"):
    return Section([(0.0, 0.0), (10_000.0, 0.0)], name=name)


# ---------------------------------------------------------------------------
# 1. Dedup logic is inline in app.py — documentation test
# ---------------------------------------------------------------------------

class TestPolygonDedupIsInlineInApp:
    # The "dedup" logic for polygon regeneration is inline in
    # app.py._on_generate_polygons (around line 2266).  It consists of:
    #   1. Listing existing polygons for the section.
    #   2. Showing a QMessageBox.question asking the user to replace or keep.
    #   3. Calling state.remove_polygon for each if user says Yes.
    # There is no algorithmic similarity/duplicate-detection function;
    # the logic is entirely UI-driven and cannot be unit-tested without
    # a running Qt application and mocked user interaction.

    def test_dedup_logic_is_inline_in_app(self):
        pytest.skip(
            "dedup logic is inline in app.py._on_generate_polygons; "
            "needs extraction for unit testability"
        )


# ---------------------------------------------------------------------------
# 2. SectionPolygon.is_bound and is_free attributes accessible
# ---------------------------------------------------------------------------

class TestPolygonIsBoundAttribute:

    def test_is_free_returns_true_for_vertex_polygon(self):
        poly = _rect_polygon(name="FreeRect")
        assert poly.is_free() is True

    def test_is_bound_returns_false_for_vertex_polygon(self):
        poly = _rect_polygon(name="FreeRect")
        assert poly.is_bound() is False

    def test_is_bound_returns_true_when_bounds_given(self):
        poly = _rect_polygon(name="Bound")
        poly.bounds = [PolygonBoundary(category="Horizons", index=0)]
        assert poly.is_bound() is True

    def test_is_free_returns_false_when_bounds_given(self):
        poly = _rect_polygon(name="Bound")
        poly.bounds = [PolygonBoundary(category="Horizons", index=0)]
        assert poly.is_free() is False

    def test_is_bound_callable_on_fresh_polygon(self):
        """Regression: is_bound() should not raise regardless of constructor args."""
        poly = SectionPolygon(
            [(0, 0), (100, 0), (100, 100)],
            name="tri",
            fill_color="#aabbcc",
        )
        result = poly.is_bound()
        assert isinstance(result, bool)

    def test_is_free_callable_on_fresh_polygon(self):
        poly = SectionPolygon(
            [(0, 0), (100, 0), (100, 100)],
            name="tri",
        )
        result = poly.is_free()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# 3. Polygon dedup via state regenerate
#    (Requires UI prompt — cannot test programmatically)
# ---------------------------------------------------------------------------

class TestPolygonDedupViaStateRegenerate:

    def test_regenerate_requires_ui(self):
        # AppState.add_polygon / remove_polygon work without UI.
        # _on_generate_polygons lives in app.py (the main window),
        # requires an active QApplication and shows modal dialogs —
        # cannot be driven programmatically without a test double.
        pytest.skip(
            "_on_generate_polygons requires QApplication and modal dialogs; "
            "extraction to a testable function is needed"
        )

    def test_adding_polygon_twice_doubles_count(self):
        """Without the UI dedup, adding the same polygon twice doubles the count."""
        state = AppState()
        sec = _section()
        state.add_section(sec)

        poly = _rect_polygon(name="LayerA", section_name="EW")
        state.add_polygon(poly)
        state.add_polygon(poly)   # add again — no dedup at AppState level

        assert len(state.project.polygons) == 2

    def test_remove_polygon_reduces_count(self):
        state = AppState()
        poly = _rect_polygon(name="LayerA")
        state.add_polygon(poly)
        assert len(state.project.polygons) == 1

        state.remove_polygon(poly)
        assert len(state.project.polygons) == 0


# ---------------------------------------------------------------------------
# 4. Confirm no standalone polygon_similarity or _detect_duplicate function
# ---------------------------------------------------------------------------

class TestPolygonSimilarityNoFunction:
    # Confirmed by grepping the entire section_tool package:
    # - No function named 'polygon_similarity' exists.
    # - No function named '_detect_duplicate' (related to polygons) exists.
    # - No function named '_detect_duplicate_polygons' exists.
    # The only deduplication in the project is in topology_audit.py
    # (_deduplicate_picks), which deduplicates horizon/fault picks,
    # not polygons.

    def test_no_standalone_polygon_similarity_function(self):
        pytest.skip(
            "No standalone similarity function; "
            "dedup is UI-only (confirmed by grep of section_tool package)"
        )

    def test_polygon_module_has_no_similarity_function(self):
        """Assert polygon_similarity is not importable from section_tool.core.polygons."""
        import section_tool.core.polygons as poly_module
        assert not hasattr(poly_module, "polygon_similarity"), (
            "polygon_similarity unexpectedly found in core.polygons"
        )

    def test_no_detect_duplicate_polygons_in_app_state(self):
        """AppState should not have a _detect_duplicate_polygons method."""
        state = AppState()
        assert not hasattr(state, "_detect_duplicate_polygons"), (
            "_detect_duplicate_polygons found on AppState; "
            "if added, write a dedicated unit test for it"
        )


# ---------------------------------------------------------------------------
# 5. SectionPolygon bounds/vertices accessible after construction
# ---------------------------------------------------------------------------

class TestSectionPolygonBoundsAccessible:

    def test_vertices_property_returns_ndarray(self):
        poly = _rect_polygon()
        v = poly.vertices
        assert isinstance(v, np.ndarray)

    def test_vertices_shape(self):
        poly = _rect_polygon()
        v = poly.vertices
        assert v.ndim == 2
        assert v.shape[1] == 2
        assert v.shape[0] >= 3

    def test_vertices_non_none(self):
        poly = _rect_polygon()
        assert poly.vertices is not None

    def test_n_vertices_matches(self):
        poly = _rect_polygon()
        assert poly.n_vertices == len(poly.vertices)

    def test_free_points_accessible(self):
        poly = _rect_polygon()
        assert poly.free_points is not None
        assert isinstance(poly.free_points, np.ndarray)

    def test_bounds_list_accessible_and_initially_empty(self):
        poly = _rect_polygon()
        assert isinstance(poly.bounds, list)
        assert len(poly.bounds) == 0

    def test_vertices_are_copy_not_reference(self):
        """SectionPolygon.vertices should return a copy."""
        poly = _rect_polygon()
        v1 = poly.vertices
        v1[0, 0] = 9999.0
        v2 = poly.vertices
        assert v2[0, 0] != 9999.0, (
            "SectionPolygon.vertices returned a reference instead of a copy"
        )

    def test_closed_distances_has_extra_element(self):
        poly = _rect_polygon()
        cd = poly.closed_distances()
        # closed: first vertex appended at end
        assert len(cd) == poly.n_vertices + 1
        assert cd[0] == cd[-1]

    def test_closed_depths_has_extra_element(self):
        poly = _rect_polygon()
        cz = poly.closed_depths()
        assert len(cz) == poly.n_vertices + 1
        assert cz[0] == cz[-1]

    def test_area_positive(self):
        poly = _rect_polygon()
        assert poly.area > 0

    def test_triangle_area(self):
        """Triangle with known area."""
        # Triangle (0,0)-(1000,0)-(0,1000): area = 500000
        poly = SectionPolygon([(0, 0), (1000, 0), (0, 1000)], name="tri")
        assert poly.area == pytest.approx(500_000.0, rel=1e-6)

    def test_vertices_values_match_constructor_input(self):
        verts = [(10.0, 20.0), (30.0, 20.0), (20.0, 50.0)]
        poly = SectionPolygon(verts)
        v = poly.vertices
        assert v[0, 0] == pytest.approx(10.0)
        assert v[0, 1] == pytest.approx(20.0)
        assert v[2, 0] == pytest.approx(20.0)
        assert v[2, 1] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# 6. AppState polygon CRUD signals
# ---------------------------------------------------------------------------

class TestAppStatePolygonSignals:

    def test_add_polygon_emits_polygon_added(self):
        state = AppState()
        received = []
        state.polygon_added.connect(lambda p: received.append(p))
        poly = _rect_polygon(name="Test")
        state.add_polygon(poly)
        assert len(received) == 1
        assert received[0] is poly

    def test_remove_polygon_emits_polygon_removed(self):
        state = AppState()
        received = []
        state.polygon_removed.connect(lambda p: received.append(p))
        poly = _rect_polygon(name="Remove")
        state.add_polygon(poly)
        state.remove_polygon(poly)
        assert len(received) == 1

    def test_update_polygon_emits_polygon_modified(self):
        state = AppState()
        modified_args = []
        state.polygon_modified.connect(lambda idx, p: modified_args.append((idx, p)))
        poly = _rect_polygon(name="Update")
        state.add_polygon(poly)
        new_poly = _rect_polygon(name="Updated")
        state.update_polygon(0, new_poly)
        assert len(modified_args) == 1
        assert modified_args[0][0] == 0
        assert modified_args[0][1] is new_poly


# ---------------------------------------------------------------------------
# 7. PolygonBoundary dataclass
# ---------------------------------------------------------------------------

class TestPolygonBoundaryDataclass:

    def test_boundary_construction(self):
        b = PolygonBoundary(category="Horizons", index=2)
        assert b.category == "Horizons"
        assert b.index == 2
        assert b.reversed is False  # default

    def test_boundary_reversed(self):
        b = PolygonBoundary(category="Faults", index=0, reversed=True)
        assert b.reversed is True

    def test_boundary_equality(self):
        b1 = PolygonBoundary("Horizons", 1)
        b2 = PolygonBoundary("Horizons", 1)
        assert b1 == b2

    def test_boundary_inequality(self):
        b1 = PolygonBoundary("Horizons", 1)
        b2 = PolygonBoundary("Horizons", 2)
        assert b1 != b2
