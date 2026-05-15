"""Comprehensive tests for the 1D decompaction engine.

All tests are headless (no Qt, no GUI).  All assertions are derived from
first-principles physics or verified numerically.

Physical note on decompaction
-----------------------------
When overburden is removed, the remaining layers *expand* (decompact):
less confining pressure → higher porosity → greater thickness.
A layer that was, say, 1000 m thick at burial depth will be >1000 m thick
when decompacted back to the surface.  Tests check this direction carefully.

Sclater & Christie (1980) rock-type parameters used throughout:
  shale:    phi0=0.63, c=0.00051
  sand:     phi0=0.49, c=0.00027
  chalk:    phi0=0.70, c=0.00071
  limestone:phi0=0.40, c=0.00040
  salt:     phi0=0.01, c=0.0  (incompressible)
"""
from __future__ import annotations

import numpy as np
import pytest

from section_tool.core.decompaction import (
    burial_history,
    decompact_column,
    decompact_layer,
    porosity_athy,
    solid_thickness,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_layers(*specs) -> list[dict]:
    """Build a list of layer dicts from (name, z_top, z_bottom, phi0, c, age_top, age_base)."""
    layers = []
    for age_idx, (name, z_top, z_bottom, phi0, c) in enumerate(specs):
        layers.append({
            "name": name,
            "z_top": z_top, "z_bottom": z_bottom,
            "phi0": phi0, "c": c,
            "age_top": age_idx * 10, "age_base": (age_idx + 1) * 10,
        })
    return layers


# ---------------------------------------------------------------------------
# 1. porosity_athy
# ---------------------------------------------------------------------------

class TestPorosityAthy:

    def test_surface_equals_phi0(self):
        assert porosity_athy(0, 0.5, 0.0005) == pytest.approx(0.5)

    def test_surface_with_zero_c(self):
        assert porosity_athy(0, 0.63, 0.0) == pytest.approx(0.63)

    def test_deep_approaches_zero(self):
        assert porosity_athy(10_000, 0.5, 0.0005) < 0.01

    def test_zero_c_constant_porosity(self):
        """c=0 (incompressible): porosity is constant with depth."""
        for z in [0, 500, 1000, 5000]:
            assert porosity_athy(z, 0.01, 0.0) == pytest.approx(0.01)

    def test_monotone_decrease_with_depth(self):
        phi_shallow = porosity_athy(100, 0.63, 0.00051)
        phi_deep    = porosity_athy(2000, 0.63, 0.00051)
        assert phi_shallow > phi_deep

    def test_array_input(self):
        z = np.array([0, 500, 1000, 2000])
        result = porosity_athy(z, 0.49, 0.00027)
        assert result.shape == (4,)
        assert result[0] == pytest.approx(0.49)
        assert np.all(np.diff(result) < 0)  # monotone decreasing

    def test_shale_sclater_christie(self):
        """Shale at 1000 m: phi ≈ 0.63*exp(-0.51) ≈ 0.375."""
        phi = porosity_athy(1000, 0.63, 0.00051)
        assert phi == pytest.approx(0.63 * np.exp(-0.51), rel=1e-6)

    def test_sand_sclater_christie(self):
        phi = porosity_athy(2000, 0.49, 0.00027)
        assert phi == pytest.approx(0.49 * np.exp(-0.54), rel=1e-6)

    def test_porosity_between_zero_and_phi0(self):
        for z in [0, 100, 1000, 5000]:
            phi = porosity_athy(z, 0.7, 0.00071)
            assert 0 <= phi <= 0.7


# ---------------------------------------------------------------------------
# 2. solid_thickness
# ---------------------------------------------------------------------------

class TestSolidThickness:

    def test_incompressible_salt(self):
        """c=0: solid thickness = total thickness (no porosity change)."""
        T_s = solid_thickness(0, 1000, 0.01, 0.0)
        assert T_s == pytest.approx(1000.0, abs=1.0)

    def test_incompressible_at_depth(self):
        T_s = solid_thickness(2000, 3000, 0.0, 0.0)
        assert T_s == pytest.approx(1000.0, abs=1e-9)

    def test_compactible_less_than_total(self):
        T_s = solid_thickness(0, 1000, 0.63, 0.00051)
        assert 0 < T_s < 1000

    def test_shale_solid_fraction(self):
        """Sanity check: at surface, shale ~37% solid (phi0=0.63)."""
        T_s = solid_thickness(0, 100, 0.63, 0.00051)
        T_total = 100
        assert T_s / T_total < 0.5   # mostly pore space near surface

    def test_solid_increases_monotone_with_burial(self):
        """Deeper burial → more compaction → higher solid fraction."""
        T_s_shallow = solid_thickness(0,    1000, 0.63, 0.00051)
        T_s_deep    = solid_thickness(5000, 6000, 0.63, 0.00051)
        # Same total thickness, but fraction is higher deeper:
        assert T_s_deep > T_s_shallow

    def test_symmetry_with_numerical_integral(self):
        """Compare analytical formula to numerical integral."""
        phi0, c = 0.49, 0.00027
        z_top, z_bottom = 1000.0, 2500.0
        T_analytic = solid_thickness(z_top, z_bottom, phi0, c)
        # Numerical: sum (1 - phi(z)) dz with fine grid
        z = np.linspace(z_top, z_bottom, 10_000)
        dz = z[1] - z[0]
        T_numeric = np.sum(1.0 - porosity_athy(z, phi0, c)) * dz
        assert T_analytic == pytest.approx(T_numeric, rel=1e-4)

    def test_zero_thickness_layer(self):
        assert solid_thickness(500, 500, 0.49, 0.00027) == pytest.approx(0.0)

    def test_additivity(self):
        """Splitting a layer at mid-depth gives the same total solid."""
        phi0, c = 0.63, 0.00051
        z_top, z_mid, z_bot = 0.0, 700.0, 1400.0
        T_whole = solid_thickness(z_top, z_bot,  phi0, c)
        T_upper = solid_thickness(z_top, z_mid,  phi0, c)
        T_lower = solid_thickness(z_mid, z_bot,  phi0, c)
        assert T_whole == pytest.approx(T_upper + T_lower, rel=1e-9)


# ---------------------------------------------------------------------------
# 3. decompact_layer
# ---------------------------------------------------------------------------

class TestDecompactLayer:

    def test_incompressible_layer(self):
        """Salt (c=0): new bottom = new top + T_solid exactly."""
        T_s = solid_thickness(1000, 1500, 0.01, 0.0)
        z_bot = decompact_layer(0, T_s, 0.01, 0.0)
        assert z_bot == pytest.approx(T_s, abs=0.1)

    def test_decompact_to_surface_gives_thicker_layer(self):
        """Removing overburden → layer expands: new thickness > original."""
        phi0, c = 0.49, 0.00027
        z_top_orig, z_bot_orig = 2000.0, 3000.0   # 1000 m thick at depth
        T_s = solid_thickness(z_top_orig, z_bot_orig, phi0, c)
        z_bot_new = decompact_layer(0.0, T_s, phi0, c)
        new_thickness = z_bot_new - 0.0
        assert new_thickness > 1000.0, "decompacted layer must be thicker"

    def test_solid_thickness_preserved(self):
        """After decompaction, solid_thickness(new) == solid_thickness(orig)."""
        phi0, c = 0.63, 0.00051
        z_top_orig, z_bot_orig = 500.0, 1500.0
        T_s = solid_thickness(z_top_orig, z_bot_orig, phi0, c)
        z_bot_new = decompact_layer(0.0, T_s, phi0, c)
        T_s_check = solid_thickness(0.0, z_bot_new, phi0, c)
        assert T_s_check == pytest.approx(T_s, abs=0.05)

    def test_deeper_top_gives_thinner_layer(self):
        """With more overburden (deeper top), layer is more compacted → thinner."""
        phi0, c = 0.63, 0.00051
        T_s = 400.0
        z_bot_shallow = decompact_layer(0.0,    T_s, phi0, c)
        z_bot_deep    = decompact_layer(3000.0, T_s, phi0, c)
        thickness_shallow = z_bot_shallow - 0.0
        thickness_deep    = z_bot_deep    - 3000.0
        assert thickness_shallow > thickness_deep

    def test_zero_solid_thickness(self):
        z_bot = decompact_layer(1000.0, 0.0, 0.49, 0.00027)
        assert z_bot == pytest.approx(1000.0, abs=0.01)

    def test_roundtrip_decompact_recompact(self):
        """Decompact to surface then recompact to original depth → original bottom."""
        phi0, c = 0.49, 0.00027
        z_top_orig, z_bot_orig = 1500.0, 2500.0
        T_s = solid_thickness(z_top_orig, z_bot_orig, phi0, c)
        # Decompact to surface
        z_bot_surface = decompact_layer(0.0, T_s, phi0, c)
        T_s_check = solid_thickness(0.0, z_bot_surface, phi0, c)
        # Recompact to original top
        z_bot_back = decompact_layer(z_top_orig, T_s_check, phi0, c)
        assert z_bot_back == pytest.approx(z_bot_orig, abs=0.5)

    def test_limestone(self):
        """Limestone (phi0=0.40, c=0.00040) at 2000–3500m decompacts correctly."""
        phi0, c = 0.40, 0.00040
        T_s = solid_thickness(2000, 3500, phi0, c)
        z_bot = decompact_layer(0.0, T_s, phi0, c)
        assert z_bot > 1500.0  # thicker at surface than 1500 m


# ---------------------------------------------------------------------------
# 4. decompact_column
# ---------------------------------------------------------------------------

class TestDecompactColumn:

    @pytest.fixture
    def three_layers(self):
        return _make_layers(
            ("Unit A", 0,    500,  0.49, 0.00027),   # sand
            ("Unit B", 500,  1500, 0.63, 0.00051),   # shale
            ("Unit C", 1500, 3000, 0.40, 0.00040),   # limestone
        )

    def test_present_day_unchanged(self, three_layers):
        result = decompact_column(three_layers, target_time_index=0)
        assert len(result) == 3
        assert result[0]["z_top"]    == pytest.approx(0)
        assert result[0]["z_bottom"] == pytest.approx(500)
        assert result[2]["z_bottom"] == pytest.approx(3000)

    def test_present_day_thickness_fields(self, three_layers):
        result = decompact_column(three_layers, target_time_index=0)
        for r in result:
            assert r["decompacted_thickness"] == pytest.approx(r["original_thickness"])

    def test_remove_top_gives_two_layers(self, three_layers):
        result = decompact_column(three_layers, target_time_index=1)
        assert len(result) == 2
        assert result[0]["name"] == "Unit B"
        assert result[1]["name"] == "Unit C"

    def test_decompacted_top_is_at_surface(self, three_layers):
        result = decompact_column(three_layers, target_time_index=1)
        assert result[0]["z_top"] == pytest.approx(0.0)

    def test_layers_are_continuous(self, three_layers):
        """No gaps or overlaps between adjacent layers."""
        result = decompact_column(three_layers, target_time_index=1)
        assert result[1]["z_top"] == pytest.approx(result[0]["z_bottom"], abs=1e-6)

    def test_decompacted_unit_b_thicker_than_original(self, three_layers):
        """Unit B at surface (without Unit A's overburden) must be THICKER than
        its original 1000 m (less compaction without overburden pressure)."""
        result = decompact_column(three_layers, target_time_index=1)
        unit_b = result[0]
        assert unit_b["decompacted_thickness"] > unit_b["original_thickness"]
        assert unit_b["z_bottom"] > 1000.0

    def test_remove_two_layers(self, three_layers):
        result = decompact_column(three_layers, target_time_index=2)
        assert len(result) == 1
        assert result[0]["name"] == "Unit C"
        assert result[0]["z_top"] == pytest.approx(0.0)
        # Unit C should be thicker without overburden
        assert result[0]["decompacted_thickness"] > result[0]["original_thickness"]

    def test_remove_all_layers(self, three_layers):
        result = decompact_column(three_layers, target_time_index=3)
        assert result == []

    def test_solid_thickness_conserved(self, three_layers):
        """Solid thickness of each layer must not change after decompaction."""
        solids_before = [
            solid_thickness(lyr["z_top"], lyr["z_bottom"], lyr["phi0"], lyr["c"])
            for lyr in three_layers[1:]  # layers B and C
        ]
        result = decompact_column(three_layers, target_time_index=1)
        solids_after = [
            solid_thickness(lyr["z_top"], lyr["z_bottom"], lyr["phi0"], lyr["c"])
            for lyr in result
        ]
        for T_before, T_after in zip(solids_before, solids_after):
            assert T_after == pytest.approx(T_before, abs=0.1)

    def test_single_layer_column(self):
        layers = _make_layers(("Sand", 0, 1000, 0.49, 0.00027))
        result = decompact_column(layers, target_time_index=0)
        assert len(result) == 1
        assert result[0]["z_top"] == pytest.approx(0)
        assert result[0]["z_bottom"] == pytest.approx(1000)

    def test_salt_layer_preserves_thickness(self):
        """Incompressible salt layer: thickness stays ~500 m regardless of overburden."""
        layers = _make_layers(
            ("Overburden", 0,    1000, 0.49, 0.00027),
            ("Salt",       1000, 1500, 0.01, 0.0),     # c=0 → incompressible
        )
        result = decompact_column(layers, target_time_index=1)
        salt = result[0]
        salt_thickness = salt["z_bottom"] - salt["z_top"]
        assert salt_thickness == pytest.approx(500.0, abs=5.0)

    def test_depths_monotone_increasing(self):
        """All z_top / z_bottom values must increase with depth."""
        layers = _make_layers(
            ("A", 0,    500,  0.49, 0.00027),
            ("B", 500,  1200, 0.63, 0.00051),
            ("C", 1200, 2500, 0.40, 0.00040),
            ("D", 2500, 4000, 0.70, 0.00071),
        )
        result = decompact_column(layers, target_time_index=1)
        depths = [r["z_top"] for r in result] + [result[-1]["z_bottom"]]
        for a, b in zip(depths, depths[1:]):
            assert b > a, f"depth not increasing: {a} → {b}"


# ---------------------------------------------------------------------------
# 5. burial_history
# ---------------------------------------------------------------------------

class TestBurialHistory:

    @pytest.fixture
    def two_layers(self):
        return _make_layers(
            ("A", 0,   500,  0.49, 0.00027),
            ("B", 500, 1500, 0.63, 0.00051),
        )

    @pytest.fixture
    def three_layers(self):
        return _make_layers(
            ("A", 0,    500,  0.49, 0.00027),
            ("B", 500,  1500, 0.63, 0.00051),
            ("C", 1500, 3000, 0.40, 0.00040),
        )

    def test_shape_two_layers(self, two_layers):
        depths = burial_history(two_layers)
        assert depths.shape == (3, 3)   # n_steps=3, n_boundaries=3

    def test_shape_three_layers(self, three_layers):
        depths = burial_history(three_layers)
        assert depths.shape == (4, 4)

    def test_shape_single_layer(self):
        layers = _make_layers(("X", 0, 1000, 0.49, 0.00027))
        depths = burial_history(layers)
        assert depths.shape == (2, 2)

    def test_present_day_surface_is_zero(self, two_layers):
        depths = burial_history(two_layers)
        assert depths[0, 0] == pytest.approx(0.0)

    def test_present_day_matches_input(self, two_layers):
        depths = burial_history(two_layers)
        # boundaries at step 0 should match input
        assert depths[0, 1] == pytest.approx(500.0)   # A/B interface
        assert depths[0, 2] == pytest.approx(1500.0)  # B base

    def test_step1_b_top_at_surface(self, two_layers):
        """After stripping A, B's top is decompacted to the paleo-surface."""
        depths = burial_history(two_layers)
        assert depths[1, 1] == pytest.approx(0.0)

    def test_step1_b_base_shallower_than_present(self, two_layers):
        """B's absolute base depth is shallower when A is removed (less burial)."""
        depths = burial_history(two_layers)
        assert depths[1, 2] < depths[0, 2]

    def test_step1_b_thicker_than_present(self, two_layers):
        """B's thickness at step 1 is greater than present (decompacted)."""
        depths = burial_history(two_layers)
        thickness_present    = depths[0, 2] - depths[0, 1]   # 1000 m
        thickness_decompacted = depths[1, 2] - depths[1, 1]
        assert thickness_decompacted > thickness_present

    def test_salt_in_burial_history(self):
        """Incompressible salt thickness is constant across all time steps."""
        layers = _make_layers(
            ("Overburd", 0,    1000, 0.49, 0.00027),
            ("Salt",     1000, 1500, 0.01, 0.0),
        )
        depths = burial_history(layers)
        # Step 1: only Salt remains, decompacted
        salt_thickness_step1 = depths[1, 2] - depths[1, 1]
        assert salt_thickness_step1 == pytest.approx(500.0, abs=5.0)

    def test_zero_depth_cells_for_undeposited_layers(self, two_layers):
        """Boundaries of layers not yet deposited stay at zero."""
        depths = burial_history(two_layers)
        # At step 2, neither layer exists → all zero
        assert depths[2, 0] == pytest.approx(0.0)
        assert depths[2, 1] == pytest.approx(0.0)
        assert depths[2, 2] == pytest.approx(0.0)

    def test_four_layer_column_shapes(self):
        layers = _make_layers(
            ("A", 0,    200,  0.49, 0.00027),
            ("B", 200,  700,  0.63, 0.00051),
            ("C", 700,  1800, 0.40, 0.00040),
            ("D", 1800, 3500, 0.70, 0.00071),
        )
        depths = burial_history(layers)
        assert depths.shape == (5, 5)
        # At step 0, boundaries are at the input depths
        assert depths[0, 0] == pytest.approx(0.0)
        assert depths[0, 4] == pytest.approx(3500.0)

    def test_non_negative_depths(self, three_layers):
        """All depths must be >= 0."""
        depths = burial_history(three_layers)
        assert np.all(depths >= 0)

    def test_depths_non_decreasing_in_space(self, three_layers):
        """Within each time step, boundary depths are non-decreasing."""
        depths = burial_history(three_layers)
        for step in range(depths.shape[0]):
            row = depths[step]
            for i in range(len(row) - 1):
                assert row[i] <= row[i + 1] + 1e-6, (
                    f"step={step}, boundaries {i} and {i+1} not non-decreasing: "
                    f"{row[i]} > {row[i+1]}"
                )


# ---------------------------------------------------------------------------
# 6. Physics validation — self-consistent checks
# ---------------------------------------------------------------------------

class TestPhysicsConsistency:

    def test_solid_invariant_through_burial_history(self):
        """Solid thickness must be conserved across all time steps."""
        layers = _make_layers(
            ("A", 0,    500,  0.49, 0.00027),
            ("B", 500,  1500, 0.63, 0.00051),
            ("C", 1500, 3000, 0.40, 0.00040),
        )
        depths = burial_history(layers)
        n_layers = len(layers)

        # Solid thickness of layer i in present day
        T_s_present = [
            solid_thickness(layers[i]["z_top"], layers[i]["z_bottom"],
                            layers[i]["phi0"], layers[i]["c"])
            for i in range(n_layers)
        ]

        # Check each decompacted step
        for step in range(1, n_layers + 1):
            surviving_start = step
            for i, lyr in enumerate(layers[surviving_start:]):
                orig_i = surviving_start + i
                z_t = depths[step, orig_i]
                z_b = depths[step, orig_i + 1]
                if z_b <= z_t:
                    continue  # undeposited boundary
                T_s_step = solid_thickness(z_t, z_b, lyr["phi0"], lyr["c"])
                assert T_s_step == pytest.approx(T_s_present[orig_i], abs=0.5), (
                    f"Layer {lyr['name']} solid thickness changed at step {step}: "
                    f"{T_s_present[orig_i]:.2f} → {T_s_step:.2f}"
                )

    def test_porosity_athy_integral_matches_solid_thickness(self):
        """Numerical integral of (1-phi) must match analytical solid_thickness."""
        phi0, c = 0.63, 0.00051
        z_top, z_bottom = 0.0, 2000.0
        T_analytic = solid_thickness(z_top, z_bottom, phi0, c)
        z = np.linspace(z_top, z_bottom, 100_000)
        T_numeric = np.trapezoid(1.0 - porosity_athy(z, phi0, c), z)
        assert T_analytic == pytest.approx(T_numeric, rel=1e-4)

    def test_decompaction_increases_porosity(self):
        """Moving a layer to shallower depth increases its average porosity."""
        phi0, c = 0.63, 0.00051
        z_top_deep, z_bot_deep = 2000.0, 3000.0
        T_s = solid_thickness(z_top_deep, z_bot_deep, phi0, c)

        z_bot_surface = decompact_layer(0.0, T_s, phi0, c)
        thickness_deep    = z_bot_deep    - z_top_deep
        thickness_surface = z_bot_surface - 0.0
        # More porosity at surface → greater thickness
        assert thickness_surface > thickness_deep

    def test_single_layer_burial_history(self):
        """Single-layer column: step 0 = present, step 1 = empty."""
        layers = _make_layers(("X", 0, 500, 0.49, 0.00027))
        depths = burial_history(layers)
        assert depths.shape == (2, 2)
        assert depths[0, 0] == pytest.approx(0.0)
        assert depths[0, 1] == pytest.approx(500.0)
        assert depths[1, 0] == pytest.approx(0.0)   # no layer → zeros
        assert depths[1, 1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 7. Spec examples (verbatim, with correct physical assertions)
# ---------------------------------------------------------------------------

class TestSpecExamples:

    def test_porosity_athy_spec(self):
        assert porosity_athy(0,      0.5, 0.0005) == pytest.approx(0.5)
        assert porosity_athy(10_000, 0.5, 0.0005) < 0.01

    def test_solid_thickness_incompressible_spec(self):
        assert solid_thickness(0, 1000, 0.01, 0.0) == pytest.approx(1000, abs=1)

    def test_solid_thickness_compactible_spec(self):
        T_s = solid_thickness(0, 1000, 0.63, 0.00051)
        assert T_s < 1000
        assert T_s > 0

    def test_decompact_to_surface_spec(self):
        """Layer buried at 2000–3000 m, decompacted to surface."""
        T_s = solid_thickness(2000, 3000, 0.49, 0.00027)
        z_bottom = decompact_layer(0, T_s, 0.49, 0.00027)
        # Decompacted thickness > original 1000 m (less compaction at surface)
        assert z_bottom > 1000

    def test_decompact_column_spec(self):
        layers = [
            {"name": "Unit A", "z_top": 0,    "z_bottom": 500,
             "phi0": 0.49, "c": 0.00027, "age_top": 0,  "age_base": 10},
            {"name": "Unit B", "z_top": 500,  "z_bottom": 1500,
             "phi0": 0.63, "c": 0.00051, "age_top": 10, "age_base": 50},
            {"name": "Unit C", "z_top": 1500, "z_bottom": 3000,
             "phi0": 0.40, "c": 0.00040, "age_top": 50, "age_base": 100},
        ]
        result = decompact_column(layers, target_time_index=1)
        assert len(result) == 2
        assert result[0]["z_top"] == pytest.approx(0.0)        # new surface
        # Unit B is THICKER without overburden (decompacted, not thinner):
        assert result[0]["z_bottom"] > 1000.0
        assert result[1]["z_top"] == pytest.approx(result[0]["z_bottom"], abs=1e-6)

    def test_burial_history_spec(self):
        layers = [
            {"name": "A", "z_top": 0,   "z_bottom": 500,
             "phi0": 0.49, "c": 0.00027, "age_top": 0,  "age_base": 10},
            {"name": "B", "z_top": 500, "z_bottom": 1500,
             "phi0": 0.63, "c": 0.00051, "age_top": 10, "age_base": 50},
        ]
        depths = burial_history(layers)
        assert depths.shape == (3, 3)
        assert depths[0, 0] == pytest.approx(0.0)   # present surface
        assert depths[1, 1] == pytest.approx(0.0)   # B top at surface in step 1
        assert depths[1, 2] < depths[0, 2]          # B base is shallower when A absent

    def test_salt_incompressible_spec(self):
        layers = [
            {"name": "Overburden", "z_top": 0,    "z_bottom": 1000,
             "phi0": 0.49, "c": 0.00027, "age_top": 0,  "age_base": 10},
            {"name": "Salt",       "z_top": 1000, "z_bottom": 1500,
             "phi0": 0.01, "c": 0.0,     "age_top": 10, "age_base": 50},
        ]
        result = decompact_column(layers, target_time_index=1)
        salt_thickness = result[0]["z_bottom"] - result[0]["z_top"]
        assert salt_thickness == pytest.approx(500, abs=5)
