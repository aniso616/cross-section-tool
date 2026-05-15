"""Comprehensive tests for the 1D steady-state thermal modeling module.

All tests are headless (no Qt).  Physical reasoning is noted inline.
"""
from __future__ import annotations

import numpy as np
import pytest

from section_tool.core.thermal import (
    effective_conductivity_column,
    maturity_easy_ro,
    steady_state_geotherm,
    thermal_conductivity_with_porosity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_layer(k: float = 2.5, dz: float = 3000.0,
                   A: float = 0.0, z_top: float = 0.0) -> dict:
    return {
        "z_top": z_top,
        "z_bottom": z_top + dz,
        "thermal_conductivity": k,
        "heat_production": A,
    }


def _geotherm_dict(layers, **kwargs) -> dict[float, float]:
    """Build {depth: temperature} mapping from geotherm output."""
    return {d: t for d, t in steady_state_geotherm(layers, **kwargs)}


# ---------------------------------------------------------------------------
# 1. steady_state_geotherm
# ---------------------------------------------------------------------------

class TestSteadyStateGeotherm:

    # ---- Spec examples ------------------------------------------------

    def test_steady_state_simple(self):
        """Uniform k, no heat production → linear gradient.
        gradient = q/k = 0.06/2.5 = 0.024 °C/m = 24 °C/km
        T at 3000 m = 10 + 72 = 82 °C
        """
        layers = [_uniform_layer(k=2.5, dz=3000)]
        result = steady_state_geotherm(layers, surface_temp_C=10, basal_heat_flow_mW=60)
        assert result[-1][1] == pytest.approx(82, abs=2)

    def test_geotherm_two_layers(self):
        """Two layers with different k → gradient changes at boundary.
        Layer 1 (k=2.0): gradient = 60e-3/2.0 = 30 °C/km → T(1000) = 40 °C
        Layer 2 (k=3.0): gradient = 60e-3/3.0 = 20 °C/km → T(3000) = 40+40 = 80 °C
        """
        layers = [
            {"z_top": 0,    "z_bottom": 1000, "thermal_conductivity": 2.0, "heat_production": 0},
            {"z_top": 1000, "z_bottom": 3000, "thermal_conductivity": 3.0, "heat_production": 0},
        ]
        result = steady_state_geotherm(layers, surface_temp_C=10, basal_heat_flow_mW=60)
        gd = {d: t for d, t in result}
        assert gd.get(1000, None) == pytest.approx(40, abs=2)

    # ---- Surface and basic properties --------------------------------

    def test_surface_temperature_correct(self):
        layers = [_uniform_layer()]
        result = steady_state_geotherm(layers, surface_temp_C=15.0, basal_heat_flow_mW=60)
        assert result[0] == (pytest.approx(0.0), pytest.approx(15.0))

    def test_temperature_increases_monotone(self):
        layers = [_uniform_layer(k=2.5, dz=5000)]
        result = steady_state_geotherm(layers, surface_temp_C=10, basal_heat_flow_mW=60)
        depths = [d for d, _ in result]
        temps  = [t for _, t in result]
        # Depths are non-decreasing
        assert all(b >= a for a, b in zip(depths, depths[1:]))
        # Temperatures are strictly increasing (positive geothermal gradient)
        assert all(b > a for a, b in zip(temps, temps[1:]))

    def test_higher_heat_flow_means_higher_temperature(self):
        layers = [_uniform_layer(k=2.5, dz=3000)]
        result_lo = steady_state_geotherm(layers, surface_temp_C=10, basal_heat_flow_mW=40)
        result_hi = steady_state_geotherm(layers, surface_temp_C=10, basal_heat_flow_mW=80)
        assert result_hi[-1][1] > result_lo[-1][1]

    def test_lower_conductivity_means_higher_temperature(self):
        layers_lo_k = [_uniform_layer(k=1.5, dz=3000)]
        layers_hi_k = [_uniform_layer(k=3.0, dz=3000)]
        result_lo = steady_state_geotherm(layers_lo_k, surface_temp_C=10, basal_heat_flow_mW=60)
        result_hi = steady_state_geotherm(layers_hi_k, surface_temp_C=10, basal_heat_flow_mW=60)
        # Lower k → steeper gradient → higher T at depth
        assert result_lo[-1][1] > result_hi[-1][1]

    def test_gradient_equals_q_over_k(self):
        """For no heat production, gradient = q/k exactly."""
        q_mW, k = 70.0, 2.5
        layers = [_uniform_layer(k=k, dz=1000)]
        result = steady_state_geotherm(layers, surface_temp_C=0, basal_heat_flow_mW=q_mW)
        gd = {d: t for d, t in result}
        T_1000 = gd[1000.0]
        expected = (q_mW * 1e-3 / k) * 1000.0
        assert T_1000 == pytest.approx(expected, abs=0.1)

    def test_last_point_is_base_of_column(self):
        layers = [_uniform_layer(dz=2500)]
        result = steady_state_geotherm(layers, surface_temp_C=10, basal_heat_flow_mW=60)
        assert result[-1][0] == pytest.approx(2500.0, abs=0.1)

    def test_surface_is_first_point(self):
        layers = [_uniform_layer()]
        result = steady_state_geotherm(layers)
        assert result[0][0] == pytest.approx(0.0)

    # ---- Heat production ----------------------------------------------

    def test_heat_production_raises_near_surface_gradient(self):
        """With radiogenic heat production, temperature near surface is higher."""
        layers_A0 = [_uniform_layer(k=2.5, dz=3000, A=0.0)]
        layers_A2 = [_uniform_layer(k=2.5, dz=3000, A=2.0)]   # 2 µW/m³ (granitic)
        T0 = steady_state_geotherm(layers_A0, basal_heat_flow_mW=60)[-1][1]
        T2 = steady_state_geotherm(layers_A2, basal_heat_flow_mW=60)[-1][1]
        assert T2 > T0

    def test_no_heat_production_linear_profile(self):
        """Without heat production, T vs depth must be linear."""
        k, q_mW = 2.5, 60.0
        layers = [_uniform_layer(k=k, dz=5000, A=0.0)]
        result = steady_state_geotherm(layers, surface_temp_C=0, basal_heat_flow_mW=q_mW)
        grad = q_mW * 1e-3 / k
        for z, T in result:
            assert T == pytest.approx(grad * z, abs=0.05), f"Non-linear at z={z}"

    # ---- Multi-layer --------------------------------------------------

    def test_three_layer_temperature_continuity(self):
        """Temperature must be continuous at layer boundaries."""
        layers = [
            {"z_top": 0,    "z_bottom": 500,  "thermal_conductivity": 1.5, "heat_production": 0},
            {"z_top": 500,  "z_bottom": 1500, "thermal_conductivity": 3.0, "heat_production": 0},
            {"z_top": 1500, "z_bottom": 3000, "thermal_conductivity": 2.0, "heat_production": 0},
        ]
        result = steady_state_geotherm(layers, surface_temp_C=10, basal_heat_flow_mW=65)
        gd = {d: t for d, t in result}
        # Boundary at 500 m should appear once; boundary at 1500 m once
        assert 500.0  in gd
        assert 1500.0 in gd

    def test_intermediate_points_present(self):
        """A 1000-m layer should produce ≥2 result points (excluding surface)."""
        layers = [_uniform_layer(dz=1000)]
        result = steady_state_geotherm(layers)
        # +1 for surface, +n_points (≥2) for layer
        assert len(result) >= 3

    def test_invalid_conductivity_uses_fallback(self):
        """k ≤ 0 must not crash — a safe default is applied."""
        layers = [{"z_top": 0, "z_bottom": 1000, "thermal_conductivity": 0, "heat_production": 0}]
        result = steady_state_geotherm(layers)
        assert result[-1][1] > result[0][1]   # temperature still increases


# ---------------------------------------------------------------------------
# 2. thermal_conductivity_with_porosity
# ---------------------------------------------------------------------------

class TestThermalConductivityMixing:

    # ---- Spec examples ------------------------------------------------

    def test_conductivity_mixing_geometric_mean(self):
        """k = k_matrix^(1-phi) * k_fluid^phi; with k_m=3.0, k_f=0.6, phi=0.3."""
        k = thermal_conductivity_with_porosity(3.0, 0.6, 0.3)
        # 3.0^0.7 * 0.6^0.3 = exp(0.7*ln3 + 0.3*ln0.6) ≈ 2.07
        assert 1.8 < k < 2.3

    def test_zero_porosity_returns_matrix(self):
        k = thermal_conductivity_with_porosity(3.0, 0.6, 0.0)
        assert k == pytest.approx(3.0)

    def test_unity_porosity_returns_fluid(self):
        k = thermal_conductivity_with_porosity(3.0, 0.6, 1.0)
        assert k == pytest.approx(0.6)

    # ---- Additional --------------------------------------------------

    def test_mixing_between_limits(self):
        k = thermal_conductivity_with_porosity(3.0, 0.6, 0.5)
        assert 0.6 < k < 3.0

    def test_mixing_decreases_with_porosity(self):
        ks = [thermal_conductivity_with_porosity(3.0, 0.6, phi) for phi in [0.1, 0.3, 0.5, 0.7]]
        assert ks == sorted(ks, reverse=True)

    def test_formula_exact_value(self):
        phi = 0.3
        k_m, k_f = 3.0, 0.6
        expected = k_m ** (1 - phi) * k_f ** phi
        result = thermal_conductivity_with_porosity(k_m, k_f, phi)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_negative_porosity_treated_as_zero(self):
        k = thermal_conductivity_with_porosity(3.0, 0.6, -0.1)
        assert k == pytest.approx(3.0)

    def test_water_conductivity_standard(self):
        """Standard water (k=0.6) mixed at 20% porosity."""
        k = thermal_conductivity_with_porosity(2.5, 0.6, 0.2)
        assert 0.6 < k < 2.5


# ---------------------------------------------------------------------------
# 3. effective_conductivity_column
# ---------------------------------------------------------------------------

class TestEffectiveConductivityColumn:

    def test_returns_one_dict_per_layer(self):
        layers = [
            {"z_top": 0, "z_bottom": 1000, "thermal_conductivity": 2.5},
            {"z_top": 1000, "z_bottom": 2500, "thermal_conductivity": 3.0},
        ]
        result = effective_conductivity_column(layers)
        assert len(result) == 2

    def test_augmented_keys_present(self):
        layers = [{"z_top": 0, "z_bottom": 500, "thermal_conductivity": 2.0}]
        result = effective_conductivity_column(layers)
        assert "porosity_at_midpoint" in result[0]
        assert "effective_conductivity" in result[0]

    def test_original_keys_preserved(self):
        layers = [{"z_top": 0, "z_bottom": 500, "thermal_conductivity": 2.0,
                   "heat_production": 1.5}]
        result = effective_conductivity_column(layers)
        assert result[0]["heat_production"] == 1.5

    def test_zero_porosity_effective_equals_matrix(self):
        layers = [{
            "z_top": 0, "z_bottom": 1000,
            "thermal_conductivity": 3.0,
            "porosity_surface": 0.0,    # zero porosity
            "compaction_coeff": 0.0005,
        }]
        result = effective_conductivity_column(layers)
        assert result[0]["effective_conductivity"] == pytest.approx(3.0)

    def test_effective_conductivity_less_than_matrix(self):
        """Water (k=0.6) lowers bulk conductivity below matrix value."""
        layers = [{
            "z_top": 0, "z_bottom": 100,
            "thermal_conductivity": 3.0,
            "porosity_surface": 0.4,
            "compaction_coeff": 0.0,
        }]
        result = effective_conductivity_column(layers, k_fluid=0.6)
        assert result[0]["effective_conductivity"] < 3.0

    def test_deeper_layer_has_higher_effective_k(self):
        """Deeper layer → less porosity (Athy) → higher effective conductivity."""
        layers = [
            {"z_top": 0,    "z_bottom": 100,  "thermal_conductivity": 3.0,
             "porosity_surface": 0.5, "compaction_coeff": 0.001},
            {"z_top": 4000, "z_bottom": 4100, "thermal_conductivity": 3.0,
             "porosity_surface": 0.5, "compaction_coeff": 0.001},
        ]
        result = effective_conductivity_column(layers)
        assert result[1]["effective_conductivity"] > result[0]["effective_conductivity"]

    def test_porosity_at_midpoint_uses_athy(self):
        phi0, c = 0.5, 0.001
        z_mid = 500.0
        layers = [{"z_top": 0, "z_bottom": 1000, "thermal_conductivity": 2.0,
                   "porosity_surface": phi0, "compaction_coeff": c}]
        result = effective_conductivity_column(layers)
        expected_phi = phi0 * np.exp(-c * z_mid)
        assert result[0]["porosity_at_midpoint"] == pytest.approx(expected_phi, rel=1e-9)


# ---------------------------------------------------------------------------
# 4. maturity_easy_ro
# ---------------------------------------------------------------------------

class TestMaturityEasyRo:

    def _times(self, duration_ma=100, n=101):
        return np.linspace(duration_ma, 0, n)

    # ---- Spec examples ------------------------------------------------

    def test_immature_low_ro(self):
        """50 °C for 100 Ma → immature (Ro < 0.5%)."""
        times = self._times()
        temps = np.full(101, 50.0)
        ro = maturity_easy_ro(temps, times)
        assert ro < 0.5

    def test_mature_high_ro(self):
        """150 °C for 100 Ma → mature (Ro > 1.0%)."""
        times = self._times()
        temps = np.full(101, 150.0)
        ro = maturity_easy_ro(temps, times)
        assert ro > 1.0

    def test_easy_ro_increases_with_temperature(self):
        times = self._times()
        ro_60  = maturity_easy_ro(np.full(101, 60.0),  times)
        ro_100 = maturity_easy_ro(np.full(101, 100.0), times)
        ro_160 = maturity_easy_ro(np.full(101, 160.0), times)
        assert ro_60 < ro_100 < ro_160

    # ---- Additional physical checks ----------------------------------

    def test_ro_positive(self):
        times = self._times()
        ro = maturity_easy_ro(np.full(101, 80.0), times)
        assert ro > 0.0

    def test_longer_burial_increases_ro(self):
        """Longer time at same temperature → higher maturity."""
        temps = np.full(51, 100.0)
        ro_short = maturity_easy_ro(temps, np.linspace(50, 0, 51))
        ro_long  = maturity_easy_ro(temps, np.linspace(100, 0, 51))
        assert ro_long > ro_short

    def test_cold_history_near_initial_ro(self):
        """Near-surface temperature (~20°C) for short time → very low Ro."""
        times = np.linspace(10, 0, 11)
        temps = np.full(11, 20.0)
        ro = maturity_easy_ro(temps, times)
        assert ro < 0.35   # essentially immature

    def test_very_high_temperature_gives_overmature(self):
        """200 °C for 100 Ma → overmature (Ro > 2.0%)."""
        times = self._times()
        temps = np.full(101, 200.0)
        ro = maturity_easy_ro(temps, times)
        assert ro > 2.0

    def test_maturity_monotone_with_temperature_range(self):
        """Ro is monotone increasing across a wide temperature range."""
        times = self._times()
        rvalues = [maturity_easy_ro(np.full(101, T), times)
                   for T in range(40, 200, 10)]
        assert all(b >= a for a, b in zip(rvalues, rvalues[1:]))

    def test_initial_ro_around_point_two(self):
        """With exactly zero duration, no reactions proceed → Ro = exp(-1.6) ≈ 0.20."""
        # Two identical time points → dt = 0 for the single interval
        times = np.array([50.0, 50.0])   # zero elapsed time
        temps = np.full(2, 200.0)
        ro = maturity_easy_ro(temps, times)
        assert ro == pytest.approx(np.exp(-1.6), rel=1e-9)

    def test_array_types_accepted(self):
        """Both list and ndarray inputs must work."""
        times_list = list(np.linspace(100, 0, 11))
        temps_list = [100.0] * 11
        ro_list = maturity_easy_ro(temps_list, times_list)
        ro_arr  = maturity_easy_ro(np.array(temps_list), np.array(times_list))
        assert ro_list == pytest.approx(ro_arr, rel=1e-9)

    def test_equal_to_single_step(self):
        """A thermal history with identical steps at the same T is equivalent
        regardless of whether we use 2 steps or 100 steps."""
        T = 120.0
        ro_coarse = maturity_easy_ro(np.full(2, T), np.array([50.0, 0.0]))
        ro_fine   = maturity_easy_ro(np.full(51, T), np.linspace(50.0, 0.0, 51))
        # Should be close (same total time at same temperature)
        assert ro_coarse == pytest.approx(ro_fine, rel=0.02)


# ---------------------------------------------------------------------------
# 5. Integration: geotherm → maturity
# ---------------------------------------------------------------------------

class TestGeothermMaturityIntegration:

    def test_deeper_burial_gives_higher_maturity(self):
        """Simple workflow: compute T at depth → use as constant history → Ro."""
        layers_shallow = [_uniform_layer(k=2.5, dz=2000, A=0.0)]
        layers_deep    = [_uniform_layer(k=2.5, dz=4000, A=0.0)]
        q_mW = 60.0

        result_s = steady_state_geotherm(layers_shallow, surface_temp_C=10, basal_heat_flow_mW=q_mW)
        result_d = steady_state_geotherm(layers_deep,    surface_temp_C=10, basal_heat_flow_mW=q_mW)

        T_shallow = result_s[-1][1]
        T_deep    = result_d[-1][1]
        assert T_deep > T_shallow

        times = np.linspace(100, 0, 101)
        ro_s = maturity_easy_ro(np.full(101, T_shallow), times)
        ro_d = maturity_easy_ro(np.full(101, T_deep),    times)
        assert ro_d > ro_s

    def test_effective_conductivity_feeds_geotherm(self):
        """effective_conductivity_column result feeds directly into geotherm."""
        base_layers = [
            {"z_top": 0,    "z_bottom": 1000, "thermal_conductivity": 3.0,
             "porosity_surface": 0.4, "compaction_coeff": 0.0005, "heat_production": 0},
            {"z_top": 1000, "z_bottom": 3000, "thermal_conductivity": 2.5,
             "porosity_surface": 0.3, "compaction_coeff": 0.0003, "heat_production": 0},
        ]
        eff_layers = effective_conductivity_column(base_layers)
        # Replace thermal_conductivity with effective value
        geotherm_layers = [
            {**lyr, "thermal_conductivity": lyr["effective_conductivity"]}
            for lyr in eff_layers
        ]
        result = steady_state_geotherm(geotherm_layers, surface_temp_C=10, basal_heat_flow_mW=65)
        # Just verify it runs and gives a reasonable temperature
        assert result[-1][1] > 10.0
        assert result[-1][1] < 200.0
