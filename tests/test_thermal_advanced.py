"""Tests for advanced thermal modeling: transient solver, AFT/AHe/ZHe kinetics,
and Monte Carlo inverse modeling.

All headless — no Qt, no GUI.
"""
from __future__ import annotations

import numpy as np
import pytest

from section_tool.core.thermal import (
    aft_age,
    aft_track_length_distribution,
    ahe_age,
    transient_1d_heat,
    zhe_age,
)
from section_tool.core.thermal_inverse import (
    _predict_observation,
    good_paths_envelope,
    monte_carlo_search,
)


# ---------------------------------------------------------------------------
# Transient heat solver
# ---------------------------------------------------------------------------

def _simple_column(n=20):
    depths = np.linspace(0.0, 5000.0, n)
    T0 = 10.0 + depths * 0.025    # 25 °C/km gradient
    return depths, T0


def test_transient_shape():
    depths, T0 = _simple_column()
    n = len(depths)
    times = np.array([100.0, 50.0, 0.0])
    result = transient_1d_heat(depths, T0, None, times, thermal_conductivity=2.5)
    assert result.shape == (3, n)


def test_transient_surface_bc():
    depths, T0 = _simple_column()
    times = np.array([50.0, 0.0])
    result = transient_1d_heat(depths, T0, None, times, thermal_conductivity=2.5,
                               surface_temp_C=15.0)
    # Surface node should stay at surface_temp_C at all time steps
    assert result[0, 0] == pytest.approx(T0[0], abs=1.0)  # initial
    assert result[1, 0] == pytest.approx(15.0, abs=0.1)   # after step


def test_transient_initial_preserved_on_zero_steps():
    depths, T0 = _simple_column()
    # Only one time step → shape is (1, n), no evolution
    result = transient_1d_heat(depths, T0, None, np.array([0.0]), thermal_conductivity=2.5)
    assert result.shape == (1, len(depths))
    np.testing.assert_allclose(result[0], T0, rtol=1e-10)


def test_transient_temperatures_positive():
    depths, T0 = _simple_column()
    times = np.array([200.0, 100.0, 50.0, 0.0])
    result = transient_1d_heat(depths, T0, None, times, thermal_conductivity=2.5,
                               surface_temp_C=10.0)
    # All temperatures should be physically reasonable (not NaN or negative)
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)


def test_transient_with_burial_history():
    n = 10
    depths = np.linspace(0.0, 1000.0, n)
    T0 = 10.0 + depths * 0.03
    times = np.array([50.0, 0.0])
    bh = np.zeros((2, n))
    bh[0] = depths * 0.5
    bh[1] = depths
    result = transient_1d_heat(depths, T0, bh, times, thermal_conductivity=2.5)
    assert result.shape == (2, n)
    assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# AFT kinetics
# ---------------------------------------------------------------------------

def test_aft_age_immature():
    """Cold history — AFT age should be close to initial age."""
    times = np.array([100.0, 0.0])
    temps = np.array([20.0, 20.0])
    age = aft_age(temps, times, initial_age_ma=100)
    assert age > 80, f"Expected > 80 Ma, got {age:.2f} Ma"


def test_aft_age_reset():
    """Hot history — AFT age should be significantly reduced."""
    times = np.array([100.0, 50.0, 0.0])
    temps = np.array([20.0, 150.0, 20.0])
    age = aft_age(temps, times, initial_age_ma=100)
    assert age < 60, f"Expected < 60 Ma, got {age:.2f} Ma"


def test_aft_age_positive():
    """AFT age is always non-negative."""
    times = np.array([200.0, 100.0, 50.0, 0.0])
    temps = np.array([30.0, 180.0, 120.0, 25.0])
    age = aft_age(temps, times, initial_age_ma=200)
    assert age >= 0.0


def test_aft_age_zero_initial():
    times = np.array([50.0, 0.0])
    temps = np.array([50.0, 50.0])
    age = aft_age(temps, times, initial_age_ma=0)
    assert age == pytest.approx(0.0)


def test_aft_age_single_step():
    times = np.array([10.0, 0.0])
    temps = np.array([30.0, 30.0])
    age = aft_age(temps, times, initial_age_ma=50)
    assert 0.0 <= age <= 50.0


def test_aft_hotter_more_annealing():
    """Higher peak temperature → more annealing → lower age."""
    times = np.array([50.0, 0.0])
    age_cold = aft_age(np.array([50.0, 50.0]),  times, initial_age_ma=50)
    age_hot  = aft_age(np.array([150.0, 150.0]), times, initial_age_ma=50)
    assert age_hot < age_cold


# ---------------------------------------------------------------------------
# AFT track length distribution
# ---------------------------------------------------------------------------

def test_aft_tld_returns_array():
    times = np.array([100.0, 50.0, 0.0])
    temps = np.array([20.0, 80.0, 20.0])
    lengths = aft_track_length_distribution(temps, times, n_tracks=50)
    assert isinstance(lengths, np.ndarray)
    assert len(lengths) > 0


def test_aft_tld_lengths_positive():
    times = np.array([50.0, 0.0])
    temps = np.array([20.0, 20.0])
    lengths = aft_track_length_distribution(temps, times, n_tracks=30)
    assert np.all(lengths >= 0.0)


def test_aft_tld_cold_near_l0():
    """Cold history → track lengths near initial l0 (≈ 16.3 µm)."""
    times = np.array([50.0, 0.0])
    temps = np.array([10.0, 10.0])
    lengths = aft_track_length_distribution(temps, times, n_tracks=20)
    assert np.mean(lengths) > 10.0   # near initial length


# ---------------------------------------------------------------------------
# AHe kinetics
# ---------------------------------------------------------------------------

def test_ahe_age_basic():
    """AHe age should be positive for any thermal history."""
    times = np.array([50.0, 0.0])
    temps = np.array([30.0, 30.0])
    age = ahe_age(temps, times)
    assert age > 0.0
    assert age < 100.0


def test_ahe_age_nonnegative():
    times = np.array([200.0, 100.0, 50.0, 0.0])
    temps = np.array([40.0, 200.0, 100.0, 30.0])
    age = ahe_age(temps, times)
    assert age >= 0.0


def test_ahe_age_cold_large():
    """Very cold history → high He retention → age ≈ total time span."""
    times = np.array([50.0, 0.0])
    temps = np.array([5.0, 5.0])
    age = ahe_age(temps, times, ft_correction=False)
    # Should retain nearly all He — apparent age ≈ 50 Ma
    assert age > 30.0


def test_ahe_hotter_younger():
    """Hotter history → more diffusion loss → younger apparent age."""
    times = np.array([50.0, 0.0])
    age_cold = ahe_age(np.array([20.0, 20.0]), times, ft_correction=False)
    age_hot  = ahe_age(np.array([90.0, 90.0]), times, ft_correction=False)
    assert age_hot < age_cold


def test_ahe_single_step():
    times = np.array([10.0, 0.0])
    temps = np.array([25.0, 25.0])
    age = ahe_age(temps, times)
    assert 0.0 <= age <= 50.0


# ---------------------------------------------------------------------------
# ZHe kinetics
# ---------------------------------------------------------------------------

def test_zhe_age_basic():
    times = np.array([50.0, 0.0])
    temps = np.array([50.0, 50.0])
    age = zhe_age(temps, times)
    assert age >= 0.0


def test_zhe_higher_tc_than_ahe():
    """ZHe closure T ≈ 200°C > AHe ≈ 70°C → ZHe age ≥ AHe age for same history."""
    times = np.array([100.0, 0.0])
    temps = np.array([80.0, 80.0])   # between AHe Tc and ZHe Tc
    ahe = ahe_age(temps, times, ft_correction=False)
    zhe = zhe_age(temps, times, ft_correction=False)
    assert zhe >= ahe


def test_zhe_nonnegative():
    times = np.array([200.0, 0.0])
    temps = np.array([300.0, 20.0])
    age = zhe_age(temps, times)
    assert age >= 0.0


# ---------------------------------------------------------------------------
# Monte Carlo inverse modeling
# ---------------------------------------------------------------------------

def test_monte_carlo_finds_solution():
    """MC search should find paths consistent with a synthetic AFT observation."""
    true_times = np.array([100.0, 50.0, 0.0])
    true_temps = np.array([20.0,  80.0, 20.0])

    synthetic_aft = aft_age(true_temps, true_times)
    observations = [
        {"type": "AFT", "value": synthetic_aft, "uncertainty": max(synthetic_aft * 0.3, 10.0)},
    ]

    paths = monte_carlo_search(
        observations,
        time_bounds_ma=(100.0, 0.0),
        temp_bounds_C=(10.0, 180.0),
        n_paths=2000,
        acceptance_threshold=5.0,
    )

    assert len(paths) > 0, "MC search found no acceptable paths"


def test_monte_carlo_no_observations():
    """With no observations, all paths are accepted (chi2 = 0 < threshold)."""
    paths = monte_carlo_search(
        [],
        time_bounds_ma=(50.0, 0.0),
        temp_bounds_C=(10.0, 150.0),
        n_paths=100,
        acceptance_threshold=1.0,
    )
    assert len(paths) == 100


def test_monte_carlo_impossible_constraint():
    """Impossible observations → very few or no accepted paths."""
    observations = [
        {"type": "AFT", "value": -999.0, "uncertainty": 1.0},
    ]
    paths = monte_carlo_search(
        observations,
        time_bounds_ma=(50.0, 0.0),
        temp_bounds_C=(10.0, 150.0),
        n_paths=500,
        acceptance_threshold=0.01,
    )
    # May find 0; definitely < 50% of paths
    assert len(paths) < 250


def test_monte_carlo_path_structure():
    """Accepted paths have the expected structure."""
    paths = monte_carlo_search(
        [],
        time_bounds_ma=(50.0, 0.0),
        temp_bounds_C=(10.0, 100.0),
        n_paths=20,
        n_inflection_points=3,
        acceptance_threshold=999.0,
    )
    assert len(paths) > 0
    path = paths[0]
    assert "ages" in path
    assert "temps" in path
    assert "chi_squared" in path
    assert len(path["ages"]) == len(path["temps"])
    # 3 inflections + 2 endpoints = 5
    assert len(path["ages"]) == 5


def test_good_paths_envelope_shape():
    paths = monte_carlo_search(
        [],
        time_bounds_ma=(50.0, 0.0),
        temp_bounds_C=(10.0, 100.0),
        n_paths=50,
        acceptance_threshold=999.0,
    )
    time_grid = np.linspace(50.0, 0.0, 20)
    t_min, t_max, t_mean = good_paths_envelope(paths, time_grid)
    assert t_min is not None
    assert t_min.shape == (20,)
    assert t_max.shape == (20,)
    assert t_mean.shape == (20,)
    assert np.all(t_min <= t_max)


def test_good_paths_envelope_empty():
    t_min, t_max, t_mean = good_paths_envelope([], np.linspace(50.0, 0.0, 10))
    assert t_min is None
    assert t_max is None
    assert t_mean is None


def test_predict_observation_ro():
    ages  = np.array([100.0, 0.0])
    temps = np.array([80.0, 80.0])
    ro = _predict_observation("Ro", ages, temps)
    assert ro is not None
    assert ro > 0.0


def test_predict_observation_aft():
    ages  = np.array([100.0, 0.0])
    temps = np.array([60.0, 60.0])
    aft = _predict_observation("AFT", ages, temps)
    assert aft is not None
    assert aft >= 0.0


def test_predict_observation_unknown_returns_none():
    ages  = np.array([50.0, 0.0])
    temps = np.array([50.0, 50.0])
    result = _predict_observation("MagicAge", ages, temps)
    assert result is None
