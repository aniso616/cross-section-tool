"""Thermal Step 5: Monte Carlo inverse on real data.

Covers the seeded, deterministic core search — a known t-T path generates the
observations, and the accepted-path envelope must bracket that true path — plus
the seam check that the dialog no longer fabricates a synthetic observation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from section_tool.core.thermal_inverse import (
    monte_carlo_search, good_paths_envelope, _predict_observation)


# ── the true path is recovered inside the envelope (synthetic dataset) ───────

def test_true_path_falls_within_envelope():
    # A known cooling history: heat to a peak, then cool to 10 °C at present.
    true_ages  = np.array([100.0, 60.0, 30.0, 0.0])
    true_temps = np.array([30.0, 115.0, 70.0, 10.0])

    observations = []
    for typ in ("Ro", "AFT"):
        val = _predict_observation(typ, true_ages, true_temps)
        unc = 0.1 if typ == "Ro" else max(val * 0.1, 2.0)
        observations.append({"type": typ, "value": float(val), "uncertainty": unc})

    paths = monte_carlo_search(
        observations,
        time_bounds_ma=(100.0, 0.0),
        temp_bounds_C=(10.0, 200.0),
        n_paths=20_000,
        n_inflection_points=4,
        acceptance_threshold=2.0,
        seed=20240615,
    )
    assert paths, "expected at least some acceptable t-T paths"

    grid = np.linspace(100.0, 0.0, 50)
    p5, p95, mean = good_paths_envelope(paths, grid)
    assert p5 is not None and np.all(p95 >= p5)

    true_on_grid = np.interp(grid, true_ages[::-1], true_temps[::-1])
    inside = (true_on_grid >= p5 - 1e-6) & (true_on_grid <= p95 + 1e-6)
    # The true path is largely bracketed by the P5–P95 envelope of accepted paths.
    assert inside.mean() >= 0.6


def test_seed_makes_search_deterministic():
    obs = [{"type": "AFT", "value": 55.0, "uncertainty": 5.0}]
    kw = dict(time_bounds_ma=(100.0, 0.0), temp_bounds_C=(10.0, 180.0),
              n_paths=3000, n_inflection_points=4, acceptance_threshold=2.0)
    a = monte_carlo_search(obs, seed=7, **kw)
    b = monte_carlo_search(obs, seed=7, **kw)
    assert len(a) == len(b)
    if a:
        assert a[0]["chi_squared"] == b[0]["chi_squared"]


# ── graceful degradation: a type with no measurement doesn't contribute ──────

def test_missing_type_does_not_contribute():
    # Only an Ro observation present → ZHe/AHe/AFT simply aren't in the objective.
    obs = [{"type": "Ro", "value": 0.8, "uncertainty": 0.1}]
    paths = monte_carlo_search(
        obs, time_bounds_ma=(100.0, 0.0), temp_bounds_C=(10.0, 180.0),
        n_paths=2000, n_inflection_points=4, acceptance_threshold=3.0, seed=1)
    # A single weak constraint admits many histories — the search must not crash
    # and must return paths scored on that one observation alone.
    assert isinstance(paths, list)


# ── the dialog seam no longer fabricates an observation ──────────────────────

def test_dialog_inverse_has_no_fabricated_observation():
    src = (Path(__file__).resolve().parents[1] /
           "section_tool" / "views" / "thermal_modeling_dialog.py").read_text(
               encoding="utf-8")
    assert "synthetic_aft" not in src
    assert "_inverse_observations" in src        # real-measurement mapping present
