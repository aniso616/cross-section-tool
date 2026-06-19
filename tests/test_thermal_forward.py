"""Thermal Step 3: Easy%Ro benchmark, forward T(t) model, and the honesty pass on
kinetic labels (no UI string may claim a model that isn't implemented)."""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from section_tool.core.thermal import (
    maturity_easy_ro, forward_temperature_history, KINETIC_MODEL_LABELS)
from section_tool.core.burial import BurialHistory


def _ro_const(t_start, t_peak, total_ma, n=200):
    times = np.linspace(total_ma, 0.0, n)          # oldest first → present
    temps = np.linspace(t_start, t_peak, n)
    return maturity_easy_ro(temps, times)


# ── Easy%Ro benchmark (Sweeney-Burnham 1990) ────────────────────────────────

def test_easy_ro_benchmark_against_sweeney_burnham():
    """The canonical Easy%Ro curve: Ro0 ≈ 0.2 %, oil window 0.6–1.3 % at
    100–150 °C, practical maximum ≈ 4.6 %. Fails if the normalization is wrong."""
    assert _ro_const(20, 20, 100) == pytest.approx(0.28, abs=0.06)     # immature Ro0
    assert 0.55 < _ro_const(20, 100, 100) < 0.70                       # oil-window onset
    assert 0.95 < _ro_const(20, 150, 100) < 1.30                       # oil-window peak/end
    rmax = _ro_const(350, 350, 100)
    assert abs(rmax - 4.6) / 4.6 < 0.05                                # max within 5 % rel


# ── forward model ───────────────────────────────────────────────────────────

def test_forward_temperature_history_tracks_geotherm():
    # slow linear burial 0→3000 m over 100 Ma; q=60 mW/m², k=2.5 → 24 °C/km
    bh = BurialHistory(points=[(100.0, 0.0), (50.0, 1500.0), (0.0, 3000.0)],
                       source="user-specified")
    res = forward_temperature_history(bh, basal_heat_flow_mW=60.0,
                                      conductivity=2.5, surface_temp_C=10.0)
    assert res.gradient_C_per_km == pytest.approx(24.0, abs=0.5)       # derived q/k
    assert res.temps_C[-1] == pytest.approx(82.0, abs=0.5)            # 10 + 3000·0.024
    assert res.temps_C[0] == pytest.approx(10.0, abs=0.5)            # surface at the oldest step
    assert res.temps_C[0] < res.temps_C[-1]                           # hotter now (deeper)
    assert res.surface_temp_C == 10.0 and res.source == "user-specified"


def test_forward_higher_heat_flow_is_hotter():
    bh = BurialHistory(points=[(50.0, 1000.0), (0.0, 3000.0)], source="user-specified")
    lo = forward_temperature_history(bh, basal_heat_flow_mW=40.0,
                                     conductivity=2.5, surface_temp_C=10.0)
    hi = forward_temperature_history(bh, basal_heat_flow_mW=80.0,
                                     conductivity=2.5, surface_temp_C=10.0)
    assert hi.temps_C[-1] > lo.temps_C[-1]


# ── honesty pass: labels must not claim unimplemented models ────────────────

_FORBIDDEN = ("Ketcham07", "Ketcham 2007", "RDAAM", "ZRDAAM", "Farley 2000")


def test_kinetic_labels_dont_claim_unimplemented_models():
    for label in KINETIC_MODEL_LABELS.values():
        for tok in _FORBIDDEN:
            assert tok not in label, f"label {label!r} claims unimplemented {tok!r}"
    # the simplified proxies say so
    assert "simplified" in KINETIC_MODEL_LABELS["aft"].lower()
    assert "simplified" in KINETIC_MODEL_LABELS["ahe"].lower()
    # Easy%Ro is the one model allowed to name its source (it IS implemented)
    assert "Sweeney-Burnham" in KINETIC_MODEL_LABELS["easy_ro"]


def test_no_false_model_claims_in_display_layer():
    """The view layer never advertises a model that isn't there (re-labeling drift
    regression)."""
    views = pathlib.Path(__file__).resolve().parents[1] / "section_tool" / "views"
    for f in views.glob("*.py"):
        text = f.read_text(encoding="utf-8")
        for tok in _FORBIDDEN:
            assert tok not in text, f"{f.name} advertises unimplemented {tok!r}"
