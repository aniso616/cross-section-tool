"""Thermal Step 4: predicted thermochronometric values from a T(t) path and the
predicted-vs-observed goodness of fit."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from section_tool.core.thermochron_fit import (
    predict_ro, predict_aft_age, predict_ahe_age, predict_zhe_age,
    predict_for_type, goodness_of_fit)


def _path(t_start, t_peak, total_ma, n=200):
    ages = np.linspace(total_ma, 0.0, n)            # oldest first
    temps = np.linspace(t_start, t_peak, n)
    return list(zip(ages, temps))


def _meas(mtype, value, unc=None):
    return SimpleNamespace(measurement_type=mtype, value=value, uncertainty=unc)


# ── prediction functions vs known outcomes ──────────────────────────────────

def test_predict_ro_matches_easy_ro_benchmark():
    assert 0.95 < predict_ro(_path(20, 150, 100)) < 1.30      # oil window ~1.0-1.3 %


def test_predict_aft_age_cold_retains_hot_resets():
    cold = predict_aft_age(_path(20, 20, 100))               # below Tc → no annealing
    hot = predict_aft_age(_path(20, 180, 100))               # above Tc → annealed/reset
    assert cold == pytest.approx(100.0, abs=2.0)             # ≈ depositional age
    assert hot < cold                                         # reset younger


def test_predict_ahe_younger_than_zhe_for_same_path():
    # ZHe has a higher closure temperature → older apparent age than AHe
    path = _path(20, 160, 100)
    assert predict_zhe_age(path) >= predict_ahe_age(path)


def test_predict_for_type_dispatch_and_unknown():
    p = _path(20, 150, 100)
    assert predict_for_type("vitrinite_ro", p) == pytest.approx(predict_ro(p))
    assert predict_for_type("bht", p) is None                # no kinetic predictor


def test_path_too_short_raises():
    with pytest.raises(ValueError):
        predict_ro([(0.0, 20.0)])


# ── goodness of fit ─────────────────────────────────────────────────────────

def test_goodness_of_fit_known_chi2():
    path = _path(20, 150, 100)
    pred = predict_ro(path)                                   # ~1.08 %Ro
    obs = pred - 0.1                                          # known offset
    fit = goodness_of_fit(path, [_meas("vitrinite_ro", obs, unc=0.1)])
    assert fit.n_obs == 1
    tf = fit.per_type["vitrinite_ro"]
    assert tf.predicted == pytest.approx(pred)
    assert tf.chi2 == pytest.approx(((pred - obs) / 0.1) ** 2)    # = 1.0
    assert fit.total_chi2 == pytest.approx(1.0) and fit.reduced_chi2 == pytest.approx(1.0)
    assert tf.discrepancy_pct == pytest.approx(100.0 * (pred - obs) / obs)


def test_goodness_of_fit_no_measurements_is_graceful():
    fit = goodness_of_fit(_path(20, 120, 100), [])
    assert fit.per_type == {} and fit.n_obs == 0
    assert np.isnan(fit.reduced_chi2)


def test_goodness_of_fit_skips_types_without_measurements():
    # only an AFT measurement present → only AFT in the result, no Ro/AHe/ZHe
    path = _path(20, 120, 100)
    fit = goodness_of_fit(path, [_meas("aft_age", 60.0, unc=5.0)])
    assert set(fit.per_type) == {"aft_age"} and fit.n_obs == 1
    assert fit.per_type["aft_age"].model_key == "aft"
