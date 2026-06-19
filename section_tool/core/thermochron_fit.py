"""Predicted thermochronometric values from a forward T(t) path, and the
predicted-vs-observed goodness of fit (Thermal Step 4).

Pure functions T(t) → prediction, wrapping the simplified-but-honest kinetic
proxies (Step 3). Labels never claim an unimplemented model. This is the objective
function the inverse (Step 5) will optimize.

A *t_path* is an ``(N≥2, 2)`` array/list of ``(age_Ma, T_C)`` pairs (any order —
sorted oldest-first internally). Temperatures are °C at the boundary; the kinetics
convert to K internally.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from section_tool.core import thermal as _thermal


def _split_path(t_path):
    arr = np.asarray(t_path, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 2:
        raise ValueError("t_path must be an (N>=2, 2) array of (age_Ma, T_C)")
    order = np.argsort(arr[:, 0])[::-1]              # oldest age first
    a = arr[order]
    return a[:, 1], a[:, 0]                          # temps_C, ages_ma (oldest first)


def predict_ro(t_path) -> float:
    temps, times = _split_path(t_path)
    return float(_thermal.maturity_easy_ro(temps, times))


def predict_aft_age(t_path) -> float:
    temps, times = _split_path(t_path)
    # the depositional (oldest) age caps the apparent age (no annealing → that age)
    return float(_thermal.aft_age(temps, times, initial_age_ma=float(times[0])))


def predict_ahe_age(t_path) -> float:
    temps, times = _split_path(t_path)
    return float(_thermal.ahe_age(temps, times))


def predict_zhe_age(t_path) -> float:
    temps, times = _split_path(t_path)
    return float(_thermal.zhe_age(temps, times))


# measurement_type → (kinetic model key, predictor). model_key indexes
# thermal.KINETIC_MODEL_LABELS / _NOTES (honest, simplified-proxy labels).
PREDICTORS = {
    "vitrinite_ro": ("easy_ro", predict_ro),
    "aft_age":      ("aft",     predict_aft_age),
    "ahe_age":      ("ahe",     predict_ahe_age),
    "zhe_age":      ("zhe",     predict_zhe_age),
}


def predict_for_type(mtype: str, t_path):
    spec = PREDICTORS.get(mtype)
    return None if spec is None else spec[1](t_path)


def _default_uncertainty(mtype: str, value: float) -> float:
    if mtype == "vitrinite_ro":
        return max(0.05, abs(value) * 0.1)          # %Ro
    return max(1.0, abs(value) * 0.1)               # ages: 10 % or 1 Ma floor


@dataclass
class TypeFit:
    measurement_type: str
    model_key: str
    predicted: float
    observed: list                                  # observed value(s)
    chi2: float

    @property
    def mean_observed(self) -> float:
        return float(np.mean(self.observed)) if self.observed else float("nan")

    @property
    def discrepancy_pct(self) -> float:
        o = self.mean_observed
        return 100.0 * (self.predicted - o) / o if abs(o) > 1e-9 else float("nan")


@dataclass
class FitResult:
    per_type: dict                                  # mtype → TypeFit
    total_chi2: float
    n_obs: int

    @property
    def reduced_chi2(self) -> float:
        return self.total_chi2 / self.n_obs if self.n_obs else float("nan")


def goodness_of_fit(t_path, measurements) -> FitResult:
    """χ² misfit of the predicted observables (from *t_path*) vs *measurements*.

    Predicts once per measurement type that has a predictor; per measurement,
    ``chi² += ((predicted − observed) / σ)²`` with σ the measurement's uncertainty
    (or a typed default). Types with no measurement are skipped — graceful, no
    crash. Not a formal inversion; feeds Step 5's objective.
    """
    per: dict = {}
    total, n = 0.0, 0
    for mtype, (model_key, fn) in PREDICTORS.items():
        obs = [m for m in measurements
               if getattr(m, "measurement_type", None) == mtype]
        if not obs:
            continue
        pred = fn(t_path)
        chi2 = 0.0
        for m in obs:
            unc = getattr(m, "uncertainty", None) or _default_uncertainty(mtype, m.value)
            chi2 += ((pred - m.value) / unc) ** 2
        per[mtype] = TypeFit(mtype, model_key, pred, [m.value for m in obs], chi2)
        total += chi2
        n += len(obs)
    return FitResult(per_type=per, total_chi2=total, n_obs=n)
