"""Inverse thermal history modeling.

Given observed thermochronometer ages and/or maturity data, find the
thermal histories that reproduce them.  Uses Monte Carlo random sampling
of piecewise-linear time-temperature (t-T) paths.

Functions
---------
monte_carlo_search      — run MC search; returns accepted paths
good_paths_envelope     — min/max/mean temperature envelope from accepted paths
"""
from __future__ import annotations

import numpy as np
from section_tool.core import thermal as _thermal


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def monte_carlo_search(
    observations: list[dict],
    time_bounds_ma: tuple[float, float],
    temp_bounds_C: tuple[float, float],
    n_paths: int = 10_000,
    n_inflection_points: int = 5,
    acceptance_threshold: float = 2.0,
    seed: int | None = None,
) -> list[dict]:
    """Run a Monte Carlo search for acceptable thermal histories.

    Parameters
    ----------
    observations:
        List of observation dicts, each with:

        ============  =================================================
        Key           Description
        ============  =================================================
        type          ``'AFT'``, ``'AHe'``, ``'ZHe'``, or ``'Ro'``
        value         Measured value (Ma or %Ro)
        uncertainty   1-sigma uncertainty
        ============  =================================================

    time_bounds_ma:
        ``(oldest_Ma, youngest_Ma)`` — the time window to explore.
        Example: ``(200.0, 0.0)``
    temp_bounds_C:
        ``(T_min, T_max)`` temperature range.
        Example: ``(10.0, 250.0)``
    n_paths:
        Number of random t-T paths to try.
    n_inflection_points:
        Number of interior inflection points on each random path.
    acceptance_threshold:
        Chi-squared per observation below which a path is accepted.

    Returns
    -------
    list[dict]
        Accepted paths, each a dict with keys:

        * ``'ages'``   — list of ages (Ma), length n_inflection_points + 2
        * ``'temps'``  — list of temperatures (°C), same length
        * ``'chi_squared'`` — chi-squared per observation
    """
    accepted: list[dict] = []
    t_old, t_young = max(time_bounds_ma), min(time_bounds_ma)
    T_lo,  T_hi    = min(temp_bounds_C),  max(temp_bounds_C)
    n_obs = len(observations)

    rng = np.random.default_rng(seed)

    for _ in range(n_paths):
        # Interior inflection points plus oldest endpoint
        ages_interior = sorted(
            rng.uniform(t_young, t_old, size=n_inflection_points),
            reverse=True,
        )
        # All temperatures (including oldest) are random; present-day is fixed at 10°C
        temps_all = rng.uniform(T_lo, T_hi, size=n_inflection_points + 1)

        path_ages  = [t_old] + list(ages_interior) + [float(t_young)]
        path_temps = list(temps_all) + [10.0]

        path_ages_arr  = np.array(path_ages,  dtype=float)
        path_temps_arr = np.array(path_temps, dtype=float)

        chi2 = 0.0
        valid = True
        for obs in observations:
            predicted = _predict_observation(obs["type"], path_ages_arr, path_temps_arr)
            if predicted is None:
                valid = False
                break
            unc = obs.get("uncertainty", 1.0) or 1.0
            chi2 += ((predicted - obs["value"]) / unc) ** 2

        if not valid:
            continue

        chi2_per_obs = chi2 / max(n_obs, 1)
        if chi2_per_obs < acceptance_threshold:
            accepted.append({
                "ages":        path_ages,
                "temps":       path_temps,
                "chi_squared": chi2_per_obs,
            })

    return accepted


def good_paths_envelope(
    accepted_paths: list[dict],
    time_grid_ma: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Compute the P5/P95 temperature envelope from accepted paths.

    Parameters
    ----------
    accepted_paths:
        Output of :func:`monte_carlo_search`.
    time_grid_ma:
        Uniform time grid (Ma) at which to evaluate the envelope.

    Returns
    -------
    (t_min, t_max, t_mean)
        Each is an ndarray aligned with *time_grid_ma*, or
        ``(None, None, None)`` when *accepted_paths* is empty.
    """
    if not accepted_paths:
        return None, None, None

    time_grid_ma = np.asarray(time_grid_ma, dtype=float)
    n_times = len(time_grid_ma)
    interpolated = np.zeros((len(accepted_paths), n_times))

    for i, path in enumerate(accepted_paths):
        ages  = np.array(path["ages"],  dtype=float)
        temps = np.array(path["temps"], dtype=float)
        # np.interp requires increasing x; ages decrease → reverse
        interpolated[i] = np.interp(time_grid_ma, ages[::-1], temps[::-1])

    t_min  = np.percentile(interpolated, 5,  axis=0)
    t_max  = np.percentile(interpolated, 95, axis=0)
    t_mean = np.mean(interpolated, axis=0)
    return t_min, t_max, t_mean


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _predict_observation(
    obs_type: str,
    path_ages: np.ndarray,
    path_temps: np.ndarray,
) -> float | None:
    """Predict a thermochronometric value for a given t-T path."""
    try:
        if obs_type == "Ro":
            return _thermal.maturity_easy_ro(path_temps, path_ages)
        elif obs_type == "AFT":
            return _thermal.aft_age(path_temps, path_ages)
        elif obs_type == "AHe":
            return _thermal.ahe_age(path_temps, path_ages)
        elif obs_type == "ZHe":
            return _thermal.zhe_age(path_temps, path_ages)
    except Exception:
        pass
    return None
