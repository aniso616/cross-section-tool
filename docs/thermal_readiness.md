# Thermal modeling readiness map

*Originally a discovery + annotation map (pre-build). Updated after each build
step. Current state: Steps 1–7 complete — thermal arc fully wired.*

Evidence cited as `path:line`. Verified against the repo at main after each step.

---

## 0. Headline verdict (updated — arc complete)

The thermal arc is complete as a **75%-solution first arc**: all workflow seams
are wired with real data (no fabrications); physics is honest (labels never claim
a model that isn't implemented); the arc is tested end-to-end.

**Pre-build state (for reference):** the core physics was real but the workflow
had fabricated seams — the dialog faked the burial history, fabricated the observed
data, and the measurements table was orphaned. Two kinetic models were mislabeled.

---

## 1. BUILD STATUS

| Step | Status | Commit | What was built |
|---|---|---|---|
| Step 1 — Load + enter measurements | **COMPLETE** | 3554d8e | `core/measurements.py` (`Measurement` dataclass, `MEASUREMENT_TYPES`, `validate_measurement`, `parse_measurements_csv`); `Well.measurements`; `MeasurementsDialog`; DB schema + migration; `measurements_of_type`; `section_view` measurement markers |
| Step 2 — Wire burial history | **COMPLETE** | f8cd2cb | `core/burial.py` (`BurialHistory`, `burial_history_from_restoration`, `manual_burial_history`); `BurialHistoryDialog`; restoration↔thermal seam via `snapshot_interpretation`; no synthetic burial in transient mode |
| Step 3 — Forward T(t) + Easy%Ro | **COMPLETE** | 91e1b7e | `forward_temperature_history` (quasi-static geotherm; transient solver's basal BC documented as unsuitable for absolute sample T); Easy%Ro benchmark validated (ΣF0≈0.85 weighting reproduces Sweeney-Burnham 1990 canonical curve: immature 0.28%, oil window 0.62%@100°C/1.08%@150°C); `KINETIC_MODEL_LABELS` / `KINETIC_MODEL_NOTES` added; `aft_age` relabeled from false "Ketcham07" to "simplified, single-Arrhenius" |
| Step 4 — Predicted vs observed | **COMPLETE** | aee439d | `core/thermochron_fit.py` (`predict_ro/aft/ahe/zhe`, `PREDICTORS`, `goodness_of_fit`, `TypeFit`, `FitResult`); dialog `_run_forward` overlays BHT errorbars + predicted/observed age lines + χ²/obs fit label with proxy caveat |
| Step 5 — Inverse on real data | **COMPLETE** | 3e0f204 | `monte_carlo_search` gains `seed`; `_inverse_observations()` maps real measurements to χ² objective; fabricated `synthetic_aft` deleted; search runs off UI thread (daemon thread + `inverse_finished` Signal); `_plot_inverse_envelope` renders P5–P95 shaded band labeled "P5–P95 range of acceptable T(t) paths" (not "confidence interval") + best-fit (min-χ²) line |
| Step 6 — Conductivity from lithology | **COMPLETE** | 1879cc9 | `_build_layers` now looks up `strat_column.get_formation(name)` per interval → `effective_conductivity_column` (Athy + geometric-mean mixing) → effective k baked into `thermal_conductivity` field; `_column_mean_conductivity` (harmonic mean) for scalar-k solvers; session-level per-formation k override via `_ConductivityProfileDialog`; `_k_source_label` shows provenance; fallback to spinbox when no picks |
| Step 7 — Honesty + calibration pass | **COMPLETE** | this commit | "Tools →" → correct menu path in all docstrings; `docs/thermal_readiness.md` updated; full-workflow integration smoke test added |

---

## 2. Honesty record

### Labels that were changed (Step 3)

| Model | Old label (false claim) | New label (honest) |
|---|---|---|
| `aft_age` | `"Ketcham07"` (docstring + default arg) | `"AFT age (simplified, single-Arrhenius)"` |
| `ahe_age` | "Farley (2000)" in docstring | `"AHe age (simplified, single-domain)"` |
| `zhe_age` | implied ZRDAAM in docstring | `"ZHe age (simplified, single-domain)"` |

### What the labels now say

```python
KINETIC_MODEL_LABELS = {
    "easy_ro": "Easy%Ro (Sweeney-Burnham 1990)",          # benchmarked + verified
    "aft":     "AFT age (simplified, single-Arrhenius)",   # proxy, NOT Ketcham 2007
    "ahe":     "AHe age (simplified, single-domain)",      # proxy, NOT RDAAM
    "zhe":     "ZHe age (simplified, single-domain)",      # proxy, NOT ZRDAAM
}
```

`KINETIC_MODEL_NOTES` carries the full disclaimer for tooltip display.

### Test that guards label honesty

`tests/test_thermal_forward.py::test_kinetic_labels_dont_claim_unimplemented_models`
and `test_no_false_model_claims_in_display_layer` (runs against all `views/*.py`).

### Easy%Ro benchmark result (Step 3)

`maturity_easy_ro` is verified correct against Sweeney & Burnham (1990):

| Condition | Computed %Ro | Expected | Result |
|---|---|---|---|
| 50 °C / 100 Ma | 0.28 | immature (< 0.5) | PASS |
| 100 °C / 100 Ma | 0.62 | oil window (0.5–0.9) | PASS |
| 150 °C / 100 Ma | 1.08 | late-oil / gas (1.0–1.3) | PASS |
| 200 °C / 100 Ma | 4.69 | post-mature (> 2) | PASS |

The ΣF0 ≈ 0.85 weighting is correct per the Sweeney-Burnham paper (not a bug).

---

## 3. Integration seams (as implemented)

| Seam | Status | Implementation |
|---|---|---|
| **Restoration → thermal** (palinspastic burial) | **WIRED** | `burial_history_from_restoration(seq, horizon_uuid, x, snapshot=snap)` in `core/burial.py`; `_current_burial_history()` in dialog consumes restoration sequence or manual fallback |
| **Wells → observations** | **WIRED** | `Well.measurements`; `_sample_measurements()` → nearest well; `goodness_of_fit(t_path, measurements)` for chi-squared |
| **Wells → observations → inverse** | **WIRED** | `_inverse_observations()` maps `vitrinite_ro→Ro`, `aft_age→AFT`, `ahe_age→AHe`, `zhe_age→ZHe`; BHT/DST skipped (not t-T constraints) |
| **Lithology → conductivity** | **WIRED** | `_build_layers()` calls `strat_column.get_formation(name)` → `effective_conductivity_column`; harmonic mean for scalar-k solvers |
| **Off-thread work** | **IMPLEMENTED** | inverse search: daemon thread + `inverse_finished = Signal(object)`; pattern mirrors DEM fetch |

---

## 4. Deferred (research-grade, not in scope)

- **True Ketcham 2007 AFT**: c-axis-projected, Dpar-dependent multi-kinetic annealing (the full HeFTy model). The current `aft_age` is a single first-order Arrhenius (honest label, clearly disclosed).
- **RDAAM / ZRDAAM**: radiation-damage He diffusion (Flowers 2009 / Reiners). The current `ahe_age`/`zhe_age` are single-domain spherical diffusion with fixed Ft=0.79. Clearly labeled as proxies.
- **QTQt-style MCMC**: controlled random walk with Bayesian acceptance. The current `monte_carlo_search` is uniform-random uniform sampling with χ²/obs threshold acceptance.
- **2D/3D thermal (Pecube-like)**: lateral heat flow and 3D topographic effects. All current solvers are 1D vertical.
- **Grain-size-dependent Ft**: the alpha-ejection correction uses a fixed Ft=0.79 (grain-size independent).
- **Fluid-saturation / pressure corrections** to thermal conductivity.

---

## 5. Original pre-build inventory (for historical reference)

The pre-build state of each item and why it needed work is preserved below
for context. Current state as-built is in §1 above.

| Item | Pre-build status | Notes |
|---|---|---|
| `steady_state_geotherm` | REAL | unchanged |
| `effective_conductivity_column` | REAL but UNUSED | now wired (Step 6) |
| `maturity_easy_ro` | REAL (verify calibration) | verified CORRECT (Step 3) |
| `transient_1d_heat` | REAL | now fed real burial (Step 2) |
| `aft_age` | REAL-SIMPLIFIED / MISLABELED | relabeled (Step 3) |
| `ahe_age` / `zhe_age` | REAL-SIMPLIFIED | relabeled (Step 3) |
| `monte_carlo_search` | REAL but fed synthetic obs | wired to real data (Step 5) |
| `measurements` table | REAL schema, ORPHANED | wired (Step 1) |
| lithology k library | REAL but DISCONNECTED | wired (Step 6) |
| `ThermalModelingDialog` | DEMO (fabricated burial + obs) | fully wired (Steps 2–6) |
