# Thermal modeling readiness map

*Discovery + annotation only. No code was changed to produce this. Inventory of
the thermal-modeling code against today's codebase, with an honest real/stubbed/
missing assessment, the integration seams, and a recommended build order.
Reviewed before any thermal build prompt is written.*

Evidence is cited as `path:line`. Verified by reading the repo at the current
`main` (post the restoration arc, Steps 1–8).

---

## 0. Headline verdict

**Unlike restoration (which was a bookkeeping shell), the thermal *physics* is
real and tested — but the *workflow* is a demo with fabricated integration
seams.**

The core compute (`core/thermal.py`, `core/thermal_inverse.py`) is genuine,
validated 1D heat physics + thermochronometric kinetics, with two real test
files. What's missing is the wiring: the dialog **fakes the burial history**
(hardcoded 3-step deepening, not the restoration/decompaction output it claims),
**fabricates the observed data** (a synthetic AFT age, not the well measurements),
and the observed-data table is **orphaned** (full DB schema + CRUD, but never
loaded into the live model, no entry UI, never consumed). Two kinetic models are
also **simplified proxies mislabeled as the real thing** (`aft_age` advertises
"Ketcham07" but is a single Arrhenius; AHe/ZHe are single-domain diffusion, not
RDAAM).

So the gap to a working thermal arc is mostly **plumbing + honesty**, not new
physics — the inverse of the restoration situation.

---

## 1. Inventory (what exists → status)

Status: **REAL** = implemented + tested + correct · **REAL-SIMPLIFIED** = works,
but a first-order proxy (sometimes mislabeled) · **DEMO** = runs but on fabricated
inputs · **ORPHANED** = built but not wired into the workflow · **MISSING**.

| Item | Status | Notes | Evidence |
|---|---|---|---|
| `steady_state_geotherm` | **REAL** | layered steady-state, radiogenic heat production; physics-validated (gradient = q/k, linear w/o A) | `core/thermal.py:27`; tests `test_thermal.py:98,128` |
| `thermal_conductivity_with_porosity` / `effective_conductivity_column` | **REAL but UNUSED** | geometric-mean mixing + Athy porosity; tested — **imported by the dialog but never called** (it uses a single spinbox k) | `core/thermal.py:113,143`; `thermal_modeling_dialog.py:302` (imported, unused) |
| `maturity_easy_ro` (Easy%Ro) | **REAL** (verify calibration) | Sweeney & Burnham 1990, full 20-reaction Ea table; **but `total_reacted = ΣF0 − Σf` with ΣF0≈0.85 (not normalized to 1)** — non-canonical, needs benchmark validation | `core/thermal.py:208,244` |
| `transient_1d_heat` | **REAL** | implicit backward-Euler FD (scipy sparse); scalar k only (layered = future); approximate bottom Neumann BC | `core/thermal.py:252`; tests `test_thermal_advanced.py:35` |
| `aft_age` | **REAL-SIMPLIFIED / MISLABELED** | single first-order **Arrhenius** proxy (Tc≈110°C); **docstring + default say "Ketcham07" but it is NOT the Ketcham 2007 multi-kinetic annealing model**; `kinetics`/`composition` args are no-ops | `core/thermal.py:367` |
| `aft_track_length_distribution` | **REAL-SIMPLIFIED** | MC of the same Arrhenius reduction, not a real track-annealing model | `core/thermal.py:414` |
| `ahe_age` / `zhe_age` / `_he_age_diffusion` | **REAL-SIMPLIFIED** | single-domain spherical diffusion; Farley(2000)/Reiners(2004) D₀/Ea are correct but the model is **not RDAAM/ZRDAAM**; fixed Ft=0.79 (not grain-size dependent) | `core/thermal.py:470,526,559` |
| `monte_carlo_search` (inverse) | **REAL but BASIC** | uniform-random piecewise-linear t-T paths + χ²/obs acceptance; **not** a QTQt-style controlled random walk / MCMC | `core/thermal_inverse.py:22` |
| `good_paths_envelope` | **REAL** | P5/P95 + mean envelope from accepted paths | `core/thermal_inverse.py:113` |
| `decompaction.burial_history` | **REAL but DISCONNECTED** | proper decompaction burial history (strip + decompact); **not consumed by the thermal solver** | `core/decompaction.py:239` |
| `measurements` table (observed data) | **REAL schema + CRUD, ORPHANED** | `vitrinite_ro/aft_age/aft_length/ahe_age/zhe_age/bht/dst_temp/…`; add/get/delete + DB-roundtrip tests — **but not loaded into Project/Well, no entry UI, no caller outside tests** | schema `io/database.py:311`; CRUD `:1393,1419,1432`; tests `test_database_schema.py:267`; **absent** from `app_state.py`/`wells.py` |
| lithology library (`matrix_thermal_conductivity`) | **REAL but DISCONNECTED** | per-lithology k in the strat library; not used by the thermal column | `io/database.py:376-390` |
| `ThermalModelingDialog` | **DEMO** | steady-state real (crude column); **transient fakes burial** (`bh = depths×[0.5,0.75,1.0]`); **inverse fabricates a synthetic AFT obs**; no measurement-input UI | `thermal_modeling_dialog.py:205-294,300` |
| Model ▸ Thermal Modeling action | **REAL (opens dialog)** | needs an active section; docstring still says "Tools →" (stale, like restoration) | `app.py:1477` |
| Kinetics test coverage | **BEHAVIORAL only** | immature/reset/positive/hotter-younger/monotonic/Tc-ordering — **qualitative direction, not benchmark-accurate ages vs HeFTy/published Durango** | `test_thermal_advanced.py:88-345` |

---

## 2. Target capability assessment (Section's thermochronology scope)

| Target capability | Status | What it needs |
|---|---|---|
| Geothermal gradient / heat-flow model | **REAL** | use it (and feed conductivity from the lithology library) |
| Burial/exhumation history from the section | **PARTIAL / disconnected** | `decompaction.burial_history` (stratigraphic) and the restoration sequence (structural) both exist; **neither is wired to thermal** — the seam is the work |
| Forward thermal model T(t) at a point | **ENGINE REAL, fed fake input** | `transient_1d_heat` works; replace the dialog's synthetic burial with real burial → sample-point T(t) |
| Easy%Ro (vitrinite, Sweeney-Burnham) | **REAL** | benchmark-validate the normalization |
| AFT (Ketcham 2007) | **SIMPLIFIED / mislabeled** | relabel as an Arrhenius proxy now; real Ketcham07 is a deferred research item |
| AHe (Farley 2000) | **SIMPLIFIED** | single-domain proxy; RDAAM (Flowers 2009) deferred |
| ZHe | **SIMPLIFIED** | single-domain; ZRDAAM deferred |
| Uncertainty / Monte Carlo | **REAL (basic)** | feed it real observations; MCMC is a later upgrade |
| Display: T-t path | **REAL** (inverse mode) | — |
| Display: depth-T profile | **REAL** (steady/transient) | — |
| Display: age-depth | **MISSING** | needs measurements loaded |
| Display: predicted vs observed | **MISSING** | needs measurements loaded + forward model wired |

---

## 3. Integration seams

| Seam | Status | Detail |
|---|---|---|
| **Restoration → thermal** (palinspastic burial through time) | **DISCONNECTED** | the transient mode hardcodes burial; no thermal code consumes restoration output |
| **Decompaction → thermal** (burial history) | **DISCONNECTED** | `burial_history` is real but unused by the solver |
| **Wells → column** (formation tops) | **CONNECTED (crude)** | nearest-well-by-map-distance → layers (`_build_layers`); fixed k + heat_production=1 |
| **Wells → observations** (thermochron / BHT) | **MISSING SEAM** | `measurements` table + CRUD exist but are orphaned; dialog fabricates a synthetic obs instead |
| **Lithology → conductivity** | **DISCONNECTED** | `matrix_thermal_conductivity` library + `effective_conductivity_column` exist but go unused |
| **Section geometry → point/column** | **CONNECTED (crude)** | distance-along → nearest well; no true column extraction at an arbitrary point |
| **Units / provenance** | **OK, document** | display in °C, depth m, heat flow mW/m²; kinetics convert to K internally (`+273.15`). No provenance stamping on thermal results yet |

---

## 4. Recommended build order (today → working end-to-end thermal history)

The physics is largely done; the arc is **wiring real inputs/outputs around it**.

1. **Load + enter observed data.** Wire the `measurements` table into the live
   model (e.g. `Well.measurements`), add a minimal import/entry UI, and
   load/save through `app_state`. Unblocks every observed-vs-predicted step.
2. **Wire burial history into the transient solver.** Replace the dialog's
   fabricated `bh` with `decompaction.burial_history` (stratigraphic) and/or the
   restoration sequence's palinspastic depths — **the restoration↔thermal seam**.
3. **Forward T(t) at a sample point.** Compose real burial + geotherm →
   temperature history at a measurement's depth → drive Easy%Ro / AFT / AHe.
   End-to-end forward model (Easy%Ro is the simplest, most-trusted start).
4. **Predicted-vs-observed display.** Overlay modeled Ro/ages against the well's
   loaded measurements (age-depth + maturity-depth) — the audit moment.
5. **Inverse with real observations.** Feed the well's measurements into
   `monte_carlo_search` instead of the synthetic obs; show the t-T envelope
   against the constraints.
6. **Conductivity from lithology.** Use `effective_conductivity_column` + the
   lithology library instead of a single spinbox k.
7. **Honesty + calibration pass.** Relabel `aft_age` (it is not Ketcham07);
   benchmark-validate Easy%Ro and the He proxies; fix the "Tools →" docstring.

Each step is independently testable; steps 1–2 are the load-bearing plumbing.

---

## 5. Scope: 75% first arc vs research-grade (deferred)

**75% first arc (achievable, mostly wiring):** geotherm + Easy%Ro +
transient-from-*real*-burial + basic MC inverse + predicted-vs-observed display,
with the simplified AFT/AHe/ZHe proxies **clearly labeled as first-order**. The
real work is steps 1–5 (measurements wired, burial wired, forward model composed,
display, inverse-on-real-data) — not new kinetics.

**Research-grade / deferred (real algorithms, significant effort):**
- True **Ketcham 2007** c-axis-projected, Dpar-dependent multi-kinetic AFT
  annealing (the actual HeFTy model) + real track-length annealing.
- **RDAAM** (Flowers 2009) / **ZRDAAM** radiation-damage He diffusion (replacing
  the single-domain proxies).
- **QTQt-style MCMC** inversion (controlled random walk, Bayesian) replacing
  uniform random sampling.
- 2D/3D thermal (Pecube-like) and lateral heat flow — currently 1D only.

**Honest flag for the build:** the codebase advertises "Ketcham07" and
"Farley 2000" as if the full models are present. Two of those are first-order
proxies. The first build step's relabel (step 7) prevents the interface from
lying about what the numbers are.
