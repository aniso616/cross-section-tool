# F3 data inventory for depth conversion (discovery — report only)

Data root: `J:\data\F3_Demo_2023\` (OpendTect F3 Demo 2023). Test well: **F02-01**
(OpendTect native DB names it `F02-1`). Vertical well, KB = 30 m,
surface X/Y = 606554 / 6080126.

## Step 1 — Well curve inventory (`Rawdata\Well_data\F02-01_logs.las`)

lasio load: well `F02-01`, depth curve **DEPTH (M)**, MD 30 → 3150 m, irregular
sampling (median step ≈ 0.15 m, 9592 samples). 7 curves, full listing:

| # | Mnemonic | Unit | Description |
|---|----------|------|-------------|
| 1 | `DEPTH`   | M        | Domain Curve — Type DEPTH |
| 2 | `CALI`    | in       | Caliper_1 — Type Caliper |
| 3 | `RHOB`    | g/cc     | Density_1 — Type Density |
| 4 | `GR`      | API      | Gamma Ray_math — Type Gamma Ray |
| 5 | `DT:1`    | us/ft    | P-wave_1 — Type P-wave (raw sonic) |
| 6 | `DT:2`    | us/ft    | P-wave_corr — Type P-wave (corrected sonic) |
| 7 | `UNKNOWN` | fraction | Porosity_1 — Type Porosity |

Flagged curves:
- **DT (sonic): PRESENT** — two variants. `DT:1` raw, `DT:2` corrected. `DT:1`
  has 9118 valid samples, 111.8–203.8 µs/ft (≈ 1495–2725 m/s), spanning the
  whole logged interval. Dense, continuous → integrable to a TDR.
- **RHOB (density): PRESENT** (g/cc).
- **DTS (shear sonic): ABSENT.** No shear → no full-elastic/AVO, but irrelevant
  to the V(z)/Dix ladder.
- **GR: PRESENT** (API). **CALI: PRESENT** (in).
- Datum/KB: LAS header START is 30 m; KB elevation = 30 m is carried in the
  welltrack and the saved projects (`kb_elevation = 30.0`). Depth is MD-from-KB.

Auxiliary sonic file: `F02-01_DT_TVDSS.txt` — 625 rows, two columns =
**TVDSS (m, regular 5 m grid, 0–3120) vs TWT (ms), the sonic integrated to time**.
It cross-checks against the checkshot to ~3 ms (shallow) / ~25 ms (deep) — i.e.
a ready-made sonic TDR.

## Step 2 — Checkshot / time–depth data

- `Rawdata\Well_data\F02-01_TD.txt` — **checkshot / T-D table**, 25 pairs:
  **depth (m) vs TWT (s)**, 30 m → 0 s down to 3150 m → 3.234 s. This is the
  canonical independent (depth, TWT) tie for the well. (datum 30 m = KB.)
- `F02-01_DT_TVDSS.txt` — dense sonic-integrated TDR (see Step 1), 5 m, 625 pts,
  consistent with the checkshot.
- OpendTect native per-well T-D / checkshot models in `WellInfo\`:
  `F02-1.wlt` (well log-time / T-D), `F02-1.csmdl` (checkshot model),
  `F02-1.tie` (well tie). Same for F03-2, F03-4, F06-1. Native binary — the
  readable `*_TD.txt` is the practical source.

## Step 3 — Seismic velocity data

| File | Type | Format |
|------|------|--------|
| `Rawdata\Velocity_functions.txt` (2.9 MB) | **Vrms + Vint + Vavg** vs Time(ms)/Depth(m) at CDP-X/Y | ASCII table, columns `CDP-X CDP-Y Time(ms) Vrms Vint Vavg Depth(m)`. Header: *"example velocities, not measured velocities."* |
| `Seismics\Velocity_model__INT_.cbvs` (+`.par`) | **Interval velocity** volume, depth-domain, range 1500–2400 m/s | OpendTect CBVS (binary) |
| `Seismics\Velocity_model__RMS_.cbvs` (+`.par`) | **RMS velocity** volume | OpendTect CBVS (binary) |
| `Locations\RMS_velocities.pck` (174 MB) | RMS velocity **picks** | OpendTect PickSet (binary) |

The CBVS/PCK are OpendTect-native (not SEG-Y; would need odpy/conversion). The
**ASCII `Velocity_functions.txt` already exposes Vrms, Vint and Vavg directly**,
so both Dix (Vrms→Vint) and direct-interval rungs are reachable without touching
the binaries. (Note the explicit "example velocities" caveat — synthetic, fine
for a method-ladder test, not for hard QC.)

## Step 4 — Tops cross-check

- **Source file** `Rawdata\Well_data\F02-01_markers.txt`: **24 formation tops**,
  MD (m) + name, **MD-only — no TWT column**. From `Seasurface` (30) down:
  MFS11 (553.6), FS11 (612.9), MFS10, MFS9, MFS8, FS8, FS7, Truncation 1,
  Lower Low Sonic, FS6, MFS4, FS4, FS 3, FS2, MFS 2, FS1, NMRF (Mid_Mio_Unc,
  1285.09), CKGR, SGKI, SLCU, SLCMU, SLCMS, SLCML, SLCL (3150).
- **Currently loaded in the test project(s):** F02-01 is present in 5 saved
  projects (`seismic_stretch_test`, `test_02`, `test_05`, `test_06`, `test_07`),
  KB 30, correct X/Y — but **`well_tops` is EMPTY (0 rows) in every one.** Tops
  have not been imported.
- **Schema gap:** the `well_tops` table is `(formation_name, md, tvd, confidence,
  note)` — **no TWT column.** A marker's TWT is not stored; it must be derived at
  calibration time by looking the top's depth up in the checkshot (`TD.txt`) or
  the sonic TDR. (Consistent with the well-tie code: marker TWTs are entered/
  measured, never stored on the top.)

## Summary — data kind → present? → ladder rung unlocked

| Data kind | Present for F02-01? | Source | Unlocks rung |
|-----------|--------------------|--------|--------------|
| Continuous sonic DT (µs/ft) | ✅ dense, whole well (DT:1 raw, DT:2 corr) | `F02-01_logs.las` | **Sonic V(z)** (integrate DT → TDR) |
| Sonic-integrated TDR (TVDSS↔TWT) | ✅ 5 m, 625 pts | `F02-01_DT_TVDSS.txt` | **Sonic V(z)** (ready-made) |
| Checkshot (depth↔TWT) | ✅ 25 pairs | `F02-01_TD.txt` | **Checkshot-tied** |
| Density RHOB | ✅ | `F02-01_logs.las` | (impedance/synthetic, not depth-conv) |
| Formation tops (MD) | ✅ source (24); ❌ not loaded | `F02-01_markers.txt` | **Marker-tied** (once imported + tied via checkshot) |
| Tops TWT | ❌ no TWT in tops file or schema | — | marker TWT must come from checkshot/TDR |
| RMS velocity | ✅ | `Velocity_functions.txt` (ASCII), `Velocity_model__RMS_.cbvs`, `RMS_velocities.pck` | **Dix** (Vrms→Vint) |
| Interval velocity | ✅ | `Velocity_functions.txt` (Vint col), `Velocity_model__INT_.cbvs` | **Dix / direct interval** |
| Shear sonic DTS | ❌ | — | (none — no elastic rung) |

**Bottom line:** F02-01 unlocks **every rung** of the planned ladder — sonic V(z)
(two ways: integrate the LAS DT, or use the ready `DT_TVDSS` TDR), checkshot-tied
(`TD.txt`), Dix (`Velocity_functions.txt` Vrms/Vint), and marker-tied (24 tops in
`markers.txt`, *once imported* — they currently aren't, and the tops schema holds
no TWT, so marker TWT is checkshot-derived). The only genuinely absent data is
shear sonic (DTS), which the ladder doesn't need.
