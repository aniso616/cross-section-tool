# Restoration readiness map

*Discovery + annotation only. No code was changed to produce this. Inventory of
the restoration code as it stands against today's codebase, with a staleness
audit and a recommended build sequence. Reviewed before any restoration code is
written.*

Evidence is cited as `path:line`. Verified by reading the repo at the current
`main` (post depth-conversion arc, post menu-IA reorg, post Part A marker-TWT fix).

---

## BUILD STATUS — restoration arc COMPLETE (Steps 1–8)

The original inventory below described restoration as "a metadata + bookkeeping
shell with no geometry engine." That has been built out end to end. What exists now:

| Step | Built | Was (original stub) |
|---|---|---|
| 1. UUID-keyed removal | `RestorationEvent.remove_element_ids` (UUIDs) + resolver + name→UUID migration; `SectionPolygon`/`ReferenceLine` gained UUIDs | removal keyed by **name** (rename-fragile) |
| 2. Editable event content | panel event editor sets which elements an event removes (UUIDs), live-resolved | editor set only name/age/description |
| 3. Interpretation snapshot | `core/restoration_snapshot.py` — faithful, non-destructive, preserved-UUID isolated bundle | `Section.snapshot()` round-tripped only the section line |
| 4. Balance comparison | `core/balance.py` (Dahlstrom area/line-length/detachment) + deformed-vs-**restored** dialog | read-only single-state measurements |
| 5. Pin/datum role | `ReferenceLine.restoration_role` + UUID; events reference pin/datum lines, resolved live | reference-line primitive only, no role |
| 6. Kinematic engine | `core/kinematics.py` — rigid / flexural-slip / simple-shear / fault-parallel-flow; ghost overlay; **anchors-under-restoration** | no engine — hide-by-name only |
| 7. Construction-rule reversal | `restore_by_construction_rule` proposes the algorithm from the construction kind (defaults the user confirms) | construction rules were write-only metadata |
| 8. Wire + provenance | capture-baseline action; balance uses the restored state; restored geometry carries `restoration_provenance`; Restoration Stack shows per-event algorithm/assumptions; docstring sweep | — |

**Anchors-under-restoration seam (the one genuinely new question).** A restored copy
is a pre-deformation frame, NOT tied to the present seismic. `restore_snapshot`
clears each restored pick's `twt_anchor` and drops `seismic_tied` (so `tie_kind`
reads `depth_native`); the **original's anchors are never touched** (deep-copy +
in-place deform on the copy); the snapshot carries `restoration_frame=True`.

**Provenance.** Every restored entity carries `restoration_provenance = {kind:
"restored", source_uuid, algorithm, params, event_id, event_name}` — display-only
(the ghost is view-only, never a persisted entity).

**Persistence.** Events (algorithm/params/pin_line_id/datum_line_id), the
RestorationSequence, and pin/datum reference lines (role + UUID) all survive a
project reopen. The snapshot itself is in-memory by design (Step 3) — recomputed
from the persisted base interpretation on the next Capture.

---

## Construction-rule → restoration-algorithm mapping (Step 7)

`restore_by_construction_rule(entity, event)` (`core/kinematics.py`) reads
`entity.construction_rule.kind` and proposes a restoration algorithm. Proposals
are **defaults the user confirms or overrides** — never silent/forced. Confidence
is `certain` only when the construction rule explicitly encodes the kinematic
(not for inferred defaults).

| construction kind | algorithm | confidence | seeded params | rationale |
|---|---|---|---|---|
| `parallel_to_bed` | `flexural_slip` | suggested | — | a bed parallel to a reference bed restores by layer-parallel slip |
| `kink_band` | `flexural_slip` | suggested | — | kink folds are equal-area / constant bed length → flexural unfold |
| `dip_constrained` | `simple_shear` | suggested | `shear_angle=0` | a constant-dip planar bed restores to the datum by simple shear |
| `listric_fault` | `fault_parallel_flow` | **certain** | `fault_uuid=entity.uuid` | the fault geometry is explicitly recorded → hangingwall flows parallel to it |
| `freehand` | — (`None`) | none | — | no construction constraint to reverse |
| `mirror_axial_trace` | — (`None`) | none | — | no kinematic inverse here |

Pin/datum come from the event (its `pin_line_id` / `datum_line_id` or numeric
`params`), not the construction rule. The editor runs this for each element in
`remove_element_ids`, pre-populates a single agreed proposal (until the user
picks manually), and shows conflicting proposals without auto-resolving them.

---

## 0. Headline verdict

Restoration today is a **metadata + bookkeeping shell with no geometry engine.**
The data model, the panel, and three Model-menu reports all exist and run, but
the only thing "restoration" does to a section is **hide elements by name**.
There is **no kinematic deformation, no balancing comparison, and no
construction-rule reversal** anywhere in the repo.

Two assumptions in the work order are already stale and should be corrected:

- **`core/kinematics.py` and `core/balance.py` do not exist.** Nothing under
  those names is in the tree. The kinematic algorithms (flex unfold,
  fault-parallel flow, simple shear, layer-parallel slip) and the balance
  methods (Dahlstrom, fault offset) have **no implementation at all** — not
  stubbed, absent.
- **`restore_by_construction_rule` does not exist**, and there is **no
  construction-rule → algorithm mapping.** `core/restoration.py` contains zero
  references to `construction_rule`.

What *does* exist and is genuinely useful as foundation: the construction-rule
metadata (real, UUID-linked, persisted), the `ReferenceLine` primitive (pin /
datum capable), `core/structural.py` (real orientation math), the
`RestorationSequence` data model + panel + reports, and a section-line
`snapshot()`/`load_snapshot()`.

---

## 1. Inventory (capability → status → evidence)

Status legend: **WORKS** = real and functional · **STALE** = exists but assumes
an older world · **STUB** = present but does nothing real · **MISSING** = absent.

| Capability | Status | What's actually there | Evidence |
|---|---|---|---|
| `core/kinematics.py` (flex unfold, fault-parallel flow, simple shear, layer-parallel slip) | **MISSING** | File does not exist; no algorithm under any name | no file `core/kinematics.py` |
| `core/balance.py` (Dahlstrom, area balance, line-length, fault offset, detachment depth) | **MISSING (module)** | No core module. Line-length, polygon area, and area÷length "detachment depth" exist **only as a read-only dialog report** — single-state measurements, no deformed-vs-restored comparison, no Dahlstrom, no fault offset | `views/balance_check_dialog.py:23` (`_pick_line_length`), `:99` (`poly.area`), `:113` (`area ÷ length`) |
| `core/restoration.py` — `RestorationEvent`, `RestorationSequence`, serialization | **WORKS (data only)** | Dataclasses with full to/from-dict + to/from-json; sequence CRUD/reorder | `core/restoration.py:24,84,144,150` |
| Restoration **execution** (`restore_remove_layer`) | **STUB** | Explicitly a "decompaction stub — only handles element removal." Filters lists by **name** (`getattr(obj,"name")`), no geometry | `core/restoration.py:174-212` (docstring `:183-185`, name keep `:205-206`) |
| Restoration persistence | **WORKS** | Stored as a **`project_meta` JSON entry** `"restoration_sequence"`, not a table | `io/database.py:1223-1236`; `core/restoration.py:144` |
| `restore_by_construction_rule` + rule→algorithm map | **MISSING** | No such function; restoration never reads construction rules | grep: 0 hits repo-wide; `core/restoration.py` has 0 `construction_rule` |
| Construction rules themselves (`core/construction.py`) | **WORKS** | 6 rule types (freehand, parallel_to_bed, dip_constrained, kink_band, listric_fault, mirror_axial_trace), registry, serialize/deserialize; **UUID-linked** (`reference_uuid`, `hangingwall_uuid`) | `core/construction.py:34-197` |
| Construction rules — attachment & persistence | **WORKS** | `.construction_rule` on HorizonPick/Fault/Polygon; `construction_rule_json` column on `horizons`/`faults`/`polygons`; restored on load | `core/surfaces.py:494`, `core/polygons.py:92`, `io/database.py:65,104,223,844`, `app_state.py:28,546,590,685` |
| Construction rules — **consumers** | **STALE / write-only** | Created by the construction tools, stored, shown as a label — but **nothing reads them to do geometry**. They are currently write-only metadata | created `tools/construction_tools.py:83,156,226,249`; label only `core/surfaces.py:718-721` |
| `RestorationPanel` (Ctrl+6 dock) | **WORKS (partial)** | Add/remove/reorder/edit events, step nav, persists; emits `step_changed` | `views/restoration_panel.py:89-288`; wired `app.py:347-357,913-915` |
| RestorationPanel — event **content** editor | **STUB / gap** | The event dialog only sets **name / age / description** — it **cannot set `remove_elements` or `decompact_params`**. So UI-created events remove nothing | `views/restoration_panel.py:42-86,215-228` |
| Model ▸ Check Section Balance | **WORKS (report)** | Read-only line-length + area + area÷length report | `views/balance_check_dialog.py`; wired `app.py:732-734,1484-1494` |
| Model ▸ Restoration Stack | **WORKS (report)** | Read-only timeline of name-based removals + cumulative count | `views/restoration_stack_dialog.py`; wired `app.py:739-741,1497-1500` |
| Model ▸ Topology Audit | **WORKS** | Real hygiene audit with severity + auto-fix actions, delegates to `core/topology_audit.py` | `views/topology_audit_dialog.py:100-181`; wired `app.py:735-736,1431-1434` |
| Section-view effect of a restoration step | **STUB** | Step → **hide elements by name** only (`elements_visible_at_step`) | `views/section_view.py:2030-2034`; `core/restoration.py:162-171` |
| Pin lines & datums | **PRIMITIVE WORKS / role conceptual** | `ReferenceLine` (horizontal/vertical/angled) with CRUD + persistence + signals; **but no "pin" semantics and no restoration consumes them** | `core/reference_line.py:6-49`; CRUD `app_state.py:1220-1241`; add `app.py:682-686` |
| Section-as-snapshot (duplicate + deform interpretation independently) | **PARTIAL / STALE** | `Section.snapshot()`/`load_snapshot()` round-trips **only the section-line nodes + metadata** — **not** the picks/polygons/horizons. Cannot duplicate an interpreted section and deform it independently | `core/section.py:506-536` |
| Area-preservation metadata on polygons | **MISSING** | `PolygonBoundary.area` is computed on demand; no stored `original_area` / preservation invariant | `core/polygons.py:116` (only `area`); no such field |
| `core/structural.py` (orientation math) | **WORKS (but off-target)** | Real Allmendinger/Cardozo/Fisher orientation tools (strike-dip↔pole, rotations, best-fit fold axis, apparent/true dip) — **stereonet math, not section restoration** | `core/structural.py:18-233` |
| Test coverage — restoration / balance / kinematics | **MISSING** | No `test_restoration.py`, `test_balance.py`, or `test_kinematics.py`. `RestorationEvent/Sequence`, `restore_remove_layer`, and the balance dialog have **no dedicated tests** (construction rules, structural, topology audit *are* tested) | tests dir; `test_construction.py`, `test_structural.py`, `test_topology_audit.py` exist; restoration/balance absent |

---

## 2. Staleness audit — where the restoration code assumes a world that's gone

1. **Element identity = name, not UUID (the deepest staleness).**
   `restore_remove_layer` and `elements_visible_at_step` match on
   `getattr(obj,"name","")` (`core/restoration.py:205-206,168-171`), and
   `RestorationEvent.remove_elements` is a `list[str]` of names
   (`:51`). Entities now carry **UUIDs** and construction rules link by UUID
   (`reference_uuid`, `hangingwall_uuid`). Name-keyed removal is rename-fragile
   and can't disambiguate same-named elements. **Flag:** migrate
   `remove_elements` to UUIDs.

2. **Geometry shape — restoration never touches geometry, so it "works" only
   because it does nothing.** The pure functions filter lists; they never read
   `[(x,y)]`. When a real engine is built it must consume the **current**
   storage: HorizonPick picks are per-section `(distances, depths)`
   (`views/balance_check_dialog.py:68` uses `hp.picks_for_section(name)` →
   `(dist, depth)`), x = along-section metres, y = depth metres (positive down).
   That convention still holds. **Flag:** the engine, not the existing stub, is
   where this matters.

3. **Domain — section is depth-canonical, which suits restoration.** `Section`
   carries `depth_domain` (default `"depth"`) and `depth_units`
   (`core/section.py:518-519,532-533`). Restoration must assume **depth**; no
   domain assumption exists in the restoration code to fix (it has no geometry).
   Note a residual `"twt"` default lingers on *surfaces*/export
   (`app_state.py:645`, `export/print_renderer.py:238`) — not a restoration
   blocker, but confirm the engine reads depth.

4. **Anchors-under-restoration (the key interaction) — entirely unhandled.**
   Restoration **predates anchors** and has **zero** anchor awareness
   (`core/restoration.py` has no `twt_anchor`). HorizonPick now carries a
   per-point `twt_anchor` array + a tie-kind axis (`app_state.py:512-524`;
   `core/surfaces.py:701` "tie kind — a separate axis from construction_rule").
   **Flags for the build:**
   - Restored geometry is a **new pre-deformation frame** and is effectively
     **depth-native, NOT seismic-tied** — it must not claim a `twt_anchor` and
     must not be drawn/serialized as seismic-tied.
   - Restoration must **not corrupt the original's anchor** — restore into a
     copy; never mutate the live anchored pick.
   - Part A seam: anchors were seeded at pick time as
     `model.depth_to_twt(depth)` (`views/section_view.py:3210`), so an anchor's
     independence depends on what depth the section showed at pick time —
     relevant if/when restoration consumes anchored geometry as "truth."

5. **Construction-metadata reversal — schema is current, but unused.** The
   construction rules already use the rename-safe UUID-linked schema
   (`core/construction.py`), so a future `restore_by_construction_rule` can read
   them as-is. **Caveat / terminology drift:** the work order's
   `{kind, *_uuid parents, params}` describes the **VelocityModel/Surface
   lineage** dict, a *different* thing from `core/construction.py`'s typed rule
   dataclasses. The restoration reversal targets the **rules** (per-element
   geometric method), not the lineage dict. No reversal code exists yet.

6. **Menu / IA — current and correct.** The three items live under **Model**
   (`app.py:729-741`), and the panel docks as Ctrl+6 tabbed behind Properties
   (`app.py:388-389,583-584`). Note: several dialog docstrings still say
   "Tools →" (`balance_check_dialog.py:3`, `restoration_stack_dialog.py:4`,
   `topology_audit_dialog.py` comment) — **stale comments**, the wiring is Model.

7. **Provenance / units — not yet carried by restored output.** Restored
   geometry currently doesn't exist, so it carries no provenance. The build must
   stamp restored elements with provenance (which events applied, source UUIDs,
   "depth-native / not seismic-tied"); SI internal, depth display — consistent
   with the rest of the app.

---

## 3. Recommended build sequence (today → working end-to-end restoration)

Each step is independently testable; the geometry engine is the long pole.
Ordered so every step lands on something real.

1. **Identity migration: name → UUID in `RestorationEvent.remove_elements`.**
   Carry UUIDs (keep name as a denormalised label). Update
   `elements_visible_at_step` / `restore_remove_layer` / the stack dialog.
   Backfill old name-keyed sequences on load. *Closes staleness #1; unblocks
   everything else.* Add the missing `test_restoration.py`.

2. **Let the panel actually define an event's content.** Extend the event
   dialog to pick which elements (UUIDs) an event removes and optional
   decompaction params — today it can't (`restoration_panel.py:42-86`). Without
   this, no event removes anything.

3. **Section-as-snapshot for the interpretation, not just the line.** Extend
   `Section.snapshot()` (or add an interpretation snapshot) to deep-copy the
   section's picks/polygons/faults into an independent, deformable frame
   (`core/section.py:506`). This is the substrate the engine deforms.

4. **Balance as a core module + comparison, not a one-state report.** Lift
   line-length / area / detachment-depth out of the dialog into `core/balance.py`
   (pure, tested), and add the **deformed-vs-restored** comparison (Dahlstrom
   bed-length, area balance) the dialog can't currently do. Add fault-offset.

5. **Pin lines & datums as restoration inputs.** Give `ReferenceLine` a "pin"
   role and let the engine consume a pin line (fixed) + a datum (regional
   restoration target). Primitive already exists (`core/reference_line.py`).

6. **The kinematic engine (`core/kinematics.py`) — pure, depth-space.** Build
   the algorithms operating on `[(x,y)]` in (distance, depth): start with
   **vertical simple shear / flexural-slip unfolding to a datum** (most useful,
   simplest invariant), then **fault-parallel flow**, then layer-parallel slip.
   Pure functions, no Qt; heavy test coverage on synthetic folds/faults.
   **Explicit sub-step — anchors-under-restoration (staleness #4):** the engine
   operates on a *copy*, never mutating the original's `twt_anchor`; restored
   output is stamped depth-native / not-seismic-tied and carries provenance.

7. **`restore_by_construction_rule` — the construction-metadata reversal
   (staleness #5).** Map each `core/construction.py` rule kind → its inverse
   kinematic operation (e.g. `parallel_to_bed` → re-derive from the restored
   reference bed; `kink_band` → equal-area unfold; `listric_fault` →
   fault-parallel-flow restore to detachment). Reads the current UUID-linked
   rule schema directly.

8. **Wire the engine into the panel/step machinery + provenance.** Replace the
   hide-by-name effect in `section_view` (`:2030-2034`) with "deform to the
   restored frame for `current_step`," stamp provenance (staleness #7), and make
   the Restoration Stack / Balance reports read the restored state.

**Anchors-under-restoration (step 6 sub-step) and construction-metadata
reversal (step 7) are the two called-out explicit steps.** Steps 1–5 are
foundation that can proceed immediately; steps 6–8 are the real engine.

---

## 4. One-line readiness summary

Data model, panel, reports, construction-rule metadata, reference-line
primitive, and orientation math are **real**; the **kinematic engine, balancing
comparison, construction-rule reversal, UUID-keyed removal, interpretation
snapshot, and anchor handling are missing** — restoration today only hides
elements by name.
