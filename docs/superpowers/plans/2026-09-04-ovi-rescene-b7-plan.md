# OVI-MAP x ReScene B7 Dense Recovery Implementation Plan

> Execute task-by-task with strict RED-GREEN verification. Do not run the full
> suite before the method and configuration freeze.

**Goal:** Add a source-bound, visibility-safe B7-G extension that uses frozen
B4 geometric identity and rigid registration to recover dense OVI history,
then stop or unlock learned work using the pre-registered gate.

**Architecture:** Keep B3 unchanged. Estimate typed per-relation transforms,
classify transformed candidates through the same signed t1 visibility engine,
replace only the paired entity's old B3 fallback spans, preserve OVI geometry
and semantics, and emit complete provenance and evaluator-only attribution.

**Stack:** Python 3.13, NumPy, SciPy `cKDTree`, pytest; pinned ReScene conda
environment only for conditional learned gates.

---

## Task 1: Arbitrary-Point Signed Visibility

**Files:**

- Modify: `src/oviv2/two_visit_execution.py`
- Modify: `tests/oviv2/test_two_visit_execution.py`

1. Add failing tests proving arbitrary transformed points receive the same
   occupied/visible-free/occluded/unobserved aggregation as source t0 points,
   empty inputs are valid, input arrays are not mutated, and the legacy
   `derive_signed_visibility` result is content-equivalent.
2. Run only `pytest -q tests/oviv2/test_two_visit_execution.py` and capture RED.
3. Extract one private key-aggregation helper and add
   `derive_signed_visibility_for_points`; delegate both public paths to it.
4. Rerun the same test module and require GREEN.

## Task 2: Typed Rigid Registration

**Files:**

- Create: `src/oviv2/two_visit_registration.py`
- Create: `tests/oviv2/test_two_visit_registration.py`

1. Write failing contract tests for immutable arrays, canonical hash,
   transform SO(3)/bottom-row/reflection validation, exact reason schema, and
   accepted/rejected consistency.
2. Write failing estimator tests for identity, known 3D rotation+translation,
   deterministic repeatability, low support, linear degeneracy, mismatched
   extent, excessive motion, and ineligible relation states/labels.
3. Run the new module and capture RED.
4. Implement voxel averaging, deterministic cap, rank test, proper PCA
   initializations, trimmed ICP/Kabsch, directional quality measurement,
   lexicographic selection, and fail-closed evidence.
5. Rerun the new module and require GREEN; run `git diff --check`.

## Task 3: Dense Recovery And Provenance

**Files:**

- Create: `src/oviv2/two_visit_dense_recovery.py`
- Create: `tests/oviv2/test_two_visit_dense_recovery.py`

1. Write failing tests for exact no-registration B3 fallback, t1 occupied
   priority, visible-free non-revival, occluded/unobserved compatible recovery,
   incompatible rejection, source-span replacement, removed/appeared/
   uncertain/split/merge fallback, nonrigid label fallback, OVI t1 semantic
   authority, source immutability, deterministic content hash, and no orphan or
   overlapping recovered spans.
2. Run the new module and capture RED.
3. Implement frozen config/contracts, output-span masks, per-entity candidate
   grouping, compatibility checks, snapshot construction, diagnostics, and
   atomic writer.
4. Rerun the new module and require GREEN; run the existing B3 current-map
   tests to prove no behavior regression.

## Task 4: Source-Bound Standalone Runner

**Files:**

- Create: `configs/evaluation/ovi_rescene_b7_g_v1.json`
- Create: `scripts/evaluation/run_ovi_rescene_b7.py`
- Create: `tests/evaluation/test_run_ovi_rescene_b7.py`

1. Write failing tests for strict frozen-record/hash validation, B0/B2 semantic
   receipt binding, B3 provenance loading, exact original VisitMap SHA checks,
   deterministic B4 relation serialization, method-before-evaluator ordering,
   output inventory, atomic failure, and blocked Office policy.
2. Run the new runner tests and capture RED.
3. Implement the config parser, strict readers, frozen OVI reuse, B4 relation
   materialization, registration/candidate visibility/recovery execution,
   existing evaluator invocation, resource accounting, and atomic manifest.
4. Rerun the runner tests and require GREEN.

## Task 5: Evaluator-Only B7 Attribution

**Files:**

- Create: `src/evaluation/two_visit_b7_attribution.py`
- Create: `scripts/evaluation/attribute_ovi_rescene_b7.py`
- Create: `tests/evaluation/test_two_visit_b7_attribution.py`

1. Write failing tests for newly covered GT surface, introduced confirmed-free
   matches, revealed-background conflicts, rejected candidate counts,
   registration failure groups, all required grouping dimensions, source hash
   checks, and no method-output mutation.
2. Run the new attribution tests and capture RED.
3. Implement pure attribution plus atomic JSON/JSONL output; reuse existing
   5 cm metric primitives and B3 attribution conventions.
4. Rerun the module and require GREEN.

## Task 6: IT0 And Cached IT1

**Files:**

- Create/update only compact receipts under
  `configs/evaluation/results/ovi_rescene_b7/`
- Add focused integration tests only if a missing invariant is exposed.

1. Run registration, recovery, visibility, runner, and attribution modules as
   IT0; require all synthetic invariants.
2. Materialize the lexicographically selected eligible frozen B4 relation,
   execute the cached Apartment slice, and record source/config/output hashes.
3. Verify repeat execution produces equal transform/result content hashes and
   that every visible-free candidate count is rejected.
4. Stop before full Apartment on any failure.

## Task 7: Freeze And Run Apartment B7-G Once

**Files:**

- Update compact receipts under
  `configs/evaluation/results/ovi_rescene_b7/`
- Do not modify the frozen B0--B6 matrix or artifacts.

1. Freeze the implementation commit and config SHA after IT0/IT1.
2. Run one full Apartment B7-G command using the frozen OVI/B3 artifacts.
3. Run evaluator-only attribution after the method manifest is immutable.
4. Compare every pre-registered predicate against the exact B3 receipt.
5. Emit either `GO_B7_MAPPING_EXTENSION` or
   `STOP_B7_MAPPING_EXTENSION`. Do not tune thresholds from the result.

## Task 8: Conditional ReScene And Dataset Gates

**Files:**

- Update only B7 compact receipts/reports unless a failing gate requires a
  separately designed change.

1. If B7-G is NO-GO, record ReScene/learned B7 as stopped and skip all GPU work.
2. If B7-G is GO, run R0, then R1, then R2; stop at the first failed gate.
3. Do not run learned B7 unless legal B5 identity exceeds B4.
4. Keep 3RScan final ranking blocked until the frozen pilot reaches 44/44.
5. Keep Office at zero attempts unless all bound assets exist after commit and
   config freeze; then permit exactly one held-out run.

## Task 9: Verification, Review, Results, And Push

**Files:**

- Create: `docs/superpowers/reports/2026-09-04-ovi-rescene-b7-results.md`
- Create: `docs/superpowers/reports/2026-09-04-ovi-rescene-b7-handoff.md`
- Update the audit only for factual post-audit asset corrections, never results.

1. Run all B7 targeted modules, legacy B3/B4 tests, source-manifest checks,
   `py_compile`, and `git diff --check`.
2. Request a code review and personally inspect the full diff, contracts,
   numerical behavior, leakage boundary, and result receipts.
3. Run the full repository suite once. On failure, diagnose, repair with
   targeted tests, and run it at most once more.
4. Write results and answer all 20 handoff questions with exact values or
   explicit N/A/BLOCKED reasons.
5. Commit logical changes, push
   `research/ovi-rescene-b7-dense-recovery`, and verify local HEAD equals the
   remote SHA.
