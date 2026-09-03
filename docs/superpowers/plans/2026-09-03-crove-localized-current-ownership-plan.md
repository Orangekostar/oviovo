# CROVE P6 Localized Current Ownership Implementation Plan

> Execute in order. Every behavior change starts with a failing focused test.

**Goal:** Replace whole-anchor stale deletion with 5 cm signed reversible
ownership, gate dense moved geometry by causal agreement, and publish an
official-Dyn provenance sidecar without changing T1 or official metrics.

**Architecture:** Preserve the immutable OVI-MAP anchor and attach typed
copy-on-advance ownership state to unbound anchors. The composition layer reads
that state to filter native dense points. Counterfactual ID filtering exists
only behind an explicit diagnostic contract. Dense moved geometry is selected
by a fixed 5 cm agreement gate. Provenance consumes immutable evaluator and
runtime artifacts as a sidecar.

**Stack:** Python 3.12, NumPy, SciPy `cKDTree`, pytest, existing TESSE-CD
composition/bridge/evaluation scripts, canonical JSON and SHA256 bindings.

## Task 1: Counterfactual variant and feature contracts

**Files:**

- Create: `src/evaluation/crove_anchor_counterfactual.py`
- Create: `tests/evaluation/test_crove_anchor_counterfactual.py`

1. Add failing tests for deriving exactly CF0, CF1, ten CF2, ten CF3, G1, and
   G2 from ten unique P5 transitions and P2 Ghost mass.
2. Add failing tests that reject duplicate transitions, non-dormant
   transitions, missing P2 coverage, ID leakage into feature rules, and
   malformed metric values.
3. Add failing tests for anchor features: transition/support timing, semantic
   label, point/voxel counts, evidence counts, first absence, latency, and
   extent.
4. Implement immutable variant/feature dataclasses and pure deterministic
   builders with strict schemas.
5. Add metric-row construction with deltas against CF0 and CF1; preserve
   unavailable official metrics as empty CSV values plus an explicit status.
6. Run:

   ```bash
   pytest -q tests/evaluation/test_crove_anchor_counterfactual.py
   git diff --check
   ```

7. Commit: `feat/eval: add P5 counterfactual contracts`

## Task 2: Diagnostic-only counterfactual recomposition

**Files:**

- Modify: `scripts/evaluation/run_crove_ovimap_static_anchor.py`
- Modify: `tests/evaluation/test_run_crove_ovimap_static_anchor.py`
- Create: `scripts/evaluation/run_crove_anchor_counterfactuals.py`
- Modify: `tests/evaluation/test_crove_anchor_counterfactual.py`

1. Add a failing runner test for a `counterfactual_diagnostic` contract that
   intersects the causal P5 suppressed set with a supplied diagnostic set.
2. Add failing guards: the set is invalid for every production readout role;
   omitted diagnostics preserve byte-identical P5 behavior; selected anchors
   must be transitioned unbound anchors.
3. Extend visibility diagnostics to record first absence, transition support,
   distinct views at transition, and evidence counts without changing P5
   decisions.
4. Implement the diagnostic contract and record selected IDs, source P5 policy,
   and variant identity in the run manifest.
5. Add CLI `prepare` to derive variants, recompose all checkpoint packages from
   the existing source run, and publish immutable per-variant manifests. Do not
   invoke the source mapper.
6. Add CLI `collect` to validate official/common result receipts and publish
   `counterfactual_manifest.json`, `counterfactual_metrics.csv`, and
   `counterfactual_anchor_features.csv` atomically.
7. Run:

   ```bash
   pytest -q tests/evaluation/test_crove_anchor_counterfactual.py \
     tests/evaluation/test_run_crove_ovimap_static_anchor.py
   python scripts/evaluation/run_crove_anchor_counterfactuals.py --help
   git diff --check
   ```

8. Commit: `feat/eval: add diagnostic counterfactual replay`

## Task 3: Execute and close P6-A

**Files:**

- Create: `docs/superpowers/reports/P6A_COUNTERFACTUAL_REPORT.md`
- Modify: `docs/superpowers/experiments/2026-09-03-crove-localized-current-ownership-decision-ledger.md`

1. Run `prepare` against the frozen P5 composition, P2 attribution, source run,
   anchor, policy, and Apartment RGB-D manifest.
2. Validate all 24 variant compositions and hashes.
3. For each variant, export the temporal artifact, build the Khronos bridge,
   run the unchanged official evaluator, and run common-v2. Execute independent
   variants concurrently only where isolated disk/output roots are available;
   remove no source artifact.
4. Run `collect` and verify CF0/CF1 reproduce their frozen references.
5. Analyze Ghost benefit versus Object damage using features, never IDs as the
   final rule.
6. Record exact artifacts, hashes, metrics, GO/NO-GO, and reason in the report
   and ledger.
7. Commit: `docs: report P6-A counterfactual results`

## Task 4: Typed 5 cm ownership state

**Files:**

- Modify: `src/oviv2/anchor_visibility.py`
- Modify: `tests/oviv2/test_anchor_visibility.py`

1. Add failing tests for deterministic initialization over all unique anchor
   voxels and the capped P5 evidence sample.
2. Add failing tests for per-voxel PRESENT support, six ABSENT observations and
   three-view suppression, OCCLUDED neutrality, UNOBSERVED neutrality, and
   invalid-depth neutrality.
3. Add failing tests for two-frame local restoration, two-region partial state,
   monotone frame/timestamp checks, immutable inputs, and read-only arrays.
4. Implement `AnchorVoxelVisibilityEvidence`, `AnchorCurrentOwnership`,
   initialization, projection, vectorized copy-on-advance, status/stats, memory
   accounting, and bit-packed canonical mask records.
5. Keep existing whole-anchor APIs and tests unchanged.
6. Run:

   ```bash
   pytest -q tests/oviv2/test_anchor_visibility.py
   git diff --check
   ```

7. Commit: `feat: add localized reversible anchor ownership`

## Task 5: Partial dense-anchor composition

**Files:**

- Modify: `src/oviv2/ovimap_static_anchor.py`
- Modify: `tests/oviv2/test_ovimap_static_anchor.py`

1. Add failing tests that partial ownership filters points by
   `floor(point / 0.05)` while preserving native coordinates, point order,
   entity ID, and semantics.
2. Add failing tests for unchanged, partially suppressed, and dormant metadata;
   unknown/bound ownership rejection; and mutual exclusion with legacy
   `suppressed_unbound_anchor_ids`.
3. Add the typed ownership input and point filter. Preserve exact legacy output
   when ownership is absent.
4. Add active/suppressed voxel and point counts and policy identity metadata;
   do not embed masks.
5. Run:

   ```bash
   pytest -q tests/oviv2/test_ovimap_static_anchor.py
   git diff --check
   ```

6. Commit: `feat: compose partial current anchor geometry`

## Task 6: Localized causal timeline and mask artifacts

**Files:**

- Modify: `scripts/evaluation/run_crove_ovimap_static_anchor.py`
- Modify: `tests/evaluation/test_run_crove_ovimap_static_anchor.py`
- Create: `configs/evaluation/crove_ovimap_localized_visibility_l1_v1.json`

1. Add failing tests for a localized candidate readout contract, checkpoint-only
   ownership snapshots, hash-bound mask sidecars, partial trajectories, and
   whole-anchor lifecycle transitions derived from ownership status.
2. Add failing deterministic replay and future-frame noninterference tests.
3. Add failing compatibility test proving the legacy P5 role remains identical.
4. Implement one-pass localized replay from frame 263, retaining only current
   state plus official checkpoint copies.
5. Publish canonical bit-packed masks separately and bind them in checkpoint
   diagnostics and the run manifest.
6. Add the frozen L1 config using exactly the P5 thresholds and explicit
   `per_voxel` state granularity.
7. Measure implemented state bytes. Keep L2 `NOT_RUN_NOT_NEEDED` unless state
   exceeds 8 MB or replay is unstable.
8. Run:

   ```bash
   pytest -q tests/oviv2/test_anchor_visibility.py \
     tests/oviv2/test_ovimap_static_anchor.py \
     tests/evaluation/test_run_crove_ovimap_static_anchor.py
   pytest -q tests/oviv2/test_t1_exactness.py tests/oviv2/test_t1_noninterference.py
   git diff --check
   ```

9. Commit: `test: cover localized ownership causality and replay`

## Task 7: Execute and gate P6-C L1

**Files:**

- Create: `docs/superpowers/reports/P6C_LOCALIZED_VISIBILITY_REPORT.md`
- Modify: `docs/superpowers/experiments/2026-09-03-crove-localized-current-ownership-decision-ledger.md`

1. Recompose the frozen source run with L1; do not rerun the source mapper.
2. Export temporal artifacts, prepare/import the Khronos bridge, run unchanged
   official evaluation, and run common-v2 twice.
3. Apply the preregistered exact Object/Dyn/Change/current/Ghost gates.
4. Record full coverage, state size, partial/dormant counts, output hashes,
   metrics, and Office authorization.
5. Commit: `docs/eval: evaluate localized Apartment candidate`

## Task 8: Geometry agreement primitive

**Files:**

- Create: `src/oviv2/dense_moved_readout.py`
- Create: `tests/oviv2/test_dense_moved_readout.py`
- Modify: `src/evaluation/crove_dense_readout_audit.py`
- Modify: `tests/evaluation/test_crove_dense_readout_audit.py`

1. Add failing tests for both directional 5 cm coverage, median/p90 nearest
   neighbors, centroid residual, extent residual/ratio, and degenerate axes.
2. Add failing gate boundary tests for the exact preregistered thresholds.
3. Implement typed config, measurement, and decision contracts with finite
   checks and deterministic `cKDTree` queries.
4. Extend the read-only audit to emit the same bidirectional fields and gate
   decision while preserving existing fields.
5. Run:

   ```bash
   pytest -q tests/oviv2/test_dense_moved_readout.py \
     tests/evaluation/test_crove_dense_readout_audit.py
   git diff --check
   ```

6. Commit: `feat: add geometry-gated dense moved readout`

## Task 9: Hybrid composition and exact fallback

**Files:**

- Modify: `src/oviv2/ovimap_static_anchor.py`
- Modify: `scripts/evaluation/run_crove_ovimap_static_anchor.py`
- Modify: `tests/oviv2/test_ovimap_static_anchor.py`
- Modify: `tests/evaluation/test_run_crove_ovimap_static_anchor.py`
- Create: `configs/evaluation/crove_dense_moved_geometry_gate_v1.json`

1. Add failing tests that an accepted gate emits translated dense geometry and
   a failed gate emits point-for-point exact temporal compact geometry.
2. Add failing tests for fallback count, per-entity decision diagnostics,
   config binding, and incompatible role/mode rejection.
3. Implement `geometry_gated_anchor_translation` without changing movement
   detection or temporal state.
4. Run focused composition, runner, T1, and audit tests.
5. Commit: `feat: integrate hybrid dense moved readout`

## Task 10: Execute and gate P6-D standalone

**Files:**

- Create: `docs/superpowers/reports/P6D_HYBRID_DENSE_READOUT_REPORT.md`
- Modify: `docs/superpowers/experiments/2026-09-03-crove-localized-current-ownership-decision-ledger.md`

1. Recompose P4A's source with the hybrid gate and no localized visibility.
2. Measure density, nearest-neighbor spacing, PLY/readout bytes, accepted and
   fallback counts, official metrics, common-v2, and exact fallback hashes.
3. Record standalone GO/NO-GO. Do not combine with L1 if either standalone
   mechanism lacks valid measurements.
4. Commit: `docs/eval: evaluate hybrid dense readout`

## Task 11: Conditional combined candidate

**Files:**

- Create if authorized: `docs/superpowers/reports/P6_COMBINED_APARTMENT_REPORT.md`
- Modify: `docs/superpowers/experiments/2026-09-03-crove-localized-current-ownership-decision-ledger.md`

1. If and only if Tasks 7 and 10 are standalone GO, compose L1 plus hybrid at
   the frozen commit/config identities.
2. Run official and common-v2 evaluation once, apply the same hard gates, and
   record the result. Otherwise record `NOT_RUN_STANDALONE_GATE`.
3. If no Apartment candidate passes all gates, record Office as
   `NOT_RUN_HELD_OUT`.
4. If one passes, freeze its commit/config/hashes before a single Office run;
   do not retune from Office.

## Task 12: Official Dyn provenance contracts

**Files:**

- Create: `src/evaluation/crove_dyn_provenance.py`
- Create: `scripts/evaluation/build_crove_dyn_provenance.py`
- Create: `tests/evaluation/test_crove_dyn_provenance.py`

1. Add failing strict-parser tests for official `dynamic_objects.csv`, bridge
   node assignments, composition entities, temporal samples, runtime
   attribution, and official visualization files.
2. Add failing binding tests from `(MapName, QueryTime, node_symbol)` to
   prediction entity and runtime state fields.
3. Add failing tests for `exact`, `ambiguous`, and `unavailable`; coordinated
   artifact tampering; duplicate identities; and aggregate mass conservation.
4. Implement the sidecar without modifying or estimating official metrics.
   Dynamic mass lacking per-sample official association remains ambiguous or
   unavailable.
5. Publish canonical JSON and CSV atomically with all source hashes.
6. Run:

   ```bash
   pytest -q tests/evaluation/test_crove_dyn_provenance.py \
     tests/evaluation/test_crove_failure_attribution.py \
     tests/evaluation/test_crove_runtime_attribution.py
   git diff --check
   ```

7. Commit: `feat/eval: add official Dyn provenance sidecar`

## Task 13: Execute and report P6-E

**Files:**

- Create: `docs/superpowers/reports/P6E_DYN_PROVENANCE_REPORT.md`
- Modify: `docs/superpowers/experiments/2026-09-03-crove-localized-current-ownership-decision-ledger.md`

1. Run the sidecar on the best valid Apartment artifact set and existing
   official evaluator visualization outputs.
2. If required runtime attribution is absent, run the instrumentation-only A4
   path once in an isolated result root and verify formal output equivalence.
3. Report exact/ambiguous/unavailable fractions and detected/missed/hallucinated
   mass by proven mechanism bucket.
4. Recommend proposal, association, motion, or future ReScene work only when
   exact dominant evidence supports it. Otherwise retain `NO_GO / NOT_NEEDED`.
5. Commit: `docs: report official Dyn provenance`

## Task 14: Final integration, verification, and upload

**Files:**

- Create: `docs/superpowers/reports/2026-09-03-crove-localized-current-ownership-handoff.md`
- Modify: `docs/superpowers/experiments/2026-09-03-crove-localized-current-ownership-decision-ledger.md`

1. Inspect the complete diff and verify no T1 protected source changed without
   an explicit P6 need.
2. Run:

   ```bash
   git diff --check
   python -m compileall src scripts tests
   pytest -q
   ```

3. Record external dependency failures exactly; do not weaken tests to hide
   them.
4. Write the handoff with repo/base/branch/local and remote SHA, every artifact
   path/hash/byte count, reproduction commands, metrics, Office status, known
   limitations, and final decisions.
5. Commit the ledger and handoff.
6. Verify clean status and recent commits, push the branch, and require exact
   local/remote SHA equality before `UPLOAD_STATUS=VERIFIED`.
