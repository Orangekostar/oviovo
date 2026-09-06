# OVI ReScene Object-Level Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a real 3RScan RGB-D to OVI object-level temporal mapping path that separates proposal quality, cross-visit association, within-visit grouping, and historical geometry recovery.

**Architecture:** Each 3RScan visit is materialized once into the native ScanNet-style layout already accepted by OVI-MAP, then mapped independently. A new D2 loader preserves OVI entity ownership and dense surface attributes; G, F, and R consume that same immutable candidate pool. Evaluator-only GT bindings are frozen before association, composite grouping changes ownership without changing XYZ, and recovery keeps temporal identity separate from current geometry.

**Tech Stack:** Python 3.12, NumPy/SciPy, PyTorch/ReScene, OVI-MAP native environment, 3RScan RGB-D/GT, pytest, CSV/JSON/PLY artifacts.

**Spec:** `/home/ww/crove/docs/CODEX_OVI_RESCENE_OBJECT_LEVEL_TRANSFER_ALL_IN_ONE.md`

## Global Constraints

- Baseline commit is `48117bd6f502d7d289eafe81392ef55748bd6c4a`; implementation branch is `research/ovi-rescene-object-level-transfer`.
- D2 must originate from independent RGB-D OVI reconstructions, never from masked native processed arrays.
- OVI supplies dense XYZ and open-vocabulary semantics; ReScene supplies temporal evidence and never substitutes sparse tokens for the final surface.
- G/F/R use identical OVI candidates, support domain, assignment capacity, and unmatched option.
- Candidate-to-GT endpoint bindings are evaluator-only, computed before association, and identical across methods.
- Historical identity, historical shape, current shape, and current pose remain separate; confirmed-free current space is never restored.
- Main instance IoU is 0.50 with 0.25 sensitivity; voxel and distance-based 5 cm metrics retain distinct names.
- Unknown and not-applicable values are serialized as null with a reason, never silently as zero.
- Fixed resolver confidence threshold is 0.3 outside DEV6; no further threshold search is permitted on DEV6.
- Office is held out until the final architecture and parameters are frozen.
- Only real measured outputs enter result tables; unsupported conclusions and fabricated values are forbidden.

---

### Task 1: Cached resolver attribution and transfer selection

**Files:**
- Modify: `scripts/evaluation/tune_ovi_rescene_resolver.py`
- Modify: `src/evaluation/rscan_association_metrics.py`
- Test: `tests/evaluation/test_tune_ovi_rescene_resolver.py`
- Test: `tests/evaluation/test_rscan_association_metrics.py`
- Create: `configs/evaluation/manifests/ovi_rescene_object_level_pairs_v1.json`
- Generate: `configs/evaluation/results/ovi_rescene_object_level_transfer/resolver_effect_breakdown.csv`
- Generate: `configs/evaluation/results/ovi_rescene_object_level_transfer/resolver_fixed_threshold_transfer.csv`

**Interfaces:**
- Consumes: cached DEV6 `TemporalQueryEvidence`, raw masks/logits, `GroundTruthPair`, and the two eligible environments excluded by the deterministic take-first-6 rule.
- Produces: `build_fixed_endpoint_bindings(...)`, per-query attribution rows, per-environment/micro/macro fixed-threshold summaries, and a frozen pair manifest.

- [x] Write a failing example where maximum-total-IoU assignment yields fewer threshold-valid matches than cardinality-first assignment; require metric-v2 to maximize valid match count and then IoU.
- [x] Run `python -m pytest -q tests/evaluation/test_rscan_gt_instances.py tests/evaluation/test_rscan_association_metrics.py` and confirm the new assertion fails for the expected assignment definition.
- [x] Add a public evaluator-only matching mode while preserving the existing v1 result path and apply it uniformly to G/F/R.
- [x] Add failing tests showing that removing a competing low-confidence query does not alter precomputed candidate-to-GT endpoint bindings and that query outcomes classify as retained TP, endpoint failure, false re-ID, or duplicate competition.
- [x] Implement cached attribution and fixed-threshold transfer output without invoking the network for DEV6.
- [x] Freeze the two unselected eligible environments before inspecting method results; record checkpoint-validation overlap separately from resolver-held-out status.
- [x] Run the fixed 0.3 resolver on every usable transfer pair, reporting integer TP/FP/GT, micro P/R/F, environment macro P/R/F, and explicit asset limitations.
- [x] Verify with `python -m pytest -q tests/evaluation/test_tune_ovi_rescene_resolver.py tests/evaluation/test_rscan_association_metrics.py tests/evaluation/test_rscan_gt_instances.py`.
- [x] Commit the Task 1 code, manifest, and measured CSVs.

### Task 2: Real 3RScan RGB-D visit materialization

**Files:**
- Create: `scripts/evaluation/materialize_3rscan_ovi_visit.py`
- Modify: `scripts/evaluation/run_ovimap_native.py`
- Test: `tests/evaluation/test_materialize_3rscan_ovi_visit.py`
- Test: `tests/evaluation/test_run_ovimap_native.py`
- Create: `configs/evaluation/ovi_rescene_object_level_transfer_v1.json`

**Interfaces:**
- Consumes: a manifest-bound `sequence.zip`, its `_info.txt`, frame color/depth/pose members, and one visit output directory.
- Produces: ScanNet-style `color/<i>.jpg`, `depth/<i>.png`, `pose/<i>.txt`, `intrinsic/intrinsic_{color,depth}.txt`, plus `materialized_manifest.json` containing UUID, frame map, dimensions, depth shift, calibration matrices, and content bindings.

- [x] Write failing tests for parsing `_info.txt`, contiguous valid frame selection, 16-bit PGM depth preservation, 4x4 pose validation, and different RGB/depth intrinsics and dimensions.
- [x] Write a failing test requiring materialization to reject non-identity RGB-depth extrinsics until an explicit calibrated reprojection path is supplied; identity extrinsics use the native `K_depth @ inv(K_color)` warp already implemented by `ScannetLoader`.
- [x] Run `python -m pytest -q tests/evaluation/test_materialize_3rscan_ovi_visit.py` and confirm failures are caused by the missing materializer.
- [x] Implement streamed archive extraction and a canonical manifest; do not unpack unrelated mesh/GT assets into the mapper input.
- [x] Extend `run_ovimap_native.py` with an explicit `scannet_nyu` scene contract instead of weakening Replica validation; generated OVI commands must use `--dataset scannet_nyu` and real per-visit intrinsics.
- [x] Add preflight projection checks for depth scale, finite pose, image shape, and one real frame's camera/world round trip.
- [x] Materialize one selected pair, run both visits through OVI once, and store local-only paths, byte counts, SHA-256 values, runtime, and peak GPU memory.
- [x] Verify with `python -m pytest -q tests/evaluation/test_materialize_3rscan_ovi_visit.py tests/evaluation/test_run_ovimap_native.py`.
- [x] Commit Task 2 code/config and compact manifests; exclude raw RGB-D, checkpoints, and full meshes from Git.

### Task 3: D2 OVI pair loader and shared candidate contract

**Files:**
- Create: `src/evaluation/ovi_pair_views.py`
- Modify: `src/evaluation/rscan_method_views.py`
- Test: `tests/evaluation/test_ovi_pair_views.py`
- Test: `tests/evaluation/test_rscan_method_views.py`

**Interfaces:**
- Consumes: two OVI native mapping manifests and their entity/surface/semantic artifacts plus the pair's shared 3RScan global alignment.
- Produces: `OviObjectVisitView`, `OviObjectPairView`, `to_visit_maps()`, `geometric_sample()`, immutable entity-point indices, RGB, normals, OVI semantics, source-point/pixel provenance, and `D2_OVI_RECONSTRUCTION` content hashes.

- [x] Write failing tests requiring D2 construction to reject `processed_visits` and support masks, proving the old non-D0 branch cannot masquerade as D2.
- [x] Write failing synthetic loader tests for unique OVI entity ownership, dense-source conservation, per-visit independent snapshots, and exactly-once global alignment.
- [x] Run `python -m pytest -q tests/evaluation/test_ovi_pair_views.py tests/evaluation/test_rscan_method_views.py` and confirm the missing API failures.
- [x] Implement a 3RScan-specific loader with the frozen `ovimap_visit_loader.py` validation semantics and reusable `ovi_surface_attributes.py`; retain unsupported dense points in D while limiting only adapter domain A.
- [x] Build the first real D2 pair and verify entity count, point count, support fraction, RGB/normal finiteness, coordinate bounds, and source artifact identity.
- [x] Export representative dense `rgb.ply` and `instance.ply` plus fixed-camera previews from unmodified measured geometry.
- [x] Verify with `python -m pytest -q tests/evaluation/test_ovi_pair_views.py tests/evaluation/test_rscan_method_views.py tests/oviv2/test_ovi_rescene_adapter.py`.
- [x] Commit Task 3 code, tests, compact manifest, and small previews.

### Task 4: Fixed-candidate G_obj, F_obj, and R_obj association

**Files:**
- Create: `src/evaluation/object_pair_association.py`
- Modify: `src/evaluation/rscan_association_metrics.py`
- Create: `scripts/evaluation/run_ovi_rescene_object_level_transfer.py`
- Test: `tests/evaluation/test_object_pair_association.py`
- Test: `tests/evaluation/test_rscan_association_metrics.py`
- Generate: `configs/evaluation/results/ovi_rescene_object_level_transfer/shared_candidate_gfr.csv`

**Interfaces:**
- Consumes: one `OviObjectPairView`, independent per-visit Concerto features, joint ReScene `TemporalQueryEvidence`, and frozen endpoint bindings.
- Produces: one-to-one `ObjectPairPrediction` rows for `G_obj`, `F_obj`, and `R_obj`, with score, unmatched state, support, query purity, conflicts, and strict/ambiguity-aware metrics.

- [x] Write failing tests that all methods receive identical candidate IDs and that method output cannot mutate the pair view.
- [x] Write failing tests for dummy/unmatched assignment, one-to-one capacity, large displacement eligibility, independent F feature provenance, and the exact R score `max_q(c_q * a_qi_t0 * a_qj_t1)`.
- [x] Run `python -m pytest -q tests/evaluation/test_object_pair_association.py` and confirm failures reflect missing association behavior.
- [x] Implement a shared assignment solver; G uses geometry/OVI semantics, F uses independently pooled visit features, and R uses joint query support without changing candidates.
- [x] Implement fixed-endpoint metrics: end-to-end and representation-conditional recall, strict-ID and ambiguity-aware scores, endpoint failure, false re-ID, duplicates, and rigid integer TP/GT.
- [x] Run G_full/G_supported, F_obj, and R_obj on every completed D2 pair using cached OVI maps and one reusable ReScene forward per pair.
- [x] Verify with `python -m pytest -q tests/evaluation/test_object_pair_association.py tests/evaluation/test_rscan_association_metrics.py`.
- [x] Commit Task 4 code, tests, and measured shared-candidate table.

### Task 5: Composite object grouping and dense current instance readout

**Files:**
- Create: `src/evaluation/temporal_object_groups.py`
- Modify: `src/evaluation/ovi_ownership_completion.py`
- Modify: `scripts/evaluation/evaluate_ovi_ownership_completion.py`
- Test: `tests/evaluation/test_temporal_object_groups.py`
- Test: `tests/evaluation/test_ovi_ownership_completion.py`
- Generate: `configs/evaluation/results/ovi_rescene_object_level_transfer/instance_readout.csv`

**Interfaces:**
- Consumes: immutable OVI surface/entities, accepted query supports, and variants `A_ID`, `U0`, `U1`, `U2`, `U3`.
- Produces: `CompositeObject` records with visit, object ID, member entity IDs, query/support/confidence, exclusive ownership, rejected conflicts, and a dense `MapSnapshot` whose XYZ multiset is unchanged for grouping-only variants.

- [x] Write failing tests distinguishing within-object fragment merge from cross-object erroneous merge and prohibiting transitive closure beyond direct reliable query support.
- [x] Write failing tests that every source point has at most one owner, conflicts resolve deterministically by confidence/support, abstained/uncovered entities remain in the map, and grouping-only XYZ hashes remain identical.
- [x] Run `python -m pytest -q tests/evaluation/test_temporal_object_groups.py tests/evaluation/test_ovi_ownership_completion.py` and confirm the missing behavior.
- [x] Implement `A_ID` as identity-only and `U1/U2/U3` as explicit grouping variants; keep moved-object historical support out of current coordinates.
- [x] Evaluate current t1 class-agnostic instance P/R/F at IoU 0.50 and 0.25, fragments per GT, merged-GT rate, duplicate objects, support coverage, and rejected ownership.
- [x] Export the best evidence-supported dense instance map and corresponding RGB/instance previews without geometric beautification.
- [x] Verify with `python -m pytest -q tests/evaluation/test_temporal_object_groups.py tests/evaluation/test_ovi_ownership_completion.py tests/evaluation/test_evaluate_ovi_ownership_completion.py`.
- [x] Commit Task 5 code, tests, measured metrics, and compact visual artifacts.

### Task 6: D0/D1/D2 controlled transfer diagnosis and adaptation gate

**Files:**
- Modify: `scripts/evaluation/run_ovi_rescene_object_level_transfer.py`
- Modify: `scripts/evaluation/decide_ovi_rescene_adaptation.py`
- Test: `tests/evaluation/test_decide_ovi_rescene_adaptation.py`
- Generate: `configs/evaluation/results/ovi_rescene_object_level_transfer/per_pair_metrics.csv`
- Generate: `configs/evaluation/results/ovi_rescene_object_level_transfer/experiments.csv`

**Interfaces:**
- Consumes: reused D0 predictions, sensor-defined D1 support, real D2 OVI candidates, and measured raw/proposal/fixed-pool metrics.
- Produces: layer-separated domain diagnosis and one deterministic decision: no adaptation, decoder/mask-head adaptation, or upstream proposal/grouping repair.

- [x] Write failing decision tests covering D0/D1 degradation, D2-only degradation, raw-good/resolver-bad, and proposal-limited cases.
- [x] Run `python -m pytest -q tests/evaluation/test_decide_ovi_rescene_adaptation.py` and confirm expected decision failures.
- [x] Implement the diagnosis using measured thresholds declared in the config before viewing D2_EVAL results.
- [x] Run identical UUID pairs across available D0/D1/D2 layers, logging runtime, memory, support, raw query, proposal, and fixed-pool association metrics.
- [x] If and only if the gate selects domain adaptation, train one frozen-backbone decoder/mask-head candidate on environment-disjoint TRAIN_OVI_ADAPT data, select on development environments, and compare against the original checkpoint on unchanged D2_EVAL. The frozen gate selected `NO_ADAPTATION` because D1 and D2 proposal coverage were both 1/13, so no training was authorized.
- [x] Verify with `python -m pytest -q tests/evaluation/test_decide_ovi_rescene_adaptation.py tests/evaluation/test_run_3rscan_t2_matrix.py`.
- [x] Commit Task 6 decision logic, config, experiment ledger, and measured results.

### Task 7: C0/C1/C2/O1/O2 historical geometry and oracle diagnosis

**Files:**
- Create: `scripts/evaluation/run_ovi_rescene_object_recovery.py`
- Modify: `src/oviv2/two_visit_registration.py`
- Modify: `src/oviv2/two_visit_dense_recovery.py`
- Modify: `src/evaluation/ovi_ownership_completion.py`
- Test: `tests/evaluation/test_run_ovi_rescene_object_recovery.py`
- Test: `tests/oviv2/test_two_visit_registration.py`
- Test: `tests/oviv2/test_two_visit_dense_recovery.py`
- Generate: `configs/evaluation/results/ovi_rescene_object_level_transfer/completion_oracles.csv`

**Interfaces:**
- Consumes: one-to-one composite objects, G/R/GT-assisted relations, estimated or GT object transforms, t1 visibility queried at transformed candidate positions, and evaluator-only GT surfaces.
- Produces: C0/C1/C2/O1/O2 maps and a funnel from paired objects through rigid eligibility, registration, opportunity, compatibility, visibility, and new GT coverage.

- [x] Write failing synthetic SE(3) tests for official row-vector to internal column-vector conversion and normal rotation without translation.
- [x] Write failing tests showing transformed occupied/free points are rejected, unknown/occluded points remain candidates, and a moved object's old pose is not appended to current geometry.
- [x] Write failing metric tests for opportunity, new correct/wrong surface, deleted correct baseline surface, total P/R/F, and unevaluable historical surface.
- [x] Run `python -m pytest -q tests/evaluation/test_run_ovi_rescene_object_recovery.py tests/oviv2/test_two_visit_registration.py tests/oviv2/test_two_visit_dense_recovery.py` and confirm expected failures.
- [x] Add a composite-object 1:1 registration contract and an explicit oracle source type; apply identical eligibility rules to C1/C2/O1 and reserve GT transforms for O2.
- [x] Run C0/C1/C2/O1/O2 on the completed D2 pair with rigid GT opportunity, preserving negative and null outcomes.
- [x] Verify with `python -m pytest -q tests/evaluation/test_run_ovi_rescene_object_recovery.py tests/oviv2/test_two_visit_registration.py tests/oviv2/test_two_visit_dense_recovery.py tests/oviv2/test_two_visit_current_map.py`.
- [x] Commit Task 7 code, tests, maps/indexes, and measured oracle table.

### Task 8: Evidence synthesis, verification, and remote delivery

**Files:**
- Create: `docs/paper/ovi_rescene_object_level_transfer_results.md`
- Create: `docs/paper/ovi_rescene_object_level_transfer_handoff.md`
- Create: `configs/evaluation/results/ovi_rescene_object_level_transfer/compact_artifact_index.json`
- Modify: `docs/paper/ovi_rescene_object_level_transfer_plan.md`

**Interfaces:**
- Consumes: every committed manifest, experiment ledger, CSV, compact preview, local-only artifact binding, and test receipt.
- Produces: claim-bounded results, artifact index, reproducible commands, limitations, and verified remote SHA.

- [ ] Reconcile each acceptance item in the spec against an authoritative artifact; label incomplete scientific branches without hiding completed engineering work.
- [ ] Report resolver transfer, real D2 UUID/visits, shared-candidate G/F/R integer counts, grouping effects, D0/D1/D2 diagnosis, map changes, oracle bottleneck, compute reuse, and prohibited claims.
- [ ] Validate every numeric table cell against machine-readable output and preserve null/status semantics.
- [ ] Run `git diff --check` and all changed-module/direct-dependency tests; run a broader affected-suite regression only for shared contract or serialization changes.
- [ ] Run syntax compilation for every changed Python file and inspect `git status --short` for untracked required artifacts or accidental large files.
- [ ] Commit results, handoff, and compact index; push `research/ovi-rescene-object-level-transfer` without force.
- [ ] Compare `git rev-parse HEAD` with `git ls-remote origin refs/heads/research/ovi-rescene-object-level-transfer` and record `UPLOAD_STATUS=VERIFIED` only when identical.
