# OVI ReScene Dense Instance Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reassign immutable OVI dense surface points with raw ReScene temporal masks, measure where endpoint quality is lost, and evaluate P0/P1/P2 on the existing pair plus frozen D2 evaluation pairs.

**Architecture:** Keep each visit's OVI XYZ and source row order immutable. Build one explicit D-to-model reverse map, use raw query logits to create overlapping diagnostic proposals and a deterministic exclusive P2 ownership map, and preserve unsupported points as OVI residual/background/unknown. GT remains evaluator-only in a separate diagnosis module; association and current-map stages consume the method output without GT.

**Tech Stack:** Python 3.12, NumPy, SciPy, PyTorch/ReScene cached forwards, OVI-MAP native artifacts, 3RScan RGB-D/GT, pytest, CSV/JSON/PLY/PNG.

**Spec:** `/home/ww/crove/docs/CODEX_OVI_RESCENE_DENSE_INSTANCE_REPAIR_ALL_IN_ONE.md`

## Global Constraints

- Baseline is `dab81331017c7359cf2775f82351c97d2fe1c8bd`; branch is `research/ovi-rescene-dense-instance-repair`.
- P0, P1, and P2 use the same immutable OVI dense XYZ; segmentation changes ownership only.
- Raw model masks are `M x Q`; dense rows are recovered through supported source indices and original OVI vertex indices, never by treating compact indices as dense rows.
- The P2 query threshold is fixed at `0.3`; point support is `logit > 0`; winner score is `query_score * sigmoid(logit)` with raw query index as the final tie-break.
- Every dense point appears exactly once as query-owned, OVI residual, background, or unknown; unsupported points are not negatives.
- GT can enter endpoint diagnosis and evaluation only. It cannot select queries, alter ownership, assign open-vocabulary semantics, or enter map construction.
- Main instance metric is exact-floor 5 cm IoU at `0.50`; `0.25` is sensitivity. Distance diagnostics use a distinct name.
- P0/P1/P2 candidate-to-GT bindings are frozen independently before association. Null denominators remain null with a reason.
- `scene0109_00-scene0109_01` remains the development pair. `scene0359_00-scene0359_01` and `scene0459_00-scene0459_01` remain frozen D2_EVAL pairs.
- Large PLY/NPZ/checkpoints remain local-only with path, size, and SHA-256 receipts; compact code, config, CSV/JSON, selected PNG, results, and handoff are committed.
- Core design, algorithm, debugging, integration, and final review stay with the primary agent. Mechanical workers may only execute explicitly specified inventory or fixed commands.

---

### Task 1: Evaluator-only endpoint diagnosis

**Files:**
- Create: `src/evaluation/ovi_endpoint_diagnosis.py`
- Create: `tests/evaluation/test_ovi_endpoint_diagnosis.py`

**Interfaces:**
- Consumes: `OviObjectVisitView`, one visit's `GroundTruthInstance` tuple, raw dense query proposals, confident query IDs, and final P2 `PredictedInstance` values.
- Produces: `EndpointDiagnosisRow`, `RawDenseProposal`, `diagnose_visit_endpoints(...)`, and `endpoint_diagnosis_rows(...)` without mutating method inputs.

- [x] Write RED tests proving `D intersect GT` never adds GT-only voxels; exact union search covers all subsets for at most 12 intersecting candidates; larger greedy search is labeled non-exact and is never below best-single IoU.
- [x] Write RED tests for full/owned/supported coverage, best/second atomic precision-recall-IoU, raw/all versus confidence-0.3 proposals, and final-P2 IoU.
- [x] Write RED tests that missing sensor-visible GT is serialized as null with `NOT_COMPUTED`, and provisional failure evidence may remain `mixed` instead of forcing a single cause.
- [x] Run `python -m pytest -q tests/evaluation/test_ovi_endpoint_diagnosis.py` and verify failure is caused by the missing module.
- [x] Implement voxel-set operations and bounded union search; keep GT objects and selected D voxels immutable.
- [x] Run `python -m pytest -q tests/evaluation/test_ovi_endpoint_diagnosis.py tests/evaluation/test_rscan_gt_instances.py`.
- [x] Commit Task 1.

### Task 2: Dense D-to-M projection and exclusive P2 ownership

**Files:**
- Create: `src/oviv2/rescene_dense_instance_readout.py`
- Create: `tests/oviv2/test_rescene_dense_instance_readout.py`

**Interfaces:**
- Produces `DenseInstance`, `DenseVisitReadout`, `DensePairReadout` and:

```python
def build_dense_to_model_indices(
    pair: OviObjectPairView,
    *,
    surface: SurfaceAttributeBundle,
    supported_view: SupportedInferenceView,
) -> tuple[np.ndarray, np.ndarray]: ...

def build_dense_instance_readout(
    pair: OviObjectPairView,
    *,
    surface: SurfaceAttributeBundle,
    supported_view: SupportedInferenceView,
    pred_masks_mq: np.ndarray,
    pred_logits_qc: np.ndarray,
    minimum_query_score: float = 0.3,
    point_chunk_size: int = 131072,
) -> DensePairReadout: ...
```

- `DenseVisitReadout` stores dense-row-aligned `owner_instance_indices`, `owner_source_codes`, `neural_valid`, and compact instance records containing raw query, temporal hypothesis, parent OVI point counts, semantic provenance, and conflict state.

- [x] Write a RED D-to-A-to-M-to-D toy test with different visit row counts, unsupported dense rows, many source contributors to one model row, and nonidentity original vertex indices.
- [x] Write RED tests where one OVI entity is split by two queries and one query merges two OVI entities while preserving every source point exactly once.
- [x] Write RED overlap/tie tests requiring maximum `query_score * sigmoid(logit)`, then lower raw query index; verify query-owned points are removed from residual owners.
- [x] Write RED tests that unsupported owned points remain residual, unsupported unowned points remain explicit background/unknown, and cross-visit rows never mix.
- [x] Write a RED million-row-shaped chunking test that forbids allocating a dense `N x Q` result and confirms input arrays and XYZ are not mutated.
- [x] Run `python -m pytest -q tests/oviv2/test_rescene_dense_instance_readout.py` and verify the missing module is the failure.
- [x] Implement the immutable reverse map, query score semantics matching `postprocess_native_predictions()`, chunked winner selection, residual fallback, parent provenance, and content hash.
- [x] Run `python -m pytest -q tests/oviv2/test_rescene_dense_instance_readout.py tests/oviv2/test_rescene_supported_view.py tests/evaluation/test_run_ovi_rescene_object_level_transfer.py`.
- [x] Commit Task 2.

### Task 3: P0/P1/P2 evaluation, identity hypotheses, and dense artifacts

**Files:**
- Create: `src/evaluation/dense_instance_repair_metrics.py`
- Create: `src/evaluation/dense_instance_repair_artifacts.py`
- Create: `tests/evaluation/test_dense_instance_repair_metrics.py`
- Create: `tests/evaluation/test_dense_instance_repair_artifacts.py`

**Interfaces:**
- Convert P0 OVI entities, P1 U3 composite objects, and P2 exclusive owners into `PredictedInstance` tuples over the same visit surface.
- Produce full-visit instance rows at IoU 0.50/0.25, endpoint bindings per candidate pool, split/merge/duplicate counts, query/residual/background/unknown fractions, source/final counts, and `geometric_change_count=0`.
- Export both visits and t1 current readout as PLY plus fixed-camera RGB/instance PNG; PLY vertex order and XYZ equal the source visit exactly.

- [x] Write RED tests that P0/P1/P2 share source point counts and XYZ, while P2 candidate masks can cross or split OVI entity boundaries.
- [x] Write RED tests for TP/FP/FN/P/R/F1, raw/final counts, owner fractions, split/merge/duplicate fields, and null conditional recall when no GT identity has both endpoints.
- [x] Write a RED test that endpoint bindings are built before relation scoring and differ when the candidate pool changes.
- [x] Write RED artifact tests for exact vertex order, one-based PLY instance indices, deterministic colors, and explicit background/unknown encoding.
- [x] Implement metrics and exporters by reusing `evaluate_instance_geometry`, fixed endpoint semantics, and existing preview helpers.
- [x] Run `python -m pytest -q tests/evaluation/test_dense_instance_repair_metrics.py tests/evaluation/test_dense_instance_repair_artifacts.py tests/evaluation/test_temporal_object_groups.py tests/evaluation/test_rscan_association_metrics.py`.
- [x] Commit Task 3.

### Task 4: Cached development-pair runner and measured P0/P1/P2 evidence

**Files:**
- Create: `scripts/evaluation/run_ovi_rescene_dense_instance_repair.py`
- Create: `tests/evaluation/test_run_ovi_rescene_dense_instance_repair.py`
- Create: `configs/evaluation/ovi_rescene_dense_instance_repair_v1.json`
- Generate: `configs/evaluation/results/ovi_rescene_dense_instance_repair/endpoint_diagnosis.csv`
- Generate: `configs/evaluation/results/ovi_rescene_dense_instance_repair/instance_metrics.csv`
- Generate: `configs/evaluation/results/ovi_rescene_dense_instance_repair/identity_metrics.csv`
- Generate: `configs/evaluation/results/ovi_rescene_dense_instance_repair/runs.csv`

**Interfaces:**
- Rebuild and bind the existing D2 pair, rebuild its supported bundle, load the existing `native_forwards.npz`, validate M/Q shapes and metadata, then run Tasks 1–3 without a new forward.
- Publish outputs exclusively under `/home/ww/oviovo_baseline_runs/20260907_ovi_rescene_dense_instance_repair/scene0109_00-scene0109_01/` and compact CSV/JSON under the tracked results directory.

- [x] Write RED orchestration tests using real module objects and tiny temporary artifacts; require cache identity mismatch, M/Q mismatch, or pair mismatch to fail before output publication.
- [x] Write RED tests for stable CSV columns matching Tables A/B/C/E and for null/status serialization.
- [x] Implement config loading, bound pair/cache/GT restoration, P0/P1/P2 construction, endpoint diagnosis, metrics, artifacts, and measured runtime collection.
- [x] Run targeted tests, commit code/config, then run the real cached development pair at an explicit evaluated commit.
- [x] Inspect numeric output and fixed previews. Record whether P2 actually splits/merges, where raw-to-exclusive loss occurs, and which failure category dominates.
- [x] Commit compact measured development evidence and selected PNG; keep full PLY/NPZ local-only.

### Task 5: Frozen new-pair selection and independent OVI reconstruction

**Files:**
- Create: `configs/evaluation/manifests/ovi_rescene_dense_instance_repair_pairs_v1.json`
- Generate: `configs/evaluation/results/ovi_rescene_dense_instance_repair/pair_selection.csv`
- Reuse without semantic changes: `scripts/evaluation/materialize_3rscan_ovi_visit.py`, `scripts/evaluation/run_ovimap_native.py`, `src/evaluation/ovi_pair_views.py`

**Interfaces:**
- Freeze current development pair and both existing D2_EVAL pairs before P2 results from those pairs are inspected.
- Each new visit is materialized and independently mapped once; all methods reuse that OVI artifact.

- [ ] Record pair role, UUIDs, asset presence, selection reason, checkpoint-validation overlap, rigid/repeated/static strata, and frozen threshold/config before method execution.
- [ ] Materialize and map `scene0359_00-scene0359_01` and `scene0459_00-scene0459_01`, using distinct GPUs when independent commands can run concurrently.
- [ ] Build GT sidecars, OVI pair artifacts, supported bundles, and one cached ReScene forward per pair; correct reproducible implementation errors uniformly.
- [ ] Record runtime, peak allocated/reserved memory, paths, sizes, hashes, and any real asset failure. Never replace a failed EVAL pair based on its score.
- [ ] Commit the frozen selection and compact mapping/forward receipts; keep large artifacts local-only.

### Task 6: Fixed multi-pair evaluation and one evidence-selected repair

**Files:**
- Modify only the Task 1–4 module proven responsible by the development diagnosis.
- Modify its corresponding focused test file first.
- Generate: `configs/evaluation/results/ovi_rescene_dense_instance_repair/followup_decision.json`
- Generate: updated multi-pair `endpoint_diagnosis.csv`, `instance_metrics.csv`, `identity_metrics.csv`, and `runs.csv`.

**Interfaces:**
- The follow-up decision is exactly one of `P2_FINAL`, `P3_BOUNDARY`, `RECONSTRUCTION_REPAIR`, `DECODER_MASK_ADAPTATION`, selected on development evidence before D2_EVAL metrics are opened.
- `P3_BOUNDARY` may only split a raw-query parent into fixed single-visit geometric components and marks their cross-visit identity uncertain; it cannot tune thresholds per scene.
- Adaptation is permitted only if D0 and an actual re-forwarded D1 are materially better than D2 raw masks with comparable GT support; training environments remain disjoint from D2_EVAL.

- [ ] Freeze `followup_decision.json` from development endpoint rows using the Section 9 evidence table; record the chosen component, evidence rows, expected effect, and forbidden concurrent changes.
- [ ] If the decision is not `P2_FINAL`, write a RED regression reproducing the measured failure and implement only the selected repair; rerun the development pair against unchanged P0/P1.
- [ ] If partial observation is causal, construct camera/depth-defined D1 input and run an actual new D1 forward; otherwise preserve old D1 metadata as cached-projection only.
- [ ] Freeze the final method/config, run all available frozen D2_EVAL pairs once, and report per-pair plus micro/macro P0/P1/P2 results without retuning.
- [ ] Rebuild fixed endpoint bindings for every final candidate pool and evaluate system identity plus fixed-P2-candidate G/F/R with explicit dependency on the joint-query-derived pool.
- [ ] Export P0/P1/P2 visits and t1 current readouts for every completed pair. Run C0/C_G/C_R/O_ID/O_POSE only when new rigid endpoints or explicit recoverable surface exist; otherwise record `NOT_RUN_INELIGIBLE` with measured reasons.
- [ ] Run all changed-module and direct-dependency tests and commit the final method and compact results.

### Task 7: Evidence audit, handoff, and remote delivery

**Files:**
- Create: `docs/paper/ovi_rescene_dense_instance_repair_results.md`
- Create: `docs/paper/ovi_rescene_dense_instance_repair_handoff.md`
- Create: `configs/evaluation/results/ovi_rescene_dense_instance_repair/compact_artifact_index.json`
- Modify: `docs/paper/ovi_rescene_dense_instance_repair_plan.md`

- [ ] Validate every numeric table cell against machine-readable outputs and preserve null/status meanings.
- [ ] Answer every item in Section 14.3, including actual pairs, split/merge and XYZ conservation, raw-to-dense loss, D1/adapter/recovery status, claims, compute, and local-only assets.
- [ ] Include Tables A–E, representative success and failure previews, and an explicit supported/unsupported claim ledger.
- [ ] Run `git diff --check`, compile every changed Python file, and run all changed-module/direct-dependency tests. Run broader regression only for shared interfaces.
- [ ] Inspect the full branch diff, frozen baseline boundaries, accidental large files, and untracked required artifacts.
- [ ] Commit code, compact evidence, plan, results, and handoff; push without force.
- [ ] Compare local and remote branch SHA and report `UPLOAD_STATUS=VERIFIED` only when identical.
