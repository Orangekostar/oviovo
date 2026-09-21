# OVI-MAP Module Validation V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the complete S/G/Q module-validation surface, bind real local assets, execute every runnable phase, and publish measured or evidence-backed blocked outcomes on the dedicated task branch.

**Architecture:** A new `src/static_ovmap/module_validation/` package owns immutable contracts, asset/split binding, native capture adapters, the three isolated modules, evaluation, frozen selection, and reporting. A single phase-aware CLI writes atomic receipts beneath one resolved attempt directory; prediction code cannot receive evaluator or GT inputs. The official OVI integration is maintained as a tracked patch against a separate clean `f8f7bcd` checkout.

**Tech Stack:** Python 3.11, NumPy, SciPy, scikit-learn, PyTorch 2.1.1 for bounded heads, existing released OVI evaluator, pybind11/C++ patch, pytest, JSON/JSONL/NPZ/PNG artifacts.

**Spec:** `docs/paper/static_ovmap/module_validation_v1/spec/CODEX_FINAL_EXECUTION_EN.md`

## Global Constraints

- Start from project commit `b6455520c758a3413988e0827b3e1f34667bdfd1` on `research/ovimap-module-validation-v1`.
- Keep the released evaluator, trajectory, class subsets, projection tolerance, historical code, configurations, and outputs unchanged.
- Use at most 8 CPU workers and one visual model on one GPU at a time; do not launch unrelated neural pipelines.
- Resolve assets in the specified precedence order and write exact `BLOCKED_*` states instead of fabricating inputs.
- FIT/CAL/SELECT require 8/2/2 independent development families; CONFIRM requires 2 separate official-validation families.
- Prediction packages contain no GT paths, correctness labels, cause-ledger columns, or future/unacquired query features.
- Disabled scientific thresholds serialize as `KEEP_ALL`; committed JSON contains no NaN or Infinity.
- S/Q preserve scene geometry, registry, and serialized ranks; G preserves source XYZ/faces/TSDF/projection and outputs complete unique partitions.
- Record a code commit before experiments; final reports and receipts may use later commits without self-referential hashes.

---

### Task 1: Contracts, Asset Binding, Splits, and Phase Orchestration

**Files:**
- Create: `src/static_ovmap/module_validation/contracts.py`
- Create: `src/static_ovmap/module_validation/assets.py`
- Create: `src/static_ovmap/module_validation/__init__.py`
- Create: `scripts/evaluation/run_ovimap_module_study.py`
- Create: `tests/module_validation/test_assets_and_contracts.py`
- Create: `tests/module_validation/test_orchestrator.py`

**Interfaces:**
- Consumes: `PROTOCOL_SPEC.json`, three historical evaluation configs, environment overrides.
- Produces: `StudySpec.load(path)`, `resolve_assets(spec, env)`, `build_scene_splits(inventory, exclusions, spec)`, `PhaseReceipt`, and CLI phases `bind|capture|semantic|geometry|query|select|confirm|report|all`.

- [x] **Step 1: Write failing tests for strict spec loading and JSON-safe status records**

```python
def test_study_spec_rejects_unresolved_runtime_metadata(spec_path):
    spec = StudySpec.load(spec_path)
    assert spec.specification_only is True
    assert spec.phases[-1] == "all"
    assert spec.output_root == Path("/mnt/shared/ww/ovimap-module-validation-v1")

def test_phase_receipt_rejects_nonfinite_metrics():
    with pytest.raises(ValueError):
        PhaseReceipt(phase="bind", status="COMPLETE", metrics={"x": float("nan")})
```

- [x] **Step 2: Run `pytest -q tests/module_validation/test_assets_and_contracts.py` and confirm missing imports fail**
- [x] **Step 3: Implement frozen dataclasses, canonical hashing, atomic JSON writes, explicit status enums, and attempt selection**
- [x] **Step 4: Add asset-resolution tests using temporary explicit/config/receipt roots and an ambiguous two-candidate fixture**
- [x] **Step 5: Implement precedence, depth-4 bounded discovery, file identities, environment/source locks, and `BLOCKED_ASSET_IDENTITY`**
- [x] **Step 6: Add deterministic family grouping/hash-order/split isolation tests with literal expected family orders**
- [x] **Step 7: Implement inventory completeness, exposed-family exclusion, 8/2/2+2 split locking, and `BLOCKED_INDEPENDENT_SCENES`**
- [x] **Step 8: Add CLI resume tests proving a compatible complete phase is reused and a changed full cache key invalidates it**
- [x] **Step 9: Implement phase dispatch, dependency statuses, atomic receipts, progress updates, and continued reporting after blocked branches**
- [x] **Step 10: Run both Task 1 test modules and commit contracts/orchestrator**

### Task 2: Native Capture Contracts and Official OVI Patch

**Files:**
- Create: `src/static_ovmap/module_validation/native_capture.py`
- Create: `tests/module_validation/test_native_capture.py`
- Create: `third_party_patches/ovimap/module_validation_v1/apply_patch.sh`
- Create: `third_party_patches/ovimap/module_validation_v1/README.md`
- Create: `third_party_patches/ovimap/module_validation_v1/ovimap_module_validation_v1.patch`
- Modify in clean upstream checkout only: `scripts/panoptic_mapping_.py`, `scripts/view_selection.py`, `mapping_ros_ws/src/consistent_panoptic_mapping/consistent_gsm/include/consistent_mapping/global_segment_map_py.h`, `mapping_ros_ws/src/consistent_panoptic_mapping/consistent_gsm/src/global_segment_map_py.cpp`

**Interfaces:**
- Consumes: active `f8f7bcd` mapper path and native scene assets.
- Produces: `NativeScenePack`, `FrameObservation`, `RegionRequest`, `exportStudyFrameState()`, `exportStudySurfaceLabels(native_xyz)`, and immutable capture JSONL/NPZ/PNG files.

- [x] **Step 1: Write failing schema tests for aligned surface rows, immutable hashes, request lineage, and no GT fields**
- [x] **Step 2: Implement schema validators and loaders without importing the native extension**
- [x] **Step 3: Read the exact upstream active methods and write failing patch-content checks for both accessor bindings and three capture hook locations**
- [x] **Step 4: Implement read-only locked C++ accessors and Python capture hooks in the clean upstream checkout**
- [x] **Step 5: Export a deterministic patch and an apply script that verifies upstream commit `f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424` before applying**
- [x] **Step 6: Build the patched extension in an isolated build/devel prefix and record compiler, source, patch, `.so`, and loaded-module identities**
- [x] **Step 7: Run one real historical two-frame smoke proving the loaded `.so` exposes numerical frame/surface state before temporary-memory clear**
- [x] **Step 8: Compare no-op surface labels to the native mesh registry and record exact parity or a concrete `BLOCKED_NATIVE_LABEL_EXPORT` receipt**
- [x] **Step 9: Run Task 2 tests and commit the adapter and tracked patch**

### Task 3: Semantic Region Evidence and Direct Readouts

**Files:**
- Create: `src/static_ovmap/module_validation/region_evidence.py`
- Create: `src/static_ovmap/module_validation/semantic_selector.py`
- Create: `tests/module_validation/test_region_evidence.py`
- Create: `tests/module_validation/test_semantic_selector.py`

**Interfaces:**
- Consumes: `NativeScenePack`, `FrameObservation`, native text cache, optional pinned SigLIP2/WOW/MiniLM adapters.
- Produces: deterministic target/view manifests, nine-vector native evidence, five direct method rows, 32+32 selector features, safe head weights, CAL teacher/threshold decisions, and SELECT predictions.

- [x] **Step 1: Write failing tests for target hash capping, three-view order, request deduplication, lineage rejection, and fixed native crop bounds**
- [x] **Step 2: Implement target/view construction and crop identity records**
- [x] **Step 3: Write a real-adapter test that the six returned unit vectors average to the legacy native encoder result within tolerance**
- [x] **Step 4: Extend the native encoder without changing its existing `encode()` contract and add background-only crop evidence**
- [x] **Step 5: Add literal vote/area tie tests and implement `S_NATIVE_AREA`, `S_NATIVE_VOTE`, `S_SIGLIP2_AREA`, `S_SIGLIP2_VOTE` readouts**
- [x] **Step 6: Add WOW 16x16 nonempty-mask, fixed-prompt, no-fallback, raw-generation, and deterministic name-mapping contract tests**
- [x] **Step 7: Implement the WOW adapter boundary so a missing real mask-conditioned hook returns `BLOCKED_WOW_MASK_INTERFACE`**
- [x] **Step 8: Add hand-computed 32-feature fixtures covering missing KEEP text, one-view pair unavailability, background absence, scaling, clipping, and availability bits**
- [x] **Step 9: Implement FIT targets, event-support gate, weighted 64-32-3 heads, CAL early stopping, `v>threshold`, and `KEEP_ALL`**
- [x] **Step 10: Run semantic tests and one real native-request smoke per available visual model**

### Task 4: Complete Geometry Partitions and Scorers

**Files:**
- Create: `src/static_ovmap/module_validation/entity_hypotheses.py`
- Create: `src/static_ovmap/module_validation/partition_quality.py`
- Create: `tests/module_validation/test_entity_hypotheses.py`
- Create: `tests/module_validation/test_partition_quality.py`

**Interfaces:**
- Consumes: numerical native leaves, source adjacency/faces, up to 32 captured frame observations.
- Produces: disjoint conflict groups, at most eight canonical complete hypotheses per group, `G_ORIGINAL`, `G_AGREEMENT`, `G_QUALITY`, and unique final owners with ancestry.

- [x] **Step 1: Write failing leaf tests for connected `(segment, owner)` components, owner0 preservation, deterministic mutual-8NN fallback, and no source-row loss**
- [x] **Step 2: Implement leaf/contact graphs and normal validity/PCA fallback**
- [x] **Step 3: Write literal evidence tests for 16-pixel/0.6 dominance, UNKNOWN handling, and once-per-frame pair counts**
- [x] **Step 4: Implement frame evidence, conflict strength, deterministic disjoint pair groups, and singleton preservation**
- [x] **Step 5: Add completeness tests for ORIGINAL, MERGE, FRAME_ENTITY, GRAPH_SPLIT2/3, canonicalization, deduplication, and eight-hypothesis cap**
- [x] **Step 6: Implement complete hypotheses with fixed spectral affinity, eigenvector sign, and k-means settings**
- [x] **Step 7: Add hand-computed Hungarian agreement and 20+20 feature/availability fixtures**
- [x] **Step 8: Implement `G_AGREEMENT`, local PQ-like targets, support gating, 40-32-1 quality training, CAL margins, and `KEEP_ALL`**
- [x] **Step 9: Add final-map tests for unchanged XYZ/faces/TSDF/projection, complete unique ownership, stable unchanged IDs, deterministic new IDs, and point-count ranks**
- [x] **Step 10: Run geometry tests and a real numerical-snapshot smoke when Task 2 integration is available**

### Task 5: Causal Query State, Policies, and Utility Head

**Files:**
- Create: `src/static_ovmap/module_validation/query_state.py`
- Create: `src/static_ovmap/module_validation/query_gain_policy.py`
- Create: `tests/module_validation/test_query_state.py`
- Create: `tests/module_validation/test_query_gain_policy.py`

**Interfaces:**
- Consumes: contemporaneous frame observations and a capability-limited `FeatureStore.acquire(request_id, budget_token)`.
- Produces: pure raw candidates, isolated policy state, four B=200 policies, logical/physical cost ledgers, 20+20 features, utility weights, and prefix-sentinel evidence.

- [x] **Step 1: Write failing pure-candidate tests for owner/area/depth/crop requirements and unchanged native combine side effects**
- [x] **Step 2: Implement candidate generation separately from native selection and paid-evidence state**
- [x] **Step 3: Add allowance tests for `floor(B*(t+1)/F)-spent`, failed-attempt debit, exact-request skip, carry-forward, and invalid-frame time advance**
- [x] **Step 4: Implement frame barriers, lineage-safe alias merging, ten-feature retention, and uniform unseen prior**
- [x] **Step 5: Add literal ranking tests for COMBINE/AREA/UNCERTAINTY/GAIN including deterministic ties and negative gain scores**
- [x] **Step 6: Implement four policies against the same raw universe without exposing the shared cache**
- [x] **Step 7: Add hand-computed 20+20 feature fixtures proving no image/current-crop/future/final-label data enters ranking**
- [x] **Step 8: Implement random FIT/CAL trace collection, identifiable targets, clipped NLL-gain labels, support gate, and 40-32-1 Huber head**
- [x] **Step 9: Add prefix-sentinel mutation test and causal cost-ledger reconciliation**
- [x] **Step 10: Run query tests and one real current-state capture smoke if lineage is available**

### Task 6: Evaluation, Frozen Selection, and Reporting

**Files:**
- Create: `src/static_ovmap/module_validation/evaluation.py`
- Create: `src/static_ovmap/module_validation/selection.py`
- Create: `src/static_ovmap/module_validation/reporting.py`
- Create: `tests/module_validation/test_evaluation.py`
- Create: `tests/module_validation/test_selection.py`
- Create: `tests/module_validation/test_reporting.py`

**Interfaces:**
- Consumes: frozen prediction manifests, evaluator protocol, per-branch costs and FIT/CAL/SELECT statuses.
- Produces: payload-keyed evaluations, post-prediction GT diagnostics, bootstrap descriptions, teacher/module/final selection, confirmation plan, exactly five principal Markdown tables.

- [x] **Step 1: Write failing evaluator tests for fixed S/Q geometry/ranks, G geometry/partition invariants, full payload keys, null undefined metrics, and exact trace parity**
- [x] **Step 2: Implement a new scene-aware adapter around existing evaluator functions without changing historical method lists**
- [x] **Step 3: Write table-driven selection tests covering every S/G/Q eligibility condition, tolerance, tie order, simpler-choice rule, combination cap, and N0 fallback**
- [x] **Step 4: Implement CAL-only teacher/head settings, SELECT-only module/final decisions, frozen confirmation rows, and 2000-resample seed-17 intervals**
- [x] **Step 5: Add reporting tests that require five principal tables and distinct implementation/experiment/science/confirmation/publication statuses**
- [x] **Step 6: Implement machine matrices, result/handoff rendering, evidence links, exact commands, cost summaries, and blocker propagation**
- [x] **Step 7: Run Task 6 tests and commit all executable code before experiments**

### Task 7: Bind and Execute Runnable Phases

**Files:**
- Create externally: `/mnt/shared/ww/ovimap-module-validation-v1/attempt_001/`
- Create: `artifacts/static_ovmap/module_validation_v1/`
- Create: `docs/paper/static_ovmap/module_validation_v1/progress.md`

**Interfaces:**
- Consumes: committed executable code and real local assets only.
- Produces: resolved config, locks, inventory/splits, phase receipts, actual measured rows, and explicit dependency/gate blockers.

- [x] **Step 1: Run `bind`; verify all historical inputs and record missing ScanNet/model assets without substitution**
- [x] **Step 2: Run `capture`; execute the permitted Room0 native smoke/replay and any preselected independent captures only**
- [x] **Step 3: Run `semantic`; measure direct methods whose actual adapters/inputs are available and gate learned rows by independent event support**
- [x] **Step 4: Run `geometry`; measure native-snapshot rows when numerical leaves exist and otherwise preserve the exact integration blocker**
- [x] **Step 5: Run `query`; measure causal policies only with contemporaneous state/lineage and label final-map-only diagnostics separately**
- [x] **Step 6: Run `select`; freeze all CAL/SELECT-derived settings and write `selection.json` before confirmation access**
- [x] **Step 7: Run `confirm` once only when a retained candidate and two untouched confirmation scenes exist**
- [x] **Step 8: Run `report`; copy only small artifacts/weights, list large paths/sizes/hashes, and reconcile five tables to machine metrics**

### Task 8: Scoped Final Audit and Publication

**Files:**
- Create: `docs/paper/static_ovmap/MODULE_VALIDATION_RESULTS.md`
- Create: `docs/paper/static_ovmap/MODULE_VALIDATION_HANDOFF.md`
- Create externally after final commit: publication receipt with local/remote SHAs.

**Interfaces:**
- Consumes: all phase receipts and the original execution contract.
- Produces: audited reports, final commit, verified remote branch, and reproducible handoff.

- [x] **Step 1: Audit every required method row against measured payload or evidenced blocker/gate status**
- [x] **Step 2: Verify prediction isolation, selection-stage isolation, G completeness, Q causal debit, numeric reconciliation, release-size limits, and no hidden fallbacks**
- [x] **Step 3: Run focused unit/integration tests, real-boundary smokes, orchestrator resume verification, `git diff --check`, and JSON/GZIP/NPZ parsing**
- [x] **Step 4: Inspect explicit staged paths and commit code/config/small artifacts/weights/reports**
- [ ] **Step 5: Push `research/ovimap-module-validation-v1` normally and compare full local SHA with `git ls-remote`**
- [ ] **Step 6: Write the external publication receipt and report separate implementation, experiment, science, confirmation, and publication statuses**
