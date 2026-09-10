# CROVE Entity-Episode Dynamics V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a source-preserving two-visit current-map update that keeps stable entity identity separate from location episodes, inherits B3 tombstones, and permits only evidence-backed local correction.

**Architecture:** Keep the V1 OVI-native surface and public readout unchanged. Add one aligned entity-episode sidecar, one relation builder/validator, and a versioned runner; all D2/D3/D5/D6 variants call the same state transition kernel, while D4 aliases identity without changing geometry or semantics. Existing G0, ReScene D-to-A-to-M mapping, registration, prototype banks, evaluators, and exporters are reused rather than replaced.

**Tech Stack:** Python 3.12, NumPy, SciPy `cKDTree`/assignment, pytest, existing OVI-MAP/TESSE/ReScene adapters, JSON/CSV/NPZ artifacts.

**Spec:** `/home/ww/crove/docs/CODEX_CROVE_ENTITY_EPOCH_DYNAMICS_V2_ALL_IN_ONE.md`

## Global Constraints

- Evidence base is `d5c0688bc662f8e65455cb9c62909de87c941b43`; evaluated code and delivery commit remain separate identities.
- XYZ and triangles come only from bound OVI t0/t1 native surfaces; no GT geometry, Poisson fill, TSDF rebuild, or transformed historical mesh enters a scored map.
- Existing `CurrentEvidenceState` numeric values and V1 behavior remain unchanged.
- A stable identity match never revives or deletes an old location by itself.
- Initial t0 validity and retirement reason are inherited from B3; absent new evidence means no state change.
- Only one-to-one accepted relations with explicit local source rows may authorize a state action; split, merge, ambiguity, and null remain unresolved.
- D4 uses exactly the D1 state mask and per-point semantics; only the identity alias sidecar may differ.
- All variants share the same t0/t1 surfaces, RGB-D window, semantic crosswalk, evaluation context, 5 cm metrics, and DEV configuration budget.
- ReScene predictions are pair-bound and checkpoint-bound; outputs from another pair and fabricated RGB, normals, embeddings, or query vectors are forbidden.
- The runner exposes exactly `D0_T1`, `D1_B3`, `D2_INHERIT`, `D3_GEOM`, `D4_RESCENE_ID_ONLY`, `D5_RESCENE_EPOCH`, `D6_MEMORY_EPOCH`, and optional diagnostic `DX_ORACLE_REL`.
- The deterministic prototype method is named `CROVE_MEMORY`; it is not reported as a full AutoSeg3D reproduction.
- Office is confirmation-only after a single DEV configuration is frozen; Office data is never used for tuning or training.
- GT current-valid, removal, and Ghost labels are evaluator-only and never prediction inputs.

---

### Task 1: Timed Evidence and B3 Prior Semantics

**Files:**
- Modify: `src/oviv2/fine_surface_validity.py`
- Modify: `src/oviv2/fine_dynamic_policy.py`
- Test: `tests/oviv2/test_fine_surface_validity.py`
- Test: `tests/oviv2/test_fine_dynamic_policy.py`

**Interfaces:**
- Consumes: B3 `provenance.jsonl`, native t0 owner IDs, and ordered `Frame` values.
- Produces: `PriorSurfaceState`, `SurfaceRetirementReason`, `load_b3_prior_surface_state(...)`, and optional `FineSurfaceEvidence.last_absent_frames` / `last_occluded_frames` aligned with existing evidence arrays.

- [ ] **Step 1: Write failing reason-preservation tests**

```python
prior = load_b3_prior_surface_state(path, np.asarray([1, 1, 2]))
assert prior.current_valid.tolist() == [False, False, True]
assert prior.retirement_reason_codes.tolist() == [
    SurfaceRetirementReason.DIRECT_FREE,
    SurfaceRetirementReason.ENTITY_LIFT,
    SurfaceRetirementReason.NONE,
]
```

Cover every B3 action, duplicate/missing row rejection, immutable arrays, and exact source-row coverage. `suppress_t0_visible_free` maps to `DIRECT_FREE`; `suppress_t0_entity_visible_free` maps to `ENTITY_LIFT`; `suppress_t0_occupied_by_t1` maps to `REPLACED`; retained actions map to `NONE`.

- [ ] **Step 2: Run the new tests and observe the missing-interface failure**

Run: `python -m pytest -q tests/oviv2/test_fine_dynamic_policy.py`

- [ ] **Step 3: Implement the prior-state loader without changing `load_b3_surface_policy` output**

```python
class SurfaceRetirementReason(IntEnum):
    NONE = 0
    DIRECT_FREE = 1
    COARSE_NEIGHBORHOOD = 2
    ENTITY_LIFT = 3
    REPLACED = 4

@dataclass(frozen=True, slots=True)
class PriorSurfaceState:
    current_valid: np.ndarray
    retirement_reason_codes: np.ndarray
    evidence_state_codes: np.ndarray
```

Factor the provenance row traversal into one private iterator so the legacy policy and detailed prior are derived from identical validated records.

- [ ] **Step 4: Write failing timed-evidence tests**

Test that repeated use of the same frame tuple cannot create duplicate frame identities, present/absent/occluded timestamps equal the latest contributing frame, and invalid-depth/uncaptured rows remain `-1`.

- [ ] **Step 5: Add optional timestamp arrays and populate them during projection**

```python
@dataclass(frozen=True, slots=True)
class FineSurfaceEvidence:
    # existing fields remain first and source-compatible
    last_absent_frames: np.ndarray | None = None
    last_occluded_frames: np.ndarray | None = None
```

Normalize omitted arrays to aligned immutable `int32` arrays filled with `-1`; update `observe_fine_surface`, `assemble_fine_surface_evidence`, and legacy evidence construction to carry honest timestamps.

- [ ] **Step 6: Run focused and V1 regression tests**

Run: `python -m pytest -q tests/oviv2/test_fine_surface_validity.py tests/oviv2/test_fine_dynamic_policy.py tests/oviv2/test_fine_current_composer.py tests/evaluation/test_run_crove_fine_current_map.py`

- [ ] **Step 7: Commit the evidence contract**

```bash
git add src/oviv2/fine_surface_validity.py src/oviv2/fine_dynamic_policy.py tests/oviv2/test_fine_surface_validity.py tests/oviv2/test_fine_dynamic_policy.py
git commit -m "Add timed B3 prior surface evidence"
```

### Task 2: Entity-Episode State Transition Kernel

**Files:**
- Create: `src/oviv2/entity_epoch_update.py`
- Create: `tests/oviv2/test_entity_epoch_update.py`

**Interfaces:**
- Consumes: `PriorSurfaceState`, timed t1 evidence, t0 owner IDs, t1 replacement coverage, accepted `RelationSupport` records, and `EntityEpochUpdateConfig`.
- Produces: `EntityEpisodeRecord`, `SurfaceStateDelta`, `EntityEpochUpdateResult`, and `resolve_entity_epoch_update(...)`.

- [ ] **Step 1: Write failing immutable-contract tests**

```python
result = resolve_entity_epoch_update(
    prior=prior,
    t0_owner_entity_ids=owners,
    evidence=evidence,
    t1_replacement_rows=replacement_rows,
    relation_support=(),
    config=EntityEpochUpdateConfig(),
)
assert np.array_equal(result.current_valid, prior.current_valid)
assert result.surface_deltas == ()
```

Also reject misaligned rows, invalid reason codes, duplicate relation authorization, cross-owner local rows, and writeable output arrays.

- [ ] **Step 2: Run the contract tests and observe import failure**

Run: `python -m pytest -q tests/oviv2/test_entity_epoch_update.py`

- [ ] **Step 3: Implement minimal types and baseline inheritance**

```python
class EpisodeLifecycle(str, Enum):
    ACTIVE = "active"
    DORMANT = "dormant"
    LOCATION_RETIRED = "location_retired"
    UNCERTAIN = "uncertain"

@dataclass(frozen=True, slots=True)
class SurfaceStateDelta:
    source_surface_id: str
    source_vertex_index: int
    stable_entity_id_before: str | None
    stable_entity_id_after: str | None
    episode_id_before: str
    episode_id_after: str
    current_valid_before: bool
    current_valid_after: bool
    retirement_reason_before: SurfaceRetirementReason
    action_reason: str
    evidence_frame_ids: tuple[int, ...]
    used_new_measurement: bool
```

Stable IDs alias related owners, while each source visit starts a distinct location episode unless accepted static registration explicitly confirms the same location.

- [ ] **Step 4: Write failing action-table tests**

Cover: no new evidence preserves tombstones; relation-only reactivation preserves tombstones; reliable direct-free stays invalid; t1 overlap marks old row replaced; late positive support without t1 mesh coverage can locally recover; entity-lift recovery changes only supported rows; no match plus occlusion yields dormant; centroid-only movement is unresolved.

- [ ] **Step 5: Implement the ordered transition rules**

Apply in fixed precedence: `t1 replacement > reliable negative evidence > late local positive correction > inherited state`. A relation may supply identity/episode context but never substitutes for a missing row-level positive or negative observation.

- [ ] **Step 6: Add idempotence and reversibility tests**

Feed the same evidence twice and require identical arrays/deltas; demonstrate one previously invalid local row returning valid after a strictly later positive measurement while neighboring tombstones remain invalid.

- [ ] **Step 7: Run the kernel tests**

Run: `python -m pytest -q tests/oviv2/test_entity_epoch_update.py`

- [ ] **Step 8: Commit the state kernel**

```bash
git add src/oviv2/entity_epoch_update.py tests/oviv2/test_entity_epoch_update.py
git commit -m "Implement entity episode surface updates"
```

### Task 3: G1 Relation Candidates, Null Assignment, and Motion Verification

**Files:**
- Create: `src/oviv2/entity_epoch_relations.py`
- Create: `tests/oviv2/test_entity_epoch_relations.py`
- Modify: `src/oviv2/geometric_pair_reasoner.py`
- Test: `tests/oviv2/test_geometric_pair_reasoner.py`

**Interfaces:**
- Consumes: `NeuralSampleMap`, OVI entity semantics/descriptors, per-entity source points, `RegistrationConfig`, and fixed DEV thresholds.
- Produces: `RelationSupport`, `StrongGeometricRelationConfig`, `build_g1_relation_support(...)`, `relation_support_from_queries(...)`, and `verify_relation_motion(...)`.

- [ ] **Step 1: Write failing null-assignment and soft-semantics tests**

```python
relations = build_g1_relation_support(pair, StrongGeometricRelationConfig(
    minimum_assignment_margin=0.08,
    minimum_relation_score=0.55,
    null_score=0.50,
))
assert unmatched_id not in {r.t0_entity_id for r in relations if r.accepted}
assert zero_iou_moved_pair.accepted
```

Verify that low coverage does not fall back to all intersected records, label disagreement changes a score rather than vetoing a candidate, and one t1 entity cannot be assigned twice.

- [ ] **Step 2: Run the relation tests and observe missing symbols**

Run: `python -m pytest -q tests/oviv2/test_entity_epoch_relations.py tests/oviv2/test_geometric_pair_reasoner.py`

- [ ] **Step 3: Implement G1 scoring and Hungarian one-to-one assignment**

Score normalized centered-shape agreement, size/extent agreement, registration support, visible support, optional same-feature-space cosine, soft semantic compatibility, and original-location IoU. Represent unavailable features as `None` in `RelationSupport`; never substitute category equality for a descriptor.

- [ ] **Step 4: Reuse registration and add spatial-support interpretation**

```python
motion = verify_relation_motion(
    relation=relation,
    source_points_xyz=t0_points,
    target_points_xyz=t1_points,
    registration_config=registration_config,
    minimum_separated_patches=3,
    patch_voxel_size_m=0.05,
)
assert motion.state in {"static", "moved", "unresolved"}
```

Require accepted bidirectional registration, at least three occupied support patches, and a sufficient residual improvement over identity before returning `moved`; symmetric/planar/low-support cases remain unresolved.

- [ ] **Step 5: Preserve G0 and expose G1 as a separate backend identity**

Do not silently change `GeometricPairReasoner`. Add only shared summary access needed by G1 or keep the new implementation isolated if no stable helper can be reused.

- [ ] **Step 6: Run relation and registration suites**

Run: `python -m pytest -q tests/oviv2/test_entity_epoch_relations.py tests/oviv2/test_geometric_pair_reasoner.py tests/oviv2/test_two_visit_registration.py tests/oviv2/test_query_instance_projection.py`

- [ ] **Step 7: Commit the relation layer**

```bash
git add src/oviv2/entity_epoch_relations.py src/oviv2/geometric_pair_reasoner.py tests/oviv2/test_entity_epoch_relations.py tests/oviv2/test_geometric_pair_reasoner.py
git commit -m "Add conservative entity relation verification"
```

### Task 4: ReScene and CROVE Memory Evidence Adapters

**Files:**
- Create: `src/oviv2/entity_epoch_memory.py`
- Create: `tests/oviv2/test_entity_epoch_memory.py`
- Modify: `src/oviv2/query_instance_projection.py`
- Test: `tests/oviv2/test_query_instance_projection.py`
- Modify only if required by real output: `scripts/evaluation/rescene_pair_executor.py`
- Test: `tests/evaluation/test_rescene_pair_executor.py`

**Interfaces:**
- Consumes: a pair-bound `TemporalQueryEvidence`, its exact `NeuralSampleMap`, OVI candidate rows, and same-feature-space per-view descriptors.
- Produces: strict ReScene `RelationSupport` records, `CroveMemoryIndex`, and `build_memory_relation_support(...)`.

- [ ] **Step 1: Write failing strict-projection tests**

Add a versioned strict selector that returns null when no entity passes both frozen coverage and competition-margin gates. Keep legacy `project_queries_to_instances(...)` behavior unchanged for old artifacts.

- [ ] **Step 2: Run projection tests and observe the expected failure**

Run: `python -m pytest -q tests/oviv2/test_query_instance_projection.py`

- [ ] **Step 3: Implement strict query-to-fixed-OVI support**

```python
supports = project_queries_to_relation_support(
    pair,
    evidence,
    minimum_source_coverage=0.25,
    minimum_competition_margin=0.08,
)
assert all(item.relation_source == "frozen-rescene" for item in supports)
```

Record two-end coverage, purity, competing score, margin, query confidence, and exact local source rows. A query may emit zero accepted relations.

- [ ] **Step 4: Write failing memory-bank tests**

Cover active-before-dormant search, model-space separation, last-prototype versus multiview-bank modes, duplicate-frame suppression, and identity reactivation that leaves prior location validity unchanged.

- [ ] **Step 5: Implement deterministic memory retrieval**

Reuse `FeaturePrototypeBank` and `InformativeViewBank`; key every bank by `(stable_entity_id, feature_space_id, projection_version)`. Return relation evidence only, never geometry or a current mask. Label final-support replay as `OFFLINE_REPLAY_ON_FINAL_OVI_SUPPORT`.

- [ ] **Step 6: Run a real pair-bound ReScene forward or reuse an exactly matching cached forward**

Use `rescene_pair_executor.py` with the pinned checkout, checkpoint, Concerto feature checkpoint, pair hash, `rgb_normals`, and 0.02 m neural voxel size. Verify the manifest and arrays bind to the current pair before relation projection. Add executor fields only if the real trial proves an existing field is insufficient; never infer unavailable query embeddings from old logits/masks.

- [ ] **Step 7: Run focused suites**

Run: `python -m pytest -q tests/oviv2/test_entity_epoch_memory.py tests/oviv2/test_query_instance_projection.py tests/evaluation/test_rescene_pair_executor.py tests/oviv2/test_semantic_memory.py`

- [ ] **Step 8: Commit the evidence adapters**

```bash
git add src/oviv2/entity_epoch_memory.py src/oviv2/query_instance_projection.py tests/oviv2/test_entity_epoch_memory.py tests/oviv2/test_query_instance_projection.py scripts/evaluation/rescene_pair_executor.py tests/evaluation/test_rescene_pair_executor.py
git commit -m "Connect ReScene and memory entity evidence"
```

### Task 5: Source-Preserving Composition and Artifact I/O

**Files:**
- Modify: `src/oviv2/fine_current_composer.py`
- Create: `src/oviv2/entity_epoch_io.py`
- Modify: `src/oviv2/current_surface.py` only if a new evidence-state code is demonstrably required
- Create: `tests/oviv2/test_entity_epoch_io.py`
- Modify: `tests/oviv2/test_fine_current_composer.py`

**Interfaces:**
- Consumes: two `FineVisitSurface` values and `EntityEpochUpdateResult` aligned to t0.
- Produces: `EntityEpochComposition(surface, state_sidecar)` and hash-bound NPZ/JSON sidecars.

- [ ] **Step 1: Write failing composition tests**

```python
composition = compose_entity_epoch_fine_surface(
    t0=t0,
    t1=t1,
    update=update,
    surface_id="crove-entity-epoch-pair",
)
assert np.array_equal(composition.surface.vertices_xyz[:len(t0.vertices_xyz)], t0.vertices_xyz)
assert np.array_equal(composition.surface.current_valid[:len(t0.vertices_xyz)], update.current_valid)
```

Require source-surface/vertex keys, triangle locality, D4 mask and semantic equality, and one canonical surface feeding RGB/instance/semantic/state exports.

- [ ] **Step 2: Run the tests and observe missing composition failure**

Run: `python -m pytest -q tests/oviv2/test_fine_current_composer.py tests/oviv2/test_entity_epoch_io.py`

- [ ] **Step 3: Implement the new composer as a separate entry point**

Map valid inherited/recovered rows to existing valid evidence states, invalid direct/entity-free rows to `REVOKED_VISIBLE_FREE`, and invalid overlapped rows to `REPLACED_BY_CURRENT`; preserve detailed reasons exclusively in the aligned sidecar.

- [ ] **Step 4: Implement atomic NPZ/JSON write and load**

Persist stable entity IDs, episode IDs, prior/new validity, prior reason, action reason, evidence frame references, relation IDs, and source keys. The loader recomputes the NPZ hash and validates array lengths against the canonical surface.

- [ ] **Step 5: Run save/reload and export regressions**

Run: `python -m pytest -q tests/oviv2/test_entity_epoch_io.py tests/oviv2/test_fine_current_composer.py tests/oviv2/test_current_surface.py`

- [ ] **Step 6: Commit composition and I/O**

```bash
git add src/oviv2/fine_current_composer.py src/oviv2/entity_epoch_io.py tests/oviv2/test_fine_current_composer.py tests/oviv2/test_entity_epoch_io.py
git commit -m "Compose source-bound entity episode maps"
```

### Task 6: Metrics and Action Attribution

**Files:**
- Create: `src/evaluation/entity_epoch_metrics.py`
- Create: `tests/evaluation/test_entity_epoch_metrics.py`

**Interfaces:**
- Consumes: canonical surface, `EntityEpochUpdateResult`, fixed GT/evaluation support supplied only after prediction, and relation labels when authorized.
- Produces: `EntityEpochDiagnostics` plus rows for action attribution and relation diagnostics.

- [ ] **Step 1: Write failing exact-count tests**

```python
metrics = evaluate_entity_epoch_actions(update, evaluation_support)
assert metrics.deleted_supported_source_rows == 1
assert metrics.bad_recovery_source_rows == 1
assert metrics.correct_new_coverage_unique_voxels == 1
```

Cover source-row and 5 cm unique-voxel counts, free conflicts including background/unknown, correct recovery excluding t1-covered points, relation precision/recall/rejection, and large-motion ambiguity.

- [ ] **Step 2: Run the metrics tests and observe import failure**

Run: `python -m pytest -q tests/evaluation/test_entity_epoch_metrics.py`

- [ ] **Step 3: Implement diagnostics without modifying legacy metric numerators**

Call `evaluate_two_visit_snapshot(...)` unchanged for headline metrics. Compute new diagnostics beside it from the same `current_valid` mask and fixed evaluator-only support.

- [ ] **Step 4: Add D4 invariance and denominator tests**

Require D4 current mIoU, Ghost, BG F, surface F, map mask, and per-point semantics to equal D1 exactly; require every rate to include numerator and denominator.

- [ ] **Step 5: Run metric suites**

Run: `python -m pytest -q tests/evaluation/test_entity_epoch_metrics.py tests/evaluation/test_two_visit_current_metrics.py tests/evaluation/test_two_visit_snapshot_metrics.py`

- [ ] **Step 6: Commit diagnostics**

```bash
git add src/evaluation/entity_epoch_metrics.py tests/evaluation/test_entity_epoch_metrics.py
git commit -m "Add entity episode action attribution"
```

### Task 7: Unified Prepare, Run, and Summarize CLI

**Files:**
- Create: `scripts/evaluation/run_crove_entity_epoch.py`
- Create: `configs/evaluation/crove_entity_epoch_v2.json`
- Create: `tests/evaluation/test_run_crove_entity_epoch.py`

**Interfaces:**
- Consumes: frozen pair selection, native OVI manifests, B3 provenance/metrics, RGB-D manifests, ReScene artifacts, and optional frozen DEV selection.
- Produces: resumable per-pair variant directories and compact aggregate JSON/CSV artifacts.

- [ ] **Step 1: Write failing CLI/config tests**

Test `--phase prepare|run|summarize`, `--split dev|confirm`, ordered `--variants`, `--selection` required for confirm, cache identity, per-variant failure isolation, and refusing score-driven pair replacement.

- [ ] **Step 2: Run the runner tests and observe missing script failure**

Run: `python -m pytest -q tests/evaluation/test_run_crove_entity_epoch.py`

- [ ] **Step 3: Implement prepare**

Freeze the existing Apartment `[766,1021] -> [1217,1472]` pair and at most two additional Apartment DEV pairs selected only by availability/event/visibility metadata. Report Office exactly as `RAW_MISSING`, `DERIVED_NOT_BUILT`, or `READY`; when raw assets exist, materialize native OVI inputs through existing runners before declaring readiness.

- [ ] **Step 4: Implement per-pair shared preparation**

Load each native surface once, bind owners/semantics once, use the full authorized t1 window for state evidence, compute t1 replacement coverage once, build the pair adapter once, and cache model forward by pair/config/checkpoint identity.

- [ ] **Step 5: Implement D0-D6 and oracle dispatch**

Use the exact matrix: `D0_T1` t1-only; `D1_B3` B3; `D2_INHERIT` inherited kernel without relations; `D3_GEOM` G1 plus kernel; `D4_RESCENE_ID_ONLY` D1 state plus ReScene identity aliases; `D5_RESCENE_EPOCH` same ReScene relations plus kernel; `D6_MEMORY_EPOCH` memory relations plus kernel; `DX_ORACLE_REL` evaluator relation only and excluded from selection.

- [ ] **Step 6: Implement summarize and selection**

Evaluate no more than 8-12 hypothesis configurations. Select one global DEV configuration by maximum cross-pair current mIoU subject to Ghost <= 0.02 and frozen geometry/deletion constraints, then BG F and retained-history recall. Preserve all failed and ineligible rows.

- [ ] **Step 7: Emit the required compact artifact set**

Write `input_selection.json`, `run_index.json`, `metrics_per_pair.csv`, `aggregate_metrics.json`, `action_attribution.csv`, `relation_diagnostics.csv`, `state_transition_summary.json`, `selected_config.json`, `confirmation_results.json`, `model_manifest.json`, figures, and `compact_artifact_index.json` under `configs/evaluation/results/crove_entity_epoch_v2/`.

- [ ] **Step 8: Run runner and affected regression suites**

Run: `python -m pytest -q tests/evaluation/test_run_crove_entity_epoch.py tests/evaluation/test_run_crove_fine_current_map.py tests/oviv2/test_two_visit_execution.py tests/oviv2/test_two_visit_contracts.py`

- [ ] **Step 9: Commit the runner**

```bash
git add scripts/evaluation/run_crove_entity_epoch.py configs/evaluation/crove_entity_epoch_v2.json tests/evaluation/test_run_crove_entity_epoch.py
git commit -m "Add CROVE entity episode experiment runner"
```

### Task 8: Real DEV, ReScene, Memory, and Office Experiments

**Files:**
- Create/update: `configs/evaluation/results/crove_entity_epoch_v2/**`
- Keep full PLY/NPZ caches under: `$HOME/oviovo_baseline_runs/20260910_crove_entity_epoch_v2/`

**Interfaces:**
- Consumes: completed runner and frozen config.
- Produces: real D0-D6 metrics, at least one pair-bound ReScene forward and map comparison, executable memory comparison, Office attempt/result, and representative real figures.

- [ ] **Step 1: Materialize and bind input selection before scoring**

Run: `python scripts/evaluation/run_crove_entity_epoch.py --config configs/evaluation/crove_entity_epoch_v2.json --phase prepare --split dev`

- [ ] **Step 2: Run D0-D3 on the first ready real pair**

Run the inherited and G1 variants first so state-kernel behavior is measured independently of learned evidence.

- [ ] **Step 3: Execute one exact-pair ReScene forward and D4/D5**

Set one explicit GPU with `RESCENE_DEVICE=cuda:0`; retain runtime, memory, checkpoint, source commit, pair hash, raw mask/logit, and projection diagnostics.

- [ ] **Step 4: Execute D6 memory modes**

Compare last prototype with a real multiview bank using the same feature space and support. Record offline replay truthfully where used.

- [ ] **Step 5: Run the remaining frozen DEV pairs and summarize**

Run: `python scripts/evaluation/run_crove_entity_epoch.py --config configs/evaluation/crove_entity_epoch_v2.json --phase summarize --split dev`

- [ ] **Step 6: Freeze one DEV selection and attempt Office confirmation**

If Office is ready, run at least D0, D1, D3, and the selected new method without Office tuning. If raw assets are missing, preserve the exact missing bindings and attempted command in `confirmation_results.json`.

- [ ] **Step 7: Render real success and ambiguity/failure views**

Every view reads the same canonical map used by evaluation. If no real successful recovery exists, publish real failure/ambiguity views and label the recovery mechanism as diagnostic-only.

- [ ] **Step 8: Commit compact evidence only**

```bash
git add configs/evaluation/results/crove_entity_epoch_v2
git commit -m "Add CROVE entity episode experiment evidence"
```

### Task 9: Paper Results, Handoff, Verification, and Push

**Files:**
- Modify: `docs/paper/crove_entity_epoch_v2_plan.md`
- Create: `docs/paper/crove_entity_epoch_v2_results.md`
- Create: `docs/paper/crove_entity_epoch_v2_handoff.md`
- Create: `docs/paper/crove_entity_epoch_v2_tables.md`

**Interfaces:**
- Consumes: compact result artifacts and local map index.
- Produces: evidence-linked paper tables, scientific conclusion, reproducibility handoff, and verified remote branch.

- [ ] **Step 1: Write results only from recorded artifacts**

State whether the outcome is `STATE_CORRECTNESS_REPAIRED_NO_MAP_GAIN`, `CROVE_UPDATE_GAIN_NO_LEARNED_INCREMENT`, `ENTITY_EVIDENCE_SUPPORTED_WITH_LIMITED_SCOPE`, or a fully evidenced negative result. Do not convert snapshot diagnostics into official Obj/Dyn/Chg metrics.

- [ ] **Step 2: Complete the handoff identities and upload statuses**

Record evidence SHA, evaluated code SHA, delivery branch, exact DEV/confirm coverage, executed variants, forward/training counts, checkpoint source, commands, local map paths/sizes/hashes, and separate CODE/RESULTS/MODEL/MAP statuses. Reused model status is `NOT_APPLICABLE_REUSED`; full maps default to `LOCAL_ONLY_POLICY`.

- [ ] **Step 3: Run final focused verification**

Run: `python -m pytest -q tests/oviv2/test_entity_epoch_update.py tests/oviv2/test_entity_epoch_relations.py tests/oviv2/test_entity_epoch_memory.py tests/oviv2/test_entity_epoch_io.py tests/evaluation/test_entity_epoch_metrics.py tests/evaluation/test_run_crove_entity_epoch.py tests/evaluation/test_run_crove_fine_current_map.py`

Run: `python -m compileall -q src/oviv2 scripts/evaluation/run_crove_entity_epoch.py`

Run: `git diff --check && git status --short`

- [ ] **Step 4: Review V1 noninterference and artifact lineage**

Compare V1-focused tests to the 192-pass baseline, verify D4 equality, verify every compact metric has a real source path/JSON pointer, and inspect the complete diff for unrelated files or prohibited assets.

- [ ] **Step 5: Commit documentation and push**

```bash
git add docs/paper/crove_entity_epoch_v2_plan.md docs/paper/crove_entity_epoch_v2_results.md docs/paper/crove_entity_epoch_v2_handoff.md docs/paper/crove_entity_epoch_v2_tables.md
git commit -m "Document CROVE entity episode dynamics"
git push -u origin research/crove-entity-epoch-dynamics-v2
```

- [ ] **Step 6: Verify local and remote identity**

```bash
LOCAL_SHA=$(git rev-parse HEAD)
REMOTE_SHA=$(git ls-remote origin refs/heads/research/crove-entity-epoch-dynamics-v2 | awk '{print $1}')
test "$LOCAL_SHA" = "$REMOTE_SHA"
printf 'LOCAL=%s\nREMOTE=%s\n' "$LOCAL_SHA" "$REMOTE_SHA"
```
