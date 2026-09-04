# OVI-MAP x ReScene4D Two-Visit Current-State Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a source-bound two-visit current-state mapper that preserves OVI-MAP dense geometry and semantics, uses signed revisit visibility as removal authority, and measures whether ReScene adds value over deterministic baselines.

**Architecture:** Two independent OVI visit maps are deterministically surface-sampled into a reversible neural-token representation. A pluggable temporal reasoner produces cross-visit evidence, which is projected back to OVI entities; a t1-first composer emits only OVI geometry and retains t0 solely where t1 does not prove visible-free space.

**Tech Stack:** Python 3.13, NumPy, SciPy, JSON/CSV/NPZ, pytest, external pinned OVI-MAP and ReScene4D checkouts, optional PyTorch/Pointcept/Concerto environment.

**Spec:** `docs/superpowers/specs/2026-09-04-ovi-rescene-two-visit-design.md`

## Global Constraints

- Base is exactly `research/crove-benchmark-alignment-audit` at `1e849acdcba46ab92f8704a77695ce7caefa1516`.
- OVI-MAP is read-only at `f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424`; ReScene4D is read-only at `fb2fe42eb8f1e926567c48eea9acb874e608ee10`.
- OVI map resolution remains 0.01 m; primary ReScene neural resolution is 0.02 m; evaluator resolution remains 0.05 m.
- OVI t0 and t1 mapping executions are independent and share no continuous CROVE state.
- Final geometry originates only from OVI t0 or OVI t1; final semantics originate only from OVI/VLM evidence.
- ReScene disappearance never deletes t0 geometry without t1 visible-free evidence.
- Ground-truth identities, transforms, change labels, and evaluator-only visibility never enter ranking-time method inputs.
- Apartment is development; Office remains held out until all architecture, thresholds, windows, baselines, and gates are committed.
- Missing external weights or data create explicit blocked results, never fabricated numbers or random-weight claims.
- B0-B6 metrics are a Pareto vector and preserve `N/A`; no hidden weighted score selects a method.
- Large source trees, datasets, maps, meshes, checkpoints, and attribution sidecars stay outside Git.

---

### Task 1: Freeze External Sources and the Voxel Contract

**Files:**
- Create: `configs/external/ovi_rescene_sources.json`
- Create: `scripts/evaluation/audit_ovi_rescene_sources.py`
- Create: `tests/evaluation/test_ovi_rescene_sources.py`
- Create: `docs/superpowers/experiments/2026-09-04-ovi-rescene-two-visit-ledger.md`
- Create: `docs/superpowers/reports/2026-09-04-ovi-rescene-voxel-contract.md`

**Interfaces:**
- Consumes: two pinned external Git checkouts plus tracked CROVE/evaluator config paths.
- Produces: `audit_sources(manifest, checkouts) -> SourceAuditResult` and an atomic JSON receipt with `EXTERNAL_SOURCE_PASS` plus a report ending in `VOXEL_CONTRACT_PASS` or an explicit blocked status.

- [ ] **Step 1: Write RED tests for identity and representation separation**

```python
def test_audit_accepts_exact_clean_sources_and_four_distinct_voxels(source_fixture):
    result = audit_sources(source_fixture.manifest, source_fixture.checkouts)
    assert result.status == "EXTERNAL_SOURCE_PASS"
    assert result.voxel_status == "VOXEL_CONTRACT_PASS"
    assert result.voxel_sizes_m == {
        "ovi_mapping": 0.01,
        "rescene_neural": 0.02,
        "crove_legacy_temporal": 0.05,
        "evaluator": 0.05,
    }

def test_audit_rejects_working_file_different_from_pinned_commit(source_fixture):
    source_fixture.mutate_required_file()
    with pytest.raises(SourceAuditError, match="differs from pinned commit"):
        audit_sources(source_fixture.manifest, source_fixture.checkouts)
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/evaluation/test_ovi_rescene_sources.py`

Expected: import failure because the audit script and manifest do not exist.

- [ ] **Step 3: Implement strict source binding**

```python
@dataclass(frozen=True, slots=True)
class SourceAuditResult:
    status: str
    voxel_status: str
    checkpoint_status: str
    source_commits: Mapping[str, str]
    voxel_sizes_m: Mapping[str, float]
    verified_files: tuple[FileBinding, ...]
```

Resolve each declared commit, compare every required working-tree file byte-for-byte
with `git show <commit>:<path>`, verify SHA-256/byte count/license, and reject
symlinks or dirty required files. Parse declared evidence paths for 1/2/5 cm
values and verify they are assigned to different roles. Write sorted finite JSON
through fsync plus atomic rename.

- [ ] **Step 4: Run GREEN and execute the real audit**

Run:

```bash
pytest -q tests/evaluation/test_ovi_rescene_sources.py
python scripts/evaluation/audit_ovi_rescene_sources.py \
  --manifest configs/external/ovi_rescene_sources.json \
  --ovimap-checkout /home/ww/oviovo_references/baselines/OVI-MAP \
  --rescene-checkout /home/ww/oviovo_references/evaluation/rescene4d \
  --output /tmp/ovi-rescene-source-audit.json
git diff --check
```

Expected: source and voxel contracts pass; checkpoint status is independently
reported and may be `BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT`.

- [ ] **Step 5: Commit P0 separately**

```bash
git add configs/external/ovi_rescene_sources.json \
  scripts/evaluation/audit_ovi_rescene_sources.py \
  tests/evaluation/test_ovi_rescene_sources.py \
  docs/superpowers/experiments/2026-09-04-ovi-rescene-two-visit-ledger.md \
  docs/superpowers/reports/2026-09-04-ovi-rescene-voxel-contract.md
git commit -m "docs: freeze OVI ReScene source and voxel contract"
```

### Task 2: Add Immutable Two-Visit Data Contracts

**Files:**
- Create: `src/oviv2/two_visit_contracts.py`
- Create: `tests/oviv2/test_two_visit_contracts.py`

**Interfaces:**
- Consumes: `src.evaluation.contracts.MapSnapshot` and NumPy arrays.
- Produces: `VisitMap`, `NeuralSampleMap`, `TemporalQueryEvidence`, `PairRelation`, `CurrentCompositionDecision`, and stable literal enums.

- [ ] **Step 1: Write RED validation tests**

```python
def test_neural_sample_map_requires_exact_visit_coordinate_and_csr_conservation():
    pair = make_valid_pair()
    assert np.array_equal(pair.coordinates_xyzt[:, 3], pair.visit_ids)
    assert pair.source_to_token_offsets.tolist() == [0, 2, 3]
    assert len(pair.source_entity_ids) == len(pair.source_point_indices) == 3

def test_visit_map_rejects_non_independent_or_mixed_coordinate_frame():
    with pytest.raises(ValueError, match="coordinate frame"):
        make_visit_pair(frame_ids=("world", "other"))
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/oviv2/test_two_visit_contracts.py`

Expected: import failure because `two_visit_contracts` is absent.

- [ ] **Step 3: Implement validated read-only contracts**

Use frozen, slotted dataclasses. Copy arrays, require finite values and exact
shapes, set NumPy write flags false, validate SHA-256 fields, require visit IDs
in `{0,1}`, require monotonically increasing CSR offsets ending at contributor
count, and forbid a token's contributor span from mixing entities across visits.

- [ ] **Step 4: Run GREEN and regression checks**

Run: `pytest -q tests/oviv2/test_two_visit_contracts.py tests/evaluation/test_contracts.py`

Expected: all tests pass and existing `MapSnapshot` behavior is unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/oviv2/two_visit_contracts.py tests/oviv2/test_two_visit_contracts.py
git commit -m "feat: add two-visit data contracts"
```

### Task 3: Add the Deterministic OVI Surface Adapter

**Files:**
- Create: `src/oviv2/ovi_rescene_adapter.py`
- Create: `scripts/evaluation/export_ovi_rescene_pair.py`
- Create: `tests/oviv2/test_ovi_rescene_adapter.py`

**Interfaces:**
- Consumes: two validated `VisitMap` values and `AdapterConfig(neural_voxel_size_m, feature_schema)`.
- Produces: `adapt_visit_pair(t0, t1, config) -> NeuralSampleMap` and a hash-bound NPZ/JSON pair artifact.

- [ ] **Step 1: Write RED tests for deterministic reversible sampling**

```python
def test_two_centimeter_spatial_grouping_is_deterministic_and_time_is_unscaled():
    first = adapt_visit_pair(*make_surface_visits(), AdapterConfig(0.02, "rgb"))
    second = adapt_visit_pair(*make_surface_visits(), AdapterConfig(0.02, "rgb"))
    assert first.content_sha256() == second.content_sha256()
    assert set(first.coordinates_xyzt[:, 3]) == {0.0, 1.0}

def test_reverse_index_conserves_every_source_point_once():
    pair = adapt_visit_pair(*make_surface_visits(), AdapterConfig(0.02, "rgb"))
    assert sorted(pair.source_point_indices.tolist()) == list(range(pair.source_point_count))
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/oviv2/test_ovi_rescene_adapter.py`

Expected: import failure because the adapter is absent.

- [ ] **Step 3: Implement spatial-only quantization and CSR provenance**

Sort source samples by `(visit, floor(x/v), floor(y/v), floor(z/v), entity,
point_index)`. Aggregate XYZ and valid RGB/normals deterministically, keep visit
as exact integer, and build flattened entity/point contributor arrays plus token
offsets. Hash snapshots before and after adaptation and fail if content changes.
Reject mixed coordinate frames, missing required features, non-finite points,
and unsupported feature schemas.

- [ ] **Step 4: Implement atomic export and verify 1/2/4 cm acceptance**

Run: `pytest -q tests/oviv2/test_ovi_rescene_adapter.py -k 'deterministic or reverse or voxel or mutation or frame'`

Expected: deterministic artifacts, complete reverse provenance, no source
mutation, and independent acceptance of 0.01/0.02/0.04 m neural grids.

- [ ] **Step 5: Commit**

```bash
git add src/oviv2/ovi_rescene_adapter.py \
  scripts/evaluation/export_ovi_rescene_pair.py \
  tests/oviv2/test_ovi_rescene_adapter.py
git commit -m "feat: add deterministic OVI to ReScene adapter"
```

### Task 4: Add Temporal Reasoner Abstraction and Geometric Baseline

**Files:**
- Create: `src/oviv2/temporal_pair_reasoner.py`
- Create: `src/oviv2/geometric_pair_reasoner.py`
- Create: `src/oviv2/rescene_backend.py`
- Create: `tests/oviv2/test_temporal_pair_reasoner.py`
- Create: `tests/oviv2/test_geometric_pair_reasoner.py`
- Create: `tests/oviv2/test_rescene_backend_contract.py`

**Interfaces:**
- Consumes: `NeuralSampleMap`, per-entity OVI semantic metadata, and a source-bound optional backend specification.
- Produces: `TemporalPairReasoner.infer(pair) -> TemporalQueryEvidence`.

- [ ] **Step 1: Write RED protocol and blocked-backend tests**

```python
def test_geometric_reasoner_satisfies_protocol_and_is_repeatable():
    reasoner: TemporalPairReasoner = GeometricPairReasoner(make_config())
    assert reasoner.infer(make_pair()) == reasoner.infer(make_pair())

def test_missing_checkpoint_is_explicitly_blocked_without_random_inference(tmp_path):
    backend = ReSceneBackend(checkout=PINNED_RESCENE, checkpoint=tmp_path / "missing.ckpt")
    result = backend.infer(make_pair())
    assert result.status == "BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT"
    assert result.query_masks is None
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/oviv2/test_temporal_pair_reasoner.py tests/oviv2/test_rescene_backend_contract.py`

Expected: import failure for missing reasoner modules.

- [ ] **Step 3: Implement deterministic entity evidence and assignment**

Aggregate token evidence per OVI entity, compute voxel overlap, symmetric
coverage, centroid distance, size ratio, shape extent ratio, label compatibility,
and source-bound embedding cosine. Generate 1:1, moved, appeared, removed
candidate, split, merge, and uncertain query evidence without GT inputs. Sort all
ties by entity ID.

- [ ] **Step 4: Implement a fail-closed ReScene boundary**

Validate checkout SHA, backend name, checkpoint bytes/hash, config hash, feature
schema, and dependency environment before subprocess execution. Emit a blocked
result before importing torch when any external identity is missing. Do not
include random initialization or an in-process third-party source copy.

- [ ] **Step 5: Run GREEN and commit**

```bash
pytest -q tests/oviv2/test_temporal_pair_reasoner.py \
  tests/oviv2/test_geometric_pair_reasoner.py \
  tests/oviv2/test_rescene_backend_contract.py
git add src/oviv2/temporal_pair_reasoner.py \
  src/oviv2/geometric_pair_reasoner.py src/oviv2/rescene_backend.py \
  tests/oviv2/test_temporal_pair_reasoner.py \
  tests/oviv2/test_geometric_pair_reasoner.py \
  tests/oviv2/test_rescene_backend_contract.py
git commit -m "feat: add temporal pair reasoner abstraction"
```

### Task 5: Project Query Evidence Back to OVI Instances

**Files:**
- Create: `src/oviv2/query_instance_projection.py`
- Create: `tests/oviv2/test_query_instance_projection.py`

**Interfaces:**
- Consumes: `NeuralSampleMap` and `TemporalQueryEvidence` masks/scores.
- Produces: `project_queries_to_instances(pair, evidence) -> tuple[PairRelation, ...]` plus a query-by-entity-by-visit evidence tensor.

- [ ] **Step 1: Write RED tests for entity-level lossless evidence**

```python
def test_projection_counts_all_source_contributors_not_only_token_representatives():
    matrix = project_query_evidence(make_many_to_one_token_pair(), make_query())
    assert matrix["q0", "ovi:chair:0", 0].source_point_count == 3

@pytest.mark.parametrize("case", ["one_to_one", "moved", "appeared", "removed", "split", "merge", "uncertain"])
def test_projection_preserves_relation_topology(case):
    assert project_case(case).state == EXPECTED_STATES[case]
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/oviv2/test_query_instance_projection.py`

Expected: import failure because projection code is absent.

- [ ] **Step 3: Implement soft and hard evidence projection**

For each query/token mass, expand CSR contributors and record token intersection,
entity-token coverage, query-mask fraction, soft mass, source-point coverage,
centroid, and optional semantic compatibility separately by visit. Keep repeated
same-category entities distinct and allow many-sided relations.

- [ ] **Step 4: Run GREEN and commit**

```bash
pytest -q tests/oviv2/test_query_instance_projection.py
git add src/oviv2/query_instance_projection.py tests/oviv2/test_query_instance_projection.py
git commit -m "feat: project temporal queries to OVI instances"
```

### Task 6: Add the t1-First Current Map Composer

**Files:**
- Create: `src/oviv2/two_visit_current_map.py`
- Create: `scripts/evaluation/compose_ovi_rescene_current_map.py`
- Create: `tests/oviv2/test_two_visit_current_map.py`
- Modify: `src/oviv2/anchor_visibility.py` only if a small generalization is required to expose existing signed evidence without changing legacy behavior.

**Interfaces:**
- Consumes: two `VisitMap` values, pair relations, and t1 signed visibility evidence.
- Produces: `compose_current_map(...) -> TwoVisitCurrentMap` containing `MapSnapshot` plus complete entity/point-group provenance.

- [ ] **Step 1: Write RED tests for all four visibility branches**

```python
@pytest.mark.parametrize(
    ("visibility", "expected"),
    [("occupied", "emit_t1"), ("visible_free", "suppress_t0_visible_free"),
     ("occluded", "retain_t0_occluded"), ("unobserved", "retain_t0_unobserved")],
)
def test_t1_first_composition(visibility, expected):
    result = compose_fixture(visibility)
    assert result.decisions[0].decision == expected
```

- [ ] **Step 2: Write RED authority tests**

```python
def test_missing_t1_query_without_visible_free_never_deletes_t0():
    assert compose_removed_candidate(visibility="unobserved").emits_t0

def test_moved_entity_uses_dense_ovi_t1_points_and_ovi_semantics():
    result = compose_moved_fixture()
    assert np.array_equal(result.snapshot.entities[0].points_xyz, OVI_T1_POINTS)
    assert result.provenance[0].geometry_source == "ovi_t1"
    assert result.provenance[0].semantic_source.startswith("ovi_")
```

- [ ] **Step 3: Run RED**

Run: `pytest -q tests/oviv2/test_two_visit_current_map.py`

Expected: import failure because the composer is absent.

- [ ] **Step 4: Implement entity and background composition**

Deduplicate t1 occupied geometry first, apply visible-free suppression only to
covered t0 source cells, preserve occluded/unknown t0 cells, and merge background
on the configured composition grid without altering source points. Reject any
geometry provenance other than `ovi_t0`/`ovi_t1` and any semantic provenance
outside OVI. Serialize decisions and source hashes atomically.

- [ ] **Step 5: Run GREEN, legacy visibility regressions, and commit**

```bash
pytest -q tests/oviv2/test_two_visit_current_map.py tests/oviv2/test_anchor_visibility.py
git add src/oviv2/two_visit_current_map.py \
  scripts/evaluation/compose_ovi_rescene_current_map.py \
  tests/oviv2/test_two_visit_current_map.py src/oviv2/anchor_visibility.py
git commit -m "feat: add t1-first current map composer"
```

### Task 7: Freeze the TESSE Two-Visit Protocol

**Files:**
- Create: `configs/evaluation/tesse_two_visit_current_v1.json`
- Create: `scripts/evaluation/freeze_tesse_two_visit_protocol.py`
- Create: `scripts/evaluation/run_ovi_two_visit.py`
- Create: `scripts/evaluation/evaluate_ovi_two_visit_current.py`
- Create: `tests/evaluation/test_tesse_two_visit_protocol.py`
- Create: `docs/superpowers/reports/2026-09-04-ovi-rescene-two-visit-protocol.md`

**Interfaces:**
- Consumes: checked TESSE RGB-D/export/causal manifests and candidate window diagnostics.
- Produces: immutable `TESSE_TWO_VISIT_CURRENT_V1` Apartment/Office pair manifest and guarded run/evaluation CLIs.

- [ ] **Step 1: Write RED protocol leakage and held-out tests**

```python
def test_visit_windows_are_non_overlapping_source_bound_and_method_independent():
    protocol = freeze_protocol(make_candidate_metadata())
    assert protocol.t0.end_frame < protocol.t1.start_frame
    assert protocol.selection_inputs == ("source_metadata", "causal_schedule")
    assert not protocol.method_result_bindings

def test_office_cannot_run_before_all_freeze_bindings_exist():
    with pytest.raises(ProtocolError, match="OFFICE_NOT_RUN_HELD_OUT"):
        authorize_scene_run(make_unfrozen_protocol(), "Office")
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/evaluation/test_tesse_two_visit_protocol.py`

Expected: import failure because the freezer is absent.

- [ ] **Step 3: Implement deterministic candidate scoring and freeze**

Filter windows by minimum frame count, no temporal overlap, trajectory overlap,
common observable volume, changed-object support, camera distribution, and old
location visibility. Use a lexicographic source-only selection key, record every
candidate diagnostic, bind all input bytes, and exclude GT-only fields from the
method manifest.

- [ ] **Step 4: Build independent OVI run commands**

Reuse `export_tesse_cd_rgbd.py`, `run_ovimap_native.py`, and
`build_tesse_ovimap_static_anchor.py` helpers. Emit two distinct run IDs/output
roots and assert no t0 state path appears in the t1 OVI command. The evaluation
CLI validates protocol/source/config/output hashes before reading predictions.

- [ ] **Step 5: Run GREEN and commit the frozen protocol before method scores**

```bash
pytest -q tests/evaluation/test_tesse_two_visit_protocol.py \
  tests/evaluation/test_export_tesse_cd_rgbd.py \
  tests/evaluation/test_run_ovimap_native.py \
  tests/evaluation/test_build_tesse_ovimap_static_anchor.py
git add configs/evaluation/tesse_two_visit_current_v1.json \
  scripts/evaluation/freeze_tesse_two_visit_protocol.py \
  scripts/evaluation/run_ovi_two_visit.py \
  scripts/evaluation/evaluate_ovi_two_visit_current.py \
  tests/evaluation/test_tesse_two_visit_protocol.py \
  docs/superpowers/reports/2026-09-04-ovi-rescene-two-visit-protocol.md
git commit -m "eval: freeze TESSE two-visit protocol"
```

### Task 8: Add B0-B6 Execution Matrix and Completeness Metrics

**Files:**
- Create: `configs/evaluation/ovi_rescene_two_visit_matrix.json`
- Create: `src/evaluation/two_visit_current_metrics.py`
- Create: `scripts/evaluation/run_ovi_rescene_two_visit_matrix.py`
- Create: `tests/evaluation/test_two_visit_current_metrics.py`
- Create: `tests/evaluation/test_ovi_rescene_two_visit_matrix.py`

**Interfaces:**
- Consumes: frozen protocol, OVI visit maps, composer outputs, common-v2 GT, and optional identity GT.
- Produces: B0-B6 result receipts with current, geometry, identity, semantic, runtime, memory, and artifact metrics kept separate.

- [ ] **Step 1: Write RED metric tests with hand-derived fixtures**

```python
def test_unobserved_completeness_and_observed_stale_precision_are_separate():
    result = evaluate_regions(make_hand_checked_region_fixture())
    assert result.unobserved_region_recall == 0.75
    assert result.observed_region_stale_precision == 0.80

def test_empty_or_inapplicable_identity_metric_is_na_not_zero():
    assert evaluate_identity(no_identity_gt_fixture()).moved_accuracy is None
```

- [ ] **Step 2: Write RED matrix-contract tests**

Assert exact B0-B6 IDs, OVI input use, reasoner/visibility settings, final
geometry roles, purpose, source/config hashes, command/log/receipt fields, and
Office prohibition before freeze.

- [ ] **Step 3: Run RED**

Run: `pytest -q tests/evaluation/test_two_visit_current_metrics.py tests/evaluation/test_ovi_rescene_two_visit_matrix.py`

Expected: import or file-not-found failure for the new metrics and matrix.

- [ ] **Step 4: Implement metrics and guarded orchestration**

Reuse common-v2 current mIoU/Ghost/background F@5cm definitions. Add surface
precision/recall, total coverage, t1-unobserved recall, observed-region stale
precision, identity/change metrics when valid, and system counters. The runner
executes B0-B4 unconditionally and emits explicit blocked B5/B6 receipts when a
valid ReScene backend is unavailable.

- [ ] **Step 5: Run GREEN and commit**

```bash
pytest -q tests/evaluation/test_two_visit_current_metrics.py \
  tests/evaluation/test_ovi_rescene_two_visit_matrix.py \
  tests/evaluation/test_evaluate_tesse_cd_common_v2.py
git add configs/evaluation/ovi_rescene_two_visit_matrix.json \
  src/evaluation/two_visit_current_metrics.py \
  scripts/evaluation/run_ovi_rescene_two_visit_matrix.py \
  tests/evaluation/test_two_visit_current_metrics.py \
  tests/evaluation/test_ovi_rescene_two_visit_matrix.py
git commit -m "eval: add two-visit baselines and metrics"
```

### Task 9: Integrate and Validate a Reproducible ReScene Backend When Available

**Files:**
- Modify: `src/oviv2/rescene_backend.py`
- Create: `scripts/evaluation/run_rescene_pair_backend.py`
- Create: `tests/oviv2/test_rescene_backend_integration.py`
- Create: `configs/evaluation/rescene_two_visit_backend.json`

**Interfaces:**
- Consumes: serialized adapter artifact, pinned checkout, exact environment, config, and valid checkpoint.
- Produces: source-bound `TemporalQueryEvidence` with query masks, scores, runtime, memory, and backend receipt.

- [ ] **Step 1: Inventory checkpoint and training evidence before code changes**

Check official README/releases and local source-bound training assets. Record
exactly one status: official checkpoint bound, locally trained checkpoint bound,
or `BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT`. Never select a checkpoint by B5
Apartment outcome.

- [ ] **Step 2: Write RED subprocess-contract tests**

```python
def test_backend_rejects_checkpoint_hash_mismatch(tmp_path):
    spec = make_backend_spec(tmp_path, declared_sha256="0" * 64)
    with pytest.raises(ReSceneBackendError, match="checkpoint SHA-256"):
        run_rescene_backend(spec, make_pair_artifact())

def test_random_initialization_cannot_produce_ranking_eligible_output():
    assert run_plumbing_backend_without_checkpoint().ranking_eligible is False
```

- [ ] **Step 3: Implement the thin external runner only for a valid identity**

Map the adapter feature schema to the selected official config, invoke the
external environment without modifying the checkout, restore query masks to
adapter-token order, record peak GPU memory/runtime, and validate output shapes,
finite values, visit coverage, and all hashes. Concerto is primary; alternate
backbones retain distinct IDs.

- [ ] **Step 4: Run GREEN or publish the exact blocked result**

Run: `pytest -q tests/oviv2/test_rescene_backend_contract.py tests/oviv2/test_rescene_backend_integration.py`

Expected: hermetic contract tests pass. A real run is executed only when the
checkpoint/training gate is valid; otherwise B5/B6 remain blocked and no random
metric is emitted.

- [ ] **Step 5: Commit**

```bash
git add src/oviv2/rescene_backend.py \
  scripts/evaluation/run_rescene_pair_backend.py \
  tests/oviv2/test_rescene_backend_integration.py \
  configs/evaluation/rescene_two_visit_backend.json
git commit -m "feat: integrate pinned ReScene backend"
```

### Task 10: Add Exact Two-Visit Failure Attribution

**Files:**
- Create: `src/evaluation/two_visit_attribution.py`
- Create: `scripts/evaluation/attribute_two_visit_failures.py`
- Create: `tests/evaluation/test_two_visit_attribution.py`

**Interfaces:**
- Consumes: current-map prediction, evaluator matches, composition decisions, pair evidence, visibility evidence, and OVI source identities.
- Produces: exhaustive disjoint failure rows and category aggregates.

- [ ] **Step 1: Write RED partition and provenance tests**

```python
def test_every_ghost_is_assigned_once_and_mass_is_conserved():
    result = attribute_failures(make_all_category_fixture())
    assert sum(result.ghost_counts.values()) == result.total_ghost_count
    assert result.unattributed_count == 0

def test_each_row_binds_visit_entity_relation_visibility_and_sources():
    row = attribute_failures(make_single_error_fixture()).rows[0]
    assert row.source_visit in {0, 1}
    assert row.geometry_source in {"ovi_t0", "ovi_t1"}
    assert row.semantic_source.startswith("ovi_")
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/evaluation/test_two_visit_attribution.py`

Expected: import failure because attribution code is absent.

- [ ] **Step 3: Implement exhaustive categorization and atomic sidecars**

Join evaluator point/entity failures to source groups and choose exactly one
category by a documented precedence that distinguishes unsafe retention,
legitimate unobserved/occluded fallback, t1 false positives, wrong pairing,
depth/visibility failure, and background failure. Keep `UNATTRIBUTED` visible and
report its count/share rather than redistributing it.

- [ ] **Step 4: Run GREEN and commit**

```bash
pytest -q tests/evaluation/test_two_visit_attribution.py
git add src/evaluation/two_visit_attribution.py \
  scripts/evaluation/attribute_two_visit_failures.py \
  tests/evaluation/test_two_visit_attribution.py
git commit -m "eval: add two-visit failure attribution"
```

### Task 11: Execute Frozen Apartment Matrix and Make the ReScene Decision

**Files:**
- Create: `docs/superpowers/reports/2026-09-04-ovi-rescene-apartment-results.md`
- Modify: `docs/superpowers/experiments/2026-09-04-ovi-rescene-two-visit-ledger.md`
- Add: small result receipts under `configs/evaluation/results/ovi_rescene_two_visit/`

**Interfaces:**
- Consumes: frozen Apartment protocol and B0-B6 matrix.
- Produces: measured Pareto table, B2 Ghost/completeness floor, gate outcomes, failure categories, and one ReScene necessity verdict.

- [ ] **Step 1: Run B0-B2 and diagnose the floor**

Record `G_t1`, B2 completeness, input/output hashes, logs, runtime, memory, and
artifact sizes. If B2 Ghost is not low, stop temporal interpretation and diagnose
OVI t1/evaluator alignment before executing B3-B6.

- [ ] **Step 2: Run B3 and B4 under the frozen protocol**

Compare visibility-only and geometric pairing against B2 without changing
windows or thresholds. Record current metrics, completeness groups, identity
diagnostics, and exact provenance.

- [ ] **Step 3: Run B5/B6 only when Task 9 passed**

If blocked, emit `RESCENE_BLOCKED_EXTERNAL_ASSET` with the missing checkpoint or
training identity. If runnable, execute both visibility and no-visibility modes,
then run 1/2/4 cm neural-grid ablations after the primary 2 cm result.

- [ ] **Step 4: Apply preregistered Pareto gates**

Evaluate Ghost relative to B2 plus 0.02, Object F1 thresholds 0.367952 and
0.372762, pre-frozen Change tolerance, B2 completeness recovery, and improvement
over B3/B4. Select exactly one allowed ReScene verdict without averaging metrics.

- [ ] **Step 5: Commit measured small receipts and report**

```bash
git add docs/superpowers/reports/2026-09-04-ovi-rescene-apartment-results.md \
  docs/superpowers/experiments/2026-09-04-ovi-rescene-two-visit-ledger.md \
  configs/evaluation/results/ovi_rescene_two_visit
git commit -m "results: publish frozen two-visit Apartment matrix"
```

### Task 12: Execute Held-Out Office, Close Handoff, Verify, and Upload

**Files:**
- Create: `docs/superpowers/reports/2026-09-04-ovi-rescene-two-visit-handoff.md`
- Modify: `docs/superpowers/experiments/2026-09-04-ovi-rescene-two-visit-ledger.md`
- Add: Office small receipts under `configs/evaluation/results/ovi_rescene_two_visit/` only after freeze.

**Interfaces:**
- Consumes: committed Apartment-selected configuration and all prior receipts.
- Produces: one held-out Office outcome, answers to all 14 handoff questions, exact test evidence, and verified remote branch identity.

- [ ] **Step 1: Prove the freeze boundary before Office**

Record the configuration commit SHA and verify hashes for source identities,
windows, adapter, backend/checkpoint, thresholds, composer, baselines, and metrics.
The Office command rejects any missing or changed binding.

- [ ] **Step 2: Run Office once or record an asset blocker**

Execute the selected candidate and required baselines without parameter changes.
Use `OFFICE_EXECUTED_FROZEN`, `OFFICE_GENERALIZATION_PASS`, or
`OFFICE_GENERALIZATION_FAIL` only from a completed run; otherwise preserve
`OFFICE_NOT_RUN_HELD_OUT` with exact missing evidence.

- [ ] **Step 3: Write the final handoff**

Include repository/external identities, authority model, four-voxel table,
Apartment/Office protocol hashes, complete B0-B6 results with `N/A`, ReScene
verdict, dominant attribution categories, exact commands/results, large artifact
records, 3RScan status, paper-ready/diagnostic/blocked claims, and uncompleted
work.

- [ ] **Step 4: Run focused historical regression and static validation**

```bash
pytest -q tests/oviv2/test_ovimap_static_anchor.py \
  tests/oviv2/test_t1_exactness.py tests/oviv2/test_t1_noninterference.py \
  tests/evaluation/test_benchmark_alignment.py \
  tests/evaluation/test_benchmark_alignment_preregistration.py \
  tests/evaluation/test_benchmark_alignment_reports.py
git diff --check
python -m compileall -q src scripts tests
```

Discover and add existing OVI-native, visibility, snapshot, TESSE RGB-D export,
common-v2, and static-anchor tests with `git ls-files tests | rg` before running
the focused suite. Run Ruff on every changed Python file when available.

- [ ] **Step 5: Run full tests, commit handoff, and upload**

```bash
pytest -q
git add docs/superpowers/reports/2026-09-04-ovi-rescene-two-visit-handoff.md \
  docs/superpowers/experiments/2026-09-04-ovi-rescene-two-visit-ledger.md \
  configs/evaluation/results/ovi_rescene_two_visit
git commit -m "docs: publish two-visit experiment decision and handoff"
git status --short
git diff --check
git rev-parse HEAD
git push -u origin research/ovi-rescene-two-visit
git ls-remote --heads origin research/ovi-rescene-two-visit
```

Expected: no unexplained regression, clean worktree, and remote SHA exactly equal
to local HEAD before reporting `UPLOAD_STATUS=VERIFIED`.
