# OVI-MAP x ReScene C2 Repair and Apartment B5 Implementation Plan

> **Execution:** Follow this plan task by task with test-driven development.
> Core contracts, algorithms, integration, debugging, review, and scientific
> decisions remain with the primary agent. Mechanical workers may only perform
> explicitly specified fixture or metadata leaves.

**Goal:** Repair the frozen Apartment OVI pair into a source-bound, complete
ReScene input; run one real B5 inference; compare its relations with frozen B4;
and publish truthful, hash-bound evidence.

**Architecture:** Keep V1 and all map paths untouched. Add immutable surface and
model-input sidecars, obtain A-to-M from the pinned native GridSample, recover
each M feature from one same-visit RGB-D-supported PLY vertex, then expand model
predictions back to A before entering the existing backend and projector.

**Tech stack:** Python 3.13 repository tests, Python 3.10 pinned Persist4D model
environment, NumPy, Pillow, plyfile, PyTorch, Sonata/Concerto, Hydra, pytest,
JSON/NPZ, CUDA.

**Spec:**
`docs/superpowers/specs/2026-09-05-ovi-rescene-c2-repair-b5-v2-design.md`

## Invariants

- Base is exactly `f3cb93a9c49df3891279b67330e58401c20cc540`.
- Frozen VisitMaps, PLYs, logs, OVI output, V1 receipts, B3, B4, B7, and public
  backend schema are never modified.
- TESSE GT, change masks, B5 output, and Office do not affect input construction.
- Palette RGB, zero normals, cross-visit color, coordinate perturbation, hidden
  point drops, and identity duplication are forbidden.
- One formal depth tolerance is frozen before inference. There is one complete
  Apartment forward, with one retry only for a recorded bug or OOM.
- Large arrays remain under the external run root and are represented in Git by
  path, bytes, SHA-256, producer command, and evaluated commit.

---

### Task 1: Exact PLY surface alignment and projection primitives

**Files:**
- Create: `src/oviv2/ovi_surface_attributes.py`
- Create: `tests/oviv2/test_ovi_surface_attributes.py`

**Interfaces:**
- `group_surface_attributes_by_color(points_xyz, colors_rgb, normals_xyz,
  original_indices, requested_colors) -> dict[color, SurfaceGroup]`
- `validate_surface_group(entity_points, group) -> None`
- `project_world_points(points, camera_to_world, intrinsics) -> Projection`
- `select_depth_consistent_support(...) -> SurfaceSupport`

- [ ] Write RED tests for stable XYZ/normal/index grouping, duplicate XYZ rows,
  exact frozen-entity comparison, invalid/zero normal handling, and palette
  source rejection.
- [ ] Write RED projection tests for pose inversion, camera-z depth, millimetre
  conversion, fixed pixel rounding, known RGB, invalid depth, occlusion, depth
  edge rejection, and visit mismatch.
- [ ] Implement only the pure typed arrays and fail-closed validation needed by
  those tests. Reuse the existing OVI RGB-code convention.
- [ ] Run:

```bash
/home/ww/miniconda3/bin/python -m pytest -q tests/oviv2/test_ovi_surface_attributes.py
/home/ww/miniconda3/bin/python -m py_compile src/oviv2/ovi_surface_attributes.py
git diff --check
```

### Task 2: Vectorized D-to-A bridge and model-domain contracts

**Files:**
- Create: `src/oviv2/rescene_input_bridge.py`
- Create: `tests/oviv2/test_rescene_input_bridge.py`

**Interfaces:**
- `build_adapter_geometry(earlier, later, voxel_size_m) -> AdapterGeometry`
- `validate_native_sampling(adapter, sampling) -> SamplingMap`
- `select_model_representatives(adapter, sampling, surface) -> CandidateMap`
- `build_model_input(...) -> ReSceneModelInput`
- `expand_model_predictions(values_qm, adapter_to_model) -> ndarray`
- serialization for `OVI_SURFACE_ATTRIBUTES_V2` and
  `OVI_RESCENE_MODEL_INPUT_V2` with strict array/manifest schemas.

- [ ] Write RED tests proving identical small-fixture output to V1 D-to-A
  grouping and CSR, including source order, shared center, and unchanged input
  snapshots.
- [ ] Add RED mapping tests for identity, within-entity many-to-one,
  cross-entity many-to-one, reverse order, invalid range, missing A row, and
  cross-visit rejection.
- [ ] Add RED representative/input tests for real source coordinate, same native
  grid, legal normal/RGB, 9 channels, true sequence batch 0, temporal stages
  0/1, global identity `point2segment`, and explicit unsupported failure.
- [ ] Add RED expansion tests requiring exact `P[:, g]` and final adapter order.
- [ ] Implement the vectorized arrays and serializers without changing
  `NeuralSampleMap` or the V1 adapter.
- [ ] Run:

```bash
/home/ww/miniconda3/bin/python -m pytest -q \
  tests/oviv2/test_rescene_input_bridge.py \
  tests/oviv2/test_ovi_rescene_adapter.py
/home/ww/miniconda3/bin/python -m py_compile src/oviv2/rescene_input_bridge.py
git diff --check
```

### Task 3: Pinned native GridSample capture and C2-V2 preparer

**Files:**
- Create: `scripts/evaluation/prepare_ovi_rescene_input_v2.py`
- Create: `tests/evaluation/test_prepare_ovi_rescene_input_v2.py`

**Interfaces:**
- `native-sample` subcommand, run only with the pinned model Python, accepts A
  coordinates/visits and atomically emits selected indices, grid coordinates,
  offsets, `g`, RNG policy, module identity, and hashes.
- `calibrate` subcommand emits the bounded 2048-per-visit residual diagnosis and
  one selected tolerance.
- `build` subcommand validates frozen assets, performs one formal per-visit
  recovery, and emits the adapter pair, attribute/model-input sidecars, coverage
  summary, sampling summary, and C2-V2 receipt.
- `audit` subcommand rereads sidecars without decoding source RGB-D.

- [ ] Write RED tests for exact CLI schema, source-hash mismatch, RNG restoration,
  visit-local native inverse composition, atomic output, overwrite rejection,
  parent hash mismatch, same-visit frame binding, calibration formula, and
  complete-vs-partial status.
- [ ] Implement orchestration using `load_two_visit_ovi_inputs()` and
  `load_ovimap_visit()` only; do not call evaluator setup or map construction.
- [ ] Implement frame streaming and bounded candidate rounds using Task 1/2
  primitives. Ensure every decoded frame is from the declared visit.
- [ ] Verify the native sampler in the pinned environment on the tiny fixture,
  and all other tests in the repository test environment.
- [ ] Run:

```bash
/home/ww/miniconda3/bin/python -m pytest -q \
  tests/evaluation/test_prepare_ovi_rescene_input_v2.py \
  tests/oviv2/test_ovi_surface_attributes.py \
  tests/oviv2/test_rescene_input_bridge.py
/home/ww/miniconda3/envs/persist4d/bin/python \
  scripts/evaluation/prepare_ovi_rescene_input_v2.py native-self-test
/home/ww/miniconda3/bin/python -m py_compile \
  scripts/evaluation/prepare_ovi_rescene_input_v2.py
git diff --check
```

### Task 4: Source-bound executor and existing backend integration

**Files:**
- Create: `scripts/evaluation/rescene_pair_executor.py`
- Create: `tests/evaluation/test_rescene_pair_executor.py`
- Create: `configs/evaluation/rescene_two_visit_backend_persist4d_repro.json`
- Test: `tests/oviv2/test_rescene_backend_integration.py`

**Interfaces:**
- Retain existing backend arguments and accept one command-prefix option
  `--input-sidecar PATH`.
- Emit the existing output manifest keys and NPZ arrays
  `token_indices`, `query_masks`, `token_scores`, `query_scores`.
- Store raw Q x M logits/masks in a separately bound external artifact.

- [ ] Write RED unit tests with a CPU stub model for all input identities,
  strict model-key consumption, M x Q/Q x (C+1) shape handling, one transpose,
  raw-query identity, empty query pruning, no class-top-k duplication, exact
  Q x A expansion, output hashing, and atomic publication.
- [ ] Write one RED integration test that runs the existing ReScene backend with
  a fake executor sidecar and proves the public manifest/schema is unchanged.
- [ ] Implement source-bound Hydra composition, exact 796-key model extraction,
  strict load, explicit Point construction, global identity segments, query
  scoring, expansion, and runtime/memory receipt.
- [ ] Run:

```bash
/home/ww/miniconda3/bin/python -m pytest -q \
  tests/evaluation/test_rescene_pair_executor.py \
  tests/oviv2/test_rescene_backend_contract.py \
  tests/oviv2/test_rescene_backend_integration.py
/home/ww/miniconda3/bin/python -m py_compile \
  scripts/evaluation/rescene_pair_executor.py
git diff --check
```

### Task 5: B4/B5 relation diagnostics

**Files:**
- Create: `scripts/evaluation/compare_ovi_rescene_relations.py`
- Create: `tests/evaluation/test_compare_ovi_rescene_relations.py`

**Interfaces:**
- `rebuild_b4_relations(...) -> tuple[PairRelation, ...]`
- `classify_b5_relations(evidence, relations, sampling) -> diagnostics`
- `compare_relation_topologies(b4, b5) -> RelationDelta`

- [ ] Write RED tests for topology normalization, per-state counts,
  intersection/only sets, qualified/fallback split, collision fraction,
  contradictory one-to-one hypotheses, deterministic registration top-20, and
  mismatched pair/config/source rejection.
- [ ] Implement diagnostics around the unchanged geometric builder, reasoner,
  query projector, and registration module. Do not add a new matcher.
- [ ] Run:

```bash
/home/ww/miniconda3/bin/python -m pytest -q \
  tests/evaluation/test_compare_ovi_rescene_relations.py \
  tests/oviv2/test_query_instance_projection.py \
  tests/oviv2/test_geometric_pair_reasoner.py \
  tests/oviv2/test_two_visit_registration.py
/home/ww/miniconda3/bin/python -m py_compile \
  scripts/evaluation/compare_ovi_rescene_relations.py
git diff --check
```

### Task 6: Freeze executable code and one formal input configuration

**Files:**
- Create: `configs/evaluation/ovi_rescene_c2_repair_b5_v2.json`
- Update: design status to `FROZEN_BEFORE_RESULT_BEARING_B5`

- [ ] Run input-only calibration once: exactly 2048 M indices per visit, no GT
  and no checkpoint forward. Save compact residual evidence externally.
- [ ] Generate one formal config containing the selected tolerance, frozen asset
  records, source records, frame mapping, policies, external run root, and
  Office attempt count 0. Reject manual config values that disagree with the
  calibration receipt.
- [ ] Run all Task 1--5 focused tests, compile changed Python, and
  `git diff --check`.
- [ ] Primary agent reviews the complete code/config diff, source boundaries,
  shape semantics, provenance, memory behavior, and failure ordering.
- [ ] Commit design, implementation, tests, and exact config. Record this commit
  as `evaluated_code_sha`; no result-bearing inference is legal before it.

### Task 7: Build full C2-V2 input once and run one Apartment B5

**External outputs:**
- `${run_root}/adapter_pair/`
- `${run_root}/surface_attributes/{t0,t1}/`
- `${run_root}/model_input/`
- `${run_root}/backend/`

- [ ] Recheck GPU occupancy without terminating any process; choose the card
  with sufficient free memory and record the snapshot.
- [ ] Run the formal `build` once. Require exact parent hashes, unchanged
  snapshots, M RGB/normal coverage 1.0, valid 9-channel rows, complete native
  inverse, and `C2_V2_PASS`. Otherwise publish `C2_REPAIR_BLOCKED` and skip GPU.
- [ ] Invoke `run_rescene_pair_backend.py` once through the pinned Python and
  source-bound executor. This one complete forward is both smoke and final.
- [ ] Permit one retry only if the first receipt records a concrete bug or OOM;
  preserve the failed attempt and keep all scientific parameters unchanged.
- [ ] Audit the result in a new process and require finite Q x A evidence,
  original/expanded shapes, exact pair/checkpoint/input hashes, and both visits.

### Task 8: Rebuild B4 once, project B5, and publish the evidence package

**Files:**
- Create under `configs/evaluation/results/ovi_rescene_c2_repair_b5_v2/`:
  `reuse_ledger.json`, `attribute_coverage.json`, `input_contract_v2.json`,
  `sampling_map_summary.json`, `b5_backend_receipt.json`, `b4_relations.json`,
  `b5_relations.json`, `relation_delta.json`, optional
  `registration_diagnostics.json`, `decision.json`, and `artifact_manifest.json`.
- Create: `docs/superpowers/reports/2026-09-05-ovi-rescene-c2-repair-b5-results.md`
- Create: `docs/superpowers/reports/2026-09-05-ovi-rescene-c2-repair-b5-handoff.md`

- [ ] Confirm no complete trusted B4 relation artifact exists; if absent,
  rebuild exactly once and cache the full relations. Do not rebuild a second
  time to force the historical count 120.
- [ ] Project B5 once with frozen `ProjectionConfig`, compare unique topology,
  classify fallback/collision/contradiction, and run at most 20 eligible
  registration diagnostics.
- [ ] Emit exactly one legal final status with
  `gt_identity_evidence_available=false` and
  `paper_superiority_established=false`. Missing measurements remain null, not
  zero or estimates.
- [ ] Run the focused Task 1--5 tests once more, changed-file compilation, and
  `git diff --check`. Keep full suite status
  `FULL_SUITE_NOT_RUN_IMPACT_SCOPED` because existing public contracts remain
  untouched.
- [ ] Primary agent reviews all code, artifacts, commands, hashes, numbers,
  claim boundaries, Git scope, and external file identities.
- [ ] Commit compact results/reports, push
  `research/ovi-rescene-c2-repair-b5`, and verify local HEAD equals the remote
  branch SHA. Write the final remote check to an external upload receipt without
  introducing a self-hash loop.
