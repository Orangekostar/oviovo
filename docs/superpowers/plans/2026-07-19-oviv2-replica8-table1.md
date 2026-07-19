# OVIV2 Replica-8 Table 1 Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze one GT-independent OVIV2 algorithm, run it on all eight Replica scenes, aggregate Replica-8/7 metrics, and bind a VERIFIED OVIV2 result to the eight Replica cells in Table 1.

**Architecture:** Keep the current YOLO-World+SAM object cache and add the missing depth-normal `STRUCTURE` observation path inside `src/oviv2`; structure observations write semantic evidence but never entities. A multi-scene orchestrator materializes stride-10 views, validates identical frontend provenance, invokes the existing per-scene runner, aggregates direct evaluator metrics, and emits a hash-complete OVIV2 result manifest. Table 1 uses OVIV2-specific tokens so legacy OVIOVO artifacts cannot be imported.

**Tech Stack:** Python 3.10, NumPy, SciPy ndimage, Open3D, Pillow, pytest, ConceptGraphs YOLO-World+MobileSAM cache generator, TSV/JSON benchmark registry.

---

### Task 1: Depth-Normal Structural Observations

**Files:**
- Create: `src/oviv2/structure.py`
- Create: `tests/oviv2/test_structure.py`
- Modify: `src/oviv2/observations.py`
- Modify: `src/oviv2/__init__.py`

- [ ] **Step 1: Write failing synthetic tests**

```python
def test_structure_frontend_labels_planes_without_entities():
    observations = frontend.observe(frame, object_observations=())
    assert {item.kind for item in observations} == {ObservationKind.STRUCTURE}
    assert {item.label for item in observations} == {"wall", "floor", "ceiling"}

def test_structure_frontend_excludes_object_masks():
    observations = frontend.observe(frame, object_observations=(chair,))
    assert all(not (item.mask & chair.mask).any() for item in observations)
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest -q tests/oviv2/test_structure.py`
Expected: FAIL because `src.oviv2.structure` does not exist.

- [ ] **Step 3: Implement the standalone frontend**

Implement `DepthStructureConfig` and `DepthStructureFrontend`. Back-project valid depth, calculate central-difference camera-space normals, classify floor only below `cy` with `ny > threshold`, ceiling only above `cy` with `ny < -threshold`, and walls with `abs(ny) < wall_vertical_threshold`. Dilate and subtract all object masks before connected-component filtering. Convert masks through a shared `lift_mask_to_voxels(...)` helper and emit `FrameObservation(kind=STRUCTURE)` with frozen Replica IDs.

- [ ] **Step 4: Verify focused and architecture tests**

Run: `python -m pytest -q tests/oviv2/test_structure.py tests/oviv2/test_observations.py tests/architecture/test_oviv2_voxel_first.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/oviv2/structure.py src/oviv2/observations.py src/oviv2/__init__.py tests/oviv2/test_structure.py
git commit -m "feat: add OVIV2 structural observations"
```

### Task 2: Integrate Structure Into the Frozen Runner

**Files:**
- Modify: `scripts/run_oviv2_replica.py`
- Modify: `configs/oviv2_replica_room0.json`
- Modify: `tests/evaluation/test_run_oviv2_replica_cli.py`

- [ ] **Step 1: Add a failing runner test**

```python
def test_runner_fuses_structure_without_allocating_structure_entities(tmp_path):
    run(args)
    snapshot = VoxelMapSnapshot.load(output / "final/oviv2_voxel_snapshot.npz")
    assert any(candidate.label_id in {1, 2, 3} for key in keys for candidate in snapshot.evidence.semantic_candidates(key))
    assert all(entity["semantic_id"] not in {1, 2, 3} for entity in entities)
```

- [ ] **Step 2: Verify the test fails**

Run: `python -m pytest -q tests/evaluation/test_run_oviv2_replica_cli.py -k structure`
Expected: FAIL because the runner only forwards cached object observations.

- [ ] **Step 3: Append structure observations before runtime processing**

Construct `DepthStructureFrontend` once from JSON config. For every frame, obtain object observations from `CachedFrontendAdapter`, obtain structure observations with the object masks as exclusions, concatenate deterministically, and record object/structure counts separately in `timing.json`. Include the complete structural config in `run_manifest.json`.

- [ ] **Step 4: Verify runner tests**

Run: `python -m pytest -q tests/evaluation/test_run_oviv2_replica_cli.py tests/oviv2/test_runtime.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_oviv2_replica.py configs/oviv2_replica_room0.json tests/evaluation/test_run_oviv2_replica_cli.py
git commit -m "feat: fuse OVIV2 structural semantics"
```

### Task 3: Replica Stride-10 Views And Frontend Cache Orchestrator

**Files:**
- Create: `scripts/materialize_replica_stride_view.py`
- Create: `scripts/precompute_oviv2_replica_frontend.py`
- Create: `configs/oviv2_replica8.json`
- Create: `tests/evaluation/test_oviv2_replica_frontend_orchestrator.py`

- [ ] **Step 1: Write failing materialization and dry-run tests**

```python
def test_stride_view_links_exact_source_frames(tmp_path):
    summary = materialize(source, target, start=0, stop=2000, stride=10)
    assert summary["source_frame_ids"] == list(range(0, 2000, 10))
    assert (target / "results/frame000001.jpg").resolve().name == "frame000010.jpg"

def test_frontend_dry_run_uses_one_frozen_command_for_every_scene(tmp_path):
    commands = build_commands(manifest, gpu_ids=(0, 1))
    assert len(commands) == 8
    assert {command.config_hash for command in commands} == {commands[0].config_hash}
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest -q tests/evaluation/test_oviv2_replica_frontend_orchestrator.py`
Expected: FAIL because both scripts are absent.

- [ ] **Step 3: Implement atomic stride views**

Create 200 symlinked RGB/depth pairs with sampled filenames `frame000000..199` and a 200-row sampled `traj.txt`. Write `frame_manifest.json` containing source IDs and SHA256 values; reject broken source frames before publishing the target directory.

- [ ] **Step 4: Implement frontend orchestration and validation**

Generate ConceptGraphs Hydra commands using the frozen model/class paths already recorded by room0. Support `--dry-run`, `--scene`, and `--gpu`; after each process exits, validate 200 gzip-pickle files, mask resolution, vector lengths, class list hash, model hashes, and write `frontend_manifest.json`. Never read GT.

- [ ] **Step 5: Verify and commit**

Run: `python -m pytest -q tests/evaluation/test_oviv2_replica_frontend_orchestrator.py`
Expected: PASS.

```bash
git add scripts/materialize_replica_stride_view.py scripts/precompute_oviv2_replica_frontend.py configs/oviv2_replica8.json tests/evaluation/test_oviv2_replica_frontend_orchestrator.py
git commit -m "feat: orchestrate OVIV2 Replica frontends"
```

### Task 4: Multi-Scene OVIV2 Runner

**Files:**
- Create: `scripts/run_oviv2_replica8.py`
- Create: `tests/evaluation/test_run_oviv2_replica8.py`
- Modify: `scripts/run_oviv2_replica.py`

- [ ] **Step 1: Write failing scene-contract tests**

```python
def test_replica8_runner_requires_exact_frozen_scene_set(tmp_path):
    with pytest.raises(ValueError, match="eight scenes"):
        run_replica8(config_without_office4)

def test_replica8_runner_passes_identical_algorithm_config(tmp_path, monkeypatch):
    calls = capture_scene_calls(monkeypatch)
    run_replica8(config)
    assert {call.algorithm_hash for call in calls} == {calls[0].algorithm_hash}
```

- [ ] **Step 2: Verify failures**

Run: `python -m pytest -q tests/evaluation/test_run_oviv2_replica8.py`
Expected: FAIL because the multi-scene runner is absent.

- [ ] **Step 3: Implement scene config expansion and execution**

Read scene/GT mappings from `configs/evaluation/manifests/replica8.json`; permit only paths to vary per scene. Invoke `run_oviv2_replica.run` in frozen manifest order, skip already VERIFIED outputs only after checksum validation, and write an atomic batch manifest with per-scene run-manifest hashes.

- [ ] **Step 4: Verify and commit**

Run: `python -m pytest -q tests/evaluation/test_run_oviv2_replica8.py tests/evaluation/test_run_oviv2_replica_cli.py`
Expected: PASS.

```bash
git add scripts/run_oviv2_replica8.py scripts/run_oviv2_replica.py tests/evaluation/test_run_oviv2_replica8.py
git commit -m "feat: run OVIV2 across Replica-8"
```

### Task 5: Room0 Freeze Gate

**Files:**
- Generate: `outputs/oviv2_replica8_freeze/room0-smoke20/`
- Generate: `outputs/oviv2_replica8_freeze/room0-200f/`

- [ ] **Step 1: Run 20 frames with the structural branch**

Run: `python scripts/run_oviv2_replica.py --config configs/oviv2_replica_room0.json --output outputs/oviv2_replica8_freeze/room0-smoke20 --num-frames 20`
Expected: revision 20, finite metrics, nonzero structure evidence, no structural entity.

- [ ] **Step 2: Run 200 frames and compare with the pre-structure baseline**

Run: `python scripts/run_oviv2_replica.py --config configs/oviv2_replica_room0.json --output outputs/oviv2_replica8_freeze/room0-200f --num-frames 200`
Expected: semantic coverage and mIoU exceed `outputs/oviv2_room0_200f_final`; AP/F5 remain finite. Freeze the resulting algorithm/config hash before any held-out scene run.

### Task 6: Generate Frontends And Run Replica-8

**Files:**
- Generate: `/home/ww/vv/dataset/Replica/*_s10_200f/`
- Generate: `outputs/oviv2_replica8_frozen/<scene>/`

- [ ] **Step 1: Materialize all missing stride views**

Run: `python scripts/materialize_replica_stride_view.py --config configs/oviv2_replica8.json`
Expected: eight validated 200-frame views; existing room0 is validated rather than replaced.

- [ ] **Step 2: Generate seven missing frontend caches**

Run one frozen command per scene through `scripts/precompute_oviv2_replica_frontend.py`; use available GPUs without interrupting unrelated processes. Expected: 200 valid cache files and one manifest per scene.

- [ ] **Step 3: Run all eight scenes**

Run: `python scripts/run_oviv2_replica8.py --config configs/oviv2_replica8.json --output outputs/oviv2_replica8_frozen`
Expected: every scene revision 200, nonempty mesh, finite six metrics, and verified checksums.

- [ ] **Step 4: Repeat evaluation from every final snapshot**

Expected: every `metrics.json`, aligned NPY, and GT audit PLY is byte-identical to the original evaluation.

### Task 7: Aggregate Replica-8/7 And Finalize OVIV2 Result

**Files:**
- Create: `src/evaluation/oviv2_result.py`
- Create: `scripts/evaluation/finalize_oviv2_replica_result.py`
- Create: `tests/evaluation/test_oviv2_result.py`
- Generate: `docs/paper/results/oviv2/replica/20260719-s10-200f/result.json`

- [ ] **Step 1: Write failing aggregation/provenance tests**

```python
def test_aggregate_accepts_exact_eight_direct_oviv2_metrics():
    result = aggregate_oviv2_replica(scene_metrics)
    assert result["replica_7_heldout"]["scene_ids"] == list(REPLICA7_SCENES)

def test_finalize_rejects_mixed_algorithm_hashes():
    with pytest.raises(ValueError, match="algorithm hash"):
        finalize(scene_runs_with_one_changed_hash)
```

- [ ] **Step 2: Implement strict aggregation and result finalization**

Read only evaluator `metrics.json` values. Require exact scenes, identical algorithm/vocabulary/manifest hashes, revision 200, valid per-run checksums, and repeated-evaluation hashes. Emit method `{key: "OVIV2", display_label: "OVIV2", mode: "online"}` plus eight Replica token bindings.

- [ ] **Step 3: Verify, generate, and commit**

Run: `python -m pytest -q tests/evaluation/test_oviv2_result.py`
Expected: PASS.

```bash
git add src/evaluation/oviv2_result.py scripts/evaluation/finalize_oviv2_replica_result.py tests/evaluation/test_oviv2_result.py docs/paper/results/oviv2/replica/20260719-s10-200f/result.json
git commit -m "feat: finalize OVIV2 Replica result"
```

### Task 8: OVIV2 Table 1 Tokens And Import

**Files:**
- Modify: `docs/paper/benchmark_tokens.tsv`
- Modify: `docs/paper/benchmark_tables.md`
- Modify: `docs/paper/benchmark_tables.tex`
- Modify: `tools/import_benchmark_results.py`
- Modify: `tests/evaluation/test_import_benchmark_results.py`
- Modify: `tests/test_benchmark_table_package.py`
- Generate: `docs/paper/benchmark_tables_baselines.md`
- Generate: `docs/paper/benchmark_tables_baselines.tex`

- [ ] **Step 1: Write failing OVIV2 import tests**

Replace only the twelve Table 1 method tokens `T1_OVIOVO_*` with `T1_OVIV2_*`; keep OVIOVO tokens in other tables unchanged. Assert OVIV2 VERIFIED results import, legacy OVIOVO artifacts remain rejected, and all eight Replica placeholders resolve.

- [ ] **Step 2: Implement method/token migration**

Change the Table 1 display row to `OVIV2`. Add `OVIV2` to the Table 1 importer allowlist. Do not weaken status, pointer, precision, dataset, finite-value, or duplicate-binding checks.

- [ ] **Step 3: Import all verified baseline and OVIV2 results**

Run `tools/import_benchmark_results.py` with the four existing verified baseline result files and the new OVIV2 result. Expected: all Replica-8/7 cells render as numbers; ScanNet200-5 cells remain explicit UNFILLED until licensed data exists.

- [ ] **Step 4: Verify and commit**

Run: `python -m pytest -q tests/evaluation/test_import_benchmark_results.py tests/test_benchmark_table_package.py`
Expected: PASS.

```bash
git add docs/paper/benchmark_tokens.tsv docs/paper/benchmark_tables.md docs/paper/benchmark_tables.tex docs/paper/benchmark_tables_baselines.md docs/paper/benchmark_tables_baselines.tex tools/import_benchmark_results.py tests/evaluation/test_import_benchmark_results.py tests/test_benchmark_table_package.py
git commit -m "docs: fill OVIV2 Replica Table 1 results"
```

### Task 9: Final Verification And ScanNet Audit

**Files:**
- Modify only if verification exposes an OVIV2 defect.

- [ ] **Step 1: Run the complete relevant suite**

Run: `python -m pytest -q tests/oviv2 tests/architecture tests/evaluation tests/test_benchmark_table_package.py`
Expected: PASS.

- [ ] **Step 2: Verify Table 1 provenance**

Run the benchmark package verifier. Expected: all eight OVIV2 Replica token rows are VERIFIED and point to the tracked OVIV2 result JSON; no value is copied from a PLY or handwritten.

- [ ] **Step 3: Audit ScanNet200-5 availability**

Require official ScanNet RGB-D/pose/mesh/label inputs and publisher-approved access. If absent, preserve all four OVIV2 ScanNet token rows as UNFILLED with the existing license blocker; never substitute another dataset or paper-reported number.
