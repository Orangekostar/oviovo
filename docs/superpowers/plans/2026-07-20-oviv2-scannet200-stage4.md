# OVIV2 ScanNet200 Stage4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Independently generate and verify OVIV2 ScanNet200-5 results, then fill the four remaining OVIV2 Table 1 tokens.

**Architecture:** A frozen ScanNet execution manifest supplies explicit RGB-D frame IDs and hashes to a dataset adapter. OVIV2 independently generates YOLO-World+MobileSAM observations and 200-class RADSeg caches, reuses the audited Stage3 mapper, evaluates semantic and instance heads separately in official axis-aligned coordinates, and finalizes a five-scene hash-bound result.

**Tech Stack:** Python 3.10, NumPy, Pillow, Open3D, plyfile, SciPy, pytest, YOLO-World, MobileSAM, RADSeg/RADIO, SigLIP2, JSON/SHA-256 provenance.

---

### Task 1: Freeze Stage4 Inputs and Load ScanNet RGB-D Frames

**Files:**
- Create: `data/input/scannet200_classes.json`
- Create: `data/input/scannet200_classes.txt`
- Create: `src/datasets/scannet200.py`
- Create: `scripts/evaluation/freeze_oviv2_scannet200_manifest.py`
- Create: `tests/datasets/test_scannet200.py`
- Create: `tests/evaluation/test_freeze_oviv2_scannet200_manifest.py`
- Generate: `configs/evaluation/manifests/oviv2_scannet200_5.json`

- [ ] **Step 1: Write failing dataset tests**

Define the desired API with a synthetic RGB image, depth image, depth intrinsics, and two explicit poses:

```python
dataset = ScanNet200Dataset(scene_root, source_frame_ids=(0, 10), expected_image_shape=(2, 2))
frame = dataset[1]
assert frame.frame_id == 10
assert frame.rgb.shape == (2, 2, 3)
assert frame.depth.dtype == np.float32
np.testing.assert_allclose(frame.depth, depth_mm / 1000.0)
```

Add independent tests that reject an unlisted index, missing file, non-finite pose, non-invertible pose, wrong hash, wrong depth shape, and symlinked input.

- [ ] **Step 2: Run and verify RED**

Run: `/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest tests/datasets/test_scannet200.py -q`

Expected: FAIL because `src.datasets.scannet200` does not exist.

- [ ] **Step 3: Implement the dataset adapter**

```python
class ScanNet200Dataset:
    def __init__(self, root: str | Path, *, source_frame_ids: Sequence[int], input_hashes: Mapping[int, Mapping[str, str]] | None = None, expected_image_shape: tuple[int, int] = (480, 640), depth_scale: float = 1000.0) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Frame: ...
```

Load `color/<id>.jpg`, `depth/<id>.png`, `pose/<id>.txt`, and `intrinsic/intrinsic_depth.txt`. Resize RGB with `Image.Resampling.BILINEAR`, preserve depth pixels, divide uint16 depth by 1000, and return camera-to-world poses unchanged. Validate selected paths and hashes before returning frames.

- [ ] **Step 4: Verify GREEN**

Run the dataset test command again. Expected: PASS.

- [ ] **Step 5: Write failing manifest-freezer tests**

Require exact scene order, explicit source IDs, per-frame color/depth/pose hashes, intrinsic hashes, official GT hashes, official numeric class IDs, and byte-identical output. A small synthetic fixture verifies selection and invalid-pose omission; after the real freezer command, validate the real counts separately:

```python
assert [len(scene["source_frame_ids"]) for scene in manifest["scenes"]] == [238, 465, 444, 190, 147]
assert 4650 not in manifest["scenes"][1]["source_frame_ids"]
assert manifest["vocabulary"]["class_count"] == 200
```

- [ ] **Step 6: Run and verify RED**

Run: `/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest tests/evaluation/test_freeze_oviv2_scannet200_manifest.py -q`

Expected: FAIL because the freezer script is missing.

- [ ] **Step 7: Implement and run the manifest freezer**

CLI: `--raw-manifest PATH --exported-root PATH --official-gt-root PATH --official-constants PATH --classes-json PATH --output PATH --replace`.

Use official `VALID_CLASS_IDS_200` and `CLASS_LABELS_200` to generate the tracked JSON/TXT assets in official order. Do not reuse the alphabetically ordered ConceptGraphs class file. Require exact JSON/TXT agreement, select `range(0, frame_count, 10)`, discard only invalid poses, hash every execution input, and atomically publish sorted JSON.

Run: `/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python scripts/evaluation/freeze_oviv2_scannet200_manifest.py --raw-manifest /home/ww/oviovo_baseline_runs/20260714_non_oviovo_baselines/scannet200/scannet200_5.json --exported-root /home/ww/oviovo_benchmark_assets/scannet200_release/exported --official-gt-root /home/ww/oviovo_benchmark_assets/scannet200_release/scannet200_official/val --official-constants /home/ww/oviovo_baseline_builds/scannet-official/BenchmarkScripts/ScanNet200/scannet200_constants.py --classes-json data/input/scannet200_classes.json --output configs/evaluation/manifests/oviv2_scannet200_5.json`

- [ ] **Step 8: Verify and commit**

Run both Task 1 test files. Expected: PASS.

Commit: `git add data/input/scannet200_classes.json data/input/scannet200_classes.txt src/datasets/scannet200.py scripts/evaluation/freeze_oviv2_scannet200_manifest.py tests/datasets/test_scannet200.py tests/evaluation/test_freeze_oviv2_scannet200_manifest.py configs/evaluation/manifests/oviv2_scannet200_5.json && git commit -m "feat: freeze OVIV2 ScanNet200 inputs"`

### Task 2: Generate an Independent OVIV2 Frontend Cache

**Files:**
- Create: `scripts/materialize_oviv2_scannet200_view.py`
- Create: `scripts/precompute_oviv2_scannet200_frontend.py`
- Create: `tests/evaluation/test_materialize_oviv2_scannet200_view.py`
- Create: `tests/evaluation/test_precompute_oviv2_scannet200_frontend.py`
- Create: `configs/oviv2_scannet200_stage4.json`

- [ ] **Step 1: Write failing view tests**

```python
materialize_scene(manifest, "scene0011_00", output)
assert (output / "results/frame000000.jpg").is_file()
assert (output / "results/depth000000.png").is_file()
assert json.loads((output / "view_manifest.json").read_text())["source_frame_ids"] == [0, 10]
```

Assert deterministic RGB resize, copied uint16 depth, ordered `traj.txt`, exact output file set, source hash verification, refusal to overwrite, and no symlink publication.

- [ ] **Step 2: Verify RED, implement, and verify GREEN**

Run: `/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest tests/evaluation/test_materialize_oviv2_scannet200_view.py -q`

Implement atomic directory publication through a sibling temporary directory. Run again; expected PASS.

- [ ] **Step 3: Write failing frontend tests**

```python
commands = build_commands(config, manifest, gpu_ids=(0, 1, 2))
assert commands[0].scene == "scene0011_00"
assert commands[0].cache_dir.is_relative_to(Path(config["view_root"]).resolve())
assert not commands[0].cache_dir.is_relative_to(Path("/home/ww/oviovo_baseline_runs"))
```

Validator fixtures bind source-frame order, the 197 object classes after excluding wall/floor/ceiling, 480x640 masks, finite CLIP features, model hashes, input-manifest hash, and per-frame cache hashes.

- [ ] **Step 4: Verify RED, implement, and verify GREEN**

Adapt the proven Replica orchestration but generate outputs only under the Stage4 root. GPU queues run pinned `streamlined_detections.py` against newly materialized views. Add `--scene`, repeatable `--gpu`, and `--dry-run`.

Run: `/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest tests/evaluation/test_precompute_oviv2_scannet200_frontend.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Commit: `git add scripts/materialize_oviv2_scannet200_view.py scripts/precompute_oviv2_scannet200_frontend.py tests/evaluation/test_materialize_oviv2_scannet200_view.py tests/evaluation/test_precompute_oviv2_scannet200_frontend.py configs/oviv2_scannet200_stage4.json && git commit -m "feat: add independent ScanNet200 frontend"`

### Task 3: Generalize Dense Semantics to 200 Classes

**Files:**
- Modify: `scripts/precompute_oviv2_dense_semantics.py`
- Modify: `tests/evaluation/test_precompute_oviv2_dense_semantics.py`

- [ ] **Step 1: Add failing variable-vocabulary tests**

```python
preflight = _preflight(config_path, requested_frames=2, resume=False)
assert len(preflight.classes) == 200
metadata = _metadata(worker_response(class_count=200), preflight)
assert metadata.class_count == 200
```

Add failures for 199/201 classes, worker order mismatch, top-k greater than class count, and a source-frame list differing from the scene manifest.

- [ ] **Step 2: Run and verify RED**

Run the focused dense precompute test. Expected: FAIL at the fixed 41-class checks.

- [ ] **Step 3: Implement minimal generalization**

Remove `CLASS_COUNT`; use `len(preflight.classes)`. Select `ReplicaRoom0Dataset` for `Replica` and `ScanNet200Dataset` for `ScanNet200`. For ScanNet, read explicit scene source IDs; retain arithmetic selection for Replica. Preserve strict hash, resource, and resume checks.

- [ ] **Step 4: Verify and commit**

Run: `/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest tests/evaluation/test_precompute_oviv2_dense_semantics.py tests/evaluation/test_run_oviv2_replica_cli.py -q`

Commit: `git add scripts/precompute_oviv2_dense_semantics.py tests/evaluation/test_precompute_oviv2_dense_semantics.py && git commit -m "feat: support ScanNet200 dense semantics"`

### Task 4: Run Stage3 Mapping on Explicit ScanNet Frames

**Files:**
- Modify: `scripts/run_oviv2_replica.py`
- Create: `tests/evaluation/test_run_oviv2_scannet_cli.py`
- Create: `configs/oviv2_scannet200_stage4_base.json`

- [ ] **Step 1: Write failing ScanNet runner tests**

```python
dataset, benchmark, source_ids, *_ = _preflight(config, 2, skip_evaluation=True)
assert benchmark["dataset"] == "ScanNet200"
assert source_ids == [0, 20]
assert dataset[1].frame_id == 20
```

Run the CLI with `--skip-evaluation`; require snapshot, registry, owner/dense/fused meshes, timing, run manifest, cache-index/source-ID records, and checksums. Use hostile GT paths and assert mapping does not open them.

- [ ] **Step 2: Run and verify RED**

Expected: FAIL because the runner rejects non-Replica manifests.

- [ ] **Step 3: Implement dataset dispatch and explicit IDs**

Dispatch on `benchmark["dataset"]`, obtain one selected scene record, and use explicit IDs for ScanNet. Generalize dense validation. Add dataset name, source-ID hash, and input-manifest hash to the run manifest outside the cross-scene algorithm hash. Derive `configs/oviv2_scannet200_stage4_base.json` from the audited Stage3 fused config, changing only dataset, vocabulary, path, count, and capacity fields.

- [ ] **Step 4: Verify and commit**

Run: `/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest tests/evaluation/test_run_oviv2_scannet_cli.py tests/evaluation/test_run_oviv2_replica_cli.py tests/evaluation/test_run_oviv2_replica8.py -q`

Commit: `git add scripts/run_oviv2_replica.py tests/evaluation/test_run_oviv2_scannet_cli.py configs/oviv2_scannet200_stage4_base.json && git commit -m "feat: run OVIV2 Stage3 on ScanNet200"`

### Task 5: Evaluate Separate Semantic and Instance Heads

**Files:**
- Modify: `src/evaluation/baselines/static_metrics.py`
- Create: `scripts/evaluation/evaluate_oviv2_scannet200.py`
- Create: `tests/evaluation/test_evaluate_oviv2_scannet200.py`
- Modify: `scripts/run_oviv2_replica.py`

- [ ] **Step 1: Write failing separate-head tests**

```python
metrics = evaluate_static_predictions(semantic_prediction=semantic_snapshot, instance_prediction=instance_snapshot, geometry_prediction=geometry_snapshot, ground_truth=ground_truth, semantic_vocabulary=classes, instance_vocabulary=classes, distance_threshold_m=0.05)
assert metrics["semantic"]["miou"] == 1.0
assert metrics["instance"]["predicted_instance_count"] == 1
```

Use two semantic labels owned by one entity and require one predicted instance. Add exact 5 cm and scene mismatch tests.

- [ ] **Step 2: Verify RED, implement, and verify GREEN**

`evaluate_static_snapshot` becomes a wrapper passing the same snapshot for all heads. The new API validates scene IDs, then calls the existing metric functions with the appropriate snapshot.

- [ ] **Step 3: Write failing ScanNet evaluator tests**

Create a tiny official PLY with non-contiguous class IDs and instances. Apply a known axis-alignment transform and require perfect metrics. Require fused semantic grouping, ownership instance grouping, deterministic confidence, GT access only in evaluation, and deterministic audit files.

- [ ] **Step 4: Implement the evaluator**

CLI: `--snapshot PATH --entity-info PATH --gt-ply PATH --metadata PATH --manifest PATH --scene ID --output PATH --min-instance-points 100 --semantic-head fused_uncertainty --fusion-entity-weight-scale 0.49`.

Transform with `points @ axisAlignment[:3,:3].T + axisAlignment[:3,3]`; group semantic points by positive semantic ID and instances by positive entity ID; load official labels/instances; atomically write metrics, per-class JSON, aligned arrays, and colored PLY audits.

- [ ] **Step 5: Dispatch and commit**

Select evaluator arguments from manifest dataset while preserving Replica behavior.

Run: `/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest tests/evaluation/test_evaluate_oviv2_scannet200.py tests/evaluation/test_evaluate_oviv2_replica_cli.py tests/evaluation/test_baseline_static_metrics.py -q`

Commit: `git add src/evaluation/baselines/static_metrics.py scripts/evaluation/evaluate_oviv2_scannet200.py tests/evaluation/test_evaluate_oviv2_scannet200.py scripts/run_oviv2_replica.py && git commit -m "feat: evaluate OVIV2 on ScanNet200"`

### Task 6: Add Five-Scene Batch and Strict Finalizer

**Files:**
- Create: `src/evaluation/oviv2_scannet200_result.py`
- Create: `scripts/run_oviv2_scannet200_5.py`
- Create: `scripts/evaluation/finalize_oviv2_scannet200_result.py`
- Create: `tests/evaluation/test_run_oviv2_scannet200_5.py`
- Create: `tests/evaluation/test_finalize_oviv2_scannet200_result.py`

- [ ] **Step 1: Write failing batch tests**

Require exact five-scene order, scene-specific counts/roots/cache/GT, one algorithm hash, fresh outputs, validated resume, and a batch manifest hashing batch/base/scene configs, runs, and evaluations.

- [ ] **Step 2: Implement batch runner and verify GREEN**

Use the Replica-8 batch structure but read explicit counts and paths. Reject any unverified pre-existing scene output.

- [ ] **Step 3: Write failing finalizer tests**

```python
aggregate = aggregate_oviv2_scannet200(scene_metrics)
assert aggregate["scannet200_5_heldout"]["scene_count"] == 5
assert aggregate["scannet200_5_heldout"]["semantic"]["miou"] == pytest.approx(expected)
```

Require complete revisions, shared clean commit, shared algorithm/frontend/dense/vocabulary contracts, exact artifacts, fresh byte-identical repeat evaluation, and finite metrics. Reject every violated condition independently.

- [ ] **Step 4: Implement finalizer, verify, and commit**

Result dataset is `ScanNet200`, split is `scannet200_5_heldout`, and metric paths match the importer.

Run both new focused test files. Expected: PASS.

Commit: `git add src/evaluation/oviv2_scannet200_result.py scripts/run_oviv2_scannet200_5.py scripts/evaluation/finalize_oviv2_scannet200_result.py tests/evaluation/test_run_oviv2_scannet200_5.py tests/evaluation/test_finalize_oviv2_scannet200_result.py && git commit -m "feat: finalize OVIV2 ScanNet200 results"`

### Task 7: Run Smoke, GPU Preprocessing, and Full Mapping

**External outputs:**
- `/home/ww/oviovo_scannet_stage4_views/`
- `/home/ww/oviovo_frontend_cache/scannet200_stage4/`
- `/home/ww/oviovo_dense_cache/scannet200_stage4/`
- `/home/ww/oviovo_final_outputs/oviv2_scannet200_stage4_<commit>/`

- [ ] **Step 1: Verify clean implementation**

Run: `git status --short && /home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q -p no:cacheprovider tests/oviv2 tests/evaluation tests/datasets tests/test_benchmark_table_package.py`

Expected: clean status and all PASS.

- [ ] **Step 2: Preprocess `scene0011_00` independently**

Materialize its Stage4 view, run its YOLO-World+MobileSAM cache on GPU0, then run its 238-frame 200-class RADSeg cache on GPU0. Validate both manifests and all hashes.

- [ ] **Step 3: Run the 20-frame smoke gate**

Run: `/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python scripts/run_oviv2_replica.py --config /home/ww/oviovo_scannet_stage4_configs/scene0011_00.json --num-frames 20 --output /home/ww/oviovo_final_outputs/oviv2_scannet200_stage4_smoke_scene0011_20f`

Require non-zero frontend observations, entities, dense updates, TSDF blocks, mesh vertices, matched GT vertices, and finite metrics.

- [ ] **Step 4: Generate remaining caches on GPUs 0/1/2**

Run frontend GPU queues, then one dense worker per GPU with distinct scenes. Validate all 1,484 frontend and dense cache files before mapping.

- [ ] **Step 5: Run five mapping scenes**

Run: `/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python scripts/run_oviv2_scannet200_5.py --config configs/oviv2_scannet200_stage4.json --output /home/ww/oviovo_final_outputs/oviv2_scannet200_stage4_<commit>`

Monitor CPU, memory, block usage, and logs. Resume only complete hash-validated scenes.

- [ ] **Step 6: Finalize**

Run the strict finalizer into `docs/paper/results/oviv2/scannet200/<run-id>/result.json`. Require status `VERIFIED`, five byte-identical fresh repeats, and finite mIoU/AP25/AP50/F5.

### Task 8: Fill Table 1 and Verify the Paper Package

**Files:**
- Modify: `docs/paper/benchmark_tokens.tsv`
- Modify: `docs/paper/benchmark_tables_baselines.md`
- Modify: `docs/paper/benchmark_tables_baselines.tex`
- Add: `docs/paper/results/oviv2/scannet200/<run-id>/result.json`

- [ ] **Step 1: Import the VERIFIED result**

Run `scripts/evaluation/import_benchmark_results.py` with the current verified result set and the new result. Do not edit rendered values manually.

- [ ] **Step 2: Verify exactly four tokens changed**

Run: `rg -n 'T1_OVIV2_SCANNET5_(MIOU|AP25|AP50|F5)' docs/paper/benchmark_tokens.tsv`

Require four `VERIFIED` rows with JSON pointers into the new result and no OVIV2 ScanNet placeholders in rendered tables.

- [ ] **Step 3: Run final verification**

Run: `/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q -p no:cacheprovider tests/oviv2 tests/evaluation tests/datasets tests/test_benchmark_table_package.py && git diff --check && git status --short`

- [ ] **Step 4: Commit result and table**

Commit only the result plus importer-generated registry/Markdown/LaTeX files with message `docs: fill OVIV2 ScanNet200 Table 1 results`.
